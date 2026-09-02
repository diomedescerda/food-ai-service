"""F30: clasificador ligero sobre embeddings CLIP congelados.

CLIP frozen (requires_grad=False) -> embedding 512 -> Linear | MLP -> clases.
Train: food-bench-v1 + prototype-src (licencias permisivas) · Test: food-us
y Food-101 (interno) — sin leakage (datasets distintos).

Uso: python scripts/f30_train.py --group rice,french_fries,fried_chicken [--mlp] [--epochs 40]
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]

TRAIN_DIRS = [
    BASE / "datasets/food-bench-v1",
    BASE / "datasets/prototype-src",
    BASE / "datasets/multi-food/images",
    BASE / "tests/assets",
]
TEST_DIRS = {
    "food-us": BASE / "datasets/food-us-v0.1/images/train",
    "food101": BASE / "datasets/food101-subset",
}


def collect_train(classes: set[str]) -> list[tuple[str, str]]:
    items = []
    for d in TRAIN_DIRS:
        if not d.exists():
            continue
        for img in d.rglob("*.jpg"):
            cls = img.parent.name
            if cls == "images" and d.name == "multi-food":
                cls = {"mf_003": "hamburger", "mf_010": "french_fries", "mf_011": "pizza"}.get(img.stem, "")
            if cls in classes:
                items.append((str(img), cls))
    return items


def collect_test(classes: set[str]) -> dict[str, list[tuple[str, str]]]:
    out = {}
    for name, d in TEST_DIRS.items():
        if not d.exists():
            continue
        items = []
        if name == "food-us":
            for img in sorted(d.glob("*.jpg")):
                cls = img.stem.rsplit("_", 1)[0]
                if cls in classes:
                    items.append((str(img), cls))
        else:
            for img in sorted(d.rglob("*.jpg")):
                cls = img.parent.name
                if cls in classes:
                    items.append((str(img), cls))
        out[name] = items
    return out


def embed_batch(clf, paths: list[str], batch: int = 32) -> np.ndarray:
    import torch  # noqa: PLC0415

    embs = []
    for i in range(0, len(paths), batch):
        chunk = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inputs = clf._processor(images=chunk, return_tensors="pt")
        with torch.no_grad():
            feats = clf._features(clf._model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.numpy())
    return np.concatenate(embs)


def train_model(X, y, n_classes: int, mlp: bool, epochs: int, seed: int = 42):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    if mlp:
        model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))
    else:
        model = nn.Linear(512, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    Xt = torch.tensor(X[train_idx], dtype=torch.float32)
    yt = torch.tensor(y[train_idx], dtype=torch.long)
    Xv = torch.tensor(X[val_idx], dtype=torch.float32)
    yv = torch.tensor(y[val_idx], dtype=torch.long)

    best_acc, best_state, best_epoch = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xv).argmax(1) == yv).float().mean())
        if acc > best_acc:
            best_acc, best_state, best_epoch = acc, {k: v.clone() for k, v in model.state_dict().items()}, ep
    model.load_state_dict(best_state)
    return model, best_acc, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=None, help="comas: rice,french_fries,fried_chicken")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()

    classes = set(args.group.split(",")) if args.group else None
    if classes is None:
        all_classes = sorted({p.parent.name for d in TRAIN_DIRS if d.exists() for p in d.rglob("*.jpg")})
        classes = {c for c in all_classes if c != "images"}

    train_items = collect_train(classes)
    if not train_items:
        print("sin datos de train para:", classes)
        return
    # balancear: limitar a 8 imágenes por clase (las clases tienen 8-10)
    per_class: dict[str, list] = {}
    for p, c in train_items:
        per_class.setdefault(c, []).append(p)
    train_paths = [p for ps in per_class.values() for p in ps[:8]]
    train_labels = [c for c, ps in per_class.items() for _ in ps[:8]]

    print(f"train: {len(train_paths)} imágenes, {len(classes)} clases, distribución: {dict(Counter(train_labels))}")
    t0 = time.perf_counter()
    X = embed_batch(clf, train_paths)
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    y = np.array([label_map[c] for c in train_labels])

    model, val_acc, best_ep = train_model(X, y, len(classes), args.mlp, args.epochs)
    print(f"entrenado: {time.perf_counter()-t0:.0f}s · val_acc={val_acc*100:.1f}% (epoch {best_ep}) · modelo={'MLP' if args.mlp else 'Linear'}")

    import torch  # noqa: PLC0415

    model.eval()
    inv_map = {v: k for k, v in label_map.items()}
    for tname, items in collect_test(classes).items():
        if not items:
            continue
        paths = [p for p, _ in items]
        labels = [c for _, c in items]
        Xt = torch.tensor(embed_batch(clf, paths), dtype=torch.float32)
        with torch.no_grad():
            preds = [inv_map[int(i)] for i in model(Xt).argmax(1)]
        correct = sum(1 for p, l in zip(preds, labels) if p == l)
        print(f"  TEST {tname:<8} {correct}/{len(labels)} = {correct/len(labels)*100:.1f}%")
        for gt in sorted(set(labels)):
            idxs = [i for i, l in enumerate(labels) if l == gt]
            hits = sum(1 for i in idxs if preds[i] == gt)
            print(f"    {gt:<16} {hits}/{len(idxs)}")

    out = BASE / "benchmarks/f30/reports"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"f30_exp_{'mlp' if args.mlp else 'linear'}_{'-'.join(sorted(classes))[:40]}.json", "w", encoding="utf-8") as fh:
        json.dump({"group": sorted(classes), "mlp": args.mlp, "train": len(train_paths),
                   "val_acc": val_acc, "best_epoch": best_ep}, fh, indent=2)


if __name__ == "__main__":
    main()