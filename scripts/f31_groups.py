"""F31: ConfusionGroup â€” entrenamiento y evaluaciÃ³n de grupos de confusiÃ³n.

Arquitectura: legacy detecta la clase candidata â†’ si pertenece a un grupo â†’
clasificador especializado (Linear sobre CLIP frozen) â†’ gating por confianza
softmax â†’ final. Grupos derivados de errores reales (no categorÃ­as).

Uso: python scripts/f31_groups.py [--train] [--eval] [--global]
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
MODELS = BASE / "data/models/f31"
MODELS.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "rice_fries_fried_chicken": ["rice", "french_fries", "fried_chicken"],
    "pizza_naan": ["pizza", "naan"],
    "hamburger_sandwich": ["hamburger", "sandwich"],
    "taco_quesadilla_nachos": ["taco", "quesadilla", "nachos"],
    "fried_chicken_nuggets": ["fried_chicken", "chicken_nuggets"],
}
# GT disponible en food-us por grupo (test independiente, sin leakage)
FOOD_US_GT = {
    "rice_fries_fried_chicken": ["french_fries", "fried_chicken"],
    "pizza_naan": ["pizza"],
    "hamburger_sandwich": ["hamburger", "sandwich"],
    "taco_quesadilla_nachos": [],
    "fried_chicken_nuggets": ["fried_chicken"],
}

TRAIN_DIRS = [
    BASE / "datasets/food-bench-v1",
    BASE / "datasets/prototype-src",
    BASE / "datasets/multi-food/images",
    BASE / "tests/assets",
]


def collect_images(classes: set[str], limit: int = 12) -> list[tuple[str, str]]:
    items = []
    for d in TRAIN_DIRS:
        if not d.exists():
            continue
        for img in d.rglob("*.jpg"):
            cls = img.parent.name
            if d.name == "multi-food" and cls == "images":
                cls = {"mf_003": "hamburger", "mf_010": "french_fries", "mf_011": "pizza"}.get(img.stem, "")
            if cls in classes:
                items.append((str(img), cls))
    per: dict[str, list] = {}
    for p, c in items:
        per.setdefault(c, []).append(p)
    return [(p, c) for c, ps in per.items() for p in ps[:limit] for _ in [c]][:0] or [
        (p, c) for c, ps in per.items() for p in ps[:limit]
    ]


def food_us_items(classes: set[str]) -> list[tuple[str, str]]:
    d = BASE / "datasets/food-us-v0.1/images/train"
    out = []
    for img in sorted(d.glob("*.jpg")):
        cls = img.stem.rsplit("_", 1)[0]
        if cls in classes:
            out.append((str(img), cls))
    return out


def embed(clf, paths: list[str]) -> np.ndarray:
    import torch  # noqa: PLC0415

    embs = []
    for i in range(0, len(paths), 32):
        chunk = [Image.open(p).convert("RGB") for p in paths[i:i + 32]]
        inputs = clf._processor(images=chunk, return_tensors="pt")
        with torch.no_grad():
            feats = clf._features(clf._model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.numpy())
    return np.concatenate(embs)


def train_group(clf, classes: list[str], seed: int = 42) -> tuple[object, float, dict]:
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    torch.manual_seed(seed)
    items = collect_images(set(classes))
    paths = [p for p, _ in items]
    labels = [c for _, c in items]
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    X = embed(clf, paths)
    y = np.array([label_map[c] for c in labels])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    model = nn.Linear(512, len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X[train_idx], dtype=torch.float32)
    yt = torch.tensor(y[train_idx], dtype=torch.long)
    Xv = torch.tensor(X[val_idx], dtype=torch.float32)
    yv = torch.tensor(y[val_idx], dtype=torch.long)
    best_acc, best_state = 0.0, None
    for _ in range(120):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xv).argmax(1) == yv).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    dist = dict(Counter(labels))
    return model, best_acc, {"train": len(paths), "distribution": dist, "label_map": label_map}


def eval_group(clf, model, label_map: dict, classes: list[str], items: list[tuple[str, str]]) -> dict:
    import torch  # noqa: PLC0415

    if not items:
        return {"n": 0}
    model.eval()
    inv = {v: k for k, v in label_map.items()}
    paths = [p for p, _ in items]
    labels = [c for _, c in items]
    X = torch.tensor(embed(clf, paths), dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = [inv[int(i)] for i in logits.argmax(1)]
        confs = [float(probs[i].max()) for i in range(len(preds))]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    per = {}
    for gt in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == gt]
        per[gt] = {"correct": sum(1 for i in idxs if preds[i] == gt), "total": len(idxs)}
    return {"n": len(labels), "correct": correct, "accuracy": correct / len(labels), "per_class": per,
            "mean_conf": float(np.mean(confs))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--global-eval", action="store_true", help="jerÃ¡rquico sobre food-us completo")
    args = parser.parse_args()

    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
        top_k=999,
    )
    clf.load()

    if args.train:
        import torch  # noqa: PLC0415
        for gid, classes in GROUPS.items():
            model, val_acc, meta = train_group(clf, classes)
            torch_path = MODELS / f"{gid}.pt"
            torch.save({"state": model.state_dict(), "classes": classes, "label_map": meta["label_map"],
                        "val_acc": val_acc, "train": meta["train"], "distribution": meta["distribution"]}, torch_path)
            print(f"[train] {gid:<28} val={val_acc*100:.1f}% train={meta['train']} dist={meta['distribution']}")

    if args.eval:
        import torch  # noqa: PLC0415

        print("\n=== GRUPOS (test food-us, sin leakage) ===")
        print(f"{'grupo':<28} {'acc':<8} {'per-class':<50} conf")
        for gid, classes in GROUPS.items():
            ckpt = torch.load(MODELS / f"{gid}.pt", map_location="cpu", weights_only=False)
            model = torch.nn.Linear(512, len(classes))
            model.load_state_dict(ckpt["state"])
            model.eval()
            gt_classes = FOOD_US_GT[gid]
            items = food_us_items(set(gt_classes)) if gt_classes else []
            r = eval_group(clf, model, ckpt["label_map"], classes, items)
            if r["n"]:
                per = " ".join(f"{k}:{v['correct']}/{v['total']}" for k, v in r["per_class"].items())
                print(f"{gid:<28} {r['accuracy']*100:<8.1f} {per:<50} {r['mean_conf']:.2f}")

    if args.global_eval:
        import torch  # noqa: PLC0415

        legacy = ZeroShotFoodClassifier(
            "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
            crop_padding=settings.clip_crop_padding,
            prompt_template=settings.clip_prompt_template,
            prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
)
        legacy.load()
        ENABLED = {"rice_fries_fried_chicken"}
        groups = {}
        for gid in ENABLED:
            ckpt = torch.load(MODELS / f"{gid}.pt", map_location="cpu", weights_only=False)
            m = torch.nn.Linear(512, len(ckpt["classes"]))
            m.load_state_dict(ckpt["state"])
            m.eval()
            groups[gid] = {"model": m, "classes": set(ckpt["classes"]), "inv": {v: k for k, v in ckpt["label_map"].items()}}

        total = correct = 0
        d = BASE / "datasets/food-us-v0.1/images/train"
        for img in sorted(d.glob("*.jpg")):
            gt = img.stem.rsplit("_", 1)[0]
            total += 1
            image = Image.open(img).convert("RGB")
            rec = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))[img.name]
            rl = rec["yolo"] or rec["dino"]
            if rl:
                b = max(rl, key=lambda r: r.get("conf", 0.5))
                image = image.crop((max(0, int(b["x"] - b["width"] * 0.1)), max(0, int(b["y"] - b["height"] * 0.1)),
                                    min(image.width, int(b["x"] + b["width"] * 1.1)), min(image.height, int(b["y"] + b["height"] * 1.1))))
            ranking = legacy._score_crop(image)
            best_by: dict[str, float] = {}
            for r in ranking:
                canon = r.name
                if canon not in best_by or r.score > best_by[canon]:
                    best_by[canon] = r.score
            pred = max(best_by, key=best_by.get) if best_by else "unknown"
            # Â¿la clase legacy estÃ¡ en algÃºn grupo?
            for gid, g in groups.items():
                if pred in g["classes"]:
                    emb = embed(clf, [str(img)])[0]
                    with torch.no_grad():
                        logits = g["model"](torch.tensor(emb, dtype=torch.float32).unsqueeze(0))
                        pred = g["inv"][int(logits.argmax(1))]
                    break
            if pred == gt:
                correct += 1
            else:
                if gt in ('french_fries', 'fried_chicken'):
                    print('  PERDIDA ' + gt + ': ' + pred)
        print(f"\n=== JERÃRQUICO food-us: {correct}/{total} = {correct/total*100:.1f}% ===")


if __name__ == "__main__":
    main()





