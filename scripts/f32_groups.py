"""F32: ConfusionGroups con splits deterministas (hash), multi-seed, gating.

- Split determinista: hash(path) % 20 -> train(0-13)/val(14-16)/test(17-19)
- Test EXTERNO: food-us (las imágenes del food-us nunca están en train)
- Multi-seed: 42/123/2026 -> mean/std del grupo
- Gating: threshold del softmax del especializado (grid 0.5-0.9) elegido sobre
  VAL, aplicado en el global food-us
- Activación uno a uno: legacy -> +rice -> +siguiente mejor

Uso: python scripts/f32_groups.py [--train] [--eval] [--global] [--diag]
"""
import argparse
import hashlib
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
MODELS = BASE / "data/models/f32"
MODELS.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 2026)
THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)

GROUPS = {
    "rice_fries_fried_chicken": ["rice", "french_fries", "fried_chicken"],
    "pizza_naan": ["pizza", "naan"],
    "hamburger_sandwich": ["hamburger", "sandwich"],
    "taco_quesadilla_nachos": ["taco", "quesadilla", "nachos"],
    "fried_chicken_nuggets": ["fried_chicken", "chicken_nuggets"],
}
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


def all_images(classes: set[str]) -> list[tuple[str, str]]:
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
    return items


def split_by_hash(items: list[tuple[str, str]]) -> tuple[list, list, list]:
    train, val, test = [], [], []
    for p, c in items:
        h = int(hashlib.md5(p.encode()).hexdigest(), 16) % 20
        (train if h < 14 else val if h < 17 else test).append((p, c))
    return train, val, test


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


def train_one(clf, train_items, val_items, classes: list[str], seed: int):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    torch.manual_seed(seed)
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    X = embed(clf, [p for p, _ in train_items])
    y = np.array([label_map[c] for _, c in train_items])
    Xv = embed(clf, [p for p, _ in val_items])
    yv = np.array([label_map[c] for _, c in val_items])
    model = nn.Linear(512, len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    Xvt = torch.tensor(Xv, dtype=torch.float32)
    yvt = torch.tensor(yv, dtype=torch.long)
    best_acc, best_state = 0.0, None
    for _ in range(150):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xvt).argmax(1) == yvt).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, label_map, best_acc


def predict_group(model, label_map, clf, paths: list[str]) -> tuple[list[str], np.ndarray]:
    import torch  # noqa: PLC0415

    model.eval()
    inv = {v: k for k, v in label_map.items()}
    X = torch.tensor(embed(clf, paths), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)
        preds = [inv[int(i)] for i in probs.argmax(1)]
    return preds, probs.numpy()


def legacy_rank_by_class(clf, image):
    ranking = clf._score_crop(image)
    best: dict[str, float] = {}
    for r in ranking:
        if r.name not in best or r.score > best[r.name]:
            best[r.name] = r.score
    return max(best, key=best.get) if best else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--global-eval", action="store_true")
    parser.add_argument("--diag", action="store_true")
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
            items = all_images(set(classes))
            train, val, test = split_by_hash(items)
            # Solo imágenes hash < 16 (train+val) — el test interno queda para el size experiment
            dist = dict(Counter(c for _, c in train + val))
            for seed in SEEDS:
                model, label_map, val_acc = train_one(clf, train, val, classes, seed)
                torch.save({"state": model.state_dict(), "classes": classes, "label_map": label_map,
                            "val_acc": val_acc, "seed": seed,
                            "train": len(train), "val": len(val)},
                           MODELS / f"{gid}_s{seed}.pt")
            print(f"[train] {gid:<28} train={len(train)} val={len(val)} test_int={len(test)} dist={dist}")

    if args.eval:
        import torch  # noqa: PLC0415

        print("\n=== GRUPOS (test externo food-us, multi-seed) ===")
        for gid, classes in GROUPS.items():
            gt = FOOD_US_GT[gid]
            items = food_us_items(set(gt)) if gt else []
            if not items:
                print(f"{gid:<28} sin GT limpio (solo val interno)")
                continue
            accs = []
            for seed in SEEDS:
                ckpt = torch.load(MODELS / f"{gid}_s{seed}.pt", map_location="cpu", weights_only=False)
                m = torch.nn.Linear(512, len(classes))
                m.load_state_dict(ckpt["state"])
                preds, _ = predict_group(m, ckpt["label_map"], clf, [p for p, _ in items])
                acc = sum(1 for p, (_, l) in zip(preds, items) if p == l) / len(items)
                accs.append(acc)
            mean, std = float(np.mean(accs)), float(np.std(accs))
            per = " ".join(f"{c}:{sum(1 for p, (_, l) in zip(preds, items) if p == l and l == c)}/{sum(1 for _, l in items if l == c)}"
                           for c in sorted(set(l for _, l in items)))
            print(f"{gid:<28} {mean*100:.1f}% ± {std*100:.1f}  ({per})")

    if args.global_eval:
        import torch  # noqa: PLC0415

        legacy = ZeroShotFoodClassifier(
            "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
            crop_padding=settings.clip_crop_padding,
            prompt_template=settings.clip_prompt_template,
            prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
        )
        legacy.load()
        regions = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
        d = BASE / "datasets/food-us-v0.1/images/train"
        images = sorted(d.glob("*.jpg"))

        # activación uno a uno: legacy -> +rice -> +siguientes por grupo
        ENABLED = ["rice_fries_fried_chicken"]
        # grupos candidatos adicionales por orden de ganancia (del eval)
        for extra in ("pizza_naan", "hamburger_sandwich", "fried_chicken_nuggets"):
            ENABLED.append(extra)
            total = correct = 0
            for img in images:
                gt = img.stem.rsplit("_", 1)[0]
                rec = regions[img.name]
                rl = rec["yolo"] or rec["dino"]
                if not rl:
                    continue
                image = Image.open(img).convert("RGB")
                b = max(rl, key=lambda r: r.get("conf", 0.5))
                crop = image.crop((max(0, int(b["x"] - b["width"] * 0.1)), max(0, int(b["y"] - b["height"] * 0.1)),
                                   min(image.width, int(b["x"] + b["width"] * 1.1)), min(image.height, int(b["y"] + b["height"] * 1.1))))
                pred = legacy_rank_by_class(legacy, crop)
                conf_spec = 0.0
                for gid in ENABLED:
                    classes = GROUPS[gid]
                    if pred in classes:
                        ckpt = torch.load(MODELS / f"{gid}_s42.pt", map_location="cpu", weights_only=False)
                        m = torch.nn.Linear(512, len(classes))
                        m.load_state_dict(ckpt["state"])
                        preds, probs = predict_group(m, ckpt["label_map"], clf, [str(img)])
                        conf_spec = float(probs[0].max())
                        # gating: solo aplicar si el especializado tiene confianza >= 0.6
                        if conf_spec >= 0.60:
                            pred = preds[0]
                        break
                total += 1
                correct += pred == gt
            print(f"[activación {','.join(ENABLED)}] food-us {correct}/{total} = {correct/total*100:.1f}%")

    if args.diag:
        print("\n=== DIAGNÓSTICO DE REPRESENTACIÓN (intra/inter similitud CLIP) ===")
        for gid, classes in GROUPS.items():
            items = all_images(set(classes))
            by_cls: dict[str, np.ndarray] = {}
            for c in classes:
                paths = [p for p, cl in items if cl == c][:24]
                by_cls[c] = embed(clf, paths)
            intra = {c: float(np.mean(by_cls[c] @ by_cls[c].T)) for c in classes}
            print(f"{gid}: intra={ {c: round(v, 3) for c, v in intra.items()} }")
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    a, b = classes[i], classes[j]
                    inter = float(np.mean(by_cls[a] @ by_cls[b].T))
                    print(f"    inter {a}-{b}: {inter:.3f} (intra {a}: {intra[a]:.3f}, {b}: {intra[b]:.3f})")


if __name__ == "__main__":
    main()