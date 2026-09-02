"""F35: benchmark de backbones visuales sobre los grupos de confusiÃ³n.

Cada backbone (congelado) -> embeddings -> diagnÃ³stico intra/inter ->
Linear por grupo (seeds 42/123/2026) -> accuracy del grupo en food-us.
La Ãºnica variable es la REPRESENTACIÃ“N.

Uso: python scripts/f35_backbones.py [--embeddings] [--diag] [--train-eval]
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
EMB_DIR = BASE / "data/embeddings/f35"
EMB_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 2026)

GROUPS = {
    "rice_fries_fried_chicken": ["rice", "french_fries", "fried_chicken"],
    "pizza_naan": ["pizza", "naan"],
    "fried_chicken_nuggets": ["fried_chicken", "chicken_nuggets"],
    "hamburger_sandwich": ["hamburger", "sandwich"],
}
FOOD_US_GT = {
    "rice_fries_fried_chicken": ["french_fries", "fried_chicken"],
    "pizza_naan": ["pizza"],
    "fried_chicken_nuggets": ["fried_chicken"],
    "hamburger_sandwich": ["hamburger", "sandwich"],
}

BACKBONES = {
    "clip_b32": {"model": "openai/clip-vit-base-patch32", "type": "clip"},
    "dino_l": {"model": "facebook/dinov2-large", "type": "dino"},
    "siglip": {"model": "google/siglip-base-patch16-224", "type": "siglip"},
    "clip_l": {"model": "openai/clip-vit-large-patch14", "type": "clip"},
}


def load_backbone(name: str):
    import torch  # noqa: PLC0415
    from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

    cfg = BACKBONES[name]
    kind = cfg["type"]
    if kind == "clip":
        from transformers import CLIPModel  # noqa: PLC0415

        model = CLIPModel.from_pretrained(cfg["model"]).eval()
        dim = model.config.vision_config.hidden_size
    elif kind == "siglip":
        from transformers import SiglipModel  # noqa: PLC0415

        model = SiglipModel.from_pretrained(cfg["model"]).eval()
        dim = model.config.vision_config.hidden_size
    else:
        model = AutoModel.from_pretrained(cfg["model"]).eval()
        dim = getattr(model.config, "hidden_size", 0)
    processor = AutoImageProcessor.from_pretrained(cfg["model"])
    return model, processor, dim, kind


def embed_backbone(name: str, model, processor, paths: list[str], kind: str, batch: int = 8) -> np.ndarray:
    import torch  # noqa: PLC0415

    embs = []
    for i in range(0, len(paths), batch):
        chunk = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inputs = processor(images=chunk, return_tensors="pt")
        with torch.no_grad():
            if kind in ("clip", "siglip"):
                out = model.get_image_features(**inputs)
                feats = out.pooler_output if hasattr(out, "pooler_output") else out
            else:
                out = model(**inputs)
                feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.numpy())
    return np.concatenate(embs)

def images_by_class() -> dict[str, list[str]]:
    manifest = json.loads((BASE / "data/training/f33/manifest.json").read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for m in manifest.values():
        out.setdefault(m["class"], []).append(m["path"])
    return out


def food_us_paths(classes: set[str]) -> list[tuple[str, str]]:
    d = BASE / "datasets/food-us-v0.1/images/train"
    return [(str(img), img.stem.rsplit("_", 1)[0]) for img in sorted(d.glob("*.jpg"))
            if img.stem.rsplit("_", 1)[0] in classes]


def train_linear(X, y, n_classes: int, seed: int):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    model = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X[train_idx], dtype=torch.float32)
    yt = torch.tensor(y[train_idx], dtype=torch.long)
    Xv = torch.tensor(X[val_idx], dtype=torch.float32)
    yv = torch.tensor(y[val_idx], dtype=torch.long)
    best_acc, best_state = 0.0, None
    for _ in range(150):
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
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--train-eval", action="store_true")
    args = parser.parse_args()

    by_class = images_by_class()
    # imÃ¡genes: manifest (train) + food-us (test)
    all_paths = sorted({p for ps in by_class.values() for p in ps})

    if args.embeddings:
        for name, cfg in BACKBONES.items():
            t0 = time.perf_counter()
            model, processor, dim, kind = load_backbone(name)
            embs = embed_backbone(name, model, processor, all_paths, kind)
            np.save(EMB_DIR / f"{name}_train.npy", embs)
            load_ms = (time.perf_counter() - t0) * 1000
            print(f"[emb] {name}: dim={dim} n={len(embs)} load={load_ms:.0f}ms "
                  f"lat={load_ms / max(len(embs), 1):.0f}ms/img")

    if args.diag:
        print("\n=== DIAGNÃ“STICO intra/inter por backbone ===")
        print(f"{'grupo':<26} {'backbone':<9} {'intra':<8} {'inter':<8}")
        for gid, classes in GROUPS.items():
            paths = [p for c in classes for p in by_class.get(c, [])[:24]]
            if not paths:
                continue
            for name in BACKBONES:
                if not (EMB_DIR / f'{name}_train.npy').exists():
                    continue
                embs = np.load(EMB_DIR / f'{name}_train.npy')
                # Ã­ndice de cada path
                idx = {p: i for i, p in enumerate(all_paths)}
                mats = {c: np.stack([embs[idx[p]] for p in by_class.get(c, [])[:24]]) for c in classes if c in by_class}
                if len(mats) < 2:
                    continue
                intra = {c: float(np.mean(m @ m.T)) for c, m in mats.items()}
                vals = []
                keys = list(mats)
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        vals.append(float(np.mean(mats[keys[i]] @ mats[keys[j]].T)))
                print(f"{gid:<26} {name:<9} {np.mean(list(intra.values())):.3f} {np.mean(vals):.3f}")

    if args.train_eval:
        import torch  # noqa: PLC0415

        print("\n=== LINEAR POR GRUPO (food-us test, multi-seed) ===")
        print(f"{'grupo':<26} {'backbone':<9} {'meanÂ±std':<14} per-class")
        for gid, classes in GROUPS.items():
            gt = FOOD_US_GT[gid]
            test_items = food_us_paths(set(gt))
            if not test_items:
                continue
            # embeddings de test por backbone
            for name in BACKBONES:
                if not (EMB_DIR / f'{name}_train.npy').exists():
                    continue
                model, processor, dim, kind = load_backbone(name)
                test_embs = embed_backbone(name, model, processor, [p for p, _ in test_items], kind)
                train_paths = [p for c in classes for p in by_class.get(c, [])]
                if not train_paths:
                    continue
                train_embs = embed_backbone(name, model, processor, train_paths, kind)
                labels = [c for c in classes for _ in by_class.get(c, [])]
                label_map = {c: i for i, c in enumerate(sorted(set(labels)))}
                y = np.array([label_map[l] for l in labels])
                accs = []
                for seed in SEEDS:
                    m = train_linear(train_embs, y, len(label_map), seed)
                    with torch.no_grad():
                        preds = m(torch.tensor(test_embs, dtype=torch.float32)).argmax(1)
                    inv = {v: k for k, v in label_map.items()}
                    acc = sum(1 for i, (_, l) in enumerate(test_items) if inv[int(preds[i])] == l) / len(test_items)
                    accs.append(acc)
                mean, std = float(np.mean(accs)), float(np.std(accs))
                print(f"{gid:<26} {name:<9} {mean*100:.1f}Â±{std*100:.1f}")


if __name__ == "__main__":
    main()




