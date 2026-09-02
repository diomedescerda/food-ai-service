"""F33: entrena grupos SOLO con imÃ¡genes VALID del manifest.

Los grupos con < 8 VALID por clase se reportan INSUFFICIENT_VALID_DATA.
Fried/nuggets (representaciÃ³n limitada) recibe LoRA experimental si el
frozen no supera el baseline.

Uso: python scripts/f33_train.py [--train] [--eval] [--lora] [--global]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((BASE / "data/training/f33/manifest.json").read_text(encoding="utf-8"))
MODELS = BASE / "data/models/f33"
MODELS.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 2026)

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


def group_items(classes: list[str]) -> list[tuple[str, str]]:
    return [(m["path"], m["class"]) for m in MANIFEST.values() if m["class"] in classes]


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


def train_linear(items, classes: list[str], seed: int):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    paths = [p for p, _ in items]
    labels = [c for _, c in items]
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    X = embed(clf, paths)
    y = np.array([label_map[c] for c in labels])
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
    return model, label_map, best_acc


def train_lora(clf, items, classes: list[str], seed: int = 42, rank: int = 8, epochs: int = 40, lr: float = 5e-4):
    """LoRA en el encoder visual de CLIP (QKV linears) + head Linear. El resto congelado."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    paths = [p for p, _ in items]
    labels = [c for _, c in items]
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    idx = rng.permutation(len(paths))
    n_val = max(1, len(idx) // 5)

    # Parches LoRA en los attention layers del vision model
    lora_params = []
    for name, module in clf._model.vision_model.encoder.layers.named_modules():
        if name.endswith(".self_attn.q_proj") or name.endswith(".self_attn.k_proj") or name.endswith(".self_attn.v_proj"):
            in_f = module.in_features
            lora_a = nn.Linear(in_f, rank, bias=False)
            lora_b = nn.Linear(rank, in_f, bias=False)
            nn.init.zeros_(lora_b.weight)
            lora_params.extend([lora_a.weight, lora_b.weight])
            module.lora_a, module.lora_b = lora_a, lora_b
            orig_forward = module.forward

            def make_forward(orig, a, b):
                def forward(x):
                    return orig(x) + b(a(x))
                return forward
            module.forward = make_forward(orig_forward, lora_a, lora_b)

    head = nn.Linear(512, len(classes))
    lora_params.extend(head.parameters())
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        opt.zero_grad()
        total_loss = 0.0
        for i in range(0, len(train_idx := idx[n_val:]), 16):
            batch_paths = [paths[j] for j in train_idx[i:i + 16]]
            batch_labels = [label_map[labels[j]] for j in train_idx[i:i + 16]]
            inputs = clf._processor(images=[Image.open(p).convert("RGB") for p in batch_paths], return_tensors="pt")
            feats = clf._features(clf._model.get_image_features(**inputs))
            logits = head(feats)
            loss = crit(logits, torch.tensor(batch_labels, dtype=torch.long))
            loss.backward()
            total_loss += float(loss)
        opt.step()
    # restaurar forward original
    for name, module in clf._model.vision_model.encoder.layers.named_modules():
        if hasattr(module, "lora_a"):
            module.forward = module.__class__.forward
    return head, label_map, total_loss / epochs


def predict(model, label_map, clf, paths: list[str]):
    import torch  # noqa: PLC0415

    model.eval()
    inv = {v: k for k, v in label_map.items()}
    X = torch.tensor(embed(clf, paths), dtype=torch.float32)
    with torch.no_grad():
        preds = [inv[int(i)] for i in model(X).argmax(1)]
    return preds


def food_us_items(classes: set[str]) -> list[tuple[str, str]]:
    d = BASE / "datasets/food-us-v0.1/images/train"
    return [(str(img), img.stem.rsplit("_", 1)[0]) for img in sorted(d.glob("*.jpg")) if img.stem.rsplit("_", 1)[0] in classes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    global clf
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
        top_k=999,
    )
    clf.load()

    if args.train or args.lora:
        import torch  # noqa: PLC0415

        for gid, classes in GROUPS.items():
            items = group_items(classes)
            per = Counter(c for _, c in items)
            if any(per.get(c, 0) < 8 for c in classes):
                print(f"[skip] {gid}: VALID insuficiente {dict(per)}")
                continue
            if args.lora and gid == "fried_chicken_nuggets":
                head, label_map, loss = train_lora(clf, items, classes)
                torch.save({"head": head.state_dict(), "classes": classes, "label_map": label_map}, MODELS / f"{gid}_lora.pt")
                print(f"[lora] {gid}: entrenado (loss={loss:.3f})")
                continue
            for seed in SEEDS:
                model, label_map, val_acc = train_linear(items, classes, seed)
                torch.save({"state": model.state_dict(), "classes": classes, "label_map": label_map, "seed": seed, "val_acc": val_acc},
                           MODELS / f"{gid}_s{seed}.pt")
            print(f"[train] {gid}: {len(items)} VALID {dict(per)}")

    if args.eval:
        import torch  # noqa: PLC0415

        print("\n=== GRUPOS VALIDADOS (test food-us, multi-seed) ===")
        for gid, classes in GROUPS.items():
            gt = FOOD_US_GT[gid]
            items = food_us_items(set(gt)) if gt else []
            if not items:
                print(f"{gid}: sin GT limpio")
                continue
            accs = []
            if not (MODELS / f'{gid}_s42.pt').exists():
                print(f'{gid}: no entrenado (VALID insuficiente)')
                continue
            for seed in SEEDS:
                ckpt = torch.load(MODELS / f"{gid}_s{seed}.pt", map_location="cpu", weights_only=False)
                m = torch.nn.Linear(512, len(classes))
                m.load_state_dict(ckpt["state"])
                preds = predict(m, ckpt["label_map"], clf, [p for p, _ in items])
                accs.append(sum(1 for p, (_, l) in zip(preds, items) if p == l) / len(items))
            mean, std = float(np.mean(accs)), float(np.std(accs))
            print(f"{gid}: {mean*100:.1f}% Â± {std*100:.1f}")
            # LoRA si existe
            lora_path = MODELS / f"{gid}_lora.pt"
            if lora_path.exists():
                ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)
                head = torch.nn.Linear(512, len(classes))
                head.load_state_dict(ckpt["head"])
                preds = predict(head, ckpt["label_map"], clf, [p for p, _ in items])
                acc = sum(1 for p, (_, l) in zip(preds, items) if p == l) / len(items)
                print(f"  LoRA: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
