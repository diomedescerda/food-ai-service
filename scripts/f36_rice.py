"""F36: grupo rice con DINOv2-L y fries VALID ampliadas (validaciÃ³n relaxed).

Â¿El 48.8% de F35 era datos o representaciÃ³n? Train: manifest_relaxed
(rice/fries/fried_chicken). Test: food-us (28 imÃ¡genes). 3 seeds.
Uso: python scripts/f36_rice.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
CLASSES = ["rice", "french_fries", "fried_chicken"]
TEST_CLASSES = ["french_fries", "fried_chicken"]


def embed(model, processor, paths: list[str], kind: str = "dino") -> np.ndarray:
    import torch  # noqa: PLC0415

    embs = []
    for i in range(0, len(paths), 8):
        chunk = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
        inputs = processor(images=chunk, return_tensors="pt")
        with torch.no_grad():
            if kind == "dino":
                out = model(**inputs)
                feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
            else:
                out = model.get_image_features(**inputs)
                feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.numpy())
    return np.concatenate(embs)


def train_linear(X, y, n: int, seed: int):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    model = nn.Linear(X.shape[1], n)
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
    import torch  # noqa: PLC0415
    from transformers import CLIPModel, AutoImageProcessor  # noqa: PLC0415

    manifest = json.loads((BASE / "data/training/f36/manifest_relaxed.json").read_text(encoding="utf-8"))
    by_class: dict[str, list[str]] = {}
    for m in manifest.values():
        by_class.setdefault(m["class"], []).append(m["path"])
    train_paths = [p for c in CLASSES for p in by_class.get(c, [])]
    labels = [c for c in CLASSES for _ in by_class.get(c, [])]
    label_map = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([label_map[l] for l in labels])

    d = BASE / "datasets/food-us-v0.1/images/train"
    test_items = [(str(img), img.stem.rsplit("_", 1)[0]) for img in sorted(d.glob("*.jpg"))
                  if img.stem.rsplit("_", 1)[0] in TEST_CLASSES]
    test_paths = [p for p, _ in test_items]

    settings = Settings()
    legacy = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    legacy.load()

    print("train por clase:", {c: len(by_class.get(c, [])) for c in CLASSES})
    legacy_correct = 0
    for p, gt in test_items:
        ranking = legacy._score_crop(Image.open(p).convert("RGB"))
        names = [r.name for r in ranking]
        legacy_correct += bool(names and names[0] == gt)
    print(f"legacy grupo: {legacy_correct}/{len(test_items)} = {legacy_correct/len(test_items)*100:.1f}%")

    for name, load_fn in (
        ("dino_l", lambda: (_load("facebook/dinov2-large", "dino"), 1024)),
        ("clip_b32", lambda: (CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(), 768)),
    ):
        model, dim = load_fn()
        processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-large" if name == "dino_l" else "openai/clip-vit-base-patch32")
        if name == "dino_l":
            from transformers import AutoModel  # noqa: PLC0415

            model = AutoModel.from_pretrained("facebook/dinov2-large").eval()
        X = embed(model, processor, train_paths, 'dino' if name == 'dino_l' else 'clip')
        Xt = embed(model, processor, test_paths, 'dino' if name == 'dino_l' else 'clip')
        accs = []
        for seed in SEEDS:
            m = train_linear(X, y, 3, seed)
            with torch.no_grad():
                preds = m(torch.tensor(Xt, dtype=torch.float32)).argmax(1)
            inv = {v: k for k, v in label_map.items()}
            acc = sum(1 for i, (_, l) in enumerate(test_items) if inv[int(preds[i])] == l) / len(test_items)
            accs.append(acc)
        per = {}
        for seed in SEEDS:
            m = train_linear(X, y, 3, seed)
            with torch.no_grad():
                preds = m(torch.tensor(Xt, dtype=torch.float32)).argmax(1)
            inv = {v: k for k, v in label_map.items()}
            for i, (_, l) in enumerate(test_items):
                per.setdefault(l, []).append(inv[int(preds[i])])
        per_sum = {k: f"{sum(1 for x in v if x == k)}/{len(v)}" for k, v in per.items()}
        print(f"{name}: {np.mean(accs)*100:.1f}% Â± {np.std(accs)*100:.1f}  per-class(seed42)={per_sum}")


def _load(mid: str, kind: str):
    from transformers import AutoModel  # noqa: PLC0415

    return AutoModel.from_pretrained(mid).eval()


if __name__ == "__main__":
    main()
