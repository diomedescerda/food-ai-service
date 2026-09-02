"""F39: guarda el specialist DINO-base pizza/naan (del F38 inline) y el módulo shadow.

SpecialistShadow: ejecuta la decisión del specialist SIN modificar el resultado
legacy (shadow). Solo registra telemetría. Nunca cambia nutrition/portion/final.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
MODELS = BASE / "data/models/f38"
MODELS.mkdir(parents=True, exist_ok=True)


def embed(model, processor, paths, kind="dino"):
    import torch  # noqa: PLC0415

    embs = []
    for i in range(0, len(paths), 8):
        chunk = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
        inputs = processor(images=chunk, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.numpy())
    return np.concatenate(embs)


def main() -> None:
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

    manifest = json.loads((BASE / "data/training/f33/manifest.json").read_text(encoding="utf-8"))
    by_class: dict[str, list[str]] = {}
    for m in manifest.values():
        by_class.setdefault(m["class"], []).append(m["path"])
    classes = ["pizza", "naan"]
    paths = [p for c in classes for p in by_class.get(c, [])]
    labels = [c for c in classes for _ in by_class.get(c, [])]
    label_map = {c: i for i, c in enumerate(classes)}

    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    X = embed(model, processor, paths)
    y = np.array([label_map[l] for l in labels])
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    head = nn.Linear(X.shape[1], 2)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X[train_idx], dtype=torch.float32)
    yt = torch.tensor(y[train_idx], dtype=torch.long)
    Xv = torch.tensor(X[val_idx], dtype=torch.float32)
    yv = torch.tensor(y[val_idx], dtype=torch.long)
    best_acc, best_state = 0.0, None
    for _ in range(150):
        head.train()
        opt.zero_grad()
        loss = crit(head(Xt), yt)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            acc = float((head(Xv).argmax(1) == yv).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    torch.save({"head": head.state_dict(), "classes": classes, "label_map": label_map, "val_acc": best_acc},
               MODELS / "pizza_naan_dino_base.pt")
    print(f"[guardado] pizza_naan_dino_base.pt (val={best_acc*100:.1f}%, {len(paths)} imágenes)")


if __name__ == "__main__":
    main()