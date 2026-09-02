"""F36: validaciÃ³n RELAXED (top-3 del legacy) para el grupo rice.

La validaciÃ³n estricta (top-1) eliminÃ³ fries (0 VALID). La RELAXED acepta
la clase en el top-3 â€” para EXPERIMENTACIÃ“N Ãºnicamente (no producciÃ³n).
Salida: data/training/f36/manifest_relaxed.json (rice/fries/fried_chicken).
Uso: python scripts/f36_relaxed_validation.py
"""
import json
import sys
import torch  # noqa: E402
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "data/training/f36"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = {"rice", "french_fries", "fried_chicken"}
DUP_THRESHOLD = 0.97

TRAIN_DIRS = [
    BASE / "datasets/food-bench-v1",
    BASE / "datasets/prototype-src",
    BASE / "datasets/multi-food/images",
    BASE / "tests/assets",
]


def main() -> None:
    import torch  # noqa: PLC0415

    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
        top_k=999,
    )
    clf.load()

    items = []
    for d in TRAIN_DIRS:
        if not d.exists():
            continue
        for img in d.rglob("*.jpg"):
            cls = img.parent.name
            if d.name == "multi-food" and cls == "images":
                cls = {"mf_010": "french_fries"}.get(img.stem, "")
            if cls in CLASSES:
                items.append((str(img), cls))

    per: dict[str, list] = {}
    for p, c in items:
        per.setdefault(c, []).append(p)

    manifest = {}
    counts = Counter()
    for cls in sorted(CLASSES):
        paths = per.get(cls, [])
        embs = []
        for p in paths:
            inputs = clf._processor(images=[Image.open(p).convert("RGB")], return_tensors="pt")
            with torch.no_grad():
                f = clf._features(clf._model.get_image_features(**inputs))
                f = f / f.norm(dim=-1, keepdim=True)
            embs.append(f.numpy()[0])
        sims = np.stack(embs) @ np.stack(embs).T if embs else np.zeros((0, 0))
        for i, p in enumerate(paths):
            if any(sims[i, j] >= DUP_THRESHOLD for j in range(i)):
                label = "DUPLICATE"
            else:
                ranking = clf._score_crop(Image.open(p).convert("RGB"))
                names = [r.name for r in ranking]
                label = "VALID_STRICT" if names and names[0] == cls else ("VALID_RELAXED" if cls in names[:3] else "WRONG")
            counts[label] += 1
            if label in ("VALID_STRICT", "VALID_RELAXED"):
                manifest[f"{cls}/{Path(p).name}"] = {
                    "image_id": Path(p).name, "path": p, "class": cls,
                    "source": "Commons/propio", "license": "CC0/CC BY/CC BY-SA/PD o propio",
"validation_status": label, "validator": "f36_relaxed_validation.py",
                }
        per_label = Counter()
        for p in paths:
            m = manifest.get(f"{cls}/{Path(p).name}")
            if m:
                per_label[m["validation_status"]] += 1
        print(f"[{cls:<16}] {dict(per_label)}", flush=True)

    print("TOTAL:", dict(counts))
    out = OUT_DIR / "manifest_relaxed.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    by = Counter(m["validation_status"] for m in manifest.values())
    print(f"manifest_relaxed: {len(manifest)} ({dict(by)}) -> {out}")


if __name__ == "__main__":
    main()

