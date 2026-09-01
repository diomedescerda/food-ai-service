"""F28-A: pipeline de calidad de prototipos (reproducible).

Etiquetas por imagen:
- VALID: el clasificador legacy 38 (CLIP) la clasifica como su clase.
- AMBIGUOUS: su clase está en el top-5 pero no top-1.
- WRONG: su clase ni siquiera está en el top-5 (ruido de título).
- DUPLICATE: similitud de embedding >= 0.97 con otro prototipo de la misma clase.

F28_clean = VALID sin DUPLICATE. Salida: data/prototypes/f28_clean.json.
Uso: python scripts/prototype_quality.py
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
SRC = BASE / "data/prototypes/f27_prototypes.json"
OUT = BASE / "data/prototypes/f28_clean.json"
DUP_THRESHOLD = 0.97


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

    store = json.loads(SRC.read_text(encoding="utf-8"))
    clean: dict[str, list[dict]] = {}
    report: dict[str, list[dict]] = {}

    for cls, items in store.items():
        embeds = [np.array(p["embedding"], dtype=np.float32) for p in items]
        sims = np.stack(embeds) @ np.stack(embeds).T
        keep = [True] * len(items)
        for i in range(len(items)):
            for j in range(i):
                if sims[i, j] >= DUP_THRESHOLD:
                    keep[i] = False  # conservar el primero
        report[cls] = []
        clean[cls] = []
        for p, emb, k in zip(items, embeds, keep):
            if not k:
                report[cls].append({**{x: p[x] for x in ("source", "image_id", "license")}, "label": "DUPLICATE"})
                continue
            image = Image.open(BASE / _resolve_path(p)).convert("RGB")
            ranking = clf._score_crop(image)
            names = [r.name for r in ranking]
            if names and names[0] == cls:
                label = "VALID"
            elif cls in names[:5]:
                label = "AMBIGUOUS"
            else:
                label = "WRONG"
            report[cls].append({**{x: p[x] for x in ("source", "image_id", "license")}, "label": label})
            if label in ("VALID", "AMBIGUOUS"):
                clean[cls].append(p)

    clean = {k: v for k, v in clean.items() if v}
    OUT.write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    counts = Counter()
    for cls, items in report.items():
        for it in items:
            counts[it["label"]] += 1
    print("ETIQUETAS:", dict(counts))
    print("clases limpias:", len(clean), "prototipos limpios:", sum(len(v) for v in clean.values()))
    for cls, items in report.items():
        labels = Counter(it["label"] for it in items)
        print(f"  {cls:<18} {dict(labels)}")


def _resolve_path(p: dict) -> str:
    src = p.get("source", "")
    image_id = p.get("image_id", "")
    if "multi-food" in src:
        return f"datasets/multi-food/images/{image_id}"
    if "assets" in src:
        return f"tests/assets/{image_id}"
    return f"datasets/food-bench-v1/{p.get('class_id')}/{image_id}"


if __name__ == "__main__":
    main()