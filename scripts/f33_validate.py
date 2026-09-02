"""F33: validaciÃ³n obligatoria de TODAS las imÃ¡genes de entrenamiento.

- Clases del legacy 38: top-1 del legacy == clase -> VALID; top-5 -> AMBIGUOUS;
  fuera del top-5 -> WRONG.
- Clases fuera del legacy (naan): score CLIP del candidate >= 0.26 -> VALID
  (laxo, documentado); < 0.22 -> WRONG; entre -> AMBIGUOUS.
- DUPLICATE: similitud de embedding >= 0.97 con otra imagen de la misma clase.
- AMBIGUOUS/WRONG/DUPLICATE NO entran al entrenamiento.

Salida: data/training/f33/manifest.json (solo VALID) + reporte por clase.
Uso: python scripts/f33_validate.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.food_catalog import FOOD_CATALOG  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "data/training/f33"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DUP_THRESHOLD = 0.97
NAAN_VALID = 0.26
NAAN_WRONG = 0.22

GROUPS = {
    "rice_fries_fried_chicken": ["rice", "french_fries", "fried_chicken"],
    "pizza_naan": ["pizza", "naan"],
    "hamburger_sandwich": ["hamburger", "sandwich"],
    "taco_quesadilla_nachos": ["taco", "quesadilla", "nachos"],
    "fried_chicken_nuggets": ["fried_chicken", "chicken_nuggets"],
}
CLASSES = sorted({c for g in GROUPS.values() for c in g})
LEGACY = {e.canonical_name for e in FOOD_CATALOG}

TRAIN_DIRS = [
    BASE / "datasets/food-bench-v1",
    BASE / "datasets/prototype-src",
    BASE / "datasets/multi-food/images",
    BASE / "tests/assets",
]


def all_images() -> list[tuple[str, str, str]]:
    items = []
    for d in TRAIN_DIRS:
        if not d.exists():
            continue
        for img in d.rglob("*.jpg"):
            cls = img.parent.name
            if d.name == "multi-food" and cls == "images":
                cls = {"mf_003": "hamburger", "mf_010": "french_fries", "mf_011": "pizza"}.get(img.stem, "")
            if cls in CLASSES:
                src = "food-bench-v1" if "food-bench" in str(d) else ("prototype-src" if "prototype-src" in str(d) else ("multi-food" if "multi-food" in str(d) else "assets"))
                items.append((str(img), cls, src))
    return items


def naan_candidates() -> list[str]:
    for e in FOOD_CATALOG:
        if e.canonical_name == "naan":
            return list(e.clip_candidates)
    return ["naan", "naan bread"]


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
    master = json.loads((BASE / "data/catalogs/food_master.json").read_text(encoding="utf-8"))
    CANDIDATES = {f["canonical_name"]: list(f["clip_candidates"]) for f in master["foods"] if f["canonical_name"] in CLASSES}

    items = all_images()
    per_class: dict[str, list[tuple[str, str, str]]] = {}
    for p, c, s in items:
        per_class.setdefault(c, []).append((p, c, s))

    manifest: dict[str, dict] = {}
    counts = Counter()
    for cls, cls_items in sorted(per_class.items()):
        # embeddings para dedup
        embs = []
        for p, _, _ in cls_items:
            image = Image.open(p).convert("RGB")
            inputs = clf._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                feats = clf._features(clf._model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.numpy()[0])
        sims = np.stack(embs) @ np.stack(embs).T
        for i, (p, c, s) in enumerate(cls_items):
            if any(sims[i, j] >= DUP_THRESHOLD for j in range(i)):
                label = "DUPLICATE"
            elif c in LEGACY:
                ranking = clf._score_crop(Image.open(p).convert('RGB'))
                names = [r.name for r in ranking]
                label = 'VALID' if names and names[0] == c else ('AMBIGUOUS' if c in names[:5] else 'WRONG')
            else:  # naan: score del candidate (del master — no está en food_catalog)
                prompts = [t.format(food=x) for x in CANDIDATES[c] for t in clf._prompt_templates]
                image = Image.open(p).convert("RGB")
                inputs = clf._processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    feats = clf._features(clf._model.get_image_features(**inputs))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                emb = feats.numpy()[0]
                text_inputs = clf._processor(text=prompts, padding=True, return_tensors="pt")
                with torch.no_grad():
                    tf = clf._features(clf._model.get_text_features(**text_inputs))
                    tf = tf / tf.norm(dim=-1, keepdim=True)
                score = float((emb @ tf.numpy().T).max())
                label = "VALID" if score >= NAAN_VALID else ("AMBIGUOUS" if score >= NAAN_WRONG else "WRONG")
            counts[label] += 1
            if label == "VALID":
                manifest[f"{c}/{Path(p).name}"] = {
                    "image_id": Path(p).name, "path": p, "class": c, "source": s,
                    "license": "Commons CC0/CC BY/CC BY-SA/PD o propio", "validation_status": "VALID",
                    "validator": "f33_validate.py",
                }
        print(f"[{cls:<16}] {dict(Counter(label for p, c, s in cls_items))}", flush=True)

    print("TOTAL:", dict(counts))
    out = OUT_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {len(manifest)} VALID -> {out}")


if __name__ == "__main__":
    main()


