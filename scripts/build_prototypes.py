"""F27.1-2: construye el Visual Prototype Store.

Fuentes PERMISIVAS para producciÃ³n (verificadas):
- food-bench-v1 (Wikimedia Commons: CC0/CC BY/CC BY-SA/PD â€” en metadata.json)
- multi-food (imÃ¡genes propias del proyecto)
- tests/assets (imÃ¡genes propias)

NO usa Food-101 (non-commercial) ni Nutrition5k (no descargado).

Salida: data/prototypes/f27_prototypes.json â€” {class: [embeddings + metadata]}
Uso: python scripts/build_prototypes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data/prototypes/f29_full.json"

# Clases problemÃ¡ticas (F21/F22/F26) + las 6 del regression set food-us.
TARGET_CLASSES = sorted({p.parent.name for p in (BASE / "datasets/food-bench-v1").rglob("*.jpg")} | {p.parent.name for p in (BASE / "datasets/prototype-src").rglob("*.jpg")}) + [
    "french_fries", "fried_chicken", "hot_dog", "hamburger", "taco",
    "quesadilla", "nachos", "steak", "grilled_chicken", "salmon", "pizza",
    "sandwich", "mac_and_cheese", "lasagna", "chicken_nuggets", "banana", "apple",
]

# Fuentes: (clase, ruta_imagen, source, license)
def collect_sources() -> dict[str, list[dict]]:
    sources: dict[str, list[dict]] = {c: [] for c in TARGET_CLASSES}
    bench_meta = json.loads((BASE / "datasets/food-bench-v1/metadata.json").read_text(encoding="utf-8"))
    for img_path in (BASE / "datasets/food-bench-v1").rglob("*.jpg"):
        cls = img_path.parent.name
        if cls not in TARGET_CLASSES:
            continue
        key = f"{cls}/{img_path.name}"
        meta = bench_meta.get(key, {})
        sources[cls].append({
            "path": str(img_path),
            "source": "Wikimedia Commons (food-bench-v1)",
            "license": meta.get("license", "unknown"),
            "image_id": img_path.name,
        })
    for img_path in (BASE / "datasets/prototype-src").rglob("*.jpg"):
        cls = img_path.parent.name
        if cls in TARGET_CLASSES:
            sources[cls].append({
                "path": str(img_path),
                "source": "Wikimedia Commons (prototype-src)",
                "license": "CC0/CC BY/CC BY-SA/PD (Commons)",
                "image_id": img_path.name,
            })
    for img_path in (BASE / "datasets/multi-food/images").glob("*.jpg"):
        cls_map = {"mf_003": "hamburger", "mf_010": "french_fries", "mf_011": "pizza"}
        cls = cls_map.get(img_path.stem)
        if cls and cls in TARGET_CLASSES:
            sources[cls].append({
                "path": str(img_path), "source": "multi-food (propio)",
                "license": "propietario del proyecto", "image_id": img_path.name,
            })
    for img_path in (BASE / "tests/assets").glob("*.jpg"):
        cls = img_path.stem
        if cls in TARGET_CLASSES:
            sources[cls].append({
                "path": str(img_path), "source": "tests/assets (propio)",
                "license": "propietario del proyecto", "image_id": img_path.name,
            })
    return sources


def main() -> None:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()

    def image_embedding(image: Image.Image) -> np.ndarray:
        inputs = clf._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feats = clf._features(clf._model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.numpy()[0]

    sources = collect_sources()
    store: dict[str, list[dict]] = {}
    for cls, items in sources.items():
        if not items:
            print(f"[sin imÃ¡genes] {cls}", flush=True)
            continue
        store[cls] = []
        for it in items:
            image = Image.open(it["path"]).convert("RGB")
            emb = image_embedding(image)
            store[cls].append({
                "class_id": cls,
                "class_name": cls.replace("_", " "),
                "source": it["source"],
                "image_id": it["image_id"],
                "embedding": [round(float(x), 6) for x in emb],
                "embedding_dim": len(emb),
                "license": it["license"],
                "model": "CLIP ViT-B/32",
            })
        print(f"[ok] {cls}: {len(store[cls])} prototipos", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    n = sum(len(v) for v in store.values())
    size_kb = OUT.stat().st_size / 1024
    print(f"store: {len(store)} clases, {n} prototipos, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()


