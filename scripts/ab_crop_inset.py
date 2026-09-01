"""FASE 22 P1: A/B crop inset â€” recorte interior del bbox (contexto reducido)
para fries/fried_chicken (confusiÃ³n â†’ rice). Baseline = padding 0.10.

Uso: python scripts/ab_crop_inset.py [0.05] [0.10] [0.15]
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "datasets" / "food-us-v0.1" / "images" / "train"
REGIONS = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
CLASSES = ("french_fries", "fried_chicken")


def crop_inset(image: Image.Image, box: dict, padding: float, inset: float) -> Image.Image:
    # Expande el bbox (padding) y luego recorta el interior (inset por lado).
    x1 = max(0, int(box["x"] - box["width"] * padding))
    y1 = max(0, int(box["y"] - box["height"] * padding))
    x2 = min(image.width, int(box["x"] + box["width"] * (1 + padding)))
    y2 = min(image.height, int(box["y"] + box["height"] * (1 + padding)))
    w, h = x2 - x1, y2 - y1
    ix1 = x1 + int(w * inset)
    iy1 = y1 + int(h * inset)
    ix2 = x2 - int(w * inset)
    iy2 = y2 - int(h * inset)
    if ix2 <= ix1 or iy2 <= iy1:
        return image.crop((x1, y1, x2, y2))
    return image.crop((ix1, iy1, ix2, iy2))


def main() -> None:
    settings = Settings()
    classifier = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    classifier.load()

    insets = [float(a) for a in sys.argv[1:]] or [0.0, 0.05, 0.10, 0.15]
    for inset in insets:
        per_class = defaultdict(Counter)
        total = top1 = 0
        for cls in CLASSES:
            for img_path in sorted(TRAIN.glob(f"{cls}_*.jpg")):
                rec = REGIONS[img_path.name]
                regions_list = rec["yolo"] or rec["dino"]
                if not regions_list:
                    continue
                best = max(regions_list, key=lambda r: r.get("conf", 0.5))
                image = Image.open(img_path).convert("RGB")
                crop_img = crop_inset(image, best, settings.clip_crop_padding, inset)
                ranking = classifier._score_crop(crop_img)
                names = [r.name for r in ranking]
                pred = names[0] if names else "unknown"
                total += 1
                top1 += pred == cls
                per_class[cls][pred] += 1
        print(f"=== inset {inset:.2f} (fries+fried_chicken food-us, n={total}) ===")
        print(f"top1={top1/total*100:.1f}%")
        for cls in CLASSES:
            pc = per_class[cls]
            print(f"  {cls:<16} {pc.get(cls,0)}/{sum(pc.values())} -> {dict(pc)}", flush=True)


if __name__ == "__main__":
    main()

