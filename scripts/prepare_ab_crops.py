"""FASE 20: prepara crops IDÃ‰NTICOS por imagen para el A/B de clasificadores.

- food-us-v0.1: regiones de benchmarks/detection/regions.json (hybrid ya calculado, determinista).
- food-bench-v1: detecciÃ³n hybrid en vivo con RESUME por imagen (crops .jpg).

Salida: datasets/crops-foodus/{img}.jpg y datasets/crops-v1/{cls}/{img}.jpg
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.hybrid_detector import GroundingDinoDetector, HybridFoodDetector  # noqa: E402
from app.models.yolo_food_detector import YoloFoodDetector  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
FOOD_US = BASE / "datasets" / "food-us-v0.1" / "images" / "train"
V1 = BASE / "datasets" / "food-bench-v1"
OUT_US = BASE / "datasets" / "crops-foodus"
OUT_V1 = BASE / "datasets" / "crops-v1"
REGIONS = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))


def crop(image: Image.Image, box, padding: float) -> Image.Image:
    x1 = max(0, int(box.x - box.width * padding))
    y1 = max(0, int(box.y - box.height * padding))
    x2 = min(image.width, int(box.x + box.width * (1 + padding)))
    y2 = min(image.height, int(box.y + box.height * (1 + padding)))
    return image.crop((x1, y1, x2, y2))


def main() -> None:
    settings = Settings()
    detector = HybridFoodDetector(
        YoloFoodDetector(Settings(model_path="weights/yolo11n.pt")),
        GroundingDinoDetector(settings.dino_model, settings.dino_prompt, settings.dino_threshold),
    )
    detector.load()

    done_us = 0
    for img_path in sorted(FOOD_US.glob("*.jpg")):
        rec = REGIONS[img_path.name]
        regions_list = rec["yolo"] or rec["dino"]
        if not regions_list:
            continue
        best = max(regions_list, key=lambda r: r.get("conf", 0.5))
        image = Image.open(img_path).convert("RGB")
        box = type("B", (), {"x": best["x"], "y": best["y"], "width": best["width"], "height": best["height"]})()
        OUT_US.mkdir(exist_ok=True)
        crop(image, box, settings.clip_crop_padding).save(OUT_US / f"{img_path.stem}.jpg")
        done_us += 1
    print(f"[food-us] crops: {done_us}")

    v1_meta = {}
    for cls_dir in sorted(V1.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name == "_meta":
            continue
        out_cls = OUT_V1 / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(cls_dir.glob("*.jpg")):
            out_path = out_cls / img_path.name
            if out_path.exists():
                continue
            t0 = time.perf_counter()
            image = Image.open(img_path).convert("RGB")
            detections = detector.detect(image)
            if not detections:
                continue
            best = max(detections, key=lambda d: d.confidence)
            crop(image, best.bounding_box, settings.clip_crop_padding).save(out_path)
            v1_meta[img_path.name] = {"class": cls_dir.name, "used_dino": detector.used_dino_fallback,
                                      "det_ms": round((time.perf_counter() - t0) * 1000)}
    total = sum(1 for _ in OUT_V1.rglob("*.jpg"))
    print(f"[v1] crops: {total}")


if __name__ == "__main__":
    main()

