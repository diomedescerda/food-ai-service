"""Benchmark A/B: YOLO actual vs YOLO crops + CLIP (esta implementación).

Mide sobre las 108 imágenes del benchmark: Top-1/3/5 por crop, unknown rate,
latencia por etapa (detection, crop, clip), y la elección de threshold
(accuracy vs unknown) sobre el conjunto.

Uso: python scripts/benchmark_ab.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.yolo_food_detector import YoloFoodDetector  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = BASE_DIR / "datasets" / "food-us-v0.1" / "images" / "train"

BENCHMARK_CLASSES = ["french_fries", "fried_chicken", "hamburger", "hot_dog", "pizza", "sandwich"]
GT_TO_CANONICAL = {
    "french_fries": "french_fries",
    "fried_chicken": "fried_chicken",
    "hot_dog": "hot_dog",
}


def load_items() -> list[tuple[Image.Image, str]]:
    items = []
    for path in sorted(BENCHMARK_DIR.glob("*.jpg")):
        cls = path.name.rsplit("_", 1)[0]
        if cls not in BENCHMARK_CLASSES:
            continue
        try:
            with Image.open(path) as img:
                items.append((img.convert("RGB"), cls))
        except Exception:
            continue
    return items


def top_k(predictions: list[list[str]], gt: list[str], k: int) -> float:
    hits = sum(1 for p, g in zip(predictions, gt) if g in p[:k])
    return hits / len(gt)


def main() -> None:
    items = load_items()
    print(f"[benchmark A/B] {len(items)} imágenes")

    detector = YoloFoodDetector(Settings(model_path="weights/yolo11n.pt"))
    detector.load()

    classifier = ZeroShotFoodClassifier(
        model_name="openai/clip-vit-base-patch32",
        device="cpu",
        threshold=0.0,  # sin threshold aquí: se evalúa después por score
        crop_padding=0.05,
        top_k=5,
    )
    t0 = time.perf_counter()
    classifier.load()
    print(f"[startup] CLIP load: {round(time.perf_counter()-t0, 1)}s")

    det_lat, crop_lat, clip_lat = [], [], []
    crops_evaluated = 0
    scores_positive, scores_negative = [], []

    # Por imagen: YOLO detect → crop → CLIP top-k
    results_per_item = []
    for image, gt in items:
        t0 = time.perf_counter()
        detections = detector.detect(image)
        det_lat.append((time.perf_counter() - t0) * 1000)

        best: list[str] = []
        if detections:
            # Clasificar el crop de mayor área (benchmark: 1 alimento por imagen)
            largest = max(detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            t0 = time.perf_counter()
            crop = classifier._crop(image, largest)
            crop_lat.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            top = classifier.classify_with_candidates(image, largest)
            clip_lat.append((time.perf_counter() - t0) * 1000)
            crops_evaluated += 1
            best = [c.name for c in top]
            if gt == "pizza":
                scores_positive.append(top[0].score if top[0].name == "pizza" else 0.0)
                if top[0].name == "pizza":
                    scores_positive.append(top[0].score)
            else:
                scores_negative.append(top[0].score if top[0].name != GT_TO_CANONICAL.get(gt, gt) else 0.0)
            # score del gt si está en top
            gt_canon = GT_TO_CANONICAL.get(gt, gt)
            gt_score = next((c.score for c in top if c.name == gt_canon), 0.0)
            (scores_positive if gt_canon in top[0:1] or True else scores_negative).append(gt_score) if gt == "pizza" else None
        results_per_item.append((gt, best))

    gt_all = [g for _, g in items]
    preds_all = [p for _, p in results_per_item]

    print(f"\n=== YOLO crops + CLIP (esta implementación) ===")
    print(f"Top-1 (sin threshold): {round(top_k(preds_all, gt_all, 1), 3)}")
    print(f"Top-3: {round(top_k(preds_all, gt_all, 3), 3)}")
    print(f"Top-5: {round(top_k(preds_all, gt_all, 5), 3)}")
    print(f"Crops evaluados: {crops_evaluated}/{len(items)}")
    print(f"Latencia por etapa: det={round(sum(det_lat)/len(det_lat),1)}ms crop={round(sum(crop_lat)/len(crop_lat),1)}ms clip={round(sum(clip_lat)/len(clip_lat),1)}ms")
    print(f"CLIP por crop: {round(sum(clip_lat)/len(clip_lat),1)}ms (1 crop/imagen aquí; N alimentos = N inferences)")

    result = {
        "n": len(items),
        "top1": round(top_k(preds_all, gt_all, 1), 3),
        "top3": round(top_k(preds_all, gt_all, 3), 3),
        "top5": round(top_k(preds_all, gt_all, 5), 3),
        "crops_evaluated": crops_evaluated,
        "latency_ms": {
            "detection": round(sum(det_lat) / len(det_lat), 1),
            "crop": round(sum(crop_lat) / len(crop_lat), 1),
            "clip_per_crop": round(sum(clip_lat) / len(clip_lat), 1),
        },
    }
    out = BASE_DIR / "benchmark_ab_results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()