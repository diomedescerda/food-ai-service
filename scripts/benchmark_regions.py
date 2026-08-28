"""Benchmark de regiones de comida (FASE 11).

Genera regiones por imagen con YOLO11n y Grounding DINO (fallback) sobre las
108 imágenes del benchmark y guarda regiones.json para el E2E CLIP posterior.

Métricas: imágenes con >=1 región (recall de localización), regiones/imagen,
latencia por etapa. RESUME: si el JSON parcial existe, continúa.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.yolo_food_detector import YoloFoodDetector  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = BASE_DIR / "datasets" / "food-us-v0.1" / "images" / "train"
REGIONS_FILE = BASE_DIR / "regions.json"

BENCHMARK_CLASSES = ["french_fries", "fried_chicken", "hamburger", "hot_dog", "pizza", "sandwich"]
DINO_PROMPT = "food on a plate"
DINO_THRESHOLD = 0.15


def main() -> None:
    images = sorted(BENCHMARK_DIR.glob("*.jpg"))
    items = []
    for path in images:
        cls = path.name.rsplit("_", 1)[0]
        if cls in BENCHMARK_CLASSES:
            items.append((path, cls))

    existing: dict[str, dict] = {}
    if REGIONS_FILE.exists():
        existing = json.loads(REGIONS_FILE.read_text(encoding="utf-8"))
        print(f"[resume] regiones previas: {len(existing)}")

    detector = YoloFoodDetector(Settings(model_path="weights/yolo11n.pt"))
    detector.load()
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")

    stats = {"yolo_crops": 0, "dino_fallback": 0, "total_regions_yolo": 0, "total_regions_dino": 0}
    latencies = {"yolo": [], "dino": []}

    for path, cls in items:
        filename = path.name
        if filename in existing:
            continue
        with Image.open(path) as img:
            image = img.convert("RGB")

        record: dict = {"class": cls, "yolo": [], "dino": []}

        t0 = time.perf_counter()
        detections = detector.detect(image)
        latencies["yolo"].append((time.perf_counter() - t0) * 1000)
        for d in detections:
            record["yolo"].append({
                "x": d.bounding_box.x, "y": d.bounding_box.y,
                "width": d.bounding_box.width, "height": d.bounding_box.height,
                "name": d.name, "conf": d.confidence,
            })
        stats["total_regions_yolo"] += len(detections)
        if detections:
            stats["yolo_crops"] += 1

        # DINO fallback SOLO cuando YOLO no detectó nada (híbrido) — pero para
        # el benchmark también se mide DINO puro sobre todas las imágenes.
        t0 = time.perf_counter()
        inputs = dino_processor(images=image, text=DINO_PROMPT, return_tensors="pt")
        with torch.no_grad():
            out = dino_model(**inputs)
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = dino_processor.post_process_grounded_object_detection(
            out, threshold=DINO_THRESHOLD, target_sizes=target_sizes)[0]
        latencies["dino"].append((time.perf_counter() - t0) * 1000)
        for box, score in zip(results["boxes"].tolist(), results["scores"].tolist()):
            x1, y1, x2, y2 = box
            record["dino"].append({
                "x": int(x1), "y": int(y1),
                "width": int(x2 - x1), "height": int(y2 - y1),
                "conf": round(float(score), 3),
            })
        stats["total_regions_dino"] += len(record["dino"])
        if not detections and record["dino"]:
            stats["dino_fallback"] += 1

        existing[filename] = record
        if len(existing) % 10 == 0:
            REGIONS_FILE.write_text(json.dumps(existing), encoding="utf-8")
            print(f"[progreso] {len(existing)}/{len(items)}")

    REGIONS_FILE.write_text(json.dumps(existing), encoding="utf-8")

    n = len(items)
    yolo_regions = sum(1 for r in existing.values() if r["yolo"])
    dino_regions = sum(1 for r in existing.values() if r["dino"])
    hybrid_regions = sum(1 for r in existing.values() if r["yolo"] or r["dino"])

    print(f"\n=== REGION RECALL (108 imágenes, clase única por imagen) ===")
    print(f"YOLO11n:            {yolo_regions}/{n} imágenes con región ({round(yolo_regions/n*100,1)}%)")
    print(f"DINO-tiny ('{DINO_PROMPT}'): {dino_regions}/{n} ({round(dino_regions/n*100,1)}%)")
    print(f"Híbrido YOLO+DINO:  {hybrid_regions}/{n} ({round(hybrid_regions/n*100,1)}%)")
    print(f"Regiones/imagen: YOLO={round(stats['total_regions_yolo']/n,1)} DINO={round(stats['total_regions_dino']/n,1)}")
    print(f"Latencia: YOLO={round(sum(latencies['yolo'])/len(latencies['yolo']),0)}ms DINO={round(sum(latencies['dino'])/len(latencies['dino']),0)}ms")
    print(f"Guardado: {REGIONS_FILE}")


if __name__ == "__main__":
    main()