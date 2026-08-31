"""FASE 19: benchmark food-bench-v1 con detector HYBRID (pipeline de producción).

YOLO → si no detecta → DINO fallback → crop → CLIP. Separa con datos:
- DETECTION FAILURE (ni YOLO ni DINO encuentran región)
- CLASSIFICATION FAILURE (región encontrada pero CLIP falla)

Mide: detection recall (yolo/dino/none), top-1/3/5, unknown, DINO fallback
rate, latencias por etapa. RESUME: el JSON parcial se continúa.

Uso: python scripts/benchmark_hybrid_v1.py [--only clase1 clase2 ...]
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.hybrid_detector import GroundingDinoDetector, HybridFoodDetector  # noqa: E402
from app.models.yolo_food_detector import YoloFoodDetector  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "food-bench-v1"
OUT = Path(__file__).resolve().parents[1] / "benchmark_hybrid_v1_results.json"


def crop(image: Image.Image, box, padding: float) -> Image.Image:
    x1 = max(0, int(box.x - box.width * padding))
    y1 = max(0, int(box.y - box.height * padding))
    x2 = min(image.width, int(box.x + box.width * (1 + padding)))
    y2 = min(image.height, int(box.y + box.height * (1 + padding)))
    return image.crop((x1, y1, x2, y2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    settings = Settings()
    detector = HybridFoodDetector(
        YoloFoodDetector(Settings(model_path="weights/yolo11n.pt")),
        GroundingDinoDetector(settings.dino_model, settings.dino_prompt, settings.dino_threshold),
    )
    detector.load()
    classifier = ZeroShotFoodClassifier(
        settings.clip_model, threshold=settings.clip_threshold, crop_padding=settings.clip_crop_padding
    )
    classifier.load()

    classes = sorted(d.name for d in DATASET.iterdir() if d.is_dir() and d.name != "_meta")
    if args.only:
        classes = [c for c in classes if c in args.only]

    existing = {}
    if OUT.exists():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"[resume] {len(existing)} imágenes ya procesadas")

    lat_yolo: list[float] = []
    lat_dino: list[float] = []
    lat_clip: list[float] = []
    total = top1 = top3 = top5 = unknown = yolo_hits = dino_fallback = none = 0
    per_class = defaultdict(lambda: Counter())

    for cls in classes:
        for img_path in sorted((DATASET / cls).glob("*.jpg")):
            key = f"{cls}/{img_path.name}"
            if key in existing:
                rec = existing[key]
                total += 1
                if rec["detector"] == "yolo":
                    yolo_hits += 1
                elif rec["detector"] == "dino":
                    dino_fallback += 1
                else:
                    none += 1
                if rec["top1"] == cls:
                    top1 += 1
                if cls in rec["top3"]:
                    top3 += 1
                if cls in rec["top5"]:
                    top5 += 1
                if rec["top1"] == "unknown":
                    unknown += 1
                per_class[cls][rec["top1"]] += 1
                continue

            image = Image.open(img_path).convert("RGB")
            t0 = time.perf_counter()
            detections = detector.detect(image)
            det_ms = (time.perf_counter() - t0) * 1000
            used_dino = detector.used_dino_fallback
            if used_dino:
                dino_fallback += 1
                lat_dino.append(det_ms)
            else:
                yolo_hits += 1
                lat_yolo.append(det_ms)

            total += 1
            rec = {"detector": "dino" if used_dino else "yolo", "yolo_ms": round(det_ms, 1),
                   "clip": [], "top1": "unknown", "top3": [], "top5": []}

            if not detections:
                none += 1
                per_class[cls]["no_detection"] += 1
                unknown += 1
            else:
                best = max(detections, key=lambda d: d.confidence)
                crop_img = crop(image, best.bounding_box, settings.clip_crop_padding)
                t0 = time.perf_counter()
                results = classifier.classify(crop_img, [best])
                lat_clip.append((time.perf_counter() - t0) * 1000)
                if results and results[0]:
                    ranking = [r.name for r in results]
                    rec["top1"] = ranking[0]
                    rec["top3"] = ranking[:3]
                    rec["top5"] = ranking[:5]
                    rec["clip"] = [{"name": r.name, "score": r.confidence} for r in results]
                    top1 += ranking[0] == cls
                    top3 += cls in ranking[:3]
                    top5 += cls in ranking[:5]
                    per_class[cls][ranking[0]] += 1
                else:
                    unknown += 1
                    per_class[cls]["unknown"] += 1

            existing[key] = rec
            if len(existing) % 25 == 0:
                OUT.write_text(json.dumps(existing), encoding="utf-8")
                print(f"[progreso] {len(existing)}")

    OUT.write_text(json.dumps(existing), encoding="utf-8")
    n_dino_lat = len(lat_dino) or 1
    print("\n=== FOOD-BENCH-V1 HYBRID (248, 31 clases, pipeline producción) ===")
    print(f"total={total} top1={top1/total*100:.1f}% top3={top3/total*100:.1f}% top5={top5/total*100:.1f}% unknown={unknown} ({unknown/total*100:.1f}%)")
    print(f"detection: yolo={yolo_hits} ({yolo_hits/total*100:.1f}%) dino_fallback={dino_fallback} ({dino_fallback/total*100:.1f}%) ninguna={none} ({none/total*100:.1f}%)")
    print(f"fallback_rate = {dino_fallback}/{total} = {dino_fallback/total*100:.1f}%")
    print(f"latencia: yolo_media={sum(lat_yolo)/len(lat_yolo):.0f}ms | dino_media={sum(lat_dino)/n_dino_lat:.0f}ms (n={len(lat_dino)}) | clip_media={sum(lat_clip)/max(len(lat_clip),1):.0f}ms")
    for cls in classes:
        pc = per_class[cls]
        n = sum(pc.values())
        hits = pc.get(cls, 0)
        print(f"  {cls:<20} {hits}/{n} top1={hits/n*100:.1f}% -> {dict(pc)}")


if __name__ == "__main__":
    main()