"""Benchmark multi-food (FASE 15): imágenes reales de platos compuestos.

Mide reconocimiento multi-alimento con el pipeline de producción (hybrid
YOLO→DINO + CLIP zero-shot ensemble): recall, precisión, duplicados, missed
foods y falsos positivos vs GT manual registrado en datasets/multi-food/.

Uso: python scripts/benchmark_multifood.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.detection import Detection  # noqa: E402
from app.models.hybrid_detector import GroundingDinoDetector, HybridFoodDetector  # noqa: E402
from app.models.yolo_food_detector import YoloFoodDetector  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
MF_DIR = BASE_DIR / "datasets" / "multi-food"

# GT manual mínimo por imagen (alimentos presentes, clases del catálogo).
# Registrado por nombre/categoría del archivo en Wikimedia Commons.
GT = {
    "mf_000": ["hamburger"],
    "mf_001": ["hamburger", "french_fries"],
    "mf_002": ["hamburger", "french_fries"],
    "mf_003": ["hamburger"],
    "mf_004": ["eggs"],
    "mf_005": ["bacon", "eggs"],
    "mf_006": ["bacon", "eggs"],
    "mf_007": ["bacon", "eggs"],
    "mf_008": ["french_fries"],
    "mf_009": [],  # costillas de cerdo: fuera del catálogo
    "mf_010": ["french_fries", "pizza"],
    "mf_011": ["french_fries", "pizza", "hamburger"],
    "mf_012": [],  # "chicken and company" — ambiguo sin más evidencia
    "mf_013": ["rice", "salad"],  # curry katsu + arroz + ensalada
    "mf_014": [],  # pollo nepalí: fuera del catálogo
    "mf_015": [],  # "easy lunch" — sin evidencia
}


def main() -> None:
    detector = HybridFoodDetector(
        yolo=YoloFoodDetector(Settings(model_path="weights/yolo11n.pt")),
        dino=GroundingDinoDetector(Settings().dino_model, Settings().dino_prompt, Settings().dino_threshold),
    )
    detector.load()

    classifier = ZeroShotFoodClassifier(
        model_name="openai/clip-vit-base-patch32", device="cpu",
        threshold=0.20, crop_padding=0.10, top_k=5,
        prompt_templates_extra=("a picture of {food}", "a close-up photo of {food}"),
    )
    classifier.load()

    total_gt = 0
    total_pred = 0
    total_hits = 0
    duplicates = 0
    missed = 0
    false_positives = 0
    latencies = []
    per_image = []

    for fname, gt in sorted(GT.items()):
        path = MF_DIR / "images" / f"{fname}.jpg"
        with Image.open(path) as img:
            image = img.convert("RGB")

        t0 = time.perf_counter()
        detections = detector.detect(image)
        results = classifier.classify(image, detections)
        latencies.append((time.perf_counter() - t0) * 1000)

        preds = [r.name for r in results if r is not None and r.name != "unknown"]

        gt_set = set(gt)
        pred_set = set(preds)
        hits = gt_set & pred_set
        total_gt += len(gt_set)
        total_pred += len(preds)
        total_hits += len(hits)
        missed += len(gt_set - pred_set)
        false_positives += len(pred_set - gt_set)
        dup = len(preds) - len(pred_set)  # mismo alimento en regiones distintas
        duplicates += dup

        per_image.append({
            "image": fname, "gt": gt, "preds": preds,
            "duplicates": dup, "missed": sorted(gt_set - pred_set),
            "fp": sorted(pred_set - gt_set),
        })
        print(f"{fname}: gt={gt} preds={preds} dup={dup} missed={sorted(gt_set - pred_set)} fp={sorted(pred_set - gt_set)}")

    n = len(GT)
    recall = total_hits / total_gt if total_gt else 0
    precision = total_hits / total_pred if total_pred else 0
    images_with_food = sum(1 for g in GT.values() if g)
    images_ok = sum(1 for p in per_image if p["gt"] and set(p["preds"]) >= set(p["gt"]))

    result = {
        "images": n,
        "images_with_food_gt": images_with_food,
        "images_fully_recovered": images_ok,
        "food_recall": round(recall, 3),
        "food_precision": round(precision, 3),
        "missed_foods": missed,
        "false_positives": false_positives,
        "duplicate_detections": duplicates,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "per_image": per_image,
    }
    (BASE_DIR / "multifood_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== MULTI-FOOD (16 imágenes, GT manual) ===")
    print(f"Food recall:     {total_hits}/{total_gt} = {round(recall*100,1)}%")
    print(f"Food precision: {total_hits}/{total_pred} = {round(precision*100,1)}%")
    print(f"Missed foods:   {missed}")
    print(f"False positives: {false_positives}")
    print(f"Duplicados:     {duplicates}")
    print(f"Imágenes con TODO su GT recuperado: {images_ok}/{images_with_food}")
    print(f"Latencia media: {result['avg_latency_ms']} ms")


if __name__ == "__main__":
    main()