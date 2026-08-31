"""FASE 18: benchmark multi-food POR INSTANCIA.

Evoluciona el GT de datasets/multi-food de clases a instancias
(ground_truth_foods: food + instance_id) y mide:
- instance recall / instance precision
- missed instances / false positives / duplicate detections
- doble conteo: imágenes con >=2 instancias de la misma clase deben producir
  tantas entradas nutricionales como instancias reales.

GT por instancia (⚠ provisional — evidencia espacial FASE 17, sin revisión
visual): regiones de la misma clase con IoU ~0 y sin contención se anotaron
como instancias separadas (mf_003: 2 hamburguesas; mf_006: 2 huevos).
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.hybrid_detector import GroundingDinoDetector, HybridFoodDetector  # noqa: E402
from app.models.yolo_food_detector import YoloFoodDetector  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

IMAGES = Path(__file__).resolve().parents[1] / "datasets" / "multi-food" / "images"
META = Path(__file__).resolve().parents[1] / "datasets" / "multi-food" / "metadata.json"

# GT por instancia (clase, evidencia espacial F17). 1 instancia por clase
# salvo donde se demostraron regiones separadas del mismo alimento.
GT_INSTANCES = {
    "mf_000": [("hamburger", "hamburger_1")],
    "mf_001": [("hamburger", "hamburger_1"), ("french_fries", "fries_1")],
    "mf_002": [("hamburger", "hamburger_1"), ("french_fries", "fries_1")],
    "mf_003": [("hamburger", "hamburger_1"), ("hamburger", "hamburger_2")],
    "mf_004": [("eggs", "eggs_1")],
    "mf_005": [("bacon", "bacon_1"), ("eggs", "eggs_1")],
    "mf_006": [("bacon", "bacon_1"), ("eggs", "eggs_1"), ("eggs", "eggs_2")],
    "mf_007": [("bacon", "bacon_1"), ("eggs", "eggs_1")],
    "mf_008": [("french_fries", "fries_1")],
    "mf_009": [],
    "mf_010": [("french_fries", "fries_1"), ("pizza", "pizza_1")],
    "mf_011": [("french_fries", "fries_1"), ("pizza", "pizza_1"), ("hamburger", "hamburger_1")],
    "mf_012": [],
    "mf_013": [("rice", "rice_1"), ("salad", "salad_1")],
    "mf_014": [],
    "mf_015": [],
}


def main() -> None:
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

    metadata = {}
    total_gt = tp = fp = missed = dups = 0
    double_count_ok = 0
    double_count_cases = 0

    for fname, gt_list in sorted(GT_INSTANCES.items()):
        image = Image.open(IMAGES / f"{fname}.jpg").convert("RGB")
        detections = detector.detect(image)
        results = classifier.classify(image, detections)
        preds = [r.name for r in results if r is not None and r.name != "unknown"]

        gt_classes = Counter(food for food, _ in gt_list)
        pred_counter = Counter(preds)
        per_class_tp = sum(min(gt_classes[c], pred_counter[c]) for c in gt_classes)
        per_class_extra = sum(max(0, pred_counter[c] - gt_classes[c]) for c in set(pred_counter) | set(gt_classes))
        dups += sum(max(0, pred_counter[c] - gt_classes[c]) for c in gt_classes)
        fp += sum(max(0, pred_counter[c] - gt_classes[c]) for c in set(pred_counter) - set(gt_classes))
        fp += sum(max(0, pred_counter[c] - gt_classes[c]) for c in gt_classes)
        missed += len(gt_classes) - sum(1 for c in gt_classes if pred_counter[c] > 0)
        total_gt += len(gt_list)
        tp += per_class_tp

        # Doble conteo: clase con >=2 instancias GT debe producir >=2 preds.
        multi = {c: n for c, n in gt_classes.items() if n >= 2}
        if multi:
            double_count_cases += 1
            ok = all(pred_counter[c] >= n for c, n in multi.items())
            double_count_ok += ok

        metadata[fname] = {
            "ground_truth_foods": [{"food": f, "instance_id": i} for f, i in gt_list],
            "predicted_foods": preds,
            "food_count": len(gt_list),
            "multi_food": len(gt_list) > 1,
        }
        print(f"{fname}: gt={[f for f, _ in gt_list]} preds={preds} dup={max(0, len(preds) - len(set(preds)))}")

    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    n_images = len(GT_INSTANCES)
    print("\n=== MULTI-FOOD POR INSTANCIA (16 imágenes, GT evidencia espacial F17) ===")
    print(f"Instances GT:      {total_gt}")
    print(f"Instance recall:   {tp}/{total_gt} = {tp/total_gt*100:.1f}%")
    print(f"Instance precision:{tp}/{tp+fp} = {tp/(tp+fp)*100:.1f}%")
    print(f"Missed instances:  {missed}")
    print(f"False positives:   {fp}")
    print(f"Duplicate dets:    {dups}")
    print(f"Doble conteo (clases multi-instancia GT): {double_count_ok}/{double_count_cases} correctas")
    print(f"metadata.json actualizado: {META}")


if __name__ == "__main__":
    main()