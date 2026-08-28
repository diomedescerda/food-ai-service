"""Benchmark end-to-end de cobertura nutricional (FASE 13).

Mide por etapa sobre las 108 imágenes (regions.json ya generado):
  Detection → Classification (CLIP) → Nutrition mapping → Portion → Cálculo
y reporta la métrica de producto: Nutrition End-to-End Success Rate, con la
taxonomía de fallos por etapa.

Uso: python scripts/benchmark_e2e.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, sys.path[0] and str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app.models.basic_portion_estimator import REFERENCE_GRAMS  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
REGIONS = json.loads((BASE_DIR / "regions.json").read_text(encoding="utf-8"))
CURATED = json.loads((
    BASE_DIR.parent / "coppAddresdBack" / "src" / "CoppAddresd.Api" / "Seeders" / "data" / "food_usda_curated.json"
).read_text(encoding="utf-8"))

GT = {"french_fries": "french_fries", "fried_chicken": "fried_chicken", "hot_dog": "hot_dog",
      "pizza": "pizza", "sandwich": "sandwich", "hamburger": "hamburger"}

# Alimentos con nutrition en el JSON curado (alias → nutrition disponible)
NUTRITION_AVAILABLE = {
    f["alias"] for f in CURATED["foods"] if f.get("nutrition") is not None
}


def main() -> None:
    clip = ZeroShotFoodClassifier(
        model_name="openai/clip-vit-base-patch32", device="cpu",
        threshold=0.20, crop_padding=0.10, top_k=5,
        prompt_templates_extra=("a picture of {food}", "a close-up photo of {food}"),
    )
    clip.load()

    stats = {
        "detection_ok": 0, "classification_ok": 0, "mapping_ok": 0,
        "portion_ok": 0, "calc_ok": 0,
        "unknown_rate": 0, "total": len(REGIONS),
    }
    failures: dict[str, int] = {}
    rows = []

    for fname, rec in REGIONS.items():
        gt = GT[rec["class"]]
        row = {"food": gt, "detection": True, "classification": None,
               "mapping": None, "portion": None, "calc": None, "failure": None}

        # 1) Detection (híbrido: YOLO o DINO)
        regions = rec["yolo"] or rec["dino"]
        if not regions:
            row.update(detection=False, failure="DETECTION_FAILED")
            failures["DETECTION_FAILED"] = failures.get("DETECTION_FAILED", 0) + 1
            rows.append(row)
            continue
        stats["detection_ok"] += 1

        # 2) Classification (CLIP sobre el crop de mayor score)
        best = max(regions, key=lambda r: r["conf"])
        img = Image.open(BASE_DIR / "datasets" / "food-us-v0.1" / "images" / "train" / fname).convert("RGB")
        from app.models.detection import BoundingBox, Detection
        top = clip.classify_with_candidates(img, Detection("x", 0.9, BoundingBox(
            best["x"], best["y"], best["width"], best["height"])))
        predicted = top[0].name
        row["classification"] = predicted
        if predicted == "unknown":
            stats["unknown_rate"] += 1
            row.update(failure="CLASSIFICATION_UNKNOWN")
            failures["CLASSIFICATION_UNKNOWN"] = failures.get("CLASSIFICATION_UNKNOWN", 0) + 1
            rows.append(row)
            continue
        if predicted != gt:
            row.update(failure="CLASSIFICATION_WRONG")
            failures["CLASSIFICATION_WRONG"] = failures.get("CLASSIFICATION_WRONG", 0) + 1
            rows.append(row)
            continue
        stats["classification_ok"] += 1
        row["classification_ok"] = True

        # 3) Nutrition mapping
        if gt not in NUTRITION_AVAILABLE:
            row.update(mapping=False, failure="NUTRITION_MAPPING_MISSING")
            failures["NUTRITION_MAPPING_MISSING"] = failures.get("NUTRITION_MAPPING_MISSING", 0) + 1
            rows.append(row)
            continue
        stats["mapping_ok"] += 1
        row["mapping"] = True

        # 4) Portion (referencia de porción del alimento)
        if gt not in REFERENCE_GRAMS:
            row.update(portion=False, failure="PORTION_UNAVAILABLE")
            failures["PORTION_UNAVAILABLE"] = failures.get("PORTION_UNAVAILABLE", 0) + 1
            rows.append(row)
            continue
        stats["portion_ok"] += 1
        row["portion"] = True

        # 5) Cálculo nutricional
        stats["calc_ok"] += 1
        row["calc"] = True

        rows.append(row)

    n = stats["total"]
    print(f"=== END-TO-END NUTRITION COVERAGE (108 imágenes) ===")
    print(f"Detection success:        {stats['detection_ok']}/{n} ({round(stats['detection_ok']/n*100,1)}%)")
    print(f"Classification success:   {stats['classification_ok']}/{n} ({round(stats['classification_ok']/n*100,1)}%)")
    print(f"Unknown rate:             {stats['unknown_rate']}/{n} ({round(stats['unknown_rate']/n*100,1)}%)")
    print(f"Nutrition mapping ok:     {stats['mapping_ok']}/{n} ({round(stats['mapping_ok']/n*100,1)}%)")
    print(f"Portion ok:               {stats['portion_ok']}/{n} ({round(stats['portion_ok']/n*100,1)}%)")
    print(f"Nutrition calculation ok: {stats['calc_ok']}/{n} ({round(stats['calc_ok']/n*100,1)}%)")
    print(f"\nE2E SUCCESS RATE = {round(stats['calc_ok']/n*100,1)}%")
    print(f"\n=== ERROR TAXONOMY ===")
    for error, count in sorted(failures.items(), key=lambda kv: -kv[1]):
        print(f"  {error}: {count}")
    print(f"\n=== POR CLASE (E2E ok / total) ===")
    by_class: dict[str, list] = {}
    for row in rows:
        by_class.setdefault(row["food"], []).append(row)
    for cls, cls_rows in by_class.items():
        ok = sum(1 for r in cls_rows if r["calc"])
        print(f"  {cls}: {ok}/{len(cls_rows)}")


if __name__ == "__main__":
    main()