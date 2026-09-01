"""FASE 21: identidad CLIP sobre Food-101 subset (3500 imÃ¡genes, 14 clases).

EvaluaciÃ³n de identidad a escala: imagen completa como entrada (Food-101 son
fotos centradas del plato). Reporta top-1/3/5 por clase + dÃ³nde falla CLIP.
Non-commercial: evaluaciÃ³n interna Ãºnicamente.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
SUBSET = BASE / "datasets" / "food101-subset"
OUT = BASE / "benchmarks/classification/benchmark_food101_clip_results.json"


def main() -> None:
    from app.core.config import Settings

    settings = Settings()
    classifier = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    classifier.load()

    per_class = defaultdict(lambda: Counter())
    total = top1 = top3 = top5 = 0
    t0 = time.perf_counter()
    rows = []
    for cls_dir in sorted(SUBSET.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for img_path in sorted(cls_dir.glob("*.jpg")):
            image = Image.open(img_path).convert("RGB")
            ranking = classifier._score_crop(image)
            names = [r.name for r in ranking]
            if not names:
                per_class[cls]["unknown"] += 1
                rows.append({"image": f"{cls}/{img_path.name}", "gt": cls, "top1": "unknown", "top5": []})
                total += 1
                continue
            total += 1
            top1 += names[0] == cls
            top3 += cls in names[:3]
            top5 += cls in names[:5]
            per_class[cls][names[0]] += 1
            rows.append({"image": f"{cls}/{img_path.name}", "gt": cls, "top1": names[0], "top5": names[:5]})
        hits = per_class[cls][cls]
        n = sum(per_class[cls].values())
        print(f"  {cls:<18} {hits}/{n} top1={hits/n*100:.1f}% -> top confusions: {per_class[cls].most_common(3)}", flush=True)

    print(f"\n=== CLIP FOOD-101 subset ({total}) ===")
    print(f"top1={top1/total*100:.1f}% top3={top3/total*100:.1f}% top5={top5/total*100:.1f}% tiempo={time.perf_counter()-t0:.0f}s")
    OUT.write_text(json.dumps({"total": total, "top1": top1 / total, "top3": top3 / total,
                               "top5": top5 / total, "per_class": {k: dict(v) for k, v in per_class.items()}},
                              ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
