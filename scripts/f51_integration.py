"""F51: benchmark de integración — pipeline 5.761 completo (multi-text +
grouping + reranker F48 + specialist DINO) vs F50 (retrieval solo).

Uso: python scripts/f51_integration.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.food_pipeline import FoodPipeline  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]

FOOD_US = BASE / "datasets/food-us-v0.1/images/train"
V1 = BASE / "datasets/food-bench-v1"
F101 = BASE / "datasets/food101-subset"


def main() -> None:
    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()
    pipeline = FoodPipeline(clf=clf, enabled=True)
    print(f"pipeline disponible: {pipeline.available()} (índice {pipeline.index.shape})", flush=True)

    datasets = [
        ("food-us", [(str(p), p.stem.rsplit("_", 1)[0].replace("_", " ")) for p in sorted(FOOD_US.glob("*.jpg"))]),
        ("v1", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(V1.rglob("*.jpg"))]),
        ("food101", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(F101.rglob("*.jpg"))]),
    ]
    ks = (1, 5, 10, 20, 50)
    for dname, items in datasets:
        hits = {k: 0 for k in ks}
        n = 0
        spec = 0
        fallback = 0
        lats = []
        t0 = time.perf_counter()
        for path, gt in items:
            image = Image.open(path).convert("RGB")
            res = pipeline.analyze_food(image)
            lats.append(res.get("latency_ms", 0))
            if res.get("fallback", False):
                fallback += 1
                continue
            if res.get("specialist_used"):
                spec += 1
            final = [res["canonical_name"]]
            final.extend(res["retrieval"]["top10"])
            n += 1
            for k in ks:
                hits[k] += gt in final[:k]
        lats.sort()
        print(f"{dname}: R@1={hits[1]/n*100:.1f} R@5={hits[5]/n*100:.1f} "
              f"R@10={hits[10]/n*100:.1f} R@20={hits[20]/n*100:.1f} R@50={hits[50]/n*100:.1f} "
              f"| spec={spec} fallback={fallback} | p50={lats[len(lats)//2]:.0f}ms "
              f"p95={lats[int(len(lats)*0.95)]:.0f}ms ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()