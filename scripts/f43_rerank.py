"""F43: canonical grouping + reranking del Food Retrieval.

1. retrieval top-50 (entries) -> agrupar por canonical_name
2. canonical_candidates: max/mean/top2-mean + support_count
3. reranker: final = w_r*retrieval + w_l*legacy + w_s*specialist (pizza/naan)
4. evaluación: entry R@K vs canonical R@K + unique canonicals + duplicados

Uso: python scripts/f43_rerank.py [--eval] [--errors]
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.food_retrieval import FoodRetrieval  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]

CONFIGS = {
    "A": (1.0, 0.0, 0.0),
    "B": (0.75, 0.25, 0.0),
    "C": (0.70, 0.20, 0.10),
    "D": (0.60, 0.25, 0.15),
}

FOOD_US = BASE / "datasets/food-us-v0.1/images/train"
V1 = BASE / "datasets/food-bench-v1"
F101 = BASE / "datasets/food101-subset"
LEGACY_NAMES = {"pizza", "hamburger", "hot_dog", "french_fries", "fried_chicken", "sandwich",
                "taco", "quesadilla", "nachos", "salmon", "steak", "grilled_chicken"}


def group_entries(rank: list[dict], agg: str = "max") -> list[dict]:
    """Entries (top-50) -> canonical_candidates."""
    groups: dict[str, list[float]] = {}
    for c in rank:
        groups.setdefault(c["name"], []).append(c["score"])
    cands = []
    for name, scores in groups.items():
        scores.sort(reverse=True)
        if agg == "max":
            best = scores[0]
        elif agg == "mean":
            best = float(np.mean(scores))
        else:  # top2_mean
            best = float(np.mean(scores[:2]))
        cands.append({"canonical_name": name, "best_score": best, "support_count": len(scores)})
    cands.sort(key=lambda c: -c["best_score"])
    return cands


def legacy_scores(clf, image, names: set[str]) -> dict[str, float]:
    ranking = clf._score_crop(image)
    out: dict[str, float] = {}
    for r in ranking:
        canon = r.name.replace("_", " ")
        if canon in names and canon not in out:
            out[canon] = r.score
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()
    ret = FoodRetrieval(enabled=True, clf=clf)

    ks = (1, 5, 10, 20, 50)
    result: dict[str, dict] = {name: {"entry": {k: 0 for k in ks}, "entry_t": {k: 0 for k in ks},
                                       "canonical": {k: 0 for k in ks}, "canonical_t": {k: 0 for k in ks},
                                       "rerank": {k: 0 for k in ks}, "rerank_t": {k: 0 for k in ks},
                                       "unique10": [], "unique50": []}
                               for name in ("food-us", "v1", "food101")}

    datasets = [
        ("food-us", [(str(p), p.stem.rsplit("_", 1)[0].replace("_", " ")) for p in sorted(FOOD_US.glob("*.jpg"))]),
        ("v1", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(V1.rglob("*.jpg"))]),
        ("food101", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(F101.rglob("*.jpg"))]),
    ]
    t0 = time.perf_counter()
    for dname, items in datasets:
        r = result[dname]
        for path, gt in items:
            image = Image.open(path).convert("RGB")
            rank = ret.retrieve(image, top_k=50)
            # canonical (max)
            cands = group_entries(rank, "max")
            canon_names = [c["canonical_name"] for c in cands]
            # legacy scores de las clases en el top
            leg = legacy_scores(clf, image, set(canon_names[:20]) & LEGACY_NAMES)
            # rerank: canonical con pesos A (solo retrieval) para comparar; B para el rerank final
            for k in ks:
                r["entry_t"][k] += 1
                r["entry"][k] += gt in [c["name"] for c in rank[:k]]
                r["canonical_t"][k] += 1
                r["canonical"][k] += gt in canon_names[:k]
                r["rerank_t"][k] += 1
                r["rerank"][k] += gt in canon_names[:k]  # base: grouping sin rerank (A)
            r["unique10"].append(len(set(canon_names[:10])))
            r["unique50"].append(len(set(canon_names[:50])))
        print(f"{dname}: entry R@1={r['entry'][1]/r['entry_t'][1]*100:.1f}% "
              f"canonical R@1={r['canonical'][1]/r['canonical_t'][1]*100:.1f}% "
              f"R@5={r['canonical'][5]/r['canonical_t'][5]*100:.1f}% "
              f"R@10={r['canonical'][10]/r['canonical_t'][10]*100:.1f}% "
              f"R@20={r['canonical'][20]/r['canonical_t'][20]*100:.1f}% "
              f"R@50={r['canonical'][50]/r['canonical_t'][50]*100:.1f}% "
              f"unique10={np.mean(r['unique10']):.1f} unique50={np.mean(r['unique50']):.1f}")
    print(f"tiempo total: {time.perf_counter()-t0:.0f}s")

    out = BASE / "benchmarks/f43/reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "f43_grouping.json").write_text(json.dumps(
        {k: {m: {kk: vv / v.get(f"{m}_t", {}).get(kk, 1) for kk, vv in v[m].items() if v.get(f"{m}_t", {}).get(kk)}
              for m in ("entry", "canonical", "rerank")} for k, v in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()