"""FASE 26: A/B retrieval coarse-to-fine â€” recall@N + top-1/3/5 vs legacy 38.

Retrieval score (por clase) = max coseno tpl-1 de los candidates de la clase.
Detailed score (por clase)  = max coseno ensemble (3 templates) de los candidates
                              SOLO de las top-N clases del retrieval.
Uso: python scripts/ab_retrieval.py
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "datasets" / "food-us-v0.1" / "images" / "train"
CROPS_V1 = BASE / "datasets" / "crops-v1"
SUBSET = BASE / "datasets" / "food101-subset"
REGIONS = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
MASTER = json.loads((BASE / "data/catalogs/food_master.json").read_text(encoding="utf-8"))

NS = (5, 10, 20, 30, 50)


class RetrievalModel:
    """CLIP con retrieval en memoria: clases del master con candidates."""

    def __init__(self) -> None:
        import torch  # noqa: PLC0415

        settings = Settings()
        clf = ZeroShotFoodClassifier(
            "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
            crop_padding=settings.clip_crop_padding,
            prompt_template=settings.clip_prompt_template,
            prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
            top_k=999,
        )
        clf.load()
        self.clf = clf
        self.torch = torch
        self.classes: list[str] = [f["canonical_name"] for f in MASTER["foods"]]
        self.candidates_by_class = {f["canonical_name"]: list(f["clip_candidates"]) for f in MASTER["foods"]}
        self.class_to_cand_idx: dict[str, list[int]] = {}
        all_cands: list[str] = []
        seen: dict[str, int] = {}
        for canon, cands in self.candidates_by_class.items():
            idx: list[int] = []
            for c in cands:
                if c not in seen:
                    seen[c] = len(all_cands)
                    all_cands.append(c)
                idx.append(seen[c])
            self.class_to_cand_idx[canon] = idx
        self.candidates = all_cands
        self.candidate_class = {}
        for canon, cands in self.candidates_by_class.items():
            for c in cands:
                self.candidate_class[c] = canon
        self.retrieval_feats: np.ndarray = self._text_feats(template=settings.clip_prompt_template)
        self.detailed_feats: list[np.ndarray] = [
            self._text_feats(t) for t in clf._prompt_templates
        ]

    def _text_feats(self, template: str) -> np.ndarray:
        prompts = [template.format(food=c) for c in self.candidates]
        inputs = self.clf._processor(text=prompts, padding=True, return_tensors="pt")
        with self.torch.no_grad():
            feats = self.clf._features(self.clf._model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.numpy()

    def _image_feats(self, image: Image.Image) -> np.ndarray:
        inputs = self.clf._processor(images=image, return_tensors="pt")
        with self.torch.no_grad():
            feats = self.clf._features(self.clf._model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.numpy()[0]

    def retrieval_scores(self, image: Image.Image) -> dict[str, float]:
        img = self._image_feats(image)
        scores = img @ self.retrieval_feats.T
        by_class: dict[str, float] = {}
        for canon, idxs in self.class_to_cand_idx.items():
            by_class[canon] = max(float(scores[i]) for i in idxs)
        return by_class

    def detailed_ranking(self, image: Image.Image, top_n: int) -> list[tuple[str, float]]:
        """Ensemble mean por candidate + max por clase, SOLO sobre las top-N."""
        by_class = self.retrieval_scores(image)
        top_classes = sorted(by_class, key=by_class.get, reverse=True)[:top_n]
        img = self._image_feats(image)
# mean de los 3 templates por candidate
        cand_mean: dict[str, float] = {}
        for cand in self.candidates:
            cls = self.candidate_class[cand]
            if cls not in top_classes:
                continue
            sums = []
            for template_feats in self.detailed_feats:
                sums.append(float(template_feats[self.candidates.index(cand)] @ img))
            cand_mean[cand] = sum(sums) / len(sums)
        ranked: dict[str, float] = {}
        for cand, score in cand_mean.items():
            cls = self.candidate_class[cand]
            ranked[cls] = max(ranked.get(cls, 0.0), score)
        return sorted(ranked.items(), key=lambda x: -x[1])


def legacy_ranking(clf: ZeroShotFoodClassifier, image: Image.Image) -> list[tuple[str, float]]:
    ranking = clf._score_crop(image)
    return [(r.name, r.score) for r in ranking]


def run_legacy(clf, source: str) -> dict:
    total = top1 = 0
    for img_path, cls in iterate(source):
        ranking = legacy_ranking(clf, img_path)
        pred = ranking[0][0] if ranking else "unknown"
        total += 1
        top1 += pred == cls
    return {"total": total, "top1": top1 / total}


def iterate(source: str):
    if source == "food-us":
        for img_path in sorted(TRAIN.glob("*.jpg")):
            cls = img_path.stem.rsplit("_", 1)[0]
            rec = REGIONS[img_path.name]
            regions_list = rec["yolo"] or rec["dino"]
            if not regions_list:
                continue
            best = max(regions_list, key=lambda r: r.get("conf", 0.5))
            image = Image.open(img_path).convert("RGB")
            crop_img = image.crop((max(0, int(best["x"] - best["width"] * 0.1)),
                                   max(0, int(best["y"] - best["height"] * 0.1)),
                                   min(image.width, int(best["x"] + best["width"] * 1.1)),
                                   min(image.height, int(best["y"] + best["height"] * 1.1))))
            yield crop_img, cls
    elif source == "v1":
        for img_path in sorted(CROPS_V1.rglob("*.jpg")):
            yield Image.open(img_path).convert("RGB"), img_path.parent.name
    else:
        for cls_dir in sorted(SUBSET.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for img_path in sorted(cls_dir.glob("*.jpg")):
                yield Image.open(img_path).convert("RGB"), cls


def main() -> None:
    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()
    rm = RetrievalModel()

    results: dict = {}
    for source in ("food-us", "v1", "food101"):
        # Legacy 38
        t0 = time.perf_counter()
        legacy = run_legacy(clf, source)
        legacy_lat = (time.perf_counter() - t0) * 1000 / max(legacy["total"], 1)
        print(f"=== {source}: legacy_38 top1={legacy['top1']*100:.1f}% ({legacy['total']}) ===", flush=True)

        # Retrieval con distintos N
        for n in NS:
            total = top1 = top3 = top5 = 0
            recall = {k: 0 for k in NS}
            lat = []
            for image, cls in iterate(source):
                t0 = time.perf_counter()
                by_class = rm.retrieval_scores(image)
                top_classes = sorted(by_class, key=by_class.get, reverse=True)
                ranked = rm.detailed_ranking(image, n)
                lat.append((time.perf_counter() - t0) * 1000)
                total += 1
                if cls in [x[0] for x in ranked[:5]]:
                    top5 += 1
                if cls in [x[0] for x in ranked[:3]]:
                    top3 += 1
                if ranked and ranked[0][0] == cls:
                    top1 += 1
                for k in NS:
                    if cls in top_classes[:k]:
                        recall[k] += 1
            r = {"total": total, "top1": top1 / total, "top3": top3 / total, "top5": top5 / total,
                 "recall": {k: v / total for k, v in recall.items()},
                 "lat_ms": sum(lat) / len(lat)}
            results[f"{source}_N{n}"] = r
            print(f"  N={n:<3} recall@5={r['recall'][5]*100:.1f}% @10={r['recall'][10]*100:.1f}% "
                  f"@20={r['recall'][20]*100:.1f}% @30={r['recall'][30]*100:.1f}% @50={r['recall'][50]*100:.1f}% "
                  f"top1={r['top1']*100:.1f}% top3={r['top3']*100:.1f}% top5={r['top5']*100:.1f}% "
                  f"lat={r['lat_ms']:.0f}ms", flush=True)
        results[f"{source}_legacy"] = legacy
        results[f"{source}_legacy_lat"] = legacy_lat

    out = BASE / "benchmarks/classification/ab_retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("guardado:", out)


if __name__ == "__main__":
    main()
