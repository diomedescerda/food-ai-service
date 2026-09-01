"""F27.3-5: visual prototype retrieval Ã¢â‚¬â€ A/B hybrid scoring (text + visual).

Por imagen: retrieval de texto (top-N, F26) + similitud visual contra prototipos
de las clases candidatas Ã¢â€ â€™ hybrid (1-w)*text + w*visual. Los scores se guardan
por imagen; el barrido de (w, aggregation) es post-proceso (sin re-inferencia).

Baselines: legacy 38 (pipeline) y F26 (text-only 231).

Uso: python scripts/ab_prototype.py
"""
import json
import os
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
TRAIN = BASE / "datasets/food-us-v0.1/images/train"
CROPS_V1 = BASE / "datasets/crops-v1"
SUBSET = BASE / "datasets/food101-subset"
REGIONS = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
MASTER = json.loads((BASE / "data/catalogs/food_master.json").read_text(encoding="utf-8"))
PROTOS = json.loads((BASE / os.environ.get("PROTO_STORE", "data/prototypes/f27_prototypes.json")).read_text(encoding="utf-8"))

TOP_N = int(os.environ.get("RETRIEVAL_TOP_N", "10"))
WEIGHTS = (0.25, 0.50, 0.75)
AGGREGATIONS = ("max", "mean", "topk5_mean", "medoid")


class HybridModel:
    def __init__(self) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
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
        self.classes: list[str] = [f["canonical_name"] for f in MASTER["foods"]]
        cands_by_class = {f["canonical_name"]: list(f["clip_candidates"]) for f in MASTER["foods"]}
        self.cand_idx: dict[str, list[int]] = {}
        all_cands: list[str] = []
        seen: dict[str, int] = {}
        for canon, cands in cands_by_class.items():
            idx = []
            for c in cands:
                if c not in seen:
                    seen[c] = len(all_cands)
                    all_cands.append(c)
                idx.append(seen[c])
            self.cand_idx[canon] = idx
        self.candidates = all_cands
        self.retrieval_feats: np.ndarray = self._text_feats(settings.clip_prompt_template)
        # Prototipos: clase -> matriz (n, 512)
        self.protos: dict[str, np.ndarray] = {
            cls: np.array([p["embedding"] for p in items], dtype=np.float32)
            for cls, items in PROTOS.items()
        }

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

    def scores_for(self, image: Image.Image) -> tuple[dict[str, float], dict[str, float]]:
        """text_scores (top-N clases) + visual_scores (de las mismas con prototipos)."""
        img = self._image_feats(image)
        raw = img @ self.retrieval_feats.T
        text: dict[str, float] = {}
        for canon, idxs in self.cand_idx.items():
            text[canon] = max(float(raw[i]) for i in idxs)
        top = sorted(text, key=text.get, reverse=True)[:TOP_N]
        visual: dict[str, float] = {}
        for cls in top:
            protos = self.protos.get(cls)
            if protos is None:
                continue
            sims = protos @ img
            visual[cls] = {
                "max": float(sims.max()),
                "mean": float(sims.mean()),
                "topk5_mean": float(np.sort(sims)[-5:].mean()),
                "medoid": float(sims[np.argmin(np.linalg.norm(protos - protos.mean(axis=0), axis=1))]),
            }
        return text, visual


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


def legacy_ranking(clf: ZeroShotFoodClassifier, image: Image.Image) -> list[tuple[str, float]]:
    return [(r.name, r.score) for r in clf._score_crop(image)]


def main() -> None:
    settings = Settings()
    legacy = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    legacy.load()
    hm = HybridModel()

    results: dict = {}
    import sys as _sys
    sources_arg = _sys.argv[1:] or ("food-us", "v1", "food101")
    for source in sources_arg:
        rows = []
        legacy_total = legacy_top1 = 0
        t0 = time.perf_counter()
        for image, cls in iterate(source):
            r = legacy_ranking(legacy, image)
            legacy_total += 1
            legacy_top1 += r[0][0] == cls if r else False
            text, visual = hm.scores_for(image)
            rows.append({"cls": cls, "text": text, "visual": visual})
        legacy_lat = (time.perf_counter() - t0) * 1000 / max(legacy_total, 1)
        print(f"=== {source}: legacy top1={legacy_top1/legacy_total*100:.1f}% ===", flush=True)

        # Barrido post-proceso
        for w in WEIGHTS:
            for agg in AGGREGATIONS:
                top1 = top3 = top5 = 0
                for row in rows:
                    finals: dict[str, float] = {}
                    for cls, ts in row["text"].items():
                        vs = row["visual"].get(cls, {}).get(agg)
                        finals[cls] = (1 - w) * ts + (w * vs if vs is not None else w * ts)
                    ranked = sorted(finals, key=finals.get, reverse=True)
                    top1 += ranked[0] == row["cls"]
                    top3 += row["cls"] in ranked[:3]
                    top5 += row["cls"] in ranked[:5]
                n = len(rows)
                key = f"{source}_w{w}_agg{agg}"
                results[key] = {"top1": top1 / n, "top3": top3 / n, "top5": top5 / n}
        results[f"{source}_legacy"] = {"top1": legacy_top1 / legacy_total}
        results[f"{source}_legacy_lat"] = legacy_lat

        # Reporte compacto por fuente
        best = max((k for k in results if k.startswith(f"{source}_w")),
                   key=lambda k: results[k]["top1"])
        r = results[best]
        print(f"  mejor hybrid: {best} top1={r['top1']*100:.1f}% top3={r['top3']*100:.1f}% top5={r['top5']*100:.1f}%", flush=True)

    out = BASE / "benchmarks/f27/classification/ab_prototype.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("guardado:", out)


if __name__ == "__main__":
    main()



