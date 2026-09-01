"""FASE 25 P6: A/B escalamiento del catálogo — CLIP baseline (38) vs master (231).

Mide degradación del regression set: food-us (regions), v1 (crops), Food-101.
Score POR CLASE (como producción). Uso: python scripts/ab_catalog_scale.py
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.food_catalog import candidate_to_canonical  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "datasets" / "food-us-v0.1" / "images" / "train"
CROPS_V1 = BASE / "datasets" / "crops-v1"
SUBSET = BASE / "datasets" / "food101-subset"
REGIONS = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
MASTER = json.loads((BASE / "data/catalogs/food_master.json").read_text(encoding="utf-8"))


def build_classifier(candidates: list[str] | None, top_k: int = 50):
    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
        top_k=top_k,
    )
    clf.load()
    if candidates is not None:
        import torch  # noqa: PLC0415

        clf._candidates = list(candidates)
        clf._text_features_list = []
        with torch.no_grad():
            for template in clf._prompt_templates:
                prompts = [template.format(food=f) for f in clf._candidates]
                text_inputs = clf._processor(text=prompts, padding=True, return_tensors="pt").to(clf._device)
                features = clf._features(clf._model.get_text_features(**text_inputs))
                clf._text_features_list.append(features / features.norm(dim=-1, keepdim=True))
    return clf


def rank_by_class(clf, crop_img: Image.Image) -> list[str]:
    """Score POR CLASE: max score de los candidatos de cada canonical."""
    ranking = clf._score_crop(crop_img)
    best_by_class: dict[str, float] = {}
    for r in ranking:
        canon = candidate_to_canonical(r.name)
        if canon in ("", r.name):
            canon = r.name
        if canon not in best_by_class or r.score > best_by_class[canon]:
            best_by_class[canon] = r.score
    return sorted(best_by_class, key=best_by_class.get, reverse=True)


def run(clf, source: str) -> dict:
    per_class = defaultdict(Counter)
    total = top1 = 0
    lat = []

    def score(image: Image.Image, cls: str) -> None:
        nonlocal total, top1
        t0 = time.perf_counter()
        ranking = clf._score_crop(image)
        lat.append((time.perf_counter() - t0) * 1000)
        pred = rank_by_class(clf, image)[0] if ranking else "unknown"
        total += 1
        top1 += pred == cls
        per_class[cls][pred] += 1

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
            score(crop_img, cls)
    elif source == "v1":
        for img_path in sorted(CROPS_V1.rglob("*.jpg")):
            cls = img_path.parent.name
            crop_img = Image.open(img_path).convert("RGB")
            score(crop_img, cls)
    else:
        for cls_dir in sorted(SUBSET.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for img_path in sorted(cls_dir.glob("*.jpg")):
                image = Image.open(img_path).convert("RGB")
                score(image, cls)

    n = len(lat) or 1
    return {"total": total, "top1": top1 / total,
            "per_class": {k: dict(v) for k, v in per_class.items()},
            "lat_ms": sum(lat) / n, "lat_max": max(lat) if lat else 0}


def main() -> None:
    master_candidates = [c for f in MASTER["foods"] for c in f["clip_candidates"]]
    base_candidates = [f["clip_candidates"][0] for f in MASTER["foods"]]
    results = {}
    for name, cands in (("baseline_38", None), ("master_231_descriptivo", master_candidates), ("master_231_base", base_candidates)):
        clf = build_classifier(cands)
        print(f"=== {name} ({len(clf._candidates)} candidates) ===", flush=True)
        for source in ("food-us", "v1", "food101"):
            r = run(clf, source)
            print(f"  {source:<9} top1={r['top1']*100:.1f}% ({r['total']}) lat={r['lat_ms']:.0f}ms max={r['lat_max']:.0f}ms", flush=True)
            results[f"{name}_{source}"] = r
        for cls in ("pizza", "hamburger", "hot_dog", "french_fries", "fried_chicken", "sandwich"):
            pc = results[f"{name}_food-us"]["per_class"].get(cls, {})
            hits = pc.get(cls, 0)
            print(f"    {cls:<16} {hits}/{sum(pc.values())}", flush=True)

    out = BASE / "benchmarks/classification/ab_catalog_scale.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()