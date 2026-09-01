"""FASE 22: A/B candidates descriptivos dirigidos por evidencia de confusiÃ³n.

Variante A (P1): fries/fried_chicken â€” distinguir FORMA (tiras vs piezas).
Variante B (P2-P4): hot_dog/hamburger/sandwich, taco/quesadilla/nachos,
salmon/steak/grilled_chicken â€” distinciÃ³n visual de pan/presentaciÃ³n.

Mide sobre 3 fuentes: food-us (benchmarks/detection/regions.json), v1 (crops), Food-101 (imÃ¡genes)
con el pipeline real (score por clase, ensemble). Sin modificar el catÃ¡logo.

Uso: python scripts/ab_candidates_v2.py A|B
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.food_catalog import FOOD_CATALOG, candidate_to_canonical  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "datasets" / "food-us-v0.1" / "images" / "train"
CROPS_V1 = BASE / "datasets" / "crops-v1"
SUBSET = BASE / "datasets" / "food101-subset"
REGIONS = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))

VARIANT_A = {
    "french_fries": ("thin fried potato strips", "long golden potato sticks"),
    "fried_chicken": ("large fried chicken piece", "battered chicken piece"),
}
VARIANT_B = {
    "hot_dog": ("hot dog sausage in a long bun",),
    "hamburger": ("hamburger with a round sesame bun",),
    "sandwich": ("sandwich between two slices of bread",),
    "nachos": ("nachos with melted cheese on tortilla chips",),
    "salmon": ("grilled salmon fillet with pink flesh",),
    "steak": ("grilled beef steak on a plate",),
    "grilled_chicken": ("grilled chicken breast pieces",),
}
EXTRA_MAP = {c: cls for cls, cands in {**VARIANT_A, **VARIANT_B}.items() for c in cands}


def build_classifier(extra_cands: dict):
    import torch  # noqa: PLC0415

    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()
    # Extras despuÃ©s del load: el load() reconstruye _candidates del catÃ¡logo.
    extras = [c for cands in extra_cands.values() for c in cands]
    clf._candidates = list(clf._candidates) + extras
    clf._text_features_list = []
    with torch.no_grad():
        for template in clf._prompt_templates:
            prompts = [template.format(food=f) for f in clf._candidates]
            text_inputs = clf._processor(text=prompts, padding=True, return_tensors="pt").to(clf._device)
            features = clf._features(clf._model.get_text_features(**text_inputs))
            clf._text_features_list.append(features / features.norm(dim=-1, keepdim=True))
    return clf


def canonical_of(candidate: str) -> str:
    mapped = candidate_to_canonical(candidate)
    if mapped in ("", candidate):
        return EXTRA_MAP.get(candidate, candidate)
    return mapped


def run_dataset(clf, source: str) -> dict:
    per_class = defaultdict(Counter)
    total = top1 = 0
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
            top_cands = clf._score_crop(crop_img)
            pred = canonical_of(top_cands[0].name) if top_cands else "unknown"
            total += 1
            top1 += pred == cls
            per_class[cls][pred] += 1
    elif source == "v1":
        for img_path in sorted(CROPS_V1.rglob("*.jpg")):
            cls = img_path.parent.name
            crop_img = Image.open(img_path).convert("RGB")
            top_cands = clf._score_crop(crop_img)
            pred = canonical_of(top_cands[0].name) if top_cands else "unknown"
            total += 1
            top1 += pred == cls
            per_class[cls][pred] += 1
    else:
        for cls_dir in sorted(SUBSET.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for img_path in sorted(cls_dir.glob("*.jpg")):
                image = Image.open(img_path).convert("RGB")
                top_cands = clf._score_crop(image)
                pred = canonical_of(top_cands[0].name) if top_cands else "unknown"
                total += 1
                top1 += pred == cls
                per_class[cls][pred] += 1
    return {"total": total, "top1": top1 / total, "per_class": {k: dict(v) for k, v in per_class.items()}}


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "A"
    extra = VARIANT_A if variant == "A" else VARIANT_B

    t0 = time.perf_counter()
    clf = build_classifier(extra)
    clf.load()

    results = {}
    for source in ("food-us", "v1", "food101"):
        r = run_dataset(clf, source)
        results[source] = r
        print(f"=== {variant} â€” {source} ({r['total']}) top1={r['top1']*100:.1f}% ===", flush=True)
        for cls in extra:
            pc = r["per_class"].get(cls, {})
            if pc:
                print(f"  {cls:<16} {pc.get(cls,0)}/{sum(pc.values())} = {pc.get(cls,0)/max(sum(pc.values()),1)*100:.1f}% -> {Counter(pc).most_common(2)}", flush=True)
    print(f"tiempo={time.perf_counter()-t0:.0f}s")
    with open(BASE / f"benchmarks/history/ab_candidates_v2_{variant}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


