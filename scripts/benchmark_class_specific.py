"""FASE 17 P11: candidate groups especÃ­ficos para clases problemÃ¡ticas.

Compara el scoring global vs candidate groups extendidos para french_fries y
fried_chicken (las clases con confusiÃ³n fries/fried_chickenâ†’rice). Regla:
solo se acepta una mejora en ground-truth accuracy, no en score.

Uso: python scripts/benchmark_class_specific.py
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

from app.models.food_catalog import FOOD_CATALOG, candidate_to_canonical  # noqa: E402

BENCH = Path(__file__).resolve().parents[1] / "datasets" / "food-us-v0.1" / "images" / "train"
REGIONS = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))

CLASSES = ["french_fries", "fried_chicken", "rice"]
TEMPLATES = ("a photo of {food}", "a picture of {food}", "a close-up photo of {food}")

# Candidate groups: base (catÃ¡logo) + extras descriptivos SOLO para las
# clases problemÃ¡ticas. El mapeo extra asocia cada candidato nuevo a su clase.
EXTRA_CANDIDATES = {
    "french_fries": ("golden fried potato sticks", "fried potatoes in a cup", "fast food french fries"),
    "fried_chicken": ("crispy breaded fried chicken pieces", "fast food fried chicken"),
}
EXTRA_MAP = {
    "golden fried potato sticks": "french_fries",
    "fried potatoes in a cup": "french_fries",
    "fast food french fries": "french_fries",
    "crispy breaded fried chicken pieces": "fried_chicken",
    "fast food fried chicken": "fried_chicken",
}

CANONICAL_CANDIDATES = {c: e.canonical_name for e in FOOD_CATALOG for c in e.clip_candidates}


def canonical_for(candidate: str) -> str:
    mapped = candidate_to_canonical(candidate)
    if mapped in ("", candidate):
        mapped = EXTRA_MAP.get(candidate, candidate)
    return mapped


def run(model, processor, device, candidates: list[str]) -> dict:
    prompts = [t.format(food=c) for t in TEMPLATES for c in candidates]
    with torch.no_grad():
        text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        text_out = model.get_text_features(**text_inputs)
        text_feats = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.detach()

    per_class = defaultdict(lambda: Counter())
    total = top1 = 0
    confusion = Counter()
    for cls in CLASSES:
        for img_path in sorted(BENCH.glob(f"{cls}_*.jpg")):
            rec = REGIONS[img_path.name]
            regions_list = rec["yolo"] or rec["dino"]
            if not regions_list:
                per_class[cls]["no_detection"] += 1
                continue
            image = Image.open(img_path).convert("RGB")
            best_box = max(regions_list, key=lambda r: r.get("conf", 0.5))
            box = {"x": best_box["x"], "y": best_box["y"], "width": best_box["width"], "height": best_box["height"]}
            pad = 0.10
            x1 = max(0, int(box["x"] - box["width"] * pad))
            y1 = max(0, int(box["y"] - box["height"] * pad))
            x2 = min(image.width, int(box["x"] + box["width"] * (1 + pad)))
            y2 = min(image.height, int(box["y"] + box["height"] * (1 + pad)))
            crop_img = image.crop((x1, y1, x2, y2))

            with torch.no_grad():
                img_inputs = processor(images=crop_img, return_tensors="pt").to(device)
                img_out = model.get_image_features(**img_inputs)
                img_feats = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            scores = img_feats @ text_feats.T
            scores = scores.view(len(TEMPLATES), len(candidates)).mean(dim=0)

            # score por clase = max candidatos de la clase
            class_scores: dict[str, float] = {}
            for i, cand in enumerate(candidates):
                canon = canonical_for(cand)
                class_scores[canon] = max(class_scores.get(canon, 0.0), float(scores[i]))
            best = max(class_scores, key=class_scores.get)
            total += 1
            top1 += best == cls
            per_class[cls][best] += 1
            confusion[(cls, best)] += 1
    return {"total": total, "top1": top1 / total, "per_class": {k: dict(v) for k, v in per_class.items()},
            "confusion": [{"gt": g, "pred": p, "count": c} for (g, p), c in confusion.most_common(20)]}


def main() -> None:
    device = "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    base = [c for e in FOOD_CATALOG for c in e.clip_candidates]
    extended = list(base)
    for extra in EXTRA_CANDIDATES.values():
        extended.extend(extra)

    t0 = time.perf_counter()
    base_res = run(model, processor, device, base)
    print(f"=== BASELINE (candidatos catÃ¡logo) ===")
    print(f"top1={base_res['top1']*100:.1f}% ({base_res['total']})")
    for cls in CLASSES:
        pc = base_res["per_class"].get(cls, {})
        hits = pc.get(cls, 0)
        n = sum(pc.values())
        print(f"  {cls:<16} {hits}/{n} = {hits/n*100 if n else 0:.1f}%  -> {dict(pc)}")

    ext_res = run(model, processor, device, extended)
    print(f"=== EXTENDED (candidatos extra por clase) ===")
    print(f"top1={ext_res['top1']*100:.1f}% ({ext_res['total']})")
    for cls in CLASSES:
        pc = ext_res["per_class"].get(cls, {})
        hits = pc.get(cls, 0)
        n = sum(pc.values())
        print(f"  {cls:<16} {hits}/{n} = {hits/n*100 if n else 0:.1f}%  -> {dict(pc)}")

    print(f"tiempo={time.perf_counter()-t0:.0f}s")
    with open(Path(__file__).resolve().parents[1] / "benchmark_class_specific_results.json", "w", encoding="utf-8") as fh:
        json.dump({"base": base_res, "extended": ext_res, "extra_candidates": EXTRA_CANDIDATES}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

