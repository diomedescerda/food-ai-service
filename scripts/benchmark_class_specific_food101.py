"""FASE 21: A/B dirigido — candidatos descriptivos para taco/quesadilla/
waffles/pancakes (confusiones sistemáticas detectadas en Food-101).

Regla F17: solo se acepta mejora en ground-truth accuracy, no en score.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.models.food_catalog import FOOD_CATALOG  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
SUBSET = BASE / "datasets" / "food101-subset"

CLASSES = ["taco", "quesadilla", "waffles", "pancakes"]

EXTRA = {
    "taco": ("crispy corn taco shell with meat and lettuce", "taco al pastor with cilantro"),
    "quesadilla": ("folded grilled tortilla with melted cheese inside", "quesadilla cut in triangles"),
    "waffles": ("waffle with square grid pattern", "waffles with syrup in the squares"),
    "pancakes": ("pancake stack with butter and syrup", "flat round pancakes stacked"),
}
EXTRA_MAP = {
    "crispy corn taco shell with meat and lettuce": "taco",
    "taco al pastor with cilantro": "taco",
    "folded grilled tortilla with melted cheese inside": "quesadilla",
    "quesadilla cut in triangles": "quesadilla",
    "waffle with square grid pattern": "waffles",
    "waffles with syrup in the squares": "waffles",
    "pancake stack with butter and syrup": "pancakes",
    "flat round pancakes stacked": "pancakes",
}


def load_clip():
    import torch  # noqa: PLC0415
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


TEMPLATES = ("a photo of {food}", "a picture of {food}", "a close-up photo of {food}")


def run(model, processor, candidates: list[str], extra_map: dict) -> dict:
    import torch  # noqa: PLC0415

    prompts = [t.format(food=c) for t in TEMPLATES for c in candidates]
    with torch.no_grad():
        text = processor(text=prompts, padding=True, return_tensors="pt")
        tf = model.get_text_features(**text).pooler_output
        tf = tf / tf.norm(dim=-1, keepdim=True)

    def canon(cand: str) -> str:
        for e in FOOD_CATALOG:
            if cand in e.clip_candidates:
                return e.canonical_name
        return extra_map.get(cand, cand)

    per_class = defaultdict(lambda: Counter())
    total = top1 = 0
    for cls in CLASSES:
        cls_dir = SUBSET / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.glob("*.jpg")):
            image = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                img_in = processor(images=image, return_tensors="pt")
                imgf = model.get_image_features(**img_in).pooler_output
                imgf = imgf / imgf.norm(dim=-1, keepdim=True)
            scores = (imgf @ tf.T).view(len(TEMPLATES), len(candidates)).mean(dim=0)
            class_scores = {}
            for i, cand in enumerate(candidates):
                c = canon(cand)
                class_scores[c] = max(class_scores.get(c, 0.0), float(scores[i]))
            best = max(class_scores, key=class_scores.get)
            total += 1
            top1 += best == cls
            per_class[cls][best] += 1
    return {"total": total, "top1": top1 / total, "per_class": {k: dict(v) for k, v in per_class.items()}}


def main() -> None:
    model, processor = load_clip()
    base = [c for e in FOOD_CATALOG for c in e.clip_candidates]
    extended = list(base) + [c for extra in EXTRA.values() for c in extra]

    t0 = time.perf_counter()
    base_res = run(model, processor, base, {})
    print("=== BASELINE (candidatos catálogo) ===")
    for cls in CLASSES:
        pc = base_res["per_class"].get(cls, {})
        print(f"  {cls:<12} {pc.get(cls, 0)}/{sum(pc.values())} = {pc.get(cls, 0)/max(sum(pc.values()),1)*100:.1f}%")

    ext_res = run(model, processor, extended, EXTRA_MAP)
    print("=== EXTENDED (candidatos descriptivos para 4 clases) ===")
    for cls in CLASSES:
        pc = ext_res["per_class"].get(cls, {})
        n = sum(pc.values())
        print(f"  {cls:<12} {pc.get(cls, 0)}/{n} = {pc.get(cls, 0)/max(n,1)*100:.1f}% -> {Counter(pc).most_common(3)}")

    print(f"tiempo={time.perf_counter()-t0:.0f}s")
    with open(BASE / "benchmark_class_specific_food101.json", "w", encoding="utf-8") as fh:
        json.dump({"base": base_res, "extended": ext_res, "extra": EXTRA}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()