"""Benchmark experimental de clasificación CLIP (FASE 14).

Compara configuraciones (una variable a la vez) contra el baseline de
FASE 13 (prompt "a photo of {food}", candidatos globales, padding 0.05,
bbox crop, threshold 0.22) sobre las 108 imágenes de food-us-v0.1.

Cada experimento guarda su JSON en benchmark_clasification_tuning/.

Uso: python scripts/benchmark_tuning.py [--quick]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

from app.models.food_catalog import FOOD_CATALOG  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "benchmark_clasification_tuning"
OUT_DIR.mkdir(exist_ok=True)

REGIONS = json.loads((BASE_DIR / "regions.json").read_text(encoding="utf-8"))
GT = {"french_fries": "french_fries", "fried_chicken": "fried_chicken", "hot_dog": "hot_dog",
      "pizza": "pizza", "sandwich": "sandwich", "hamburger": "hamburger"}

DEFAULT_PROMPT = "a photo of {food}"
DEFAULT_PADDING = 0.05
DEFAULT_THRESHOLD = 0.22

CATEGORY_GROUPS = {
    "fast food": ["hamburger", "hot_dog", "french_fries", "fried_chicken", "sandwich",
                  "chicken_nuggets", "pizza", "taco", "burrito", "quesadilla", "nachos"],
    "breakfast": ["eggs", "bacon", "toast", "bagel", "pancakes", "waffles", "oatmeal", "cereal"],
    "main dish": ["steak", "grilled_chicken", "salmon", "rice", "pasta", "lasagna", "mac_and_cheese", "soup"],
    "dessert": ["donut", "cake", "cookie", "brownie", "ice_cream"],
    "fruit": ["apple", "banana", "orange"],
    "vegetable": ["broccoli", "carrot", "salad"],
}

SPECIFIC_PROMPTS = {
    "hot_dog": ("hot dog",),
    "french_fries": ("french fries", "french fries on a plate"),
    "sandwich": ("sandwich", "sandwich with bread and filling"),
    "fried_chicken": ("fried chicken", "fried chicken pieces"),
}


def canonical_for_candidate(candidate: str) -> str:
    for entry in FOOD_CATALOG:
        if candidate in entry.clip_candidates:
            return entry.canonical_name
    return candidate


def build_text_features(model, processor, prompts: list[str], device: str) -> torch.Tensor:
    inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            features = out.pooler_output
        else:
            features = out.last_hidden_state[:, 0]
    return features / features.norm(dim=-1, keepdim=True)


def image_features(model, processor, image: Image.Image, device: str) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            features = out.pooler_output
        else:
            features = out.last_hidden_state[:, 0]
    return features / features.norm(dim=-1, keepdim=True)


def crop(image: Image.Image, box: dict, padding: float) -> Image.Image:
    x1 = max(0, int(box["x"] - box["width"] * padding))
    y1 = max(0, int(box["y"] - box["height"] * padding))
    x2 = min(image.width, int(box["x"] + box["width"] * (1 + padding)))
    y2 = min(image.height, int(box["y"] + box["height"] * (1 + padding)))
    return image.crop((x1, y1, x2, y2))


def masked_crop(image: Image.Image, box: dict, mask_img: Image.Image, padding: float) -> Image.Image:
    """Crop con máscara (fondo eliminado): máscara binaria aplicada sobre fondo negro."""
    cropped = crop(image, box, padding)
    mask = mask_img.resize(cropped.size, Image.NEAREST).convert("L")
    black = Image.new("RGB", cropped.size, (0, 0, 0))
    return Image.composite(cropped, black, mask)


def topk_accuracy(preds: list[list[str]], gt: list[str], k: int) -> float:
    hits = sum(1 for p, g in zip(preds, gt) if g in p[:k])
    return hits / len(gt)


def run_experiment(model, processor, device, config: dict, images: dict) -> dict:
    prompt = config.get("prompt", DEFAULT_PROMPT)
    padding = config.get("padding", DEFAULT_PADDING)
    threshold = config.get("threshold", DEFAULT_THRESHOLD)
    specific = config.get("specific_prompts", False)
    hierarchical = config.get("hierarchical", False)
    masked = config.get("masked", False)
    masks = config.get("masks", {})  # filename -> dict index -> PIL mask

    # Construir candidatos (prompts) según la configuración
    candidates: list[str] = []
    if hierarchical:
        category_prompts = [f"a photo of {cat} food" for cat in CATEGORY_GROUPS]
        category_names = list(CATEGORY_GROUPS.keys())
    else:
        all_candidates = []
        for entry in FOOD_CATALOG:
            if specific and entry.canonical_name in SPECIFIC_PROMPTS:
                all_candidates.extend(SPECIFIC_PROMPTS[entry.canonical_name])
            else:
                all_candidates.extend(entry.clip_candidates)
        candidates = all_candidates

    if hierarchical:
        text_prompts = category_prompts
    else:
        text_prompts = [prompt.format(food=c) for c in candidates]

    text_features = build_text_features(model, processor, text_prompts, device)

    preds: list[list[str]] = []
    scores_top1: list[float] = []
    latencies = []
    for fname, rec in REGIONS.items():
        gt = GT[rec["class"]]
        regions = rec["yolo"] or rec["dino"]
        if not regions:
            preds.append([])
            continue
        best = max(regions, key=lambda r: r["conf"])
        image = images[fname]
        t0 = time.perf_counter()
        if masked and fname in masks:
            crop_img = masked_crop(image, best, masks[fname], padding)
        else:
            crop_img = crop(image, best, padding)
        img_feat = image_features(model, processor, crop_img, device)
        scores = (img_feat @ text_features.T).squeeze(0)

        if hierarchical:
            cat_idx = int(scores.argmax())
            category = category_names[cat_idx]
            group = CATEGORY_GROUPS[category]
            group_prompts = [prompt.format(food=c) for c in group]
            group_features = build_text_features(model, processor, group_prompts, device)
            group_scores = (img_feat @ group_features.T).squeeze(0)
            ranked = [canonical_for_candidate(group[i]) for i in group_scores.argsort(descending=True).tolist()]
            scores_top1.append(float(scores.max()))
        else:
            ranked = [canonical_for_candidate(candidates[i]) for i in scores.argsort(descending=True).tolist()]
            scores_top1.append(float(scores[0]))
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append(ranked)

    gt_all = [GT[r["class"]] for r in REGIONS.values()]
    top1 = topk_accuracy(preds, gt_all, 1)
    top3 = topk_accuracy(preds, gt_all, 3)
    top5 = topk_accuracy(preds, gt_all, 5)

    # Unknown con threshold (rank-based top-1 score < threshold)
    unknown = sum(1 for s in scores_top1 if s < threshold)
    # Aciertos entre los no-unknown
    classified = [p for p, s in zip(preds, scores_top1) if s >= threshold]
    classified_gt = [g for g, s in zip(gt_all, scores_top1) if s >= threshold]
    acc_classified = topk_accuracy(classified, classified_gt, 1) if classified else 0.0

    return {
        "config": config,
        "top1": round(top1, 3),
        "top3": round(top3, 3),
        "top5": round(top5, 3),
        "unknown_rate": round(unknown / len(preds), 3),
        "accuracy_classified": round(acc_classified, 3),
        "latency_ms": round(sum(latencies) / len(latencies), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    images = {}
    for fname in REGIONS:
        path = BASE_DIR / "datasets" / "food-us-v0.1" / "images" / "train" / fname
        with Image.open(path) as img:
            images[fname] = img.convert("RGB")

    # Máscaras (solo para el experimento masked): YOLO-seg sobre imágenes con región YOLO
    masks: dict[str, dict] = {}
    if not args.quick:
        print("[mask] generando máscaras con YOLO-seg...")
        from app.core.config import Settings
        from app.models.yolo_food_segmenter import YoloFoodSegmenter

        segmenter = YoloFoodSegmenter(Settings(model_path="weights/yolo11n-seg.pt"))
        segmenter.load()
        from app.models.detection import BoundingBox, Detection

        for fname, rec in REGIONS.items():
            if not rec["yolo"]:
                continue
            image = images[fname]
            best = max(rec["yolo"], key=lambda r: r["conf"])
            segs = segmenter.segment(image, [Detection("x", 0.9, BoundingBox(
                best["x"], best["y"], best["width"], best["height"]))])
            if segs[0] is not None:
                import base64
                from io import BytesIO

                mask_img = Image.open(BytesIO(base64.b64decode(segs[0].mask))).convert("L")
                masks[fname] = mask_img
        print(f"[mask] {len(masks)} máscaras")

    device = "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    experiments = [
        {"name": "baseline", "prompt": DEFAULT_PROMPT, "padding": DEFAULT_PADDING},
        {"name": "prompt_picture", "prompt": "a picture of {food}"},
        {"name": "prompt_closeup", "prompt": "a close-up photo of {food}"},
        {"name": "prompt_plate", "prompt": "a photo of a plate of {food}"},
        {"name": "prompt_serving", "prompt": "a serving of {food}"},
        {"name": "prompt_ensemble_mean", "ensemble_prompts": [
            "a photo of {food}", "a picture of {food}", "a close-up photo of {food}"], "agg": "mean"},
        {"name": "prompt_specific", "specific_prompts": True},
        {"name": "hierarchical", "hierarchical": True},
        {"name": "padding_0.0", "padding": 0.0},
        {"name": "padding_0.10", "padding": 0.10},
        {"name": "padding_0.15", "padding": 0.15},
        {"name": "padding_0.20", "padding": 0.20},
        {"name": "masked_crop", "masked": True},
    ]
    if args.quick:
        experiments = experiments[:6]

    results = {}
    for exp in experiments:
        name = exp.pop("name")
        print(f"[exp] {name}...")
        if "ensemble_prompts" in exp:
            prompts = exp.pop("ensemble_prompts")
            agg = exp.pop("agg")
            # Ensemble: se computa por imagen con las N plantillas
            result = run_ensemble(model, processor, device, prompts, agg, exp, images)
        else:
            result = run_experiment(model, processor, device, exp, images)
        results[name] = result
        print(f"      {result}")
        (OUT_DIR / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {name: {k: v for k, v in res.items() if k in ("top1", "top3", "top5", "unknown_rate", "latency_ms")}
               for name, res in results.items()}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== RESUMEN ===")
    print(json.dumps(summary, indent=2))


def run_ensemble(model, processor, device, prompt_templates, agg, config, images):
    """Ensemble de plantillas: features de texto por plantilla; media/max de scores."""
    padding = config.get("padding", DEFAULT_PADDING)
    threshold = config.get("threshold", DEFAULT_THRESHOLD)
    candidates = [c for e in FOOD_CATALOG for c in e.clip_candidates]

    text_features_list = [
        build_text_features(model, processor, [t.format(food=c) for c in candidates], device)
        for t in prompt_templates
    ]

    preds, scores_top1, latencies = [], [], []
    for fname, rec in REGIONS.items():
        gt = GT[rec["class"]]
        regions = rec["yolo"] or rec["dino"]
        if not regions:
            preds.append([])
            continue
        best = max(regions, key=lambda r: r["conf"])
        t0 = time.perf_counter()
        img_feat = image_features(model, processor, crop(images[fname], best, padding), device)
        all_scores = [img_feat @ tf.T for tf in text_features_list]
        if agg == "mean":
            scores = sum(all_scores) / len(all_scores)
        else:
            scores = torch.stack(all_scores).max(dim=0).values
        scores = scores.squeeze(0)
        ranked = [canonical_for_candidate(candidates[i]) for i in scores.argsort(descending=True).tolist()]
        scores_top1.append(float(scores[0]))
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append(ranked)

    gt_all = [GT[r["class"]] for r in REGIONS.values()]
    unknown = sum(1 for s in scores_top1 if s < threshold)
    return {
        "config": {"ensemble": prompt_templates, "agg": agg},
        "top1": round(topk_accuracy(preds, gt_all, 1), 3),
        "top3": round(topk_accuracy(preds, gt_all, 3), 3),
        "top5": round(topk_accuracy(preds, gt_all, 5), 3),
        "unknown_rate": round(unknown / len(preds), 3),
        "latency_ms": round(sum(latencies) / len(latencies), 1),
    }


if __name__ == "__main__":
    main()