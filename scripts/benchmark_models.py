"""Benchmark de modelos preentrenados para reconocimiento de alimentos.

Compara: YOLO11n (actual) vs CLIP ViT-B/32 vs SigLIP base (zero-shot) sobre
el conjunto benchmark (108 imágenes descargadas legalmente en FASE 10, 6
clases con ground truth por filename).

Métricas: Top-1/3/5, latencia CPU, cobertura. Uso: python scripts/benchmark_models.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = BASE_DIR / "datasets" / "food-us-v0.1" / "images" / "train"

# Clases del benchmark (ground truth del filename) + catálogo zero-shot amplio.
BENCHMARK_CLASSES = ["french_fries", "fried_chicken", "hamburger", "hot_dog", "pizza", "sandwich"]

ZERO_SHOT_CATALOG = [
    "pizza", "hamburger", "french fries", "hot dog", "sandwich", "fried chicken",
    "chicken nuggets", "salad", "pasta", "spaghetti", "steak", "rice", "pancakes",
    "waffles", "eggs", "bacon", "toast", "donut", "cake", "ice cream", "cookies",
    "brownie", "apple", "banana", "orange", "broccoli", "carrot", "taco", "burrito",
    "nachos", "salmon", "soup", "mac and cheese", "mashed potatoes",
]

# Mapeo de la clase de benchmark al nombre del catálogo zero-shot.
GT_TO_CATALOG = {
    "french_fries": "french fries",
    "fried_chicken": "fried chicken",
    "hot_dog": "hot dog",
}


def load_benchmark() -> list[tuple[Image.Image, str, str]]:
    """(imagen, clase_gt, filename)"""
    items = []
    for path in sorted(BENCHMARK_DIR.glob("*.jpg")):
        cls = path.name.rsplit("_", 1)[0]  # french_fries_000.jpg -> french_fries
        if cls not in BENCHMARK_CLASSES:
            continue
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            items.append((rgb, cls, path.name))
        except Exception:
            continue
    return items


def top_k_accuracy(predictions: list[list[str]], ground_truth: list[str], k: int) -> float:
    hits = sum(1 for preds, gt in zip(predictions, ground_truth) if gt in preds[:k])
    return hits / len(ground_truth) if ground_truth else 0.0


def benchmark_yolo(items: list[tuple[Image.Image, str, str]]) -> dict:
    """YOLO11n COCO: top-1 = clase detectada con mayor confianza (si alguna es food)."""
    from app.core.config import Settings
    from app.models.yolo_food_detector import YoloFoodDetector

    detector = YoloFoodDetector(Settings(model_path="weights/yolo11n.pt"))
    detector.load()

    predictions: list[list[str]] = []
    latencies = []
    for image, gt, name in items:
        t0 = time.perf_counter()
        detections = detector.detect(image)
        latencies.append((time.perf_counter() - t0) * 1000)
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            predictions.append([best.name])
        else:
            predictions.append(["unknown"])

    gt = [g for _, g, _ in items]
    evaluable = [i for i, (_, g, _) in enumerate(items) if g in ("pizza", "hot_dog", "sandwich")]
    eval_gt = [gt[i] for i in evaluable]
    eval_preds = [predictions[i] for i in evaluable]

    return {
        "top1": round(top_k_accuracy(predictions, gt, 1), 3),
        "top1_coco_only": round(top_k_accuracy(eval_preds, eval_gt, 1), 3),
        "unknown_rate": round(sum(1 for p in predictions if p == ["unknown"]) / len(predictions), 3),
        "coverage": round(len(evaluable) / len(items), 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "n": len(items),
    }


def benchmark_zero_shot(model_name: str, items: list[tuple[Image.Image, str, str]]) -> dict:
    """CLIP o SigLIP con catálogo cero-shot."""
    import torch
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    t_load = time.perf_counter()
    if "clip" in model_name:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        text_key = "text_input_ids"
    else:
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        text_key = "input_ids"
    model.eval()
    load_s = round(time.perf_counter() - t_load, 1)

    prompts = [f"a photo of {c}" for c in ZERO_SHOT_CATALOG]
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt")

    def _features(output) -> torch.Tensor:
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        return output

    predictions: list[list[str]] = []
    latencies = []
    with torch.no_grad():
        text_features = _features(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for image, gt, name in items:
            t0 = time.perf_counter()
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                if hasattr(model, "get_image_features"):
                    image_features = _features(model.get_image_features(**inputs))
                else:
                    image_features = _features(model.encode_image(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = (image_features @ text_features.T).squeeze(0)
            if "siglip" in model_name:
                scores = torch.sigmoid(scores)
            latencies.append((time.perf_counter() - t0) * 1000)

            ranked = scores.argsort(descending=True).tolist()
            ranked_names = [ZERO_SHOT_CATALOG[i] for i in ranked]
            # Normalizar al nombre del catálogo equivalente al gt
            predictions.append([ranked_names[0], ranked_names[1], ranked_names[2], ranked_names[3], ranked_names[4]])

    # Ground truth normalizado al catálogo zero-shot
    gt_catalog = [GT_TO_CATALOG.get(g, g) for _, g, _ in items]

    top1 = top_k_accuracy(predictions, gt_catalog, 1)
    top3 = top_k_accuracy(predictions, gt_catalog, 3)
    top5 = top_k_accuracy(predictions, gt_catalog, 5)

    return {
        "top1": round(top1, 3),
        "top3": round(top3, 3),
        "top5": round(top5, 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "load_seconds": load_s,
        "n": len(items),
    }


def main() -> None:
    items = load_benchmark()
    print(f"[benchmark] {len(items)} imágenes, clases: {sorted({g for _, g, _ in items})}")

    results: dict = {}
    results["yolo11n_coco"] = benchmark_yolo(items)
    print(f"[yolo] {results['yolo11n_coco']}")

    results["clip_vit_b32"] = benchmark_zero_shot("openai/clip-vit-base-patch32", items)
    print(f"[clip] {results['clip_vit_b32']}")

    results["siglip_base"] = benchmark_zero_shot("google/siglip-base-patch16-224", items)
    print(f"[siglip] {results['siglip_base']}")

    out = BASE_DIR / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[benchmark] guardado en {out}")


if __name__ == "__main__":
    main()