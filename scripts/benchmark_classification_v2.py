"""FASE 17: benchmark de clasificaciÃ³n sobre food-bench-v1 (248 imÃ¡genes, 31 clases).

Uso: python scripts/benchmark_classification_v2.py [--padding 0.10] [--masked] [--topk 3]

Mide Top-1/Top-3/Top-5, unknown rate, per-class accuracy y matriz de confusiÃ³n.
La estrategia de crop es parametrizable para el experimento de fries/fried_chicken
(bbox, padding 0.10/0.20, masked crop).
"""
import argparse
import base64
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402
from app.models.yolo_food_detector import YoloFoodDetector  # noqa: E402
from app.models.yolo_food_segmenter import YoloFoodSegmenter  # noqa: E402
from app.core.config import Settings  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "food-bench-v1"


def crop(image: Image.Image, box: dict, padding: float) -> Image.Image:
    x1 = max(0, int(box["x"] - box["width"] * padding))
    y1 = max(0, int(box["y"] - box["height"] * padding))
    x2 = min(image.width, int(box["x"] + box["width"] * (1 + padding)))
    y2 = min(image.height, int(box["y"] + box["height"] * (1 + padding)))
    return image.crop((x1, y1, x2, y2))


def masked_crop(image: Image.Image, box: dict, mask_img: Image.Image, padding: float) -> Image.Image:
    """Crop con mÃ¡scara (fondo eliminado): mÃ¡scara binaria sobre fondo negro."""
    cropped = crop(image, box, padding)
    mask = mask_img.resize(cropped.size, Image.NEAREST).convert("L")
    black = Image.new("RGB", cropped.size, (0, 0, 0))
    return Image.composite(cropped, black, mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--masked", action="store_true", help="masked crop (requiere segmentador)")
    parser.add_argument("--only", nargs="*", default=None, help="solo estas clases")
    parser.add_argument("--dataset", default="food-bench-v1", help="carpeta del dataset")
    parser.add_argument("--regions", action="store_true", help="usar regiones hybrid (benchmarks/detection/regions.json) en vez de YOLO")
    args = parser.parse_args()

    dataset_dir = DATASET.parent / args.dataset
    regions = {}
    if args.regions:
        regions_file = Path(__file__).resolve().parents[1] / "benchmarks/detection/regions.json"
        regions = json.loads(regions_file.read_text(encoding="utf-8"))

    settings = Settings(model_path="weights/yolo11n.pt")
    detector = YoloFoodDetector(settings)
    detector.load()
    segmenter = YoloFoodSegmenter(Settings(model_path="weights/yolo11n-seg.pt")) if args.masked else None
    if segmenter:
        segmenter.load()
    classifier = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32",
        threshold=settings.clip_threshold,
        crop_padding=args.padding,
    )
    classifier.load()

    classes = sorted(d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name != "_meta")
    if args.only:
        classes = [c for c in args.only if (dataset_dir / c).is_dir() or list(dataset_dir.glob(f"{c}_*.jpg"))]

    total = top1 = top3 = top5 = unknown = 0
    per_class = defaultdict(lambda: Counter())
    confusion = Counter()
    t0 = time.perf_counter()
    rows = []

    for cls in classes:
        cls_dir = dataset_dir / cls
        if cls_dir.is_dir():
            paths = sorted(cls_dir.glob("*.jpg"))
        else:
            paths = sorted(dataset_dir.glob(f"{cls}_*.jpg"))
        for img_path in paths:
            image = Image.open(img_path).convert("RGB")
            if args.regions and img_path.name in regions:
                rec = regions[img_path.name]
                regions_list = rec["yolo"] or rec["dino"]
                detections = []
                for reg in regions_list:
                    from app.models.detection import BoundingBox, Detection  # noqa: PLC0415

                    detections.append(Detection(
                        name=reg.get("name", "food"),
                        confidence=reg.get("conf", 0.5),
                        bounding_box=BoundingBox(
                            x=int(reg["x"]), y=int(reg["y"]),
                            width=int(reg["width"]), height=int(reg["height"]),
                        ),
                    ))
            else:
                detections = detector.detect(image)
            if not detections:
                per_class[cls]["unknown_detection"] += 1
                unknown += 1
                total += 1
                rows.append({"image": f"{cls}/{img_path.name}", "gt": cls, "top1": "unknown_detection", "top3": [], "top5": []})
                continue
            top = max(detections, key=lambda d: d.confidence)
            box = top.bounding_box
            box_dict = {"x": box.x, "y": box.y, "width": box.width, "height": box.height}
            if args.masked and segmenter:
                segs = segmenter.segment(image, [top])
                mask_img = None
                if segs and segs[0]:
                    mask_img = Image.open(io.BytesIO(base64.b64decode(segs[0].mask))).convert("L")
                crop_img = masked_crop(image, box_dict, mask_img, args.padding) if mask_img else crop(image, box_dict, args.padding)
            else:
                crop_img = crop(image, box_dict, args.padding)
            results = classifier.classify(crop_img, [top])
            if not results:
                per_class[cls]["unknown"] += 1
                unknown += 1
                total += 1
                rows.append({"image": f"{cls}/{img_path.name}", "gt": cls, "top1": "unknown", "top3": [], "top5": []})
                continue
            ranking = [r.name for r in results]
            gt = cls
            top1 += ranking[0] == gt
            top3 += gt in ranking[:3]
            top5 += gt in ranking[:5]
            total += 1
            per_class[cls][ranking[0]] += 1
            confusion[(gt, ranking[0])] += 1
            rows.append({"image": f"{cls}/{img_path.name}", "gt": gt, "top1": ranking[0], "top3": ranking[:3], "top5": ranking[:5]})

    elapsed = time.perf_counter() - t0
    print(f"=== CLASSIFICATION v2 (padding={args.padding}, masked={args.masked}) ===")
    print(f"total={total} top1={top1/total*100:.1f}% top3={top3/total*100:.1f}% top5={top5/total*100:.1f}% unknown={unknown} ({unknown/total*100:.1f}%)")
    print(f"tiempo={elapsed:.1f}s ({elapsed/max(total,1)*1000:.0f} ms/img)")
    print("--- per-class top1 ---")
    for cls in classes:
        n = sum(per_class[cls].values())
        hits = per_class[cls][cls]
        print(f"  {cls:<20} {hits}/{n} = {hits/n*100:.1f}%")

    out = {
        "strategy": {"padding": args.padding, "masked": args.masked},
        "total": total, "top1": top1 / total, "top3": top3 / total, "top5": top5 / total,
        "unknown": unknown, "per_class": {k: dict(v) for k, v in per_class.items()},
        "confusion": [{"gt": g, "pred": p, "count": c} for (g, p), c in confusion.most_common(40)],
        "elapsed_s": elapsed,
    }
    tag = f"padding{args.padding}" + ("_masked" if args.masked else "")
    with open(Path(__file__).resolve().parents[1] / f"benchmarks/classification/benchmark_v2_results_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


