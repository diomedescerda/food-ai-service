"""FASE 20: benchmark A/B/C de clasificadores sobre crops idénticos.

Modelos: clip (openai/clip-vit-base-patch32) | dinov3 (anonymous-eval/food-recognition)
         | beit (yvelos/beit-food-384)

Mide: top-1/3/5 (canonical del catálogo), unknown, cobertura de labels mapeados,
latencia/crop, startup, RAM. E2E nutrition = canonical top-1 con mapping (35/38)
y porción disponible → nutrition available.

Uso: python scripts/benchmark_classifiers_ab.py --model clip|dinov3|beit
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402

from app.models.food_catalog import FOOD_CATALOG, candidate_to_canonical  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
CROPS_US = BASE / "datasets" / "crops-foodus"
CROPS_V1 = BASE / "datasets" / "crops-v1"
OUT = BASE / "benchmark_classifier_ab_results.json"

CANONICALS = {e.canonical_name for e in FOOD_CATALOG}
ALIASES = {}
for e in FOOD_CATALOG:
    for a in e.clip_candidates:
        ALIASES[a.lower()] = e.canonical_name
    ALIASES[e.canonical_name.replace("_", " ")] = e.canonical_name


def normalize_label(label: str) -> str:
    return label.strip().lower()


def label_to_canonical(label: str) -> str | None:
    """Mapeo conservador label→canonical: match exacto normalizado contra
    canonical y aliases. Sin equivalencia segura → None (REVIEW_REQUIRED)."""
    n = normalize_label(label)
    if n in CANONICALS or n.replace(" ", "_") in CANONICALS:
        return n.replace(" ", "_") if n in CANONICALS else n.replace(" ", "_")
    if n in ALIASES:
        return ALIASES[n]
    if n.replace(" ", "_") in ALIASES:
        return ALIASES[n.replace(" ", "_")]
    return None


def load_model(name: str):
    if name == "clip":
        from app.models.zero_shot_classifier import ZeroShotFoodClassifier

        classifier = ZeroShotFoodClassifier("openai/clip-vit-base-patch32", threshold=0.20)
        classifier.load()
        return classifier, None, None
    import torch  # noqa: PLC0415
    from transformers import AutoImageProcessor  # noqa: PLC0415

    if name == "dinov3":
        # Repo con backbone (DINOv3 ViT-L) + head linear separados.
        import torch.nn as nn  # noqa: PLC0415
        from transformers import AutoModel  # noqa: PLC0415

        model_id = "anonymous-eval/food-recognition"
        processor = AutoImageProcessor.from_pretrained(model_id)
        backbone = AutoModel.from_pretrained(model_id, subfolder="backbone")
        head_sd = torch.load(
            hf_hub_download(model_id, "classifier.pt", local_dir_use_symlinks=False), map_location="cpu", weights_only=True
        )
        head = nn.Linear(1024, 389)
        head.load_state_dict(head_sd)
        backbone.eval()
        cfg_path = hf_hub_download(model_id, "classifier_config.json", local_dir_use_symlinks=False)
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in cfg["id2label"].items()}

        class DinoV3Model:
            def __init__(self, backbone, head):
                self.backbone = backbone
                self.head = head

        return DinoV3Model(backbone, head), processor, id2label

    from transformers import AutoModelForImageClassification  # noqa: PLC0415

    model_id = "yvelos/beit-food-384"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    id2label = model.config.id2label
    return model, processor, id2label


def classify_food_model(model, processor, id2label, crop_img):
    import torch  # noqa: PLC0415

    inputs = processor(images=crop_img, return_tensors="pt")
    with torch.no_grad():
        if hasattr(model, "backbone"):
            out = model.backbone(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
            logits = model.head(feats)
        else:
            out = model(**inputs)
            logits = out.logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_idx = probs.argsort(descending=True)[:5].tolist()
    return [(id2label.get(i, f"label_{i}"), float(probs[i])) for i in top_idx]


def classify_clip(classifier, crop_img):
    """Ranking CLIP sobre el crop ya hecho (sin re-crop: crops idénticos
    entre modelos en el A/B)."""
    ranking = classifier._score_crop(crop_img)
    return [(r.name, r.score) for r in ranking]


def run_dataset(model_key, model, processor, id2label, crops_dir: Path, dataset_name: str) -> dict:
    per_class = defaultdict(lambda: Counter())
    total = top1 = top3 = top5 = unknown = 0
    lat = []
    rows = []
    done_file = BASE / f"ab_progress_{model_key}_{dataset_name}.json"
    done = {}
    if done_file.exists():
        done = json.loads(done_file.read_text(encoding="utf-8"))
        for img, rec in done.items():
            parts = img.split("/")
            cls = parts[1] if len(parts) > 2 else (parts[1] if "/" in img else img.split("_", 1)[0])
            total += 1
            if rec == "unknown":
                unknown += 1
                per_class[cls]["unknown"] += 1
            else:
                top1 += rec == cls
                per_class[cls][rec] += 1
    for img_path in sorted(crops_dir.rglob("*.jpg")):
        key = f"{dataset_name}/{img_path.parent.name}/{img_path.name}"
        if key in done:
            continue
        cls = img_path.parent.name if dataset_name == "v1" else img_path.stem.rsplit("_", 1)[0]
        t0 = time.perf_counter()
        crop_img = Image.open(img_path).convert("RGB")
        if model_key == "clip":
            ranking = classify_clip(model, crop_img)
        else:
            ranking = classify_food_model(model, processor, id2label, crop_img)
        lat.append((time.perf_counter() - t0) * 1000)

        canon_rank = []
        seen = set()
        for label, score in ranking:
            canon = label_to_canonical(label) if model_key != "clip" else label
            if canon and canon not in seen:
                canon_rank.append((canon, score))
                seen.add(canon)
        if not canon_rank:
            unknown += 1
            per_class[cls]["unknown"] += 1
            rows.append({"image": key, "gt": cls, "top1": "unknown"})
            done[key] = "unknown"
            total += 1
        else:
            top_names = [c for c, _ in canon_rank]
            total += 1
            top1 += top_names[0] == cls
            top3 += cls in top_names[:3]
            top5 += cls in top_names[:5]
            per_class[cls][top_names[0]] += 1
            rows.append({"image": key, "gt": cls, "top1": top_names[0]})
            done[key] = top_names[0]
        if len(done) % 10 == 0:
            done_file.write_text(json.dumps(done), encoding="utf-8")
            print(f"[prog {dataset_name}] {len(done)} {total} top1={top1} ({top1/total*100:.1f}%)", flush=True)
    done_file.write_text(json.dumps(done), encoding="utf-8")
    return {"dataset": dataset_name, "total": total, "top1": top1, "top3": top3, "top5": top5,
            "unknown": unknown, "lat_ms_mean": sum(lat) / max(len(lat), 1), "per_class": {k: dict(v) for k, v in per_class.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["clip", "dinov3", "beit"], required=True)
    parser.add_argument("--dataset", choices=["foodus", "v1"], default=None, help="solo un dataset")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.model == "clip":
        model, _, _ = load_model("clip")
        processor = id2label = None
    else:
        model, processor, id2label = load_model(args.model)
    startup = time.perf_counter() - t0

    if args.model != "clip":
        n_labels = len(id2label)
        mapped = sum(1 for i in id2label if label_to_canonical(id2label[i]) is not None)
        print(f"labels={n_labels} mapeados_al_catalogo={mapped} ({mapped/n_labels*100:.1f}%)")
        unmapped = [id2label[i] for i in range(n_labels) if label_to_canonical(id2label[i]) is None][:10]
        print(f"  ejemplos sin equivalencia (REVIEW_REQUIRED): {unmapped}")

    datasets = [("foodus", CROPS_US, "food-us"), ("v1", CROPS_V1, "v1")]
    if args.dataset:
        datasets = [d for d in datasets if d[0] == args.dataset]
    results = {}
    for tag, crops_dir, name in datasets:
        res = run_dataset(args.model, model, processor, id2label, crops_dir, name)
        n = res["total"]
        print(f"\n=== {args.model} — {name} ({n}) ===", flush=True)
        print(f"top1={res['top1']/n*100:.1f}% top3={res['top3']/n*100:.1f}% top5={res['top5']/n*100:.1f}% unknown={res['unknown']} ({res['unknown']/n*100:.1f}%) lat={res['lat_ms_mean']:.0f}ms", flush=True)
        results[tag] = res
    print(f"startup={startup:.1f}s")

    full = {}
    if OUT.exists():
        full = json.loads(OUT.read_text(encoding="utf-8"))
    full[args.model] = {"startup_s": round(startup, 1), **results}
    OUT.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()