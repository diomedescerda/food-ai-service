"""F42: embeddings CLIP del catálogo masivo + índice vectorial + evaluación R@K.

- Text embeddings (CLIP ViT-B/32, 3 templates) cacheados en catalog/embeddings/
- Índice numpy (1.411 x 512 — exacto; FAISS documentado para 100k+)
- Motor: app/models/food_retrieval.py
- Evaluación: food-us, food-bench-v1, Food-101 — Recall@1/5/10/20/50

Uso: python scripts/f42_retrieval.py [--embeddings] [--eval]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
EMB_DIR = BASE / "catalog/embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES = ("a photo of {food}", "a picture of {food}", "a close-up photo of {food}")

FOOD_US = BASE / "datasets/food-us-v0.1/images/train"
V1 = BASE / "datasets/food-bench-v1"
F101 = BASE / "datasets/food101-subset"


def build_index() -> tuple[np.ndarray, list[str]]:
    catalog = json.loads((BASE / "catalog/foods.json").read_text(encoding="utf-8-sig"))
    foods = catalog.get("foods", catalog)
    names = [f["canonical_name"] for f in foods]
    mats = [np.load(EMB_DIR / f"clip_text_t{i}.npy") for i in range(len(TEMPLATES))]
    index = np.mean(mats, axis=0)  # (N, 512) mean de los templates
    index = index / np.linalg.norm(index, axis=1, keepdims=True)
    return index, names


def generate_embeddings(clf) -> None:
    catalog = json.loads((BASE / "catalog/foods.json").read_text(encoding="utf-8-sig"))
    foods = catalog.get("foods", catalog)
    names = [f["canonical_name"] for f in foods]
    print(f"generando embeddings para {len(names)} alimentos...")
    import torch  # noqa: PLC0415

    for ti, template in enumerate(TEMPLATES):
        embs = []
        for i in range(0, len(names), 64):
            prompts = [template.format(food=n) for n in names[i:i + 64]]
            inputs = clf._processor(text=prompts, padding=True, return_tensors="pt")
            with torch.no_grad():
                feats = clf._features(clf._model.get_text_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.numpy())
        arr = np.concatenate(embs)
        np.save(EMB_DIR / f"clip_text_t{ti}.npy", arr)
        print(f"template {ti}: {arr.shape} guardado")
    metadata = {"model": "CLIP ViT-B/32", "dim": 512, "templates": list(TEMPLATES),
                "count": len(names), "version": "f42-1"}
    (EMB_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def normalize_gt(name: str) -> str:
    return name.replace("_", " ").lower().strip()


def recall_at_k(rank: list[str], gt: str, ks: tuple[int, ...]) -> list[bool]:
    g = normalize_gt(gt)
    canon = [normalize_gt(x) for x in rank]
    return [g in canon[:k] for k in ks]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()

    if args.embeddings:
        generate_embeddings(clf)

    if args.eval:
        import torch  # noqa: PLC0415

        index, names = build_index()
        print(f"índice: {index.shape} ({len(names)} alimentos)")
        ks = (1, 5, 10, 20, 50)
        totals = {k: 0 for k in ks}
        counts = {k: 0 for k in ks}

        def eval_images(paths_and_gt, label: str):
            t0 = time.perf_counter()
            for path, gt in paths_and_gt:
                image = Image.open(path).convert("RGB")
                inputs = clf._processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    feats = clf._features(clf._model.get_image_features(**inputs))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                emb = feats.numpy()[0]
                scores = index @ emb
                order = np.argsort(-scores)
                rank = [names[i] for i in order[:50]]
                hits = recall_at_k(rank, gt, ks)
                for k, hit in zip(ks, hits):
                    counts[k] += hit
                    totals[k] += 1
            print(f"{label}: R@1={counts[1]/totals[1]*100:.1f}% R@5={counts[5]/totals[5]*100:.1f}% "
                  f"R@10={counts[10]/totals[10]*100:.1f}% R@20={counts[20]/totals[20]*100:.1f}% "
                  f"R@50={counts[50]/totals[50]*100:.1f}% ({time.perf_counter()-t0:.0f}s)")

        eval_images([(str(p), p.stem.rsplit("_", 1)[0]) for p in sorted(FOOD_US.glob("*.jpg"))], "food-us")
        v1_items = [(str(p), p.parent.name) for p in sorted(V1.rglob("*.jpg"))]
        eval_images(v1_items, "food-bench-v1")
        f101_items = [(str(p), p.parent.name) for p in sorted(F101.rglob("*.jpg"))]
        eval_images(f101_items, "Food-101")


if __name__ == "__main__":
    main()