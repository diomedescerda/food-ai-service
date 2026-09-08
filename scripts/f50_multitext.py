"""F50: multi-text retrieval — canonical + aliases en el índice textual.

- Textos: canonical_name + aliases de cada alimento (7.498 textos reales).
- Embeddings CLIP (3 templates) cacheados (benchmarks/f50/).
- Índice sobre TODOS los textos; retrieval Top-K text-level -> grouping por
  canonical (best_score, best_rank, support, best_text) -> aggregación
  (max | top-2 mean) -> reranker F48.
- Comparación F49 (canonical-only) en R@K, bucket A, rescates por alias.

Uso: python scripts/f50_multitext.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.food_retrieval import FoodRetrieval  # noqa: E402
from app.models.retrieval_rerank import rerank_general  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "data/models/f38/pizza_naan_dino_base.pt"
OUT = BASE / "benchmarks/f50"
OUT.mkdir(parents=True, exist_ok=True)
TEMPLATES = ("a photo of {food}", "a picture of {food}", "a close-up photo of {food}")

FOOD_US = BASE / "datasets/food-us-v0.1/images/train"
V1 = BASE / "datasets/food-bench-v1"
F101 = BASE / "datasets/food101-subset"


def load_dino():
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    head = nn.Linear(768, 2)
    head.load_state_dict(ckpt["head"])
    head.eval()
    inv = {v: k for k, v in ckpt["label_map"].items()}
    return model, processor, head, inv


def dino_predict(model, processor, head, inv, image):
    import torch  # noqa: PLC0415

    inputs = processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        probs = torch.softmax(head(feats), dim=1)[0]
    return inv[int(probs.argmax())], float(probs.max())


def main() -> None:
    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()

    catalog = json.loads((BASE / "catalog/foods.json").read_text(encoding="utf-8-sig"))
    texts: list[dict] = []  # {text, canonical, type}
    for f in catalog["foods"]:
        texts.append({"text": f["canonical_name"], "canonical": f["canonical_name"], "type": "canonical"})
        for a in f["aliases"]:
            texts.append({"text": a, "canonical": f["canonical_name"], "type": "alias"})
    print(f"textos: {len(texts)} (canonical {sum(1 for t in texts if t['type']=='canonical')} + alias {sum(1 for t in texts if t['type']=='alias')})", flush=True)

    # 1) embeddings multi-text (3 templates)
    import torch  # noqa: PLC0415

    embs = []
    for ti, template in enumerate(TEMPLATES):
        npy = OUT / f"text_emb_t{ti}.npy"
        if npy.exists():
            embs.append(np.load(npy))
            continue
        arrs = []
        for i in range(0, len(texts), 128):
            prompts = [template.format(food=t["text"]) for t in texts[i:i + 128]]
            inputs = clf._processor(text=prompts, padding=True, return_tensors="pt")
            with torch.no_grad():
                feats = clf._features(clf._model.get_text_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            arrs.append(feats.numpy())
        arr = np.concatenate(arrs)
        np.save(npy, arr)
        embs.append(arr)
        print(f"template {ti}: {arr.shape}", flush=True)
    index = np.mean(embs, axis=0)
    index = index / np.linalg.norm(index, axis=1, keepdims=True)
    texts_meta = OUT / "texts.json"
    texts_meta.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"índice multi-text: {index.shape} ({index.nbytes/1e6:.0f} MB RAM)", flush=True)

    support = {}
    for f in catalog["foods"]:
        support[f["canonical_name"]] = support.get(f["canonical_name"], 0) + 1
    alias_n = {f["canonical_name"]: len(f["aliases"]) for f in catalog["foods"]}
    support_max = max(support.values()) or 1
    dino, dino_proc, head, inv = load_dino()

    # canonical-only index (F49 baseline) — reutiliza los embeddings del catálogo
    ret_base = FoodRetrieval(enabled=True, clf=clf)
    base_index = ret_base.index
    base_names = ret_base.names

    datasets = [
        ("food-us", [(str(p), p.stem.rsplit("_", 1)[0].replace("_", " ")) for p in sorted(FOOD_US.glob("*.jpg"))]),
        ("v1", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(V1.rglob("*.jpg"))]),
        ("food101", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(F101.rglob("*.jpg"))]),
    ]

    def canonical_rank(image_emb: np.ndarray, k: int, aggregation: str) -> list[dict]:
        scores = index @ image_emb
        order = np.argsort(-scores)[:k]
        group: dict[str, list[float]] = {}
        best_text = {}
        best_rank = {}
        for ei, i in enumerate(order):
            t = texts[i]
            c = t["canonical"]
            g = group.setdefault(c, [])
            if aggregation == "top2mean":
                g.append(float(scores[i]))
            else:
                if not g or scores[i] > g[0]:
                    g.insert(0, float(scores[i]))
                    best_text[c] = t["text"]
                    best_rank[c] = ei + 1
        out = []
        for c, vals in group.items():
            if aggregation == "top2mean":
                s = sorted(vals, reverse=True)[:2]
                out.append({"name": c, "score": float(np.mean(s)), "best_text": best_text.get(c, ""),
                            "best_rank": best_rank.get(c, 999), "queries": len(vals)})
            else:
                out.append({"name": c, "score": vals[0], "best_text": best_text.get(c, ""),
                            "best_rank": best_rank.get(c, 999), "queries": len(vals)})
        out.sort(key=lambda x: -x["score"])
        return out

    ks = (1, 5, 10, 20, 50, 100, 200, 500)

    for dname, items in datasets:
        print(f"\n=== {dname} ===", flush=True)
        hits_base = {k: 0 for k in ks}
        hits_max = {k: 0 for k in ks}
        hits_t2 = {k: 0 for k in ks}
        n = 0
        bucket_f49 = {50: 0, 100: 0, 200: 0, 500: 0}
        bucket_f50 = {50: 0, 100: 0, 200: 0, 500: 0}
        alias_rescues = 0
        examples = []
        t0 = time.perf_counter()
        for path, gt in items:
            image = Image.open(path).convert("RGB")
            inputs = clf._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                feats = clf._features(clf._model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            emb = feats.numpy()[0]
            # F49 baseline (canonical-only)
            sb = base_index @ emb
            ob = np.argsort(-sb)[:500]
            base_rank = [base_names[i] for i in ob]
            # F50 multi-text
            cands_max = canonical_rank(emb, 500, "max")
            cands_t2 = canonical_rank(emb, 500, "top2mean")
            n += 1
            for k in ks:
                hits_base[k] += gt in base_rank[:k]
                hits_max[k] += gt in [c["name"] for c in cands_max[:k]]
                hits_t2[k] += gt in [c["name"] for c in cands_t2[:k]]
            for k in (50, 100, 200, 500):
                bucket_f49[k] += gt not in base_rank[:k]
                bucket_f50[k] += gt not in [c["name"] for c in cands_max[:k]]
            # rescate por alias: GT fuera del top-50 base, dentro del top-50 F50
            if gt not in base_rank[:50] and gt in [c["name"] for c in cands_max[:50]]:
                alias_rescues += 1
                if len(examples) < 10:
                    c = next(c for c in cands_max if c["name"] == gt)
                    examples.append((gt, c["best_text"], c["best_rank"]))
        print(f"F49 base : R@1={hits_base[1]/n*100:.1f} R@5={hits_base[5]/n*100:.1f} "
              f"R@50={hits_base[50]/n*100:.1f} R@200={hits_base[200]/n*100:.1f} R@500={hits_base[500]/n*100:.1f}")
        print(f"F50 max  : R@1={hits_max[1]/n*100:.1f} R@5={hits_max[5]/n*100:.1f} "
              f"R@10={hits_max[10]/n*100:.1f} R@20={hits_max[20]/n*100:.1f} "
              f"R@50={hits_max[50]/n*100:.1f} R@100={hits_max[100]/n*100:.1f} "
              f"R@200={hits_max[200]/n*100:.1f} R@500={hits_max[500]/n*100:.1f}")
        print(f"F50 t2m  : R@1={hits_t2[1]/n*100:.1f} R@50={hits_t2[50]/n*100:.1f} R@500={hits_t2[500]/n*100:.1f}")
        print(f"bucket A F49: 50={bucket_f49[50]} 100={bucket_f49[100]} 200={bucket_f49[200]} 500={bucket_f49[500]}")
        print(f"bucket A F50: 50={bucket_f50[50]} 100={bucket_f50[100]} 200={bucket_f50[200]} 500={bucket_f50[500]}")
        print(f"rescates por alias (fuera top-50 F49 -> dentro F50): {alias_rescues}")
        for gt, btext, brank in examples:
            print(f"  {gt} <- '{btext[:50]}' (rank {brank})")
        print(f"({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()