"""F49: recall expansion â€” K sweep + multi-query retrieval + fusiÃ³n.

- K sweep: GT en Top-50/100/200/500 (buckets A1-A4) â€” el lÃ­mite del recall.
- F49-A: single query -> Top-200 -> grouping -> reranker F48.
- F49-B: 4 vistas (original, center crop, crop padding, flip) -> Top-200 c/u
  -> union por canonical (max/mean/query_count/best_rank) -> grouping ->
  reranker F48. Specialist F45 intacto.

Cache reanudable de los embeddings por vista (benchmarks/f49/views_cache.npz).
Uso: python scripts/f49_recall.py
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
CACHE = BASE / "benchmarks/f49/views_cache.npz"
CACHE_META = BASE / "benchmarks/f49/views_cache_meta.json"

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


def views_of(image: Image.Image) -> list[Image.Image]:
    """Q1 original (resized 224), Q2 center crop 80%, Q3 crop con padding
    0.10, Q4 horizontal flip. Augmentaciones suaves, sin modificar la imagen."""
    w, h = image.size
    q1 = image.copy()
    q2 = image.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
    q3 = image.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))  # padding 0.10
    q4 = image.transpose(Image.FLIP_LEFT_RIGHT)
    return [q1, q2, q3, q4]


def main() -> None:
    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    clf.load()
    ret = FoodRetrieval(enabled=True, clf=clf)
    index = ret.index
    names = ret.names
    catalog = json.loads((BASE / "catalog/foods.json").read_text(encoding="utf-8-sig"))
    support = {}
    aliases = {}
    for f in catalog["foods"]:
        support[f["canonical_name"]] = support.get(f["canonical_name"], 0) + 1
        aliases[f["canonical_name"]] = len(f["aliases"])
    support_max = max(support.values()) or 1
    dino, dino_proc, head, inv = load_dino()

    datasets = [
        ("food-us", [(str(p), p.stem.rsplit("_", 1)[0].replace("_", " ")) for p in sorted(FOOD_US.glob("*.jpg"))]),
        ("v1", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(V1.rglob("*.jpg"))]),
        ("food101", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(F101.rglob("*.jpg"))]),
    ]

    import torch  # noqa: PLC0415

    def embed(image: Image.Image) -> np.ndarray:
        inputs = clf._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feats = clf._features(clf._model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.numpy()[0]

    # cache de las 4 vistas por imagen
    cache = {}
    if CACHE.exists():
        arr = np.load(CACHE)
        cache = {k: arr[k] for k in arr.files}
    meta = {}
    if CACHE_META.exists():
        meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    for dname, items in datasets:
        for path, _ in items:
            if path in cache:
                continue
            image = Image.open(path).convert("RGB")
            cache[path] = np.stack([embed(v) for v in views_of(image)])
            if len(cache) % 200 == 0:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(CACHE, **cache)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **cache)
        print(f"{dname}: vistas cacheadas ({len(cache)}) {time.perf_counter()-t0:.0f}s", flush=True)

    def retrieve_k(emb: np.ndarray, k: int) -> list[str]:
        scores = index @ emb
        order = np.argsort(-scores)[:k]
        return [names[i] for i in order]

    ks_sweep = (50, 100, 200, 500)

    # 1) K sweep + buckets A1-A4 (single query, vista original)
    for dname, items in datasets:
        hits = {k: 0 for k in ks_sweep}
        n = 0
        for path, gt in items:
            emb = cache[path][0]
            rank = retrieve_k(emb, 500)
            n += 1
            for k in ks_sweep:
                hits[k] += gt in rank[:k]
        a1 = a2 = a3 = a4 = 0
        for path, gt in items:
            emb = cache[path][0]
            rank = retrieve_k(emb, 500)
            if gt in rank[:50]:
                continue
            if gt in rank[:100]:
                a1 += 1
            elif gt in rank[:200]:
                a2 += 1
            elif gt in rank[:500]:
                a3 += 1
            else:
                a4 += 1
        print(f"{dname}: R@50={hits[50]/n*100:.1f} R@100={hits[100]/n*100:.1f} "
              f"R@200={hits[200]/n*100:.1f} R@500={hits[500]/n*100:.1f} | "
              f"A1(51-100)={a1} A2(101-200)={a2} A3(201-500)={a3} A4(>500)={a4}", flush=True)

    # 2) F49-A: single Top-200 + F49-B: multi-query (union max) -> reranker
    def rank_final(dname, items, multi: bool):
        hits = {k: 0 for k in (1, 5, 10, 20, 50)}
        n = 0
        mrr = 0.0
        gt_ranks = []
        promoted = demoted = 0
        bucket_a = bucket_b = 0
        pool_sizes = []
        t_q = time.perf_counter()
        for path, gt in items:
            if multi:
                # union de las 4 vistas: scores por canonical + query_count
                group: dict[str, dict] = {}
                for vi in range(4):
                    scores = index @ cache[path][vi]
                    order = np.argsort(-scores)[:200]
                    for ei, i in enumerate(order):
                        g = group.setdefault(names[i], {"scores": [], "ranks": []})
                        g["scores"].append(float(scores[i]))
                        g["ranks"].append(ei + 1)
                cands = []
                for name, g in group.items():
                    s = np.array(g["scores"])
                    cands.append({
                        "name": name, "max": float(s.max()), "mean": float(s.mean()),
                        "rank": min(g["ranks"]), "support": support.get(name, 0),
                        "alias": aliases.get(name, 0),
                        "queries": len(g["scores"]),
                    })
            else:
                scores = index @ cache[path][0]
                order = np.argsort(-scores)[:200]
                cands = []
                for ei, i in enumerate(order):
                    name = names[i]
                    cands.append({
                        "name": name, "max": float(scores[i]), "mean": float(scores[i]),
                        "rank": ei + 1, "support": support.get(name, 0),
                        "alias": aliases.get(name, 0), "queries": 1,
                    })
            cands.sort(key=lambda c: -c["max"])
            pool_sizes.append(len(cands))
            # specialist DINO (gate F45) â€” solo si pizza/naan en candidatos
            dino_s = None
            present = [c["name"] for c in cands]
            if any(x in present for x in ("pizza", "naan")):
                try:
                    best = {}
                    for r in clf._score_crop(Image.open(path).convert("RGB")):
                        best[r.name.replace("_", " ")] = max(best.get(r.name.replace("_", " "), 0.0), r.score)
                    conf1 = max(best.values()) if best else 0.0
                    top3 = [nm for nm, _ in sorted(best.items(), key=lambda x: -x[1])[:3]]
                    if conf1 < 0.40 and any(g in top3 for g in ("pizza", "naan")):
                        sc, sf = dino_predict(dino, dino_proc, head, inv, Image.open(path).convert("RGB"))
                        if sf >= 0.75:
                            dino_s = (sc, sf)
                except Exception:  # noqa: BLE001
                    dino_s = None
            base = [c["name"] for c in cands]
            if gt in base:
                b_rank = base.index(gt) + 1
            else:
                b_rank = None
            final = rerank_general(cands, support_max, dino_s, w_rank=0.30)
            n += 1
            for k in (1, 5, 10, 20, 50):
                hits[k] += gt in final[:k]
            if gt in final:
                r = final.index(gt) + 1
                mrr += 1.0 / r
                gt_ranks.append(r)
                if b_rank is not None:
                    if r < b_rank:
                        promoted += 1
                    elif r > b_rank:
                        demoted += 1
            if gt not in base:
                bucket_a += 1
            elif b_rank > 1:
                bucket_b += 1
        label = "F49-B multi" if multi else "F49-A single"
        print(f"{dname} [{label}]: R@1={hits[1]/n*100:.1f} R@5={hits[5]/n*100:.1f} "
              f"R@10={hits[10]/n*100:.1f} R@20={hits[20]/n*100:.1f} R@50={hits[50]/n*100:.1f} "
              f"| MRR={mrr/n:.3f} med={np.median(gt_ranks) if gt_ranks else -1:.0f} "
              f"prom={promoted} dem={demoted} A={bucket_a} B={bucket_b} "
              f"pool={np.mean(pool_sizes):.0f} ({time.perf_counter()-t_q:.0f}s)", flush=True)

    for dname, items in datasets:
        rank_final(dname, items, multi=False)
    print()
    for dname, items in datasets:
        rank_final(dname, items, multi=True)


if __name__ == "__main__":
    main()

