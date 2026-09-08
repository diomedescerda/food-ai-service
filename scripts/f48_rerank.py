"""F48: reranker general determinista sobre el catálogo 5.761.

Features por canonical del Top-50 (precomputadas una vez por imagen):
- retrieval_max / retrieval_mean / retrieval_rank (mejor rank del grupo)
- support_count (entries del canonical — señal estructural INDEPENDIENTE del CLIP)
- alias_count (aliases del canonical — señal estructural)
- specialist DINO pizza/naan (solo si eligible; score = conf si >= 0.75)

Configs (pesos simples, sin grid):
A: retrieval_max + support
B: A + alias_count
D: A + specialist
E: B + specialist

Sin prototype: los prototipos del F27-F29 viven en la rama del otro agente
(no integrados) — feature unavailable, sin penalización.
Uso: python scripts/f48_rerank.py
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.food_retrieval import FoodRetrieval  # noqa: E402
from app.models.retrieval_rerank import specialist_eligible  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "data/models/f38/pizza_naan_dino_base.pt"

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


def normalize_gt(name: str) -> str:
    return name.replace("_", " ").lower().strip()


def gt_key(name: str) -> str:
    return normalize_gt(name)


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
    catalog = json.loads((BASE / "catalog/foods.json").read_text(encoding="utf-8-sig"))
    support = {}
    aliases = {}
    for f in catalog["foods"]:
        support[f["canonical_name"]] = support.get(f["canonical_name"], 0) + 1
        aliases[f["canonical_name"]] = len(f["aliases"])
    dino, dino_proc, head, inv = load_dino()

    datasets = [
        ("food-us", [(str(p), p.stem.rsplit("_", 1)[0].replace("_", " ")) for p in sorted(FOOD_US.glob("*.jpg"))]),
        ("v1", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(V1.rglob("*.jpg"))]),
        ("food101", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(F101.rglob("*.jpg"))]),
    ]
    ks = (1, 5, 10, 20, 50)
    t0 = time.perf_counter()
    per_image: dict[str, list] = {d: [] for d, _ in datasets}
    cache_file = BASE / "benchmarks/f48/features_cache.json"
    cache: dict[str, dict] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    for dname, items in datasets:
        for path, gt in items:
            if path in cache:
                per_image[dname].append((gt_key(gt), cache[path]["cands"], cache[path]["dino_s"]))
                continue
            image = Image.open(path).convert("RGB")
            rank = ret.retrieve(image, top_k=50)
            groups: dict[str, dict] = {}
            for e in rank:
                g = groups.setdefault(e["name"], {"scores": [], "rank": 51})
                g["scores"].append(e["score"])
                g["rank"] = min(g["rank"], len(g["scores"]))
            cands = []
            for n, g in groups.items():
                s = np.array(g["scores"])
                cands.append({
                    "name": n, "max": float(s.max()), "mean": float(s.mean()),
                    "rank": g["rank"], "support": support.get(n, 0),
                    "alias": aliases.get(n, 0),
                })
            cands.sort(key=lambda c: -c["max"])
            # specialist DINO solo si eligible (gate F45)
            dino_s = None
            present = [c["name"] for c in cands]
            if any(x in present for x in ("pizza", "naan")):
                try:
                    best = {}
                    for r in clf._score_crop(image):
                        best[r.name.replace("_", " ")] = max(best.get(r.name.replace("_", " "), 0.0), r.score)
                    conf1 = max(best.values()) if best else 0.0
                    top3 = [n for n, _ in sorted(best.items(), key=lambda x: -x[1])[:3]]
                    if specialist_eligible(conf1, top3):
                        sc, sf = dino_predict(dino, dino_proc, head, inv, image)
                        if sf >= 0.75:
                            dino_s = [sc, sf]
                except Exception:  # noqa: BLE001
                    dino_s = None
            cache[path] = {"cands": cands, "dino_s": dino_s}
            per_image[dname].append((gt_key(gt), cands, dino_s))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
        print(f"{dname}: features listas ({time.perf_counter()-t0:.0f}s)", flush=True)

    configs = {
        "A_retr_support": (1.0, 0.5, 0.0, 0.0),
        "B_retr_support_alias": (1.0, 0.5, 0.1, 0.0),
        "D_retr_support_spec": (1.0, 0.5, 0.0, 1.0),
        "E_all": (1.0, 0.5, 0.1, 1.0),
    }
    support_max = max(support.values()) or 1

    def finalize(cands, dino_s, w_s, w_a, w_d):
        out = []
        for c in cands:
            score = w_s * c["max"] + 0.0
            # support normalizado (0-1) — señal estructural
            score += (0.5 * (c["support"] / support_max))
            score += w_a * (0.5 * (c["alias"] / 20.0))
            name = c["name"]
            if w_d and dino_s and name == dino_s[0]:
                score += w_d * dino_s[1]  # señal independiente (DINO)
            out.append((name, score))
        out.sort(key=lambda x: -x[1])
        return [n for n, _ in out]

    def evaluate(dname, config_name, w_s, w_a, w_d):
        hits = {k: 0 for k in ks}
        n = 0
        mrr = 0.0
        gt_ranks = []
        promoted = demoted = 0
        bucket_a = bucket_b = bucket_c = 0
        for gt, cands, dino_s in per_image[dname]:
            base = [c["name"] for c in cands]
            if gt in base:
                b_rank = base.index(gt) + 1
            else:
                b_rank = None
            final = finalize(cands, dino_s, w_s, w_a, w_d)
            n += 1
            for k in ks:
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
                # bucket A: fuera del top-50; C: variante cercana en top-50
                if any(gt in c["name"] or c["name"] in gt for c in cands):
                    bucket_c += 1
                else:
                    bucket_a += 1
            elif b_rank > 1:
                bucket_b += 1
        print(f"{dname} [{config_name}]: R@1={hits[1]/n*100:.1f}% R@5={hits[5]/n*100:.1f}% "
              f"R@10={hits[10]/n*100:.1f}% R@20={hits[20]/n*100:.1f}% R@50={hits[50]/n*100:.1f}% "
              f"| MRR={mrr/n:.3f} med_rank={np.median(gt_ranks) if gt_ranks else -1:.0f} "
              f"prom={promoted} dem={demoted} | A={bucket_a} B={bucket_b} C={bucket_c}")

    # baseline: grouping por max (F47 canonical)
    evaluate("food-us", "base_max", 1.0, 0.0, 0.0)
    for name, (ws, wsp, wa, wd) in configs.items():
        evaluate("food-us", name, ws, wa, wd)
    print()
    for name, (ws, wsp, wa, wd) in configs.items():
        evaluate("v1", name, ws, wa, wd)
    print()
    for name, (ws, wsp, wa, wd) in configs.items():
        evaluate("food101", name, ws, wa, wd)


if __name__ == "__main__":
    main()