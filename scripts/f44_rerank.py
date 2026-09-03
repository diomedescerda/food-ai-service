"""F44: reranking del retrieval con el specialist DINO-base (pizza/naan).

Tras el canonical grouping: si pizza/naan estÃ¡n entre los candidatos, el
DINO-base (F38) decide el orden local y puede PROMOVER el confirmado al
top-1 (conf >= 0.6). Los demÃ¡s candidatos conservan el score retrieval
(sin penalizaciÃ³n). Nunca rompe el request.

Uso: python scripts/f44_rerank.py [--eval]
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
from app.models.food_retrieval import FoodRetrieval  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "data/models/f38/pizza_naan_dino_base.pt"
GROUP = {"pizza", "naan"}
CONF_MIN = 0.70

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


def dino_predict(model, processor, head, inv, image) -> tuple[str, float]:
    import torch  # noqa: PLC0415

    inputs = processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        probs = torch.softmax(head(feats), dim=1)[0]
    return inv[int(probs.argmax())], float(probs.max())


def main() -> None:
    parser = argparse.ArgumentParser()
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
    ret = FoodRetrieval(enabled=True, clf=clf)
    dino, dino_proc, head, inv = load_dino()

    ks = (1, 5, 10, 20, 50)
    stats = {"eligible": 0, "invoked": 0, "corr": 0, "regr": 0, "abst": 0, "dino_ms": [], "total": 0}

    datasets = [
        ("food-us", [(str(p), p.stem.rsplit("_", 1)[0].replace("_", " ")) for p in sorted(FOOD_US.glob("*.jpg"))]),
        ("v1", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(V1.rglob("*.jpg"))]),
        ("food101", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(F101.rglob("*.jpg"))]),
    ]
    t0 = time.perf_counter()
    for dname, items in datasets:
        hits = {k: 0 for k in ks}
        base_hits = {k: 0 for k in ks}
        totals = {k: 0 for k in ks}
        pizza_corr = pizza_tot = 0
        for path, gt in items:
            image = Image.open(path).convert("RGB")
            rank = ret.retrieve(image, top_k=50)
            groups: dict[str, list[float]] = {}
            for e in rank:
                groups.setdefault(e["name"], []).append(e["score"])
            cands = sorted([(n, max(s), len(s)) for n, s in groups.items()], key=lambda x: -x[1])
            base_top = [n for n, _, _ in cands]
            stats["total"] += 1
            for k in ks:
                totals[k] += 1
                base_hits[k] += gt in base_top[:k]

            final = [n for n, _, _ in cands]
            present = [n for n in final if n in GROUP]
            # gate del F38: solo invocar cuando el legacy tiene confianza baja
            # y pizza/naan están entre sus top-3 candidatos
            ranking_l = clf._score_crop(image)
            best = {}
            for r in ranking_l:
                if r.name.replace("_", " ") not in best or r.score > best[r.name.replace("_", " ")]:
                    best[r.name.replace("_", " ")] = r.score
            top1 = max(best, key=best.get) if best else "unknown"
            conf1 = best.get(top1, 0.0)
            top3l = sorted(best, key=best.get, reverse=True)[:3]
            gate_on = conf1 < 0.40 and any(g in top3l for g in ("pizza", "naan"))
            if present and gate_on:
                stats["eligible"] += 1
                t1 = time.perf_counter()
                try:
                    spec_class, spec_conf = dino_predict(dino, dino_proc, head, inv, image)
                except Exception:  # noqa: BLE001
                    spec_class, spec_conf = "pizza", 0.0
                stats["dino_ms"].append((time.perf_counter() - t1) * 1000)
                stats["invoked"] += 1
                if spec_conf >= CONF_MIN:
                    if spec_class in final:
                        final.remove(spec_class)
                        final.insert(0, spec_class)
                    if gt in GROUP and gt == spec_class:
                        stats["corr"] += 1
                    elif gt not in GROUP and spec_class == final[0]:
                        stats["regr"] += 1
                else:
                    stats["abst"] += 1
            for k in ks:
                hits[k] += gt in final[:k]
            if gt == "pizza":
                pizza_tot += 1
                pizza_corr += final[0] == "pizza"
        print(f"{dname}: base R@1={base_hits[1]/totals[1]*100:.1f}% R@5={base_hits[5]/totals[5]*100:.1f}% | "
              f"F44 R@1={hits[1]/totals[1]*100:.1f}% R@5={hits[5]/totals[5]*100:.1f}% "
              f"R@10={hits[10]/totals[10]*100:.1f}% R@20={hits[20]/totals[20]*100:.1f}% "
              f"R@50={hits[50]/totals[50]*100:.1f}% | pizza {pizza_corr}/{pizza_tot}")
    print(f"\nspecialist: eligible={stats['eligible']} invoked={stats['invoked']} "
          f"corr={stats['corr']} regr={stats['regr']} abst={stats['abst']} "
          f"dino_ms={np.mean(stats['dino_ms']) if stats['dino_ms'] else 0:.0f} "
          f"total={stats['total']} ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()

