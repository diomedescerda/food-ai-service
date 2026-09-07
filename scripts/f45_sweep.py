"""F45: sweep limitado de calibración del specialist (gate topk x threshold).

Cachea por imagen: retrieval rank (top-50), top-k legacy, predicción DINO.
El barrido de las 9 configs reordena sin re-ejecutar modelos.

Uso: python scripts/f45_sweep.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.food_retrieval import FoodRetrieval  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "data/models/f38/pizza_naan_dino_base.pt"
GROUP = {"pizza", "naan"}

FOOD_US = BASE / "datasets/food-us-v0.1/images/train"
V1 = BASE / "datasets/food-bench-v1"
F101 = BASE / "datasets/food101-subset"
CACHE = BASE / "benchmarks/f45/cache.json"

CONFIGS = [(tk, th) for tk in (3, 5, 10) for th in (0.65, 0.70, 0.75)]


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
    ret = FoodRetrieval(enabled=True, clf=clf)
    dino, dino_proc, head, inv = load_dino()

    datasets = [
        ("food-us", [(str(p), p.stem.rsplit("_", 1)[0].replace("_", " ")) for p in sorted(FOOD_US.glob("*.jpg"))]),
        ("v1", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(V1.rglob("*.jpg"))]),
        ("food101", [(str(p), p.parent.name.replace("_", " ")) for p in sorted(F101.rglob("*.jpg"))]),
    ]

    # 1) cache: retrieval rank + legacy top-k + prediccion DINO por imagen
    t0 = time.perf_counter()
    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    todo = [(d, p, g) for d, items in datasets for p, g in items if p not in cache]
    print(f"cache: {len(cache)} ok, {len(todo)} pendientes", flush=True)
    for dname, path, gt in todo:
        image = Image.open(path).convert("RGB")
        rank = ret.retrieve(image, top_k=50)
        groups: dict[str, list[float]] = {}
        for e in rank:
            groups.setdefault(e["name"], []).append(e["score"])
        cands = sorted([(n, max(s)) for n, s in groups.items()], key=lambda x: -x[1])
        best: dict[str, float] = {}
        for r in clf._score_crop(image):
            best[r.name.replace("_", " ")] = max(best.get(r.name.replace("_", " "), 0.0), r.score)
        top3 = [n for n, _ in sorted(best.items(), key=lambda x: -x[1])[:10]]
        conf1 = max(best.values()) if best else 0.0
        try:
            spec, conf = dino_predict(dino, dino_proc, head, inv, image)
        except Exception:  # noqa: BLE001
            spec, conf = "pizza", 0.0
        cache[path] = {"gt": gt, "cands": cands, "top10": top3, "conf1": conf1, "spec": spec, "conf": conf}
        if len(cache) % 200 == 0:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"cache listo ({time.perf_counter()-t0:.0f}s)", flush=True)

    # 2) barrido de las 9 configs
    rows = []
    for topk, conf_min in CONFIGS:
        per_ds = {}
        totals = {"corr": 0, "regr": 0, "abst": 0, "invoked": 0}
        for dname, _ in datasets:
            hits = {k: 0 for k in (1, 5, 10, 20, 50)}
            n = 0
            pza = pza_ok = 0
            naan = naan_ok = 0
            for d2, items in datasets:
                if d2 != dname:
                    continue
                for p, g in items:
                    item = cache[p]
                    cands = item["cands"]
                    final = [c for c, _ in cands]
                    n += 1
                    present = [c for c in final if c in GROUP]
                    gate_on = item["conf1"] < 0.40 and any(g_ in item["top10"][:topk] for g_ in ("pizza", "naan"))
                    if present and gate_on:
                        totals["invoked"] += 1
                        if item["conf"] >= conf_min:
                            if item["spec"] in final:
                                final.remove(item["spec"])
                                final.insert(0, item["spec"])
                            if g in GROUP and g == item["spec"]:
                                totals["corr"] += 1
                            elif g not in GROUP and item["spec"] == final[0]:
                                totals["regr"] += 1
                        else:
                            totals["abst"] += 1
                    for k in (1, 5, 10, 20, 50):
                        hits[k] += g in final[:k]
                    if g == "pizza":
                        pza += 1
                        pza_ok += final[0] == "pizza"
                    if g == "naan":
                        naan += 1
                        naan_ok += final[0] == "naan"
            per_ds[dname] = (hits[1] / n, hits[5] / n, pza_ok, pza, naan_ok, naan)
        rows.append((topk, conf_min, per_ds, totals))
        r = rows[-1]
        print(f"topk={topk} th={conf_min} | food-us R@1={r[2]['food-us'][0]*100:.1f}% "
              f"R@5={r[2]['food-us'][1]*100:.1f}% | v1 R@1={r[2]['v1'][0]*100:.1f}% "
              f"| F101 R@1={r[2]['food101'][0]*100:.1f}% | pizza {r[2]['food-us'][2]}/{r[2]['food-us'][3]} "
              f"| naan {r[2]['food-us'][4]}/{r[2]['food-us'][5]} | inv={r[3]['invoked']} "
              f"corr={r[3]['corr']} regr={r[3]['regr']} abst={r[3]['abst']}", flush=True)

    out = BASE / "benchmarks/f45/reports/f45_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"sweep guardado en {out}")


if __name__ == "__main__":
    main()