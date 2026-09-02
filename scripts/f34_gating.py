"""F34: gating por confianza del legacy + rescate por especialistas (top-3 routing).

Flujo: legacy top-1 + confidence -> si conf < T -> activar los grupos de las
clases del TOP-3 del legacy (la clase correcta suele estar en top-3 aunque el
top-1 falle) -> especialista con S/M -> final.

T se selecciona sobre VALIDATION (split hash: val 0-2, test 3-9).
Uso: python scripts/f34_gating.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]

GROUP_MAP = {
    "pizza": ["pizza_naan"],
    "naan": ["pizza_naan"],
    "fried_chicken": ["fried_chicken_nuggets"],
    "chicken_nuggets": ["fried_chicken_nuggets"],
}
GROUPS = {
    "pizza_naan": ["pizza", "naan"],
    "fried_chicken_nuggets": ["fried_chicken", "chicken_nuggets"],
}
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def load_specialists(clf):
    import torch  # noqa: PLC0415

    groups = {}
    for gid, classes in GROUPS.items():
        ckpt = torch.load(BASE / f"data/models/f33/{gid}_s42.pt", map_location="cpu", weights_only=False)
        m = torch.nn.Linear(512, len(classes))
        m.load_state_dict(ckpt["state"])
        m.eval()
        groups[gid] = {"model": m, "inv": {v: k for k, v in ckpt["label_map"].items()}}
    return groups


def specialist_predict(clf, groups, gid: str, path: str) -> tuple[str, float, float]:
    import torch  # noqa: PLC0415

    g = groups[gid]
    inputs = clf._processor(images=[Image.open(path).convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        feats = clf._features(clf._model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        probs = torch.softmax(g["model"](feats), dim=1)[0]
    order = probs.argsort(descending=True)
    top1 = g["inv"][int(order[0])]
    conf = float(probs[order[0]])
    margin = float(probs[order[0]] - probs[order[1]])
    return top1, conf, margin


def food_us_items(split: str) -> list[dict]:
    d = BASE / "datasets/food-us-v0.1/images/train"
    regions = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
    out = []
    for img in sorted(d.glob("*.jpg")):
        h = int(hashlib.md5(img.name.encode()).hexdigest(), 16) % 10
        if ("val" if h < 3 else "test") != split:
            continue
        rec = regions[img.name]
        rl = rec["yolo"] or rec["dino"]
        if not rl:
            continue
        out.append({"path": str(img), "gt": img.stem.rsplit("_", 1)[0], "region": max(rl, key=lambda r: r.get("conf", 0.5))})
    return out


def evaluate(split: str, threshold: float, clf, legacy, groups, s_conf: float, s_margin: float) -> dict:
    items = food_us_items(split)
    total = correct = gated = rescued = harmful = 0
    lat_extra = []
    for it in items:
        image = Image.open(it["path"]).convert("RGB")
        b = it["region"]
        crop = image.crop((max(0, int(b["x"] - b["width"] * 0.1)), max(0, int(b["y"] - b["height"] * 0.1)),
                           min(image.width, int(b["x"] + b["width"] * 1.1)), min(image.height, int(b["y"] + b["height"] * 1.1))))
        ranking = legacy._score_crop(crop)
        best: dict[str, float] = {}
        for r in ranking:
            if r.name not in best or r.score > best[r.name]:
                best[r.name] = r.score
        pred = max(best, key=best.get) if best else "unknown"
        top3 = sorted(best, key=best.get, reverse=True)[:3]
        conf = best.get(pred, 0.0)
        final = pred
        if conf < threshold:
            gated += 1
            t0 = time.perf_counter()
            group_ids = []
            for c in top3:
                for g in GROUP_MAP.get(c, []):
                    if g not in group_ids:
                        group_ids.append(g)
            for group_id in group_ids:
                spec_class, s_conf2, margin = specialist_predict(clf, groups, group_id, it["path"])
                if s_conf2 >= s_conf and margin >= s_margin:
                    final = spec_class
                    break
            lat_extra.append((time.perf_counter() - t0) * 1000)
            if pred != it["gt"] and final == it["gt"]:
                rescued += 1
            if pred == it["gt"] and final != it["gt"]:
                harmful += 1
        correct += final == it["gt"]
        total += 1
    return {"total": total, "acc": correct / total, "gated": gated / total,
            "rescue": rescued, "harmful": harmful, "net": rescued - harmful,
            "lat_extra_ms": sum(lat_extra) / max(len(lat_extra), 1)}


def main() -> None:
    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
        top_k=999,
    )
    clf.load()
    legacy = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    legacy.load()
    groups = load_specialists(clf)

    for split in ("val", "test"):
        r = evaluate(split, 0.0, clf, legacy, groups, 0.0, 0.0)
        print(f"[baseline legacy {split}] acc={r['acc']*100:.1f}% ({r['total']})")

    print("\n=== SWEEP T (validation, S=0.0 M=0.0 — potencial máximo) ===")
    best_t, best_net = None, -999
    for t in THRESHOLDS:
        r = evaluate("val", t, clf, legacy, groups, 0.0, 0.0)
        print(f"T={t:.2f} gated={r['gated']*100:.0f}% acc={r['acc']*100:.1f}% "
              f"rescue={r['rescue']} harmful={r['harmful']} net={r['net']} lat+={r['lat_extra_ms']:.0f}ms")
        if r["net"] > best_net:
            best_net, best_t = r["net"], t

    print("\n=== SWEEP S/M (validation, T del mejor net) ===")
    best_sm = (0.0, 0.0)
    best_net_sm = -999
    for s, m in ((0.0, 0.0), (0.3, 0.05), (0.4, 0.1), (0.5, 0.1), (0.5, 0.2)):
        r = evaluate("val", best_t, clf, legacy, groups, s, m)
        print(f"S={s} M={m} acc={r['acc']*100:.1f}% rescue={r['rescue']} harmful={r['harmful']} net={r['net']}")
        if r["net"] > best_net_sm:
            best_net_sm, best_sm = r["net"], (s, m)

    r = evaluate("test", best_t, clf, legacy, groups, best_sm[0], best_sm[1])
    print(f"\n[F34 TEST] T={best_t} S={best_sm[0]} M={best_sm[1]} acc={r['acc']*100:.1f}% "
          f"gated={r['gated']*100:.0f}% rescue={r['rescue']} harmful={r['harmful']} net={r['net']}")
    out = BASE / "benchmarks/f34/reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "f34_result.json").write_text(json.dumps(
        {"best_t": best_t, "best_sm": list(best_sm), "test": r}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()