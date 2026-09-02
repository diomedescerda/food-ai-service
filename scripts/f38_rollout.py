"""F38: revalidación del gate + shadow telemetría + DINO-base vs L.

- Sweep threshold 0.20-0.50 (gate B: conf<thr y pizza/naan en top-3)
- Telemetría shadow: calls, would_change, corrections, regressions, abstentions
- DINOv2-L vs DINOv2-base para pizza/naan (accuracy/latencia)

El resultado al usuario SIEMPRE sería legacy en shadow (el specialist solo
se evalúa). Uso: python scripts/f38_rollout.py
"""
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
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
ABSTAIN_SCORE = 0.60


def embed(model, processor, paths: list[str], kind: str) -> np.ndarray:
    import torch  # noqa: PLC0415

    embs = []
    for i in range(0, len(paths), 8):
        chunk = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
        inputs = processor(images=chunk, return_tensors="pt")
        with torch.no_grad():
            if kind == "dino":
                out = model(**inputs)
                feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
            else:
                out = model.get_image_features(**inputs)
                feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.numpy())
    return np.concatenate(embs)


def train_specialist(model, processor, kind: str, seed: int = 42):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    manifest = json.loads((BASE / "data/training/f33/manifest.json").read_text(encoding="utf-8"))
    by_class: dict[str, list[str]] = {}
    for m in manifest.values():
        by_class.setdefault(m["class"], []).append(m["path"])
    classes = ["pizza", "naan"]
    paths = [p for c in classes for p in by_class.get(c, [])]
    labels = [c for c in classes for _ in by_class.get(c, [])]
    label_map = {c: i for i, c in enumerate(classes)}
    X = embed(model, processor, paths, kind)
    y = np.array([label_map[l] for l in labels])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    m = nn.Linear(X.shape[1], 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X[train_idx], dtype=torch.float32)
    yt = torch.tensor(y[train_idx], dtype=torch.long)
    Xv = torch.tensor(X[val_idx], dtype=torch.float32)
    yv = torch.tensor(y[val_idx], dtype=torch.long)
    best_acc, best_state = 0.0, None
    for _ in range(150):
        m.train()
        opt.zero_grad()
        loss = crit(m(Xt), yt)
        loss.backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(Xv).argmax(1) == yv).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m, label_map, best_acc


def main() -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

    settings = Settings()
    legacy = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
    )
    legacy.load()

    backbones = {}
    for name, mid in (("dino_l", "facebook/dinov2-large"), ("dino_base", "facebook/dinov2-base")):
        t0 = time.perf_counter()
        model = AutoModel.from_pretrained(mid).eval()
        processor = AutoImageProcessor.from_pretrained(mid)
        spec, label_map, val_acc = train_specialist(model, processor, "dino")
        backbones[name] = {"model": model, "processor": processor, "spec": spec,
                           "inv": {v: k for k, v in label_map.items()},
                           "load_s": time.perf_counter() - t0, "val": val_acc,
                           "dim": getattr(model.config, "hidden_size", 0)}
        print(f"[{name}] carga={backbones[name]['load_s']:.0f}s dim={backbones[name]['dim']} val={val_acc*100:.1f}%")

    regions = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
    d = BASE / "datasets/food-us-v0.1/images/train"
    images = sorted(d.glob("*.jpg"))

    print(f"\n{'thr':<6} {'top1':<7} {'calls':<6} {'corr':<5} {'regr':<5} {'abst':<5} {'dino_ms':<8} {'acc/L':<7} {'acc/base'}")
    for thr in THRESHOLDS:
        total = correct = calls = corr = regr = abst = 0
        dino_lat = []
        legacy_correct = 0
        for img in images:
            gt = img.stem.rsplit("_", 1)[0]
            rec = regions[img.name]
            rl = rec["yolo"] or rec["dino"]
            if not rl:
                continue
            image = Image.open(img).convert("RGB")
            b = max(rl, key=lambda r: r.get("conf", 0.5))
            crop = image.crop((max(0, int(b["x"] - b["width"] * 0.1)), max(0, int(b["y"] - b["height"] * 0.1)),
                               min(image.width, int(b["x"] + b["width"] * 1.1)), min(image.height, int(b["y"] + b["height"] * 1.1))))
            ranking = legacy._score_crop(crop)
            best: dict[str, float] = {}
            for r in ranking:
                if r.name not in best or r.score > best[r.name]:
                    best[r.name] = r.score
            pred = max(best, key=best.get) if best else "unknown"
            conf = best.get(pred, 0.0)
            top3 = sorted(best, key=best.get, reverse=True)[:3]
            final = pred
            if conf < thr and ("pizza" in top3 or "naan" in top3):
                calls += 1
                t0 = time.perf_counter()
                feats = embed(backbones["dino_l"]["model"], backbones["dino_l"]["processor"], [str(img)], "dino")
                with torch.no_grad():
                    probs = torch.softmax(backbones["dino_l"]["spec"](torch.tensor(feats, dtype=torch.float32)), dim=1)
                dino_lat.append((time.perf_counter() - t0) * 1000)
                spec_pred = backbones["dino_l"]["inv"][int(probs.argmax())]
                spec_score = float(probs.max())
                if spec_score >= ABSTAIN_SCORE:
                    final = spec_pred
                    if pred == gt and final != gt:
                        regr += 1
                    if pred != gt and final == gt:
                        corr += 1
                    if final == pred:
                        pass
                else:
                    abst += 1
            total += 1
            legacy_correct += pred == gt
            correct += final == gt
        acc_l = correct / total
        # dino-base solo en las mismas llamadas
        total2 = correct2 = 0
        for img in images:
            gt = img.stem.rsplit("_", 1)[0]
            rec = regions[img.name]
            rl = rec["yolo"] or rec["dino"]
            if not rl:
                continue
            image = Image.open(img).convert("RGB")
            b = max(rl, key=lambda r: r.get("conf", 0.5))
            crop = image.crop((max(0, int(b["x"] - b["width"] * 0.1)), max(0, int(b["y"] - b["height"] * 0.1)),
                               min(image.width, int(b["x"] + b["width"] * 1.1)), min(image.height, int(b["y"] + b["height"] * 1.1))))
            ranking = legacy._score_crop(crop)
            best = {}
            for r in ranking:
                if r.name not in best or r.score > best[r.name]:
                    best[r.name] = r.score
            pred = max(best, key=best.get) if best else "unknown"
            conf = best.get(pred, 0.0)
            top3 = sorted(best, key=best.get, reverse=True)[:3]
            final = pred
            if conf < thr and ("pizza" in top3 or "naan" in top3):
                feats = embed(backbones["dino_base"]["model"], backbones["dino_base"]["processor"], [str(img)], "dino")
                with torch.no_grad():
                    probs = torch.softmax(backbones["dino_base"]["spec"](torch.tensor(feats, dtype=torch.float32)), dim=1)
                if float(probs.max()) >= ABSTAIN_SCORE:
                    final = backbones["dino_base"]["inv"][int(probs.argmax())]
            total2 += 1
            correct2 += final == gt
        print(f"{thr:<6.2f} {correct/total*100:<7.1f} {calls:<6} {corr:<5} {regr:<5} {abst:<5} "
              f"{np.mean(dino_lat) if dino_lat else 0:<8.0f} {acc_l*100:<7.1f} {correct2/total2*100:<7.1f}")

    # Latencia base vs L
    t0 = time.perf_counter()
    embed(backbones["dino_l"]["model"], backbones["dino_l"]["processor"], [str(images[0])], "dino")
    lat_l = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    embed(backbones["dino_base"]["model"], backbones["dino_base"]["processor"], [str(images[0])], "dino")
    lat_b = (time.perf_counter() - t0) * 1000
    print(f"\nlatencia 1 img: dino_l={lat_l:.0f}ms dino_base={lat_b:.0f}ms")


if __name__ == "__main__":
    main()