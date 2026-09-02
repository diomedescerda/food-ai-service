"""F37: SpecialistRouter — legacy + DINO pizza/naan selectivo, validación runtime.

Flujo: crop real (regions.json) -> legacy -> router (gates A/B/C) -> DINO
specialist pizza/naan -> final. Mide el 74.1% estimado de F36 en el pipeline.

Gates:
  A: legacy predice pizza/naan -> DINO
  B: conf legacy < T y pizza/naan en top-3 -> DINO
  C: A o B

Uso: python scripts/f37_router.py
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
MODELS = BASE / "data/models/f37"
MODELS.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 2026)
GATE_B_T = 0.40


def embed_dino(model, processor, paths: list[str]) -> np.ndarray:
    import torch  # noqa: PLC0415

    embs = []
    for i in range(0, len(paths), 8):
        chunk = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
        inputs = processor(images=chunk, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.numpy())
    return np.concatenate(embs)


def train_specialist(model, processor, classes: list[str], seed: int):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    manifest = json.loads((BASE / "data/training/f33/manifest.json").read_text(encoding="utf-8"))
    by_class: dict[str, list[str]] = {}
    for m in manifest.values():
        by_class.setdefault(m["class"], []).append(m["path"])
    paths = [p for c in classes for p in by_class.get(c, [])]
    labels = [c for c in classes for _ in by_class.get(c, [])]
    label_map = {c: i for i, c in enumerate(classes)}
    X = embed_dino(model, processor, paths)
    y = np.array([label_map[l] for l in labels])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    m = nn.Linear(X.shape[1], len(classes))
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
    return m, label_map, best_acc, len(paths)


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
    dino = AutoModel.from_pretrained("facebook/dinov2-large").eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

    # 1. Entrenar specialist pizza/naan (3 seeds)
    classes = ["pizza", "naan"]
    specialists = {}
    for seed in SEEDS:
        m, label_map, val_acc, n_train = train_specialist(dino, processor, classes, seed)
        specialists[seed] = (m, label_map)
        torch.save({"state": m.state_dict(), "classes": classes, "label_map": label_map},
                   MODELS / f"pizza_naan_dino_s{seed}.pt")
    print(f"[train] pizza/naan DINO specialist: {n_train} imágenes, 3 seeds guardados")

    regions = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
    d = BASE / "datasets/food-us-v0.1/images/train"
    images = sorted(d.glob("*.jpg"))

    # 2. Benchmark con los 3 gates (seed 42)
    inv = {v: k for k, v in specialists[42][1].items()}
    m42 = specialists[42][0]
    print(f"\n{'gate':<8} {'top1':<8} {'pizza':<8} {'calls':<7} {'dino_ms':<9} {'falsos':<7} {'oportunidades perdidas'}")
    for gate in ("A", "B", "C"):
        total = correct = pizza_correct = calls = 0
        dino_lat = 0.0
        missed = 0
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
            use_specialist = False
            if gate == "A":
                use_specialist = pred in ("pizza", "naan")
            elif gate == "B":
                use_specialist = conf < GATE_B_T and ("pizza" in top3 or "naan" in top3)
            elif gate == "C":
                use_specialist = pred in ("pizza", "naan") or (conf < GATE_B_T and ("pizza" in top3 or "naan" in top3))
            if use_specialist:
                calls += 1
                t0 = time.perf_counter()
                feats = embed_dino(dino, processor, [str(img)])
                with torch.no_grad():
                    probs = torch.softmax(m42(torch.tensor(feats, dtype=torch.float32)), dim=1)
                dino_lat += (time.perf_counter() - t0) * 1000
                spec_pred = inv[int(probs.argmax())]
                if probs.max() >= 0.6:
                    final = spec_pred
                if gt in ("pizza", "naan") and pred != gt and final == gt:
                    pass  # rescate
                if gt in ("pizza", "naan") and pred == gt and final != gt:
                    pass  # falso
            if gt in ("pizza", "naan") and pred != gt and final != gt:
                missed += 1
            total += 1
            correct += final == gt
            if gt == "pizza":
                pizza_correct += final == gt
        false_pos = 0
        for img in images:
            gt = img.stem.rsplit("_", 1)[0]
            if gt not in ("pizza", "naan"):
                continue
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
            use = pred in ("pizza", "naan") or (best.get(pred, 0) < GATE_B_T and ("pizza" in sorted(best, key=best.get, reverse=True)[:3] or "naan" in sorted(best, key=best.get, reverse=True)[:3])) if gate == "C" else (pred in ("pizza", "naan") if gate == "A" else best.get(pred, 0) < GATE_B_T and ("pizza" in sorted(best, key=best.get, reverse=True)[:3] or "naan" in sorted(best, key=best.get, reverse=True)[:3]))
            if use:
                feats = embed_dino(dino, processor, [str(img)])
                with torch.no_grad():
                    probs = torch.softmax(m42(torch.tensor(feats, dtype=torch.float32)), dim=1)
                spec = inv[int(probs.argmax())]
                if spec != gt and probs.max() >= 0.6:
                    false_pos += 1
        print(f"{gate:<8} {correct/total*100:<8.1f} {pizza_correct:<8} {calls:<7} {dino_lat/calls if calls else 0:<9.0f} {false_pos:<7} {missed}")

    # 3. Baseline legacy
    correct = 0
    total = 0
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
        total += 1
        correct += pred == gt
    print(f"\n[baseline legacy] {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    main()