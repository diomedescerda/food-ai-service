"""F29-A: análisis de la confusión rice / fries / fried_chicken en food-us.

Para cada imagen de fries/fried_chicken mal clasificada: text_score y
visual_score de las 3 clases → causa del error (retrieval miss / texto /
visual / ambos / CLIP no distingue).

Uso: python scripts/f29_rice_analysis.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "datasets/food-us-v0.1/images/train"
REGIONS = json.loads((BASE / "benchmarks/detection/regions.json").read_text(encoding="utf-8"))
PROTOS = json.loads((BASE / "data/prototypes/f28_full.json").read_text(encoding="utf-8"))
FOCUS = ("rice", "french_fries", "fried_chicken")


def main() -> None:
    import torch  # noqa: PLC0415

    settings = Settings()
    clf = ZeroShotFoodClassifier(
        "openai/clip-vit-base-patch32", threshold=settings.clip_threshold,
        crop_padding=settings.clip_crop_padding,
        prompt_template=settings.clip_prompt_template,
        prompt_templates_extra=tuple(settings.clip_prompt_ensemble.split("|")),
        top_k=999,
    )
    clf.load()

    protos = {cls: np.array([p["embedding"] for p in PROTOS[cls]], dtype=np.float32) for cls in FOCUS if cls in PROTOS}
    print("prototipos rice/fries/fried_chicken:", {k: len(v) for k, v in protos.items()})

    # Text features de las 3 clases (ensemble mean por candidate -> max)
    text_feats = {}
    for cls in FOCUS:
        cands = [c for f in json.loads((BASE / "data/catalogs/food_master.json").read_text(encoding="utf-8"))["foods"]
                 if f["canonical_name"] == cls for c in f["clip_candidates"]]
        feats = []
        for template in clf._prompt_templates:
            prompts = [template.format(food=c) for c in cands]
            inputs = clf._processor(text=prompts, padding=True, return_tensors="pt")
            with torch.no_grad():
                tf = clf._features(clf._model.get_text_features(**inputs))
                tf = tf / tf.norm(dim=-1, keepdim=True)
            feats.append(tf.numpy())
        text_feats[cls] = np.stack(feats).mean(axis=0)  # (n_cands, 512)

    stats = Counter()
    print("\nimagen                GT               pred(hybrid)  text: rice/fries/fried  visual: rice/fries/fried")
    for img_path in sorted(TRAIN.glob("french_fries_*.jpg")) + sorted(TRAIN.glob("fried_chicken_*.jpg")):
        cls = img_path.stem.rsplit("_", 1)[0]
        rec = REGIONS[img_path.name]
        regions_list = rec["yolo"] or rec["dino"]
        if not regions_list:
            continue
        best = max(regions_list, key=lambda r: r.get("conf", 0.5))
        image = Image.open(img_path).convert("RGB")
        crop = image.crop((max(0, int(best["x"] - best["width"] * 0.1)),
                           max(0, int(best["y"] - best["height"] * 0.1)),
                           min(image.width, int(best["x"] + best["width"] * 1.1)),
                           min(image.height, int(best["y"] + best["height"] * 1.1))))
        inputs = clf._processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            feats = clf._features(clf._model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        emb = feats.numpy()[0]

        text = {c: float((emb @ text_feats[c].T).max()) for c in FOCUS}
        visual = {c: float((protos[c] @ emb).max()) for c in protos}
        hybrid = {c: 0.5 * text[c] + 0.5 * visual.get(c, text[c]) for c in FOCUS}
        pred = max(hybrid, key=hybrid.get)

        tag = "OK" if pred == cls else "WRONG->" + pred
        stats[tag] += 1
        if pred != cls:
            print(f"{img_path.name:<24} {cls:<18} {pred:<14} "
                  f"t={text['rice']:.3f}/{text['french_fries']:.3f}/{text['fried_chicken']:.3f} "
                  f"v={visual.get('rice', 0):.3f}/{visual.get('french_fries', 0):.3f}/{visual.get('fried_chicken', 0):.3f}")

    print("\nRESUMEN:", dict(stats))
    print("text rice gana en los WRONG de fries:", sum(1 for k in stats if 'WRONG->rice' in k))


if __name__ == "__main__":
    main()