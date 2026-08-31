"""FASE 21: subset Food-101 (validation = test) de las clases del catálogo.

Licencia: Food-101 (ETH) es NON-COMMERCIAL research → solo EVALUACIÓN interna.
Mapeo clase Food-101 → canonical del catálogo (equivalencias seguras).

Uso: python scripts/extract_food101_subset.py [clase_food101 ...]
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "datasets" / "food101-subset"

# clase_food101 -> canonical del catálogo (solo equivalencias seguras)
CLASSES = {
    "french_fries": "french_fries",
    "grilled_salmon": "salmon",
    "hamburger": "hamburger",
    "hot_dog": "hot_dog",
    "ice_cream": "ice_cream",
    "lasagna": "lasagna",
    "macaroni_and_cheese": "mac_and_cheese",
    "nachos": "nachos",
    "pancakes": "pancakes",
    "pizza": "pizza",
    "steak": "steak",
    "tacos": "taco",
    "waffles": "waffles",
    "donuts": "donut",
}


def main() -> None:
    from datasets import load_dataset

    only = sys.argv[1:] or list(CLASSES)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ethz/food101", split="validation", streaming=True)
    label_names = ds.features["label"].names
    wanted = {i: name for i, name in enumerate(label_names) if name in only}
    print(f"clases objetivo: {len(wanted)}", flush=True)

    metadata = {}
    for row in ds:
        name = label_names[row["label"]]
        if name not in wanted:
            continue
        canonical = CLASSES[name]
        cls_dir = OUT / canonical
        cls_dir.mkdir(parents=True, exist_ok=True)
        n_existing = len(list(cls_dir.glob("*.jpg")))
        if n_existing >= 250:
            continue
        img = row["image"]
        fname = f"img_{n_existing + 1:04d}.jpg"
        img.convert("RGB").save(cls_dir / fname, quality=90)
        metadata[f"{canonical}/{fname}"] = {
            "source": "Food-101 (ETH, non-commercial research)",
            "food101_class": name,
            "ground_truth": canonical,
            "license": "unknown/non-commercial (ETH research only)",
        }
        if len(metadata) % 50 == 0:
            (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[prog] {len(metadata)}", flush=True)

    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(1 for _ in OUT.rglob("*.jpg"))
    print(f"FINAL: {total} imágenes de {len(wanted)} clases")


if __name__ == "__main__":
    main()