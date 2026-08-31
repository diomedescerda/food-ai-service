"""FASE 21: extrae el subset Food-101 (validation) de las clases del catálogo
desde los parquet locales (descargados a caché HF). Non-commercial: solo
evaluación. Uso: python scripts/extract_food101_parquet.py"""
import io
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "datasets" / "food101-subset"

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
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    OUT.mkdir(parents=True, exist_ok=True)
    label_names = list(CLASSES) + ["pizza", "donuts"]  # placeholder
    # Los índices reales se leen del parquet (columna label).
    wanted = set(CLASSES)

    metadata = {}
    for part in ("data/validation-00000-of-00003.parquet",
                 "data/validation-00001-of-00003.parquet",
                 "data/validation-00002-of-00003.parquet"):
        path = hf_hub_download("ethz/food101", part, repo_type="dataset")
        print(f"[parquet] {Path(path).name}", flush=True)
        table = pq.read_table(path)
        schema_names = None
        for name in table.column_names:
            pass
        label_col = table.column("label").to_pylist()
        img_col = table.column("image").to_pylist()
        print(f"  filas: {len(label_col)}", flush=True)
        for i, lab in enumerate(label_col):
            name = f"label_{lab}"  # placeholder; índice→nombre no disponible aquí
            canonical = None
            for cls_name, canon in CLASSES.items():
                if _label_index_of(cls_name) == lab:
                    canonical = canon
                    break
            if canonical is None:
                continue
            cls_dir = OUT / canonical
            cls_dir.mkdir(parents=True, exist_ok=True)
            n = len(list(cls_dir.glob("*.jpg")))
            if n >= 250:
                continue
            img_data = img_col[i]
            if isinstance(img_data, dict):
                img_data = img_data["bytes"]
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            fname = f"img_{n + 1:04d}.jpg"
            img.save(cls_dir / fname, quality=90)
            metadata[f"{canonical}/{fname}"] = {
                "source": "Food-101 (ETH, non-commercial research)",
                "license": "non-commercial research only",
                "ground_truth": canonical,
            }
            if len(metadata) % 100 == 0:
                print(f"  [prog] {len(metadata)}", flush=True)
        print(f"  total hasta aquí: {len(metadata)}", flush=True)

    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FINAL: {len(metadata)} imágenes")


def _label_index_of(cls_name: str) -> int:
    """Índices reales de Food-101 (definidos en el dataset card)."""
    ORDER = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
             "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
             "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
             "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
             "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
             "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
             "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
             "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
             "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
             "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
             "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
             "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
             "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
             "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
             "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
             "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
             "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
             "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
             "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
             "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"]
    return ORDER.index(cls_name) if cls_name in ORDER else -1


if __name__ == "__main__":
    main()