"""F52: descarga de nutrientes para los 5.761 canónicos (reanudable).

- USDA FDC: food/<fdc_id> (format=abridged) -> nutrientes por 100g.
  Respeto el límite diario del API (~1000 req/día con key estándar):
  --limit N por corrida; el caché reanuda sin re-descargar.
- Open Food Facts: /api/v3/product/<code>.json -> nutriments per 100g.
- Salida: nutrition/raw/<id>.json + nutrition/mappings.json (índice).

Uso: python scripts/f52_nutrition.py [--limit N] [--priority] [--off]
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env", override=True)
KEY = os.environ.get("FOODAI_USDA_API_KEY") or ""
FDC = "https://api.nal.usda.gov/fdc/v1"
OFF = "https://world.openfoodfacts.org/api/v3/product"
RAW = BASE / "nutrition/raw"
RAW.mkdir(parents=True, exist_ok=True)
INDEX = BASE / "nutrition/mappings.json"

NUTRIENT_MAP = {
    "Energy": ("calories", "kcal"),
    "Protein": ("protein", "g"),
    "Carbohydrate, by difference": ("carbohydrates", "g"),
    "Total lipid (fat)": ("fat", "g"),
    "Fiber, total dietary": ("fiber", "g"),
    "Sugars, total including NLEA": ("sugar", "g"),
    "Sodium, Na": ("sodium", "mg"),
}


def fetch(url: str, timeout: int = 25) -> dict | None:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(30)  # límite diario: espera y reintenta
                continue
            if exc.code in (400, 404):
                return None
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def fdc_nutrients(fdc_id: str) -> dict | None:
    url = f"{FDC}/food/{urllib.parse.quote(fdc_id)}?api_key={KEY}&format=abridged"
    data = fetch(url)
    if not data:
        return None
    out = {}
    for n in data.get("foodNutrients", []):
        name = n.get("name")
        amount = n.get("amount")
        key = NUTRIENT_MAP.get(name)
        if key and amount is not None:
            out[key[0]] = {"value": float(amount), "unit": key[1]}
    if not out:
        return None
    return {"source": "USDA FDC", "fdc_id": fdc_id, "reference_grams": 100, "nutrients_per_100g": out}


def off_nutrients(code: str) -> dict | None:
    data = fetch(f"{OFF}/{code}.json")
    if not data or not data.get("product"):
        return None
    nut = data["product"].get("nutriments", {})
    out = {}
    mapping = {
        "energy-kcal_100g": ("calories", "kcal"),
        "proteins_100g": ("protein", "g"),
        "carbohydrates_100g": ("carbohydrates", "g"),
        "fat_100g": ("fat", "g"),
        "fiber_100g": ("fiber", "g"),
        "sugars_100g": ("sugar", "g"),
        "sodium_100g": ("sodium", "mg"),
    }
    for key, (name, unit) in mapping.items():
        v = nut.get(key)
        if v is not None:
            out[name] = {"value": float(v), "unit": unit}
    if not out:
        return None
    return {"source": "Open Food Facts", "code": code, "reference_grams": 100, "nutrients_per_100g": out}


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=950)
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--off", action="store_true")
    args = parser.parse_args()

    catalog = json.loads((BASE / "catalog/foods.json").read_text(encoding="utf-8-sig"))
    index = {}
    if INDEX.exists():
        index = json.loads(INDEX.read_text(encoding="utf-8"))

    if args.off:
        items = [(f["food_id"], f.get("nutrition_mapping", {}).get("code") or f["food_id"].replace("off_", ""))
                 for f in catalog["foods"] if f["source"] == "Open Food Facts"]
        for food_id, code in items:
            if food_id in index:
                continue
            res = off_nutrients(code)
            if res:
                res["food_id"] = food_id
                res["canonical_name"] = next(f["canonical_name"] for f in catalog["foods"] if f["food_id"] == food_id)
                index[food_id] = res
                (RAW / f"{food_id}.json").write_text(json.dumps(res), encoding="utf-8")
                print(f"OFF {code}: OK", flush=True)
            else:
                index[food_id] = {"food_id": food_id, "status": "UNAVAILABLE"}
            time.sleep(1.0)
        INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        done = sum(1 for v in index.values() if "nutrients_per_100g" in v)
        print(f"OFF: {done}/{len(items)} con nutrientes", flush=True)
        return

    # USDA FDC: prioridad = legacy + golden + GT de los benchmarks
    foods = list(catalog["foods"])
    if args.priority:
        legacy = json.loads((BASE / "catalog/foods_legacy_1451.json").read_text(encoding="utf-8-sig"))
        legacy_names = {f["canonical_name"] for f in legacy["foods"]}
        golden = {"pizza", "hamburger", "naan", "rice", "fries", "chicken", "pasta", "sandwich",
                  "hot dog", "salad", "soup", "taco", "burrito", "cookie", "cake", "bread"}
        foods.sort(key=lambda f: (f["canonical_name"] not in (golden | legacy_names),
                                  f["canonical_name"] not in golden))

    done = 0
    errors = 0
    for f in foods:
        food_id = f["food_id"]
        if food_id in index:
            continue
        fdc_id = f.get("nutrition_mapping", {}).get("fdc_id") if isinstance(f.get("nutrition_mapping"), dict) else None
        if not fdc_id:
            index[food_id] = {"food_id": food_id, "status": "UNAVAILABLE"}
            continue
        if done >= args.limit:
            print(f"límite diario alcanzado ({args.limit}) — reanuda con: python scripts/f52_nutrition.py", flush=True)
            break
        res = fdc_nutrients(str(fdc_id))
        done += 1
        if res:
            res["food_id"] = food_id
            res["canonical_name"] = f["canonical_name"]
            res["status"] = "NUTRITION_READY"
            index[food_id] = res
            (RAW / f"{food_id}.json").write_text(json.dumps(res), encoding="utf-8")
        else:
            index[food_id] = {"food_id": food_id, "status": "NUTRITION_UNAVAILABLE"}
            errors += 1
        if done % 25 == 0:
            INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"progreso: {done} (errores {errors})", flush=True)
        time.sleep(0.3)
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    ready = sum(1 for v in index.values() if v.get("status") == "NUTRITION_READY")
    print(f"FDC: {ready} con nutrientes, {errors} errores, límite usado {done}", flush=True)


if __name__ == "__main__":
    main()