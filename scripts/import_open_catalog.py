"""F41: importador de catÃ¡logo masivo desde USDA FNDDS (CC0).

Descarga nombres del FNDDS (Survey) vÃ­a la API, normaliza a canÃ³nicos
genÃ©ricos (no cada variante), deduplica y genera aliases. Reproducible.

Fuente: USDA FoodData Central (FNDDS) â€” licencia CC0 (dominio pÃºblico).

Salida: catalog/foods.json (canÃ³nicos + aliases + categorÃ­a + fdc_id).
Uso: python scripts/import_open_catalog.py
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env", override=True)
KEY = os.environ.get("FOODAI_USDA_API_KEY") or ""
API = "https://api.nal.usda.gov/fdc/v1"
OUT_DIR = BASE / "catalog"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Patrones de preparaciÃ³n/variante que no crean una clase nueva.
PREP_PATTERNS = [
    r", nfs.*$", r" nfs.*$", r", not further specified.*$",
    r", without .*$", r", skin not eaten.*$", r", meat only.*$", r", no sauce.*$",
    r", from .*$", r" \(.*\)$", r", made with .*$", r", prepared.*$", r", cooked.*$",
    r", raw.*$", r", fresh.*$", r", frozen.*$", r", canned.*$", r", dried.*$",
]
NON_FOOD = ("beverage", "drink", "juice", "water", "tea", "coffee", "soda", "energy drink",
            "alcohol", "wine", "beer", "liquor", "cocktail", "protein powder", "formula",
            "supplement", "baby food", "infant", "oil", "sauce", "dressing", "gravy",
            "seasoning", "spice", "salt", "sugar", "syrup", "jam", "jelly", "butter substitute")


def fetch_fndds(query: str, page: int = 0, page_size: int = 50) -> tuple[list[dict], int]:
    params = urllib.parse.urlencode({
        "api_key": KEY, "query": query, "pageSize": page_size, "pageNumber": page,
    })
    data_type = urllib.parse.quote("Survey (FNDDS)", safe="()")
    url = f"{API}/foods/search?{params}&dataType={data_type}".replace("+", "%20")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("foods", []), data.get("totalHits", 0)


QUERIES = ("beef", "chicken", "vegetables", "fruit", "bread", "dessert", "soup", "salad",
           "pasta", "rice", "sandwich", "burger", "pizza", "breakfast", "fast food", "cheese",
           "fish", "pork", "egg", "potato", "snack", "cake", "cereal", "bean", "poultry",
           "pancake", "pizza", "noodle", "taco", "burrito", "seafood", "sausage", "steak", "grilled")


def normalize(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    for pat in PREP_PATTERNS:
        n = re.sub(pat, "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def is_food(normalized: str) -> bool:
    if not normalized or len(normalized.split()) > 7:
        return False
    if any(w in normalized for w in NON_FOOD):
        return False
    return True


def main() -> None:
    raw: list[tuple[str, str]] = []
    for q in QUERIES:
        foods, _ = fetch_fndds(q)
        for f in foods:
            desc = f.get("description", "")
            if desc and len(desc) < 95:
                raw.append((desc, str(f.get("fdcId"))))
        print(f"[{q}] acumulados {len(raw)}", flush=True)
        time.sleep(0.25)

    # NormalizaciÃ³n + dedup: canonical -> {aliases, fdc_ids, raw_names}
    canonical_map: dict[str, dict] = {}
    for name, fdc_id in raw:
        n = normalize(name)
        if not is_food(n):
            continue
        entry = canonical_map.setdefault(n, {"canonical_name": n, "aliases": [], "fdc_ids": [], "raw_names": []})
        entry["fdc_ids"].append(fdc_id)
        entry["raw_names"].append(name)
        if name.lower().strip() != n and name.lower().strip() not in entry["aliases"]:
            entry["aliases"].append(name.lower().strip())

    foods = []
    for n, entry in canonical_map.items():
        foods.append({
            "food_id": f"fndds_{entry['fdc_ids'][0]}",
            "canonical_name": n,
            "aliases": entry["aliases"][:5],
            "category": "unknown",
            "nutrition_mapping": {"fdc_id": entry["fdc_ids"][0], "source": "USDA FNDDS"},
            "status": "catalog",
            "source_raw_count": len(entry["raw_names"]),
        })

    foods.sort(key=lambda f: f["canonical_name"])
    catalog = {
        "schema_version": 1,
        "source": "USDA FoodData Central â€” Survey (FNDDS)",
        "license": "CC0 (public domain)",
        "imported_at": "2026-08-31",
        "total_raw": len(raw),
        "total_normalized": len(foods),
        "foods": foods,
    }
    (OUT_DIR / "foods.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"raw={len(raw)} canÃ³nicos={len(foods)}")
    print("ejemplos:", [f["canonical_name"] for f in foods[:15]])


if __name__ == "__main__":
    main()






