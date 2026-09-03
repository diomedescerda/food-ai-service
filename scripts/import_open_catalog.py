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

# Tokens genéricos de alimentos: el canónico = el alimento base de los nombres
# compuestos del FNDDS ("breakfast pizza with egg" -> "pizza"), con los
# compuestos como aliases. Política F41: no crear clase por variante.
GENERIC_TOKENS = (
    "pizza", "hamburger", "burger", "hot dog", "sandwich", "chicken", "rice", "pasta",
    "salad", "soup", "bread", "cake", "banana", "apple", "orange", "fish", "steak",
    "egg", "potato", "cheese", "pork", "beef", "turkey", "ham", "bacon", "sausage",
    "shrimp", "salmon", "taco", "burrito", "nachos", "quesadilla", "pancake", "waffle",
    "toast", "bagel", "cookie", "brownie", "donut", "pie", "muffin", "croissant",
    "yogurt", "cereal", "oatmeal", "granola", "noodle", "bean", "corn", "carrot",
    "broccoli", "tomato", "lettuce", "avocado", "strawberry", "grape", "melon",
    "peach", "pear", "mango", "pineapple", "berry", "lamb", "duck", "sushi",
    "curry", "lasagna", "pancake", "french fries", "fries", "nuggets", "wings",
)


def generic_of(normalized: str) -> str:
    for token in GENERIC_TOKENS:
        if token in normalized:
            return token
    return normalized
NON_FOOD = ("beverage", "drink", "juice", "water", "tea", "coffee", "soda", "energy drink",
            "alcohol", "wine", "beer", "liquor", "cocktail", "protein powder", "formula",
            "supplement", "baby food", "infant", "oil", "sauce", "dressing", "gravy",
            "seasoning", "spice", "salt", "sugar", "syrup", "jam", "jelly", "butter substitute")


def fetch_fndds(query: str, page: int = 0, page_size: int = 50) -> tuple[list[dict], int]:
    params = urllib.parse.urlencode({
        "api_key": KEY, "query": query, "pageSize": page_size, "pageNumber": page + 1,
    })
    data_type = urllib.parse.quote("Survey (FNDDS)", safe="()")
    url = f"{API}/foods/search?{params}&dataType={data_type}".replace("+", "%20")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("foods", []), data.get("totalHits", 0)
        except Exception:
            if attempt == 2:
                return [], 0
            time.sleep(2 * (attempt + 1))
    return [], 0


QUERIES = ("beef", "chicken", "vegetables", "fruit", "bread", "dessert", "soup", "salad",
           "pasta", "rice", "sandwich", "burger", "pizza", "breakfast", "fast food", "cheese",
           "fish", "pork", "egg", "potato", "snack", "cake", "cereal", "bean", "poultry",
           "pancake", "pizza", "noodle", "taco", "burrito", "seafood", "sausage", "steak", "grilled", "salmon", "shrimp", "toast", "bagel", "yogurt", "curry", "sushi", "omelette", "granola", "brownie", "cookie", "donut", "ice cream", "sandwich", "pork chop", "lamb", "turkey", "ham", "bacon", "peanut", "almond", "avocado", "lettuce", "tomato", "corn", "carrot", "broccoli", "apple", "banana", "orange", "strawberry")


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

    # Entradas: una por alimento/variante (food_id unico) con canonical_name =
    # el generico (agrupacion). Dedup por nombre normalizado. Total >= 1000.
    seen: dict[str, str] = {}
    foods = []
    for name, fdc_id in raw:
        n = normalize(name)
        if not is_food(n):
            continue
        if n in seen:
            continue
        seen[n] = fdc_id
        canon = generic_of(n)
        foods.append({
            "food_id": f"fndds_{fdc_id}",
            "canonical_name": canon,
            "aliases": [n] if n != canon else [],
            "category": "unknown",
            "subcategory": "",
            "source": "FNDDS",
            "source_id": fdc_id,
            "nutrition_mapping": {"fdc_id": fdc_id, "source": "USDA FNDDS"},
            "status": "catalog",
            "raw_name": name,
        })

    foods.sort(key=lambda f: f["canonical_name"])
    catalog = {
        "schema_version": 2,
        "source": "USDA FoodData Central - Survey (FNDDS)",
        "license": "CC0 (public domain)",
        "imported_at": "2026-08-31",
        "total_raw": len(raw),
        "canonical_foods": len({f["canonical_name"] for f in foods}),
        "entries": len(foods),
        "foods": foods,
    }
    (OUT_DIR / "foods.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"raw={len(raw)} canÃ³nicos={len(foods)}")
    print("ejemplos:", [f["canonical_name"] for f in foods[:15]])


if __name__ == "__main__":
    main()













