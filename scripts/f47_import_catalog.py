"""F47: escalado del catÃ¡logo a 10.000+ alimentos canÃ³nicos (sin entrenar).

Fuentes (todas abiertas, distribuciÃ³n permitida):
1. USDA FDC â€” Survey (FNDDS) COMPLETO vÃ­a /foods/list: CC0 (dominio pÃºblico)
2. USDA FDC â€” SR Legacy COMPLETO vÃ­a /foods/list: CC0 (dominio pÃºblico)
3. Open Food Facts â€” generic_name_en (US): ODbL (atribuciÃ³n en reporte)

Pipeline: raw names -> normalize -> clean -> canonical (tokens genÃ©ricos) ->
aliases. Dedup por nombre normalizado; los food_id fndds_<id> existentes se
preservan (mismo id que el catÃ¡logo 1.451). Salida: catalog/foods.json.

Uso: python scripts/f47_import_catalog.py
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
OFF_API = "https://world.openfoodfacts.org/api/v2/search"
OUT_DIR = BASE / "catalog"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAGE = 200

PREP_PATTERNS = [
    r", nfs.*$", r" nfs.*$", r", not further specified.*$",
    r", without .*$", r", skin not eaten.*$", r", meat only.*$", r", no sauce.*$",
    r", from .*$", r", made with .*$", r", prepared.*$", r", cooked.*$",
    r", raw.*$", r", fresh.*$", r", frozen.*$", r", canned.*$", r", dried.*$",
    r", breaded.*$", r", battered.*$", r", smoked.*$", r", roasted.*$", r", baked.*$",
    r", grilled.*$", r", fried.*$", r", boiled.*$", r", stewed.*$", r", microwaved.*$",
    r", whole.*$", r", pieces.*$", r", slices.*$", r", chunks.*$", r", ground.*$",
    r", fat.*$", r", lean.*$", r", skinless.*$", r", seasoned.*$", r", flavored.*$",
]

GENERIC_TOKENS = (
    # F47: SOLO platos preparados colapsan al genÃ©rico (reconocimiento visual).
    # Los ingredientes/carnes/vegetales del USDA quedan a nivel especÃ­fico
    # (el canonical = el alimento USDA limpio â€” nutriciÃ³n Ãºtil).
    "pizza", "hamburger", "burger", "hot dog", "sandwich", "taco", "burrito",
    "nachos", "quesadilla", "pancake", "waffle", "toast", "bagel", "donut",
    "muffin", "croissant", "lasagna", "curry", "sushi", "french fries", "fries",
    "nuggets", "wings", "pasta", "noodle", "ramen", "pho", "pad thai",
    "fried rice", "risotto", "paella", "chili", "stew", "casserole", "quiche",
    "frittata", "omelette", "crepe", "empanada", "samosa", "dumpling", "gyoza",
    "spring roll", "sashimi", "tempura", "gyro", "shawarma", "kebab", "falafel",
    "tagine", "gnocchi", "ravioli", "tortellini", "macaroni", "spaghetti",
    "penne", "fusilli", "fettuccine", "linguine", "rigatoni", "orzo", "udon",
    "soba", "congee", "dim sum", "biryani", "tandoori", "paneer", "dal",
    "rajma", "chana", "chaat", "gulab jamun", "cereal", "oatmeal", "granola",
    "brownie", "cookie", "cheesecake", "tiramisu", "pudding", "custard",
    "mousse", "strudel", "salad", "soup", "bread", "cake", "pie", "yogurt",
    "ice cream", "sorbet", "chocolate", "candy", "popcorn", "chips", "pretzel",
    "cracker", "granola bar", "milkshake", "smoothie", "pudding", "jelly",
    "hummus", "guacamole", "salsa", "pesto", "mayonnaise", "ketchup", "mustard",
    "tortilla", "pita", "naan", "flatbread", "biscuit", "scone", "wrap",
    "stir fry", "curry", "roti", "chapati", "paratha", "korma", "tandoori",
)

NON_FOOD = ("beverage", "drink", "juice", "water", "tea", "coffee", "soda", "energy drink",
            "alcohol", "wine", "beer", "liquor", "cocktail", "protein powder", "formula",
            "supplement", "babyfood", "baby food", "infant", "sauce", "dressing", "gravy",
            "seasoning", "spice", "salt", "sugar", "syrup", "jam", "jelly", "butter substitute")


def generic_of(normalized: str) -> str:
    for token in GENERIC_TOKENS:
        if token in normalized:
            return token
    return normalized


def normalize(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"^[\d\s.]+", "", n)  # "8 pains au lait..." -> "pains au lait..."
    n = re.sub(r"\([^)]*\)", " ", n)  # "(west indian cherry)" -> quitar antes
    n = re.sub(r"[^a-z0-9 /]", " ", n)
    n = re.sub(r"\s*/\s*", " ", n)
    for pat in PREP_PATTERNS:
        n = re.sub(pat, "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def is_food(normalized: str) -> bool:
    if not normalized or len(normalized.split()) > 15:
        return False
    if len(normalized.split()) == 1 and len(normalized) < 5:
        return False  # "50", "caf" — no alimentos
    if any(w in normalized for w in NON_FOOD):
        return False
    return True


def fetch_usda_list(data_type: str) -> list[tuple[str, str]]:
    """Trae el dataset COMPLETO paginado (list API, sin query)."""
    raw: list[tuple[str, str]] = []
    page = 1
    while True:
        url = f"{API}/foods/list?api_key={KEY}&dataType={urllib.parse.quote(data_type)}&pageSize={PAGE}&pageNumber={page}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            break
        if not data:
            break
        for f in data:
            desc = f.get("description", "")
            if desc and len(desc) < 150:
                raw.append((desc, str(f.get("fdcId"))))
        if len(data) < PAGE:
            break
        page += 1
        time.sleep(0.2)
    return raw


OFF_QUERIES = (
    "pizza", "chocolate", "cheese", "yogurt", "bread", "salad", "soup", "beef",
    "chicken", "fish", "rice", "pasta", "cake", "cookie", "cereal", "ice cream",
    "candy", "chips", "pretzel", "cracker", "biscuit", "butter", "cream", "jam",
    "honey", "peanut butter", "sauce", "dip", "dressing", "mayonnaise", "ketchup",
    "sausage", "ham", "bacon", "turkey", "pork", "lamb", "shrimp", "salmon",
    "tuna", "egg", "potato", "corn", "bean", "pea", "lentil", "tofu", "mushroom",
    "spinach", "broccoli", "carrot", "tomato", "onion", "avocado", "apple",
    "banana", "orange", "strawberry", "blueberry", "grape", "melon", "mango",
    "pineapple", "peach", "pear", "watermelon", "granola", "oatmeal", "waffle",
    "pancake", "bagel", "muffin", "croissant", "donut", "pie", "brownie",
    "noodle", "ramen", "sushi", "taco", "burrito", "sandwich", "burger",
    "hot dog", "nuggets", "fries", "popcorn", "energy bar", "smoothie", "milk",
    "juice", "syrup", "vinegar", "oil", "flour", "sugar", "salt", "spice",
    "seasoning", "pickle", "olive", "cheesecake", "pudding", "popsicle", "tea",
)


OFF_CACHE = OUT_DIR / "off_raw_cache.json"


def fetch_off(pages_per_query: int = 6, page_size: int = 100, queries=None) -> list[tuple[str, str]]:
    """Open Food Facts: search por categorías, CACHE incremental por query.
    Reanudable: cada query descargada se guarda; el re-run salta las listas."""
    cache: dict[str, list[list[str]]] = {}
    if OFF_CACHE.exists():
        cache = json.loads(OFF_CACHE.read_text(encoding="utf-8"))
    raw: list[tuple[str, str]] = []
    for q in (queries or OFF_QUERIES):
        if q in cache:
            raw.extend((g, sid) for g, sid in cache[q])
            continue
        got: list[list[str]] = []
        for page in range(1, pages_per_query + 1):
            url = (
                f"{OFF_API}?fields=code,generic_name,generic_name_en&page_size={page_size}"
                f"&page={page}&search_terms={urllib.parse.quote(q)}"
            )
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(url, timeout=8) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    for p in data.get("products", []):
                        g = (p.get("generic_name") or p.get("generic_name_en") or "").strip()
                        if g:
                            got.append([g, str(p.get("code", ""))])
                    break
                except Exception:
                    if attempt == 1:
                        pass
                    time.sleep(2)
            time.sleep(0.5)
        cache[q] = got
        OFF_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        raw.extend((g, sid) for g, sid in got)
        print(f"  off[{q}]: +{len(got)} (total {len(raw)})", flush=True)
    return raw
    return raw


def build_entries(raw: list[tuple[str, str]], source: str, prefix: str) -> list[dict]:
    seen: dict[str, dict] = {}
    for name, sid in raw:
        n = normalize(name)
        if not is_food(n):
            continue
        canon = generic_of(n)
        if n in seen:
            continue
        seen[n] = {
            "food_id": f"{prefix}_{sid}",
            "canonical_name": canon,
            "aliases": [n] if n != canon else [],
            "category": "unknown",
            "subcategory": "",
            "source": source,
            "source_id": sid,
            "nutrition_mapping": {"fdc_id": sid} if source.startswith("USDA") else {},
            "status": "catalog",
            "raw_name": name,
        }
    return list(seen.values())


def main() -> None:
    import sys as _sys

    # Chunks reanudables: --off-chunk N (1..3) descarga un tercio de las
    # queries del OFF; sin flag: descarga las que falten del cache.
    chunk = None
    if "--off-chunk" in _sys.argv:
        idx = _sys.argv.index("--off-chunk")
        chunk = int(_sys.argv[idx + 1])

    print("FNDDS completo...", flush=True)
    raw_fndds = fetch_usda_list("Survey (FNDDS)")
    print(f"  fndds raw={len(raw_fndds)}", flush=True)
    print("SR Legacy completo...", flush=True)
    raw_sr = fetch_usda_list("SR Legacy")
    print(f"  sr raw={len(raw_sr)}", flush=True)
    print("Open Food Facts (generic_name, cache)...", flush=True)
    if chunk:
        size = (len(OFF_QUERIES) + 2) // 3
        qs = OFF_QUERIES[(chunk - 1) * size: chunk * size]
        raw_off = fetch_off(queries=qs)
    else:
        raw_off = fetch_off()
    print(f"  off raw={len(raw_off)}", flush=True)

    entries = (
        build_entries(raw_fndds, "USDA FNDDS", "fndds")
        + build_entries(raw_sr, "USDA SR Legacy", "srlegacy")
        + build_entries(raw_off, "Open Food Facts", "off")
    )
    canon_order: dict[str, int] = {}
    foods: list[dict] = []
    alias_map: dict[str, list[str]] = {}
    for e in entries:
        c = e["canonical_name"]
        if c in canon_order:
            if e["aliases"]:
                alias_map[c].append(e["aliases"][0])
            continue
        canon_order[c] = len(foods)
        alias_map[c] = list(e["aliases"])
        foods.append(e)
    for f in foods:
        f["aliases"] = sorted(set(alias_map[f["canonical_name"]]))[:20]

    # Preserva el legado F42 (1.451): canónicos/aliases que la expansión no
    # re-produjo (p.ej. "naan" — el list API del FNDDS no lo trae) se añaden
    # como entries curadas para no perder alimentos legacy.
    legacy_file = BASE / "catalog/foods_legacy_1451.json"
    if not legacy_file.exists():
# Guarda el catálogo 1.451 actual UNA vez (antes de la expansión).
        import shutil  # noqa: PLC0415
        shutil.copy(OUT_DIR / "foods.json", legacy_file)
    else:
        legacy = json.loads(legacy_file.read_text(encoding="utf-8-sig"))
        names_now = {f["canonical_name"] for f in foods}
        aliases_now = set(a for f in foods for a in f["aliases"])
        for f in legacy.get("foods", []):
            c = f["canonical_name"]
            if c in names_now:
                for food in foods:
                    if food["canonical_name"] == c:
                        missing = [a for a in f.get("aliases", [])
                                   if a not in food["aliases"] and a not in names_now]
                        food["aliases"] = sorted(set(food["aliases"] + missing))
                        break
                continue
            for a in f.get("aliases", []):
                if a in aliases_now:
                    continue
                if any(a in g["canonical_name"] or a in g["aliases"] for g in foods):
                    break
            else:
                foods.append({
                    "food_id": f.get("food_id", f"legacy_{c.replace(' ', '_')}"),
                    "canonical_name": c,
                    "aliases": f.get("aliases", []),
                    "category": "unknown",
                    "subcategory": "",
                    "source": "USDA FNDDS (legacy 1.451)",
                    "source_id": f.get("source_id", ""),
                    "nutrition_mapping": f.get("nutrition_mapping", {}),
                    "status": "catalog",
                    "raw_name": f.get("raw_name", c),
                })
                names_now.add(c)
        foods.sort(key=lambda f: f["canonical_name"])

    catalog = {
        "schema_version": 3,
        "source": "USDA FDC (FNDDS + SR Legacy) + Open Food Facts",
        "license": "CC0 (USDA) + ODbL (Open Food Facts)",
        "imported_at": "2026-09-04",
        "total_raw": len(raw_fndds) + len(raw_sr) + len(raw_off),
        "raw_fndds": len(raw_fndds),
        "raw_sr_legacy": len(raw_sr),
        "raw_off": len(raw_off),
        "canonical_foods": len(foods),
        "entries": len(entries),
        "alias_count": sum(len(f["aliases"]) for f in foods),
        "foods": foods,
    }
    (OUT_DIR / "foods.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"canonicos={len(foods)} entries={len(entries)} aliases={catalog['alias_count']}")
    print("ejemplos:", [f["canonical_name"] for f in foods[:15]])


if __name__ == "__main__":
    main()
