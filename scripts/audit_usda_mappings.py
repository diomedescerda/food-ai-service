"""FASE 18: auditoría del JSON curado USDA (20 mappings + pendientes)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BACK = Path(r"C:\Users\Carlos Cortina\Documents\CoppAddresd\coppAddresdBack")
data = json.load(open(BACK / "src/CoppAddresd.Api/Seeders/data/food_usda_curated.json", encoding="utf-8"))
foods = data["foods"]

ids = Counter(f.get("fdc_id") for f in foods if f.get("fdc_id"))
dups = {k: v for k, v in ids.items() if v > 1}
print("duplicados fdc_id:", dups or "ninguno")

required = ["canonical", "fdc_id", "fdc_name", "data_type", "mapping_status",
            "mapping_confidence", "nutrition", "source", "source_version", "source_id"]
missing = [(f["canonical"], k) for f in foods for k in required if k not in f]
print("campos faltantes:", missing or "ninguno")

bad = []
for f in foods:
    nut = f.get("nutrition") or {}
    for k in ("calories", "protein", "carbohydrates", "fat", "fiber", "sugar", "sodium"):
        v = nut.get(k)
        if v is not None and not isinstance(v, (int, float)):
            bad.append((f["canonical"], k, type(v).__name__))
print("valores no numericos:", bad or "ninguno")

zero_calories = [f["canonical"] for f in foods if (f.get("nutrition") or {}).get("calories") == 0]
print("con 0 kcal:", zero_calories or "ninguno")

print("\n=== mappings con nutricion (20) ===")
for f in foods:
    nut = f.get("nutrition") or {}
    if f.get("mapping_status") in ("DIRECT_MATCH", "GOOD_EQUIVALENCE"):
        print(f"{f['canonical']:<20} {f['mapping_status']:<18} fdc={str(f.get('fdc_id')):<10} {str(f.get('fdc_name'))[:40]:<42} {nut.get('calories')} kcal  {f.get('mapping_confidence')}")

print("\n=== pendientes (15+3) ===")
for f in foods:
    if f.get("mapping_status") in ("REVIEW_REQUIRED", "NO_RELIABLE_MATCH"):
        has_id = f.get("fdc_id") is not None
        print(f"{f['canonical']:<20} {f['mapping_status']:<18} fdc_id={'SI:'+str(f.get('fdc_id')) if has_id else 'NO'} conf={f.get('mapping_confidence')}")

print("\nmeta:", {k: v for k, v in data.get("meta", {}).items() if k != "generated_by"})