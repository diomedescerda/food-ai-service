"""Auditoría FASE 17: matriz catalog × mapping × portion para los 38 alimentos."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.food_catalog import FOOD_CATALOG  # noqa: E402
from app.models.basic_portion_estimator import REFERENCE_GRAMS  # noqa: E402

BACK = Path(r"C:\Users\Carlos Cortina\Documents\CoppAddresd\coppAddresdBack")
CURATED = BACK / "src/CoppAddresd.Api/Seeders/data/food_usda_curated.json"

with open(CURATED, encoding="utf-8") as fh:
    data = json.load(fh)
curated = {f["canonical"]: f for f in data["foods"]}

def _aliases(entry):
    aliases = getattr(entry, "aliases", None)
    return aliases if isinstance(aliases, (list, tuple)) else []

rows = []
for entry in FOOD_CATALOG:
    canonical = entry.canonical_name
    mapped = curated.get(canonical)
    has_nutrition = bool(mapped and mapped.get("mapping_status") in ("DIRECT_MATCH", "GOOD_EQUIVALENCE"))
    portion = (
        canonical in REFERENCE_GRAMS
        or canonical.replace("_", " ") in REFERENCE_GRAMS
        or any(a in REFERENCE_GRAMS for a in _aliases(entry))
    )
    rows.append({
        "food": canonical,
        "catalog_aliases": len(_aliases(entry)),
        "nutrition": has_nutrition,
        "mapping_status": mapped.get("mapping_status", "NOT_MAPPED") if mapped else "NOT_MAPPED",
        "mapping_confidence": mapped.get("mapping_confidence") if mapped else None,
        "portion": portion,
        "portion_grams": REFERENCE_GRAMS.get(canonical) or REFERENCE_GRAMS.get(canonical.replace("_", " ")),
        "e2e_supported": has_nutrition and portion,
    })

n = len(rows)
nutrition = sum(1 for r in rows if r["nutrition"])
portion = sum(1 for r in rows if r["portion"])
e2e = sum(1 for r in rows if r["e2e_supported"])
print(f"TOTAL={n} nutrition={nutrition} ({nutrition/n*100:.1f}%) portion={portion} ({portion/n*100:.1f}%) e2e_supported={e2e} ({e2e/n*100:.1f}%)")
print(f"{'food':<20} {'status':<22} {'conf':<5} {'portion':<8} {'grams'}")
for r in sorted(rows, key=lambda x: x["food"]):
    print(f"{r['food']:<20} {r['mapping_status']:<22} {str(r['mapping_confidence'] or ''):<5} {str(r['portion']):<8} {r['portion_grams'] or ''}")