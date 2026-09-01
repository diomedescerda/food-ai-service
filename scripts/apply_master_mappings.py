"""FASE 25: aplica los mappings USDA verificados al Food Master Catalog."""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
master = json.loads((BASE / "data/catalogs/food_master.json").read_text(encoding="utf-8"))
sync = json.loads((BASE / "data/catalogs/usda_sync_expansion.json").read_text(encoding="utf-8"))
by = {r["canonical"]: r for r in sync}

applied = 0
for f in master["foods"]:
    r = by.get(f["canonical_name"])
    if not r or r.get("status") not in ("DIRECT_MATCH", "GOOD_EQUIVALENCE"):
        continue
    portion = None
    for p in (r.get("portions") or []):
        if p.get("gram_weight"):
            portion = {
                "reference_grams": round(p["gram_weight"]),
                "min_grams": round(p["gram_weight"] * 0.75),
                "max_grams": round(p["gram_weight"] * 1.25),
                "source": f"FDC foodPortions {p['unit']} {p['description']}".strip(),
                "source_id": r["fdc_id"],
            }
            break
    f["nutrition"] = {
        "fdc_id": r["fdc_id"], "fdc_name": r["fdc_name"], "data_type": r["data_type"],
        "mapping_status": r["status"], "mapping_confidence": r["confidence"],
        "source": "USDA FoodData Central", "source_version": "FDC 2026-08", "source_id": r["fdc_id"],
    }
    f["portion"] = portion
    f["status"] = "PRODUCTION_READY" if portion else "NUTRITION_READY"
    applied += 1

master["foods"] = sorted(master["foods"], key=lambda x: x["canonical_name"])
(BASE / "data/catalogs/food_master.json").write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
print("aplicados:", applied)
print(dict(Counter(f["status"] for f in master["foods"])))