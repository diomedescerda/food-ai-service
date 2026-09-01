"""Revisa los 53 mappings: revierte los sospechosos (nombres cortos) y fuerza
NO_RELIABLE para soup/cereal/sandwich."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
master = json.loads((BASE / "data/catalogs/food_master.json").read_text(encoding="utf-8"))
NO_RELIABLE = {"soup", "cereal", "sandwich"}
reverted = []
for f in master["foods"]:
    if f["canonical_name"] in NO_RELIABLE:
        f["nutrition"] = None
        f["portion"] = None
        f["status"] = "NO_RELIABLE_NUTRITION"
        reverted.append(f["canonical_name"])
        continue
    nut = f["nutrition"] or {}
    if nut.get("mapping_status") not in ("DIRECT_MATCH", "GOOD_EQUIVALENCE"):
        continue
    fdc_name = (nut.get("fdc_name") or "").lower()
    canonical = f["canonical_name"].replace("_", " ")
    # Regla estricta: todos los tokens del canonical deben estar en el fdc_name
    tokens = [t for t in canonical.split() if len(t) > 2]
    missing = [t for t in tokens if t not in fdc_name]
    if missing:
        f["nutrition"] = None
        f["portion"] = None
        f["status"] = "VISUAL_ONLY"
        reverted.append(f"{f['canonical_name']} (faltan tokens: {missing})")

master["foods"] = sorted(master["foods"], key=lambda x: x["canonical_name"])
(BASE / "data/catalogs/food_master.json").write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
from collections import Counter
print("revertidos:", reverted)
print(dict(Counter(f["status"] for f in master["foods"])))
# listar los mappings vigentes para inspección
for f in master["foods"]:
    nut = f["nutrition"] or {}
    if nut.get("mapping_status") in ("DIRECT_MATCH", "GOOD_EQUIVALENCE"):
        print(f"  {f['canonical_name']:<28} {nut['mapping_status']:<17} {nut['fdc_name'][:50]}")