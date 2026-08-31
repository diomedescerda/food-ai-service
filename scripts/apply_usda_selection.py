"""FASE 20: aplica la selección USDA revisada al JSON curado (formato existente).
No modifica los registros ya curados; añade los 15 nuevos con trazabilidad.
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CURATED = PROJECT.parent / "coppAddresdBack" / "src" / "CoppAddresd.Api" / "Seeders" / "data" / "food_usda_curated.json"
SELECTION = PROJECT / "usda_selection_results.json"

# alias/display_name/category de cada nuevo (del catálogo food-ai).
META = {
    "steak": ("Bistec", "steak", "restaurant"),
    "grilled_chicken": ("Pollo a la parrilla", "grilled chicken", "restaurant"),
    "salmon": ("Salmón", "salmon", "restaurant"),
    "lasagna": ("Lasaña", "lasagna", "restaurant"),
    "mac_and_cheese": ("Macarrones con queso", "mac and cheese", "restaurant"),
    "toast": ("Tostada", "toast", "breakfast"),
    "bagel": ("Bagel", "bagel", "breakfast"),
    "waffles": ("Waffles", "waffles", "breakfast"),
    "oatmeal": ("Avena", "oatmeal", "breakfast"),
    "cookie": ("Galleta", "cookie", "dessert"),
    "brownie": ("Brownie", "brownie", "dessert"),
    "ice_cream": ("Helado", "ice cream", "dessert"),
    "chicken_nuggets": ("Nuggets de pollo", "chicken nuggets", "fast_food"),
    "quesadilla": ("Quesadilla", "quesadilla", "restaurant"),
    "nachos": ("Nachos", "nachos", "restaurant"),
}


def main() -> None:
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    by_canonical = {f["canonical"]: f for f in curated["foods"]}
    added = 0
    for s in selection:
        canonical = s["canonical"]
        if canonical not in by_canonical:
            print(f"[ERROR] {canonical} no está en el catálogo")
            sys.exit(1)
        display, alias, category = META[canonical]
        existing = by_canonical[canonical]
        # Nunca pisar un registro ya curado con nutrición.
        if existing.get("nutrition") is not None:
            print(f"[SKIP] {canonical} ya tiene nutrición")
            continue
        existing.update({
            "display_name": display,
            "category": category,
            "alias": alias,
            "fdc_id": s["fdc_id"],
            "fdc_name": s["fdc_name"],
            "data_type": s["data_type"],
            "mapping_status": s["mapping_status"],
            "mapping_confidence": s["mapping_confidence"],
            "nutrition": s["nutrition"],
            "notes": s["notes"],
        })
        added += 1
        print(f"[ADD] {canonical}: {s['fdc_name'][:55]} kcal={s['nutrition'].get('calories')} status={s['mapping_status']}")

    curated["meta"]["source_version"] = "FDC 2026-08 (import API 2026-08-31)"
    CURATED.write_text(json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{added} mappings añadidos a {CURATED.name}")


if __name__ == "__main__":
    main()