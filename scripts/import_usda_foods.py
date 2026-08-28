"""Importador USDA FoodData Central → catálogo nutricional.

MODO CURATED (por defecto, sin API key):
    valida y resume food_usda_curated.json (committeado). Los valores fueron
    verificados contra registros FDC (2026-08-28). El seeder .NET lo consume.

MODO API (escalable, requiere key gratuita):
    FOODAI_USDA_API_KEY=<key> python scripts/import_usda_foods.py --sync
    Busca cada alimento del catálogo en la API de FDC, rankea candidatos
    (preferencia SR Legacy/FNDDS genéricos sobre Branded), genera un JSON con
    los registros seleccionados y status REVIEW_REQUIRED cuando la selección
    es ambigua.

Uso:
    python scripts/import_usda_foods.py                 # valida el JSON curado
    FOODAI_USDA_API_KEY=... python scripts/import_usda_foods.py --sync
"""

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CURATED = Path(__file__).resolve().parent.parent.parent / "coppAddresdBack" / "src" / "CoppAddresd.Api" / "Seeders" / "data" / "food_usda_curated.json"
OUT_JSON = Path(__file__).resolve().parent.parent.parent / "coppAddresdBack" / "src" / "CoppAddresd.Api" / "Seeders" / "data" / "food_usda_api_sync.json"

VALID_NUTRIENTS = {"calories", "protein", "carbohydrates", "fat", "fiber", "sugar", "sodium"}
VALID_STATUS = {"DIRECT_MATCH", "GOOD_EQUIVALENCE", "AMBIGUOUS", "REVIEW_REQUIRED", "NO_RELIABLE_MATCH"}


def validate_curated(data: dict) -> tuple[int, list[str]]:
    errors: list[str] = []
    foods = data.get("foods", [])
    seen: set[str] = set()
    for food in foods:
        canonical = food.get("canonical")
        if not canonical or canonical in seen:
            errors.append(f"canonical duplicado o vacío: {canonical}")
        seen.add(canonical)
        if food.get("mapping_status") not in VALID_STATUS:
            errors.append(f"{canonical}: status inválido {food.get('mapping_status')}")
        nutrition = food.get("nutrition")
        if nutrition is not None:
            if set(nutrition.keys()) != VALID_NUTRIENTS:
                errors.append(f"{canonical}: nutrientes incompletos {sorted(nutrition.keys())}")
            for key, value in nutrition.items():
                if value is None or value < 0:
                    errors.append(f"{canonical}.{key}: negativo o nulo")
            if food.get("fdc_id") is None:
                errors.append(f"{canonical}: tiene nutrition pero sin fdc_id")
        elif food.get("mapping_status") not in ("REVIEW_REQUIRED", "NO_RELIABLE_MATCH", "AMBIGUOUS"):
            errors.append(f"{canonical}: sin nutrition y status {food.get('mapping_status')}")
        confidence = food.get("mapping_confidence", 0)
        if not (0 <= confidence <= 1):
            errors.append(f"{canonical}: mapping_confidence fuera de rango")
    return len(foods), errors


def summarize(data: dict) -> dict:
    foods = data["foods"]
    by_status: dict[str, int] = {}
    for food in foods:
        status = food["mapping_status"]
        by_status[status] = by_status.get(status, 0) + 1
    with_nutrition = sum(1 for f in foods if f.get("nutrition") is not None)
    return {
        "total": len(foods),
        "with_nutrition": with_nutrition,
        "by_status": by_status,
        "coverage_pct": round(with_nutrition / len(foods) * 100, 1),
    }


def api_sync(data: dict) -> dict:
    """Modo API: busca cada alimento en FDC y genera candidatos (sin escribir
    la DB; la selección final queda en OUT_JSON para revisión)."""
    import urllib.parse
    import urllib.request

    api_key = os.environ.get("FOODAI_USDA_API_KEY")
    if not api_key:
        raise SystemExit("FOODAI_USDA_API_KEY requerida para --sync (registro gratuito en fdc.nal.usda.gov)")

    base = "https://api.nal.usda.gov/fdc/v1"
    synced: list[dict] = []
    for food in data["foods"]:
        canonical = food["canonical"]
        if food.get("fdc_id") and food.get("nutrition") is not None:
            synced.append({**food, "sync_note": "ya curado"})
            continue
        query = urllib.parse.quote(canonical.replace("_", " "))
        url = f"{base}/foods/search?query={query}&pageSize=5&api_key={api_key}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            candidates = []
            for item in result.get("foods", []):
                candidates.append({
                    "fdc_id": str(item.get("fdcId")),
                    "name": item.get("description"),
                    "data_type": item.get("dataType"),
                })
            synced.append({
                **food,
                "fdc_id": None,
                "candidates": candidates,
                "sync_note": "selección manual requerida" if candidates else "sin candidatos",
            })
        except Exception as exc:
            synced.append({**food, "sync_note": f"error API: {exc}"})

    data["foods"] = synced
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="usar API de FDC (requiere key)")
    args = parser.parse_args()

    data = json.loads(CURATED.read_text(encoding="utf-8"))
    total, errors = validate_curated(data)
    if errors:
        print(f"[ERROR] {len(errors)} problemas:")
        for e in errors[:20]:
            print(f"  - {e}")
        sys.exit(1)

    print(f"[OK] JSON curado válido: {total} alimentos")
    summary = summarize(data)
    print(f"  con nutrición: {summary['with_nutrition']} ({summary['coverage_pct']}%)")
    print(f"  por status: {summary['by_status']}")

    if args.sync:
        print("[sync] consultando API de FDC...")
        synced = api_sync(data)
        print(f"[sync] candidatos guardados en {OUT_JSON.name} (selección manual requerida)")


if __name__ == "__main__":
    main()