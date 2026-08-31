"""FASE 20: selección final de candidatos USDA genéricos con nutrientes por 100g.

Cada selección: fdc_id verificado vía API, descripción genérica (FNDDS/SR
Legacy/Foundation), sin marca, mapeada con mapping_status/confidence según la
defensabilidad. Escribe al JSON curado SOLO lo que el agente revisa después.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env", override=False)
KEY = os.environ["FOODAI_USDA_API_KEY"]
BASE = "https://api.nal.usda.gov/fdc/v1"
CURATED = PROJECT_DIR.parent / "coppAddresdBack" / "src" / "CoppAddresd.Api" / "Seeders" / "data" / "food_usda_curated.json"

NUT = {"calories": 208, "protein": 203, "carbohydrates": 205, "fat": 204, "fiber": 291, "sugar": 269, "sodium": 307}

# Selección del agente tras revisar candidatos reales de la API (búsqueda
# dirigida). Todos genéricos FNDDS/SR Legacy, sin marca.
SELECTION = {
    "steak": ("2705825", "Beef, steak, chuck (FNDDS)", "GOOD_EQUIVALENCE", 0.8),
    "grilled_chicken": ("2705968", "Chicken breast, grilled without sauce, skin not eaten (FNDDS)", "GOOD_EQUIVALENCE", 0.9),
    "salmon": ("172000", "Fish, salmon, chum, cooked, dry heat (SR Legacy)", "GOOD_EQUIVALENCE", 0.85),
    "lasagna": ("2708750", "Lasagna with meat (FNDDS)", "DIRECT_MATCH", 0.9),
    "mac_and_cheese": ("169770", "Macaroni and cheese, box mix with cheese sauce, prepared (SR Legacy)", "GOOD_EQUIVALENCE", 0.85),
    "toast": ("2707599", "Bread, white, toasted (FNDDS)", "GOOD_EQUIVALENCE", 0.95),
    "bagel": ("2707684", "Bagel (FNDDS)", "DIRECT_MATCH", 0.95),
    "waffles": ("2708325", "Waffle, plain (FNDDS)", "DIRECT_MATCH", 0.9),
    "oatmeal": ("2708380", "Oatmeal, NFS (FNDDS)", "GOOD_EQUIVALENCE", 0.85),
    "cookie": ("2707909", "Cookie, chocolate chip (FNDDS)", "GOOD_EQUIVALENCE", 0.9),
    "brownie": ("2707904", "Cookie, brownie, without icing (FNDDS)", "GOOD_EQUIVALENCE", 0.85),
    "ice_cream": ("2705630", "Ice cream, vanilla (FNDDS)", "GOOD_EQUIVALENCE", 0.9),
    "chicken_nuggets": ("2706092", "Chicken nuggets, NFS (FNDDS)", "DIRECT_MATCH", 0.9),
    "quesadilla": ("2708590", "Quesadilla, NFS (FNDDS)", "GOOD_EQUIVALENCE", 0.85),
    "nachos": ("2708577", "Nachos, cheese only (FNDDS)", "GOOD_EQUIVALENCE", 0.85),
}


def fetch(fdc_id: str) -> dict:
    url = f"{BASE}/food/{fdc_id}?api_key={KEY}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def nutrients_of(data: dict) -> dict:
    out = {}
    for n in data.get("foodNutrients", []):
        num = n.get("nutrient", {}).get("number")
        try:
            num = int(num)
        except (TypeError, ValueError):
            continue
        for key, target in NUT.items():
            if num == target:
                out[key] = round(n.get("amount", 0), 2)
    return out


def main() -> None:
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    foods = {f["canonical"]: f for f in curated["foods"]}
    results = []
    for canonical, (fdc_id, label, status, conf) in SELECTION.items():
        try:
            data = fetch(fdc_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {canonical} {fdc_id}: {exc}")
            continue
        nut = nutrients_of(data)
        actual_name = data.get("description", "")
        missing = [k for k in NUT if nut.get(k) is None]
        if missing:
            print(f"[WARN] {canonical}: faltan {missing} — usar {actual_name}")
        results.append({
            "canonical": canonical,
            "fdc_id": fdc_id,
            "fdc_name": actual_name,
            "data_type": data.get("dataType"),
            "mapping_status": status,
            "mapping_confidence": conf,
            "nutrition": nut,
            "notes": label,
        })
        print(f"[OK] {canonical}: {actual_name[:60]} kcal={nut.get('calories')} p={nut.get('protein')} c={nut.get('carbohydrates')} f={nut.get('fat')}")

    out = Path(__file__).resolve().parent.parent / "usda_selection_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(results)} seleccionados -> {out}")


if __name__ == "__main__":
    main()