"""FASE 20: búsqueda dirigida en FDC — candidatos genéricos (FNDDS/SR Legacy/Foundation)
con nutrientes por 100g para los 15 pendientes."""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env", override=False)
KEY = os.environ["FOODAI_USDA_API_KEY"]

QUERIES = {
    "steak": "beef steak cooked",
    "grilled_chicken": "chicken breast grilled",
    "salmon": "fish salmon cooked",
    "lasagna": "lasagna with meat",
    "mac_and_cheese": "macaroni and cheese prepared",
    "toast": "bread white toasted",
    "bagel": "bagel plain",
    "waffles": "waffle plain",
    "oatmeal": "oatmeal cooked",
    "cookie": "cookie chocolate chip",
    "brownie": "brownie chocolate",
    "ice_cream": "ice cream vanilla",
    "chicken_nuggets": "chicken nuggets",
    "quesadilla": "quesadilla plain",
    "nachos": "nachos with cheese",
}

NUTRIENT_IDS = {"calories": 1008, "protein": 1003, "carbohydrates": 1005, "fat": 1004, "fiber": 1079, "sugar": 2000, "sodium": 1093}
BASE = "https://api.nal.usda.gov/fdc/v1"


def nutrients(fdc_id: str) -> dict:
    url = f"{BASE}/food/{fdc_id}?nutrients={','.join(str(v) for v in NUTRIENT_IDS.values())}&api_key={KEY}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = {}
    for n in data.get("foodNutrients", []):
        nutrient = n.get("nutrient", {})
        name = nutrient.get("number")
        for key, num in NUTRIENT_IDS.items():
            if str(num) == str(nutrient.get("number")):
                out[key] = round(n.get("amount", 0), 2)
    return out


for food, query in QUERIES.items():
    url = f"{BASE}/foods/search?query={urllib.parse.quote(query)}&pageSize=10&api_key={KEY}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"=== {food}: ERROR {exc}")
        continue
    print(f"=== {food} (query: {query}) ===")
    shown = 0
    for item in result.get("foods", []):
        dt = item.get("dataType")
        desc = item.get("description", "")
        if dt == "Branded":
            continue
        fdc_id = str(item.get("fdcId"))
        nut = nutrients(fdc_id)
        kcal = nut.get("calories")
        print(f"  {fdc_id} [{dt}] {desc[:70]} | kcal={kcal} p={nut.get('protein')} c={nut.get('carbohydrates')} f={nut.get('fat')}")
        shown += 1
        if shown >= 4:
            break
    if shown == 0:
        print("  (solo Branded en el top)")