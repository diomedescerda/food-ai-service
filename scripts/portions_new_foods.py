"""FASE 20: foodPortions (referencia de porción) de los 5 nuevos sin REFERENCE_GRAMS."""
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
key = os.environ["FOODAI_USDA_API_KEY"]

for name, fid in (("lasagna", "2708750"), ("mac_and_cheese", "169770"),
                  ("grilled_chicken", "2705968"), ("cookie", "2707909"), ("brownie", "2707904")):
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fid}?api_key={key}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    print(f"=== {name} ({d.get('description')[:45]})")
    for p in (d.get("foodPortions") or [])[:4]:
        unit = p.get("measureUnit", {}).get("name", "")
        desc = p.get("portionDescription") or p.get("modifier") or ""
        print(f"  {unit} {desc} = {p.get('gramWeight')} g")