"""Completa nutrientes ausentes (=0 por 100g en FDC) del JSON curado."""
import json

p = r"C:\Users\Carlos Cortina\Documents\CoppAddresd\coppAddresdBack\src\CoppAddresd.Api\Seeders\data\food_usda_curated.json"
d = json.load(open(p, encoding="utf-8"))
for f in d["foods"]:
    nut = f.get("nutrition")
    if nut:
        for k in ("calories", "protein", "carbohydrates", "fat", "fiber", "sugar", "sodium"):
            if nut.get(k) is None:
                nut[k] = 0
                print(f"{f['canonical']}.{k} -> 0 (ausente en FDC = 0 por 100g)")
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("ok")