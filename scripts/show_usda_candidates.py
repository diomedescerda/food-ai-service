"""FASE 20: muestra candidatos FDC del sync para selección manual."""
import json

d = json.load(open(r"C:\Users\Carlos Cortina\Documents\CoppAddresd\coppAddresdBack\src\CoppAddresd.Api\Seeders\data\food_usda_api_sync.json", encoding="utf-8"))
for f in d["foods"]:
    if f.get("sync_note") == "ya curado":
        continue
    c = f.get("candidates") or []
    print(f"=== {f['canonical']} ({len(c)} candidatos) ===")
    for x in c[:6]:
        brand = f" | {x['brand']}" if x.get("brand") else ""
        print(f"  {x['fdc_id']} [{x['data_type']}] {x['name'][:85]}{brand}")