"""FASE 19: tabla por clase food-bench-v1 hybrid + disponibilidad de nutrición."""
import json
from collections import defaultdict

data = json.load(open(r"benchmark_hybrid_v1_results.json", encoding="utf-8"))
cur = json.load(open(r"C:\Users\Carlos Cortina\Documents\CoppAddresd\coppAddresdBack\src\CoppAddresd.Api\Seeders\data\food_usda_curated.json", encoding="utf-8"))
mapped = {f["canonical"] for f in cur["foods"] if f.get("mapping_status") in ("DIRECT_MATCH", "GOOD_EQUIVALENCE")}

per = defaultdict(list)
for k, r in data.items():
    per[k.split("/")[0]].append(r)

print(f"{'clase':<18} {'imgs':<5} {'top1%':<7} {'top3%':<7} {'unk':<5} {'mapping':<8} {'con_nutricion'}")
for cls in sorted(per):
    rs = per[cls]
    n = len(rs)
    t1 = sum(1 for r in rs if r["top1"] == cls)
    t3 = sum(1 for r in rs if cls in r["top3"])
    unk = sum(1 for r in rs if r["top1"] == "unknown")
    m = cls in mapped
    print(f"{cls:<18} {n:<5} {t1/n*100:<7.1f} {t3/n*100:<7.1f} {unk:<5} {'SI' if m else 'NO':<8} {t1 if m else '-'}")

print(f"\ntotal: top1 global calculado aparte; clases con mapping: {len(mapped)}/38")