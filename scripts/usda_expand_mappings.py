"""FASE 25 P4: busca mappings USDA genéricos para las clases nuevas del master.

Reglas (NO mapping ciego):
1. search FDC por canonical → filtrar data_type en (FNDDS, SR Legacy, Foundation)
2. excluir Branded
3. similitud de tokens entre canonical y fdc_name:
   >= 0.5 → DIRECT_MATCH · >= 0.3 → GOOD_EQUIVALENCE · < 0.3 → REVIEW_REQUIRED (sin mapping)
4. nutrientes + foodPortions del FDC elegido (verificados, no inventados)

Salida: data/catalogs/usda_sync_expansion.json (candidatos) — NO toca el master.
Uso: python scripts/usda_expand_mappings.py
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env", override=False)
KEY = os.environ.get("FOODAI_USDA_API_KEY") or ""
API = "https://api.nal.usda.gov/fdc/v1"
MASTER = BASE / "data/catalogs/food_master.json"
OUT = BASE / "data/catalogs/usda_sync_expansion.json"

NUT = {"calories": 208, "protein": 203, "carbohydrates": 205, "fat": 204, "fiber": 291, "sugar": 269, "sodium": 307}
GENERIC_TYPES = ("Survey (FNDDS)", "SR Legacy", "Foundation")
BLACKLIST = ("nfs,", "restaurant", "school lunch", "fast food", "fast-food")


def token_sim(a: str, b: str) -> float:
    ta = {t for t in re.sub(r"[^a-z ]", "", a.lower()).split() if len(t) > 2}
    tb = {t for t in re.sub(r"[^a-z ]", "", b.lower()).split() if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def search(canonical: str) -> list[dict]:
    url = f"{API}/foods/search?query={urllib.parse.quote(canonical.replace('_', ' '))}&pageSize=10&api_key={KEY}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8")).get("foods", [])


def fetch(fdc_id: str) -> dict:
    url = f"{API}/food/{fdc_id}?api_key={KEY}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def nutrients_of(data: dict) -> dict:
    out = {}
    for n in data.get("foodNutrients", []):
        try:
            num = int(n.get("nutrient", {}).get("number"))
        except (TypeError, ValueError):
            continue
        for key, target in NUT.items():
            if num == target:
                out[key] = round(n.get("amount", 0), 2)
    for key in NUT:
        out.setdefault(key, 0)
    return out


def portions_of(data: dict) -> list[dict]:
    out = []
    for p in (data.get("foodPortions") or [])[:3]:
        unit = p.get("measureUnit", {}).get("name", "")
        desc = p.get("portionDescription") or p.get("modifier") or ""
        if unit in ("g", "gram", "grams") or "g" == unit:
            continue  # solo porciones domésticas
        out.append({"unit": unit, "description": desc, "gram_weight": p.get("gramWeight")})
    return out


def main() -> None:
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    results = []
    for food in master["foods"]:
        if food["nutrition"] is not None:
            continue  # ya curados (38)
        canonical = food["canonical_name"]
        try:
            hits = search(canonical)
        except Exception as exc:  # noqa: BLE001
            results.append({"canonical": canonical, "error": str(exc)})
            continue
        candidates = []
        best = None
        for item in hits:
            dt = item.get("dataType", "")
            desc = item.get("description", "")
            if dt not in GENERIC_TYPES:
                continue
            if any(b in desc.lower() for b in BLACKLIST):
                continue
            sim = token_sim(canonical, desc)
            candidates.append({"fdc_id": str(item.get("fdcId")), "name": desc, "data_type": dt, "sim": round(sim, 2)})
            if best is None or sim > best["sim"]:
                best = {"fdc_id": str(item.get("fdcId")), "name": desc, "data_type": dt, "sim": sim}
        if best is None:
            results.append({"canonical": canonical, "status": "REVIEW_REQUIRED", "reason": "sin candidato genérico",
                            "candidates": candidates[:3]})
            time.sleep(0.3)
            continue
        try:
            detail = fetch(best["fdc_id"])
        except Exception as exc:  # noqa: BLE001
            results.append({"canonical": canonical, "status": "REVIEW_REQUIRED", "reason": f"fetch: {exc}"})
            continue
        status = "DIRECT_MATCH" if best["sim"] >= 0.5 else ("GOOD_EQUIVALENCE" if best["sim"] >= 0.3 else "REVIEW_REQUIRED")
        results.append({
            "canonical": canonical,
            "status": status,
            "confidence": round(max(0.5, best["sim"]), 2) if status != "REVIEW_REQUIRED" else None,
            "fdc_id": best["fdc_id"] if status != "REVIEW_REQUIRED" else None,
            "fdc_name": best["name"] if status != "REVIEW_REQUIRED" else None,
            "data_type": best["data_type"] if status != "REVIEW_REQUIRED" else None,
            "nutrition": nutrients_of(detail) if status != "REVIEW_REQUIRED" else None,
            "portions": portions_of(detail) if status != "REVIEW_REQUIRED" else None,
            "candidates": candidates[:3],
        })
        time.sleep(0.3)
        if len(results) % 25 == 0:
            print(f"[prog] {len(results)}", flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    print(f"TOTAL: {len(results)}")
    print(dict(Counter(r.get("status", "ERROR") for r in results)))


if __name__ == "__main__":
    main()