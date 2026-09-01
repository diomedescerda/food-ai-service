"""FASE 25: reconstruye el master con mappings verificados.

1. build_food_master (38 del curado + 193 nuevas sin nutrition)
2. apply mappings del sync (solo a las nuevas)
3. Revertir falsos conocidos (platos específicos no equivalentes al genérico)
4. Forzar NO_RELIABLE para soup/cereal/sandwich
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MASTER = BASE / "data/catalogs/food_master.json"
SYNC = BASE / "data/catalogs/usda_sync_expansion.json"

# Falsos positivos de la primera pasada: el FDC elegido es un plato/marca
# específico, no la equivalencia genérica del canonical → REVIEW_REQUIRED.
REVERT = {
    "avocado": "Avocado dressing (no es aguacate genérico)",
    "cod": "Cape Cod (nombre propio, no bacalao)",
    "clams": "Clams Casino (plato preparado)",
    "oysters": "Oysters Rockefeller (plato preparado)",
    "meatloaf": "Meatloaf sandwich (no meatloaf)",
    "pie": "Pie shell (masa, no pie)",
    "cornbread": "Cornbread stuffing (no cornbread)",
    "pudding": "Banana pudding (plato específico)",
    "chips": "Cracker chips (no potato chips)",
    "casserole": "Green bean casserole (plato específico)",
    "curry": "Beef curry (específico de res)",
    "biryani": "Biryani with vegetables (específico)",
    "quiche": "Spinach quiche (específico)",
    "muffin": "Muffin, zucchini (específico)",
    "gyro": "Gyro sandwich (razonable pero genérico preferible)",
    "cheeseburger": "FNDDS con marca McDonald's (revisar)",
    "nuts": "Cashew nuts (subconjunto de frutos secos)",
}


def rebuild() -> None:
    import subprocess
    import sys

    subprocess.run([sys.executable, "-B", "scripts/build_food_master.py"], cwd=BASE, check=True)
    subprocess.run([sys.executable, "-B", "scripts/apply_master_mappings.py"], cwd=BASE, check=True)


def main() -> None:
    rebuild()
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    sync = json.loads(SYNC.read_text(encoding="utf-8"))
    sync_map = {r["canonical"]: r for r in sync}

    for f in master["foods"]:
        canonical = f["canonical_name"]
        if canonical in REVERT:
            f["nutrition"] = None
            f["portion"] = None
            f["status"] = "REVIEW_REQUIRED"
            continue
        if canonical in ("soup", "cereal", "sandwich"):
            f["nutrition"] = None
            f["portion"] = None
            f["status"] = "NO_RELIABLE_NUTRITION"
            continue
        # Los mappings del sync pasan el token-check de los 38 (no tocar).
        nut = f["nutrition"] or {}
        if nut.get("mapping_status") not in ("DIRECT_MATCH", "GOOD_EQUIVALENCE"):
            continue
        fdc_name = (nut.get("fdc_name") or "").lower()
        tokens = [t for t in canonical.replace("_", " ").split() if len(t) > 2]
        missing = [t for t in tokens if t not in fdc_name]
        if missing:
            f["nutrition"] = None
            f["portion"] = None
            f["status"] = "VISUAL_ONLY"

    master["foods"] = sorted(master["foods"], key=lambda x: x["canonical_name"])
    MASTER.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
    print(dict(Counter(f["status"] for f in master["foods"])))
    print(f"TOTAL: {len(master['foods'])}")


if __name__ == "__main__":
    main()