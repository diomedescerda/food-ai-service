"""Tests del NutritionService (F52): mapping, faltante, confianza,
normalización per-100g, fallback, disabled, shadow y determinismo."""
from app.models.nutrition_service import NutritionService


def _service() -> NutritionService:
    return NutritionService(enabled=True)


def test_mapping_encontrado() -> None:
    ns = _service()
    assert ns.available()
    r = ns.nutrition_for("fndds_2708699")  # pizza (del catálogo)
    assert r["status"] == "NUTRITION_READY"
    assert "calories" in r["nutrients_per_100g"]
    assert r["reference_grams"] == 100


def test_mapping_faltante() -> None:
    ns = _service()
    r = ns.nutrition_for("no_existe")
    assert r["status"] == "NUTRITION_UNAVAILABLE"
    assert r["nutrition_confidence"] == 0.0


def test_food_id_vacio() -> None:
    ns = _service()
    r = ns.nutrition_for("")
    assert r["status"] == "NUTRITION_UNAVAILABLE"


def test_confianza_por_fuente() -> None:
    ns = _service()
    usda = ns.nutrition_for("fndds_2708699")
    assert usda["nutrition_confidence"] == 0.95
    # el primer OFF con nutrientes
    off_id = next(fid for fid, m in ns.index.items()
                  if m.get("source") == "Open Food Facts" and "nutrients_per_100g" in m)
    off = ns.nutrition_for(off_id)
    assert off["nutrition_confidence"] == 0.85


def test_normalizacion_per_100g() -> None:
    ns = _service()
    r = ns.nutrition_for("fndds_2708699")
    cal = r["nutrients_per_100g"]["calories"]
    assert cal["unit"] == "kcal"  # unidades explícitas
    assert r["reference_grams"] == 100  # referencia consistente


def test_canonical_resolucion() -> None:
    ns = _service()
    r = ns.canonical_nutrition("pizza")
    assert r["status"] == "NUTRITION_READY"
    assert r["nutrition_confidence"] == 0.95


def test_canonical_desconocido() -> None:
    ns = _service()
    r = ns.canonical_nutrition("alimento_inexistente")
    assert r["status"] == "NUTRITION_UNAVAILABLE"


def test_nutrition_disabled() -> None:
    ns = NutritionService(enabled=False)
    assert not ns.available()
    r = ns.nutrition_for("fndds_2708699")
    assert r["status"] == "NUTRITION_UNAVAILABLE"  # fallback seguro


def test_deterministico() -> None:
    ns = _service()
    a = ns.nutrition_for("fndds_2708699")
    b = ns.nutrition_for("fndds_2708699")
    assert a == b


def test_golden_foods_ready() -> None:
    ns = _service()
    for name in ("pizza", "hamburger", "rice", "fries", "chicken", "pasta", "sandwich", "bread"):
        r = ns.canonical_nutrition(name)
        assert r["status"] == "NUTRITION_READY", name
        assert r["nutrients_per_100g"]["calories"]["value"] > 0, name