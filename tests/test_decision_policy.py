"""Tests de la política de decisión (F53): confianzas separadas, fallback,
razones, DINO/nutrition/portion failures y determinismo."""
from app.models.decision import DecisionPolicy


def test_caso_a_nutrition_ready() -> None:
    p = DecisionPolicy()
    d = p.decide(visual_confidence=0.31, nutrition_status="NUTRITION_READY")
    assert d["decision"] == "NEW_RESULT_READY"
    assert d["identification_ready"] is True
    assert d["nutrition_ready"] is True
    assert d["fallback_reason"] is None


def test_caso_b_identificacion_sin_nutricion() -> None:
    p = DecisionPolicy()
    d = p.decide(visual_confidence=0.31, nutrition_status="NUTRITION_UNAVAILABLE")
    assert d["decision"] == "NEW_RESULT_NUTRITION_UNAVAILABLE"
    assert d["identification_ready"] is True  # la identificación NO se revierte
    assert d["nutrition_ready"] is False
    assert d["fallback_reason"] == "nutrition_unavailable"


def test_caso_c_confianza_baja_solo_si_enabled() -> None:
    p = DecisionPolicy(min_visual_confidence=0.20, confidence_enabled=False)
    d = p.decide(visual_confidence=0.10, nutrition_status="NUTRITION_READY")
    assert d["decision"] == "NEW_RESULT_READY"  # threshold no activado
    p2 = DecisionPolicy(min_visual_confidence=0.20, confidence_enabled=True)
    d2 = p2.decide(visual_confidence=0.10, nutrition_status="NUTRITION_READY")
    assert d2["decision"] == "NEW_RESULT_LOW_CONFIDENCE"
    assert d2["fallback_reason"] == "low_visual_confidence"


def test_caso_d_pipeline_error() -> None:
    p = DecisionPolicy()
    d = p.decide(visual_confidence=0.0, nutrition_status="NUTRITION_READY",
                 pipeline_error=True)
    assert d["decision"] == "LEGACY_FALLBACK"
    assert d["fallback_reason"] == "pipeline_error"
    assert d["identification_ready"] is False


def test_caso_e_nutrition_error_no_rompe_identificacion() -> None:
    p = DecisionPolicy()
    d = p.decide(visual_confidence=0.30, nutrition_status="NUTRITION_READY",
                 nutrition_error=True)
    assert d["decision"] == "NEW_RESULT_NUTRITION_UNAVAILABLE"
    assert d["fallback_reason"] == "nutrition_error"
    assert d["identification_ready"] is True  # el alimento NO se pierde


def test_portion_unavailable_no_inventa() -> None:
    p = DecisionPolicy()
    d = p.decide(visual_confidence=0.30, nutrition_status="NUTRITION_READY")
    # el policy no fabrica porción ni calorías: solo estados
    assert "portion_grams" not in d
    assert "calories" not in d


def test_confianzas_independientes() -> None:
    p = DecisionPolicy()
    # nutrición alta no infla la visual ni viceversa
    d1 = p.decide(0.10, "NUTRITION_READY")
    d2 = p.decide(0.10, "NUTRITION_UNAVAILABLE")
    assert d1["identification_ready"] == d2["identification_ready"] is True


def test_determinismo() -> None:
    p = DecisionPolicy()
    a = p.decide(0.25, "NUTRITION_READY")
    b = p.decide(0.25, "NUTRITION_READY")
    assert a == b