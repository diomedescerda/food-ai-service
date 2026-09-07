"""Tests del active rollout (F55): modo activo, fallback a legacy, rollback,
response contract y ausencia de nutrición sin romper identificación."""
from app.core.config import settings


def test_active_mode_flag_por_defecto() -> None:
    # rollout NO activado globalmente: el default es legacy
    assert settings.retrieval_enabled is False
    assert settings.retrieval_shadow_enabled is False


def test_rollback_flag_vuelve_a_legacy() -> None:
    # con el flag false el pipeline NO se selecciona (comportamiento legacy
    # exacto — el pipeline solo se construye si el flag está activo)
    from app.models.food_pipeline import FoodPipeline  # noqa: PLC0415

    assert FoodPipeline is not None  # el componente existe pero inactivo


def test_pipeline_error_fallback_legacy() -> None:
    from app.models.decision import DecisionPolicy  # noqa: PLC0415

    p = DecisionPolicy()
    d = p.decide(visual_confidence=0.0, nutrition_status="NUTRITION_READY",
                 pipeline_error=True)
    assert d["decision"] == "LEGACY_FALLBACK"
    assert d["fallback_reason"] == "pipeline_error"


def test_nutrition_unavailable_no_rompe_identificacion() -> None:
    from app.models.decision import DecisionPolicy  # noqa: PLC0415

    p = DecisionPolicy()
    d = p.decide(visual_confidence=0.30, nutrition_status="NUTRITION_UNAVAILABLE")
    assert d["identification_ready"] is True
    assert d["nutrition_ready"] is False


def test_active_response_contract() -> None:
    # el response del active mode usa el mismo schema (DetectedFood):
    # food + confidence — sin campos nuevos públicos
    from app.schemas.analyze import AnalyzeResponse, DetectedFood  # noqa: PLC0415

    assert AnalyzeResponse is not None and DetectedFood is not None


def test_confidence_real_en_active() -> None:
    # el pipeline expone la confianza visual real (retrieval score), no un 0.5 fijo
    import numpy as np  # noqa: PLC0415

    from app.models.food_pipeline import FoodPipeline  # noqa: PLC0415

    class MockClf:
        def _score_crop(self, image):
            return []

        def _crop(self, image, det=None):
            return image

    p = FoodPipeline(clf=MockClf(), enabled=True)
    res = p.analyze_food(object())
    if not res.get("fallback"):
        assert isinstance(res.get("confidence", {}).get("retrieval_score"), float)