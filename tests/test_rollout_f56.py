"""Tests del rollout gradual (F56): split determinista por porcentaje,
enrutamiento new/legacy y métricas por etapa."""
from app.core.config import settings


def test_split_determinista_por_porcentaje() -> None:
    def use_new(name: str, pct: float) -> bool:
        return (hash(name) % 100) / 100.0 < pct

    names = [f"img_{i:03d}.jpg" for i in range(100)]
    assert sum(use_new(n, 0.0) for n in names) == 0
    assert sum(use_new(n, 1.0) for n in names) == 100
    p10 = sum(use_new(n, 0.10) for n in names)
    assert 5 <= p10 <= 15  # ~10% con varianza de hash
    # determinismo: el mismo nombre -> la misma decisión
    assert use_new("img_007.jpg", 0.5) == use_new("img_007.jpg", 0.5)


def test_rollout_0_por_ciento_es_legacy() -> None:
    assert settings.retrieval_enabled is False  # default: 0% new


def test_rollback_flag_vuelve_a_legacy() -> None:
    # el pipeline solo se construye con el flag activo; apagado = legacy
    from app.models.food_pipeline import FoodPipeline  # noqa: PLC0415

    class MockClf:
        def _score_crop(self, image):
            return []

    p = FoodPipeline(clf=MockClf(), enabled=False)
    assert p.available() is False  # inactivo -> sin pipeline -> legacy


def test_fallback_se_mide_por_separado() -> None:
    from app.models.decision import DecisionPolicy  # noqa: PLC0415

    p = DecisionPolicy()
    d1 = p.decide(0.3, "NUTRITION_READY", pipeline_error=True)
    d2 = p.decide(0.3, "NUTRITION_UNAVAILABLE")
    d3 = p.decide(0.1, "NUTRITION_READY", pipeline_error=False)
    reasons = {d1["fallback_reason"], d2["fallback_reason"], d3["fallback_reason"]}
    assert "pipeline_error" in reasons  # fallbacks no escondidos en una métrica
    assert "nutrition_unavailable" in reasons