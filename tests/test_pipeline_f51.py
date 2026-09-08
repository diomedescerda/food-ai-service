"""Tests del pipeline integrado F51: flujo completo, fallback, flags,
specialist routing y orden de ejecución. Usa el índice real; el DINO se
carga una vez (fixture)."""
import numpy as np
import pytest

from app.core.config import settings
from app.models.food_pipeline import FoodPipeline


class MockClf:
    def __init__(self):
        self._processor = None
        self._model = None
        self._features = None

    def _score_crop(self, image):
        return []

    def _crop(self, image, det=None):
        return image


@pytest.fixture(scope="module")
def pipeline():
    return FoodPipeline(clf=MockClf(), enabled=True)


def test_pipeline_ordering_y_estructura(pipeline, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_image_embedding", lambda image: np.random.rand(512))
    res = pipeline.analyze_food(object())
    assert res["fallback"] is False
    assert isinstance(res["canonical_name"], str)
    assert res["retrieval"]["top1"] is not None
    assert res["reranker"]["used"] is True
    assert res["specialist"]["used"] is False
    assert "latency_ms" in res


def test_pipeline_fallback_si_embedding_falla(pipeline, monkeypatch) -> None:
    def boom(image):
        raise RuntimeError("clip caído")
    monkeypatch.setattr(pipeline, "_image_embedding", boom)
    res = pipeline.analyze_food(object())
    assert res["fallback"] is True  # el caller decide usar legacy


def test_pipeline_specialist_gate_cerrado(pipeline, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_image_embedding", lambda image: np.random.rand(512))
    monkeypatch.setattr(pipeline, "_dino_predict", lambda image: ("pizza", 0.99))
    # confianza legacy alta -> el gate del specialist no se abre
    res = pipeline.analyze_food(object(), legacy_conf1=0.9, legacy_top3=["pizza", "naan", "x"])
    assert res["specialist"]["used"] is False


def test_pipeline_specialist_se_usua_cuando_aplica(pipeline, monkeypatch) -> None:
    # query = el text-embedding del canonical "pizza" -> pizza rankea top-1
    pizza_row = next(i for i, t in enumerate(pipeline.texts) if t["text"] == "pizza")
    monkeypatch.setattr(pipeline, "_image_embedding", lambda image: pipeline.index[pizza_row])
    monkeypatch.setattr(pipeline, "_dino_predict", lambda image: ("pizza", 0.9))
    res = pipeline.analyze_food(object(), legacy_conf1=0.3, legacy_top3=["pizza", "naan", "x"])
    assert res["specialist"]["used"] is True
    assert res["specialist"]["confidence"] == 0.9
    assert res["canonical_name"] == "pizza"


def test_pipeline_disabled_fallback() -> None:
    p = FoodPipeline(clf=MockClf(), enabled=False)
    assert p.available() is False
    res = p.analyze_food(object())
    assert res["fallback"] is True


def test_feature_flags_defaults() -> None:
    assert settings.retrieval_enabled is False
    assert settings.retrieval_shadow_enabled is False


def test_pipeline_catalogo_5761(pipeline) -> None:
    assert len(pipeline.support) == 5761  # el catálogo completo disponible