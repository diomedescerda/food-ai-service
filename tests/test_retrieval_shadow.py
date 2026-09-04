"""Tests del RetrievalShadow (F46): disponible, fallback, telemetría,
invariancia del input y defaults de configuración. Sin modelos reales."""
from app.core.config import settings
from app.models.retrieval_shadow import RetrievalShadow


class FakeRetrieval:
    catalog_size = 1451

    def available(self) -> bool:
        return True

    def retrieve(self, image, top_k: int = 50):
        return [
            {"food_id": "f0", "name": "pizza", "score": 0.30},
            {"food_id": "f1", "name": "fries", "score": 0.29},
            {"food_id": "f2", "name": "rice", "score": 0.28},
        ]


class BrokenRetrieval(FakeRetrieval):
    def retrieve(self, image, top_k: int = 50):
        raise RuntimeError("índice corrupto")


class FakeImage:
    pass


def test_shadow_disabled_no_disponible() -> None:
    shadow = RetrievalShadow(enabled=False)
    assert not shadow.available()
    tel = shadow.shadow_evaluate(FakeImage(), "id-1", 0.31, ["fries", "rice", "pizza"])
    assert tel == {"analysis_id": "id-1", "shadow_available": False}


def test_shadow_fallback_si_retrieval_falla() -> None:
    shadow = RetrievalShadow(enabled=True, clf=object(), retrieval=BrokenRetrieval())
    shadow._dino = object()  # simula DINO cargado (available() solo mira flags)
    shadow._processor = object()
    shadow._head = object()
    shadow._label_map = {}
    tel = shadow.shadow_evaluate(FakeImage(), "id-2", 0.31, ["fries", "rice", "pizza"])
    assert tel["fallback"] is True
    assert tel["error"] is not None
    assert tel["shadow_available"] is True


def test_shadow_telemetria_completa(monkeypatch) -> None:
    shadow = RetrievalShadow(enabled=True, clf=object(), retrieval=FakeRetrieval())
    shadow._dino = object()
    shadow._processor = object()
    shadow._head = object()
    shadow._label_map = {}
    monkeypatch.setattr(shadow, "_specialist_predict", lambda image: ("pizza", 0.90))
    tel = shadow.shadow_evaluate(FakeImage(), "id-3", 0.31, ["fries", "rice", "pizza"])
    assert tel["shadow_available"] is True
    assert tel["catalog_size"] == 1451
    assert tel["retrieval_top1"] == "pizza"
    assert tel["canonical_top1"] == "pizza"
    assert tel["reranked_top1"] == "pizza"  # specialist confirma pizza
    assert tel["specialist_called"] is True
    assert tel["specialist_abstain"] is False
    assert tel["specialist_confidence"] == 0.90
    assert tel["fallback"] is False
    assert tel["latency_ms"] >= 0


def test_shadow_no_muta_el_input(monkeypatch) -> None:
    shadow = RetrievalShadow(enabled=True, clf=object(), retrieval=FakeRetrieval())
    shadow._dino = object()
    shadow._processor = object()
    shadow._head = object()
    shadow._label_map = {}
    monkeypatch.setattr(shadow, "_specialist_predict", lambda image: ("pizza", 0.90))
    img = FakeImage()
    img.mode = "RGB"
    shadow.shadow_evaluate(img, "id-4", 0.31, ["fries", "rice", "pizza"])
    assert getattr(img, "mode", None) == "RGB"  # la imagen no se toca


def test_startup_config_defaults() -> None:
    assert settings.retrieval_enabled is False
    assert settings.retrieval_shadow_enabled is False
    assert settings.retrieval_top_k == 50