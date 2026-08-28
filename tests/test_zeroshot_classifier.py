"""Tests del clasificador zero-shot: catálogo, prompts, threshold, unknown,
top-k, aliases y normalización — con scorer fake (sin descargar CLIP).

Los tests con el modelo real viven en test_zeroshot_real.py (skip si no hay
modelo en cache).
"""

from PIL import Image

from app.models.classifier_base import ClassificationResult
from app.models.detection import BoundingBox, Detection
from app.models.food_catalog import (
    CATALOG_BY_NAME,
    FOOD_CATALOG,
    all_clip_candidates,
    candidate_to_canonical,
)
from app.models.zero_shot_classifier import ZeroShotFoodClassifier

IMAGE = Image.new("RGB", (200, 200), (0, 0, 0))


class FakeScorerClassifier(ZeroShotFoodClassifier):
    """Sustituye _score_crop con un scorer determinista."""

    def __init__(self, scores: dict[str, float], top_k: int = 5):
        super().__init__("fake", threshold=0.22)
        self._scores = scores
        self._top_k = top_k
        self._candidates = list(all_clip_candidates())
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _score_crop(self, crop: Image.Image) -> list:
        ranked = sorted(self._scores.items(), key=lambda kv: -kv[1])[: self._top_k]
        from app.models.zero_shot_classifier import ClipTopCandidate

        return [ClipTopCandidate(name, score) for name, score in ranked]


def _detection() -> Detection:
    return Detection("pizza", 0.9, BoundingBox(10, 10, 100, 100))


def test_catalogo_tiene_entradas_y_candidatos():
    assert len(FOOD_CATALOG) >= 30
    names = {e.canonical_name for e in FOOD_CATALOG}
    assert {"pizza", "hamburger", "french_fries", "salad", "pancakes"} <= names
    assert all(e.canonical_name for e in FOOD_CATALOG)
    assert len(all_clip_candidates()) == len({c for e in FOOD_CATALOG for c in e.clip_candidates})


def test_catalogo_mantiene_nutrition_key_separada():
    assert CATALOG_BY_NAME["pizza"].nutrition_key == "pizza"
    assert CATALOG_BY_NAME["hamburger"].nutrition_key is None  # sin nutrición aún
    assert CATALOG_BY_NAME["banana"].nutrition_key == "banana"


def test_candidate_to_canonical():
    assert candidate_to_canonical("hamburger") == "hamburger"
    assert candidate_to_canonical("cheeseburger") == "hamburger"  # alias visual
    assert candidate_to_canonical("french fries") == "french_fries"
    assert candidate_to_canonical("no-existe") == "no-existe"


def test_prompt_generation_template():
    classifier = FakeScorerClassifier({})
    classifier._prompt_template = "a photo of {food}"
    assert classifier._prompt_template.format(food="pizza") == "a photo of pizza"


def test_threshold_unknown_si_ningun_candidato_supera():
    classifier = FakeScorerClassifier({"pizza": 0.10, "salad": 0.05})
    classifier._threshold = 0.22

    results = classifier.classify(IMAGE, [_detection()])

    assert results[0].name == "unknown"
    assert results[0].confidence == 0.0


def test_por_encima_del_threshold_devuelve_clase():
    classifier = FakeScorerClassifier({"pizza": 0.31, "salad": 0.10})
    classifier._threshold = 0.22

    results = classifier.classify(IMAGE, [_detection()])

    assert results[0].name == "pizza"
    assert results[0].confidence == 0.31


def test_top_k_devuelve_candidatos_ordenados():
    classifier = FakeScorerClassifier({"pizza": 0.31, "salad": 0.20, "rice": 0.12}, top_k=3)

    top = classifier.classify_with_candidates(IMAGE, _detection())

    assert [c.name for c in top] == ["pizza", "salad", "rice"]
    assert top[0].score >= top[1].score >= top[2].score


def test_clasificador_no_cargado_devuelve_none():
    classifier = FakeScorerClassifier({})
    classifier._loaded = False

    results = classifier.classify(IMAGE, [_detection()])

    assert results == [None]


def test_crop_respeta_bbox():
    classifier = FakeScorerClassifier({})
    classifier._crop_padding = 0.0
    detection = Detection("x", 0.9, BoundingBox(20, 20, 80, 60))

    crop = classifier._crop(IMAGE, detection)

    assert crop.size == (80, 60)


def test_crop_con_padding_expande_y_recorta():
    classifier = FakeScorerClassifier({})
    classifier._crop_padding = 0.25
    detection = Detection("x", 0.9, BoundingBox(20, 20, 80, 60))

    crop = classifier._crop(IMAGE, detection)

    # x: 20 - 80*0.25 = 0 .. 20 + 80*1.25 = 120 → 120 px; y: 5..95 → 90 px
    assert crop.size == (120, 90)


def test_crop_minimo_devuelve_none():
    classifier = FakeScorerClassifier({})
    detection = Detection("x", 0.9, BoundingBox(10, 10, 3, 3))

    assert classifier._crop(IMAGE, detection) is None


def test_confianza_redondeada_4_decimales():
    classifier = FakeScorerClassifier({"pizza": 0.34567})
    classifier._threshold = 0.2

    result = classifier.classify(IMAGE, [_detection()])[0]

    assert result is not None
    assert result.confidence == 0.3457


def test_score_por_clase_usa_max_del_candidato():
    """FASE 16: scoring por clase = max de candidatos (hamburger vs cheeseburger)."""
    classifier = FakeScorerClassifier({"hamburger": 0.30, "cheeseburger": 0.28, "salad": 0.25})
    classifier._threshold = 0.2
    classifier._score_by_class = True

    results = classifier.classify(IMAGE, [_detection()])

    assert results[0] is not None
    assert results[0].name == "hamburger"
    assert results[0].confidence == 0.30