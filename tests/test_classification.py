"""Tests de la capa de clasificación.

- Con fakes: DetectorBasedClassifier devuelve la clase/confianza del detector,
  sin inferencia adicional (integración con detection y segmentation).
- Con modelo real: clasificación de pizza/banana/apple coherente con las
  clases soportadas y confidence válida.
"""

from pathlib import Path

import pytest
from PIL import Image

from app.models.detector_based_classifier import DetectorBasedClassifier
from app.models.detection import BoundingBox, Detection

from .fakes import FakeFoodDetector


def _image():
    buf = Image.new("RGB", (64, 64), (100, 100, 100))
    return buf


def test_classifier_delega_en_detector_sin_inferencia_extra():
    detector = FakeFoodDetector(
        detections=[Detection("pizza", 0.9253, BoundingBox(7, 14, 318, 216))]
    )
    detector.load()
    classifier = DetectorBasedClassifier(detector)

    results = classifier.classify(_image(), detector.detect(_image()))

    assert len(results) == 1
    assert results[0].name == "pizza"
    assert results[0].confidence == 0.9253
    assert classifier.model_version == "detector-based-v1"
    assert classifier.is_loaded
    assert detector.detect_calls == 1  # classify no añade pasadas al detector


def test_classifier_devuelve_none_por_deteccion_sin_datos():
    detector = FakeFoodDetector(detections=[])
    detector.load()
    classifier = DetectorBasedClassifier(detector)

    results = classifier.classify(_image(), [])

    assert results == []


DET_MODEL = "weights/yolo11n.pt"

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("ultralytics", reason="ultralytics no instalado")
    or not Path(DET_MODEL).exists(),
    reason="ultralytics o modelo no disponibles",
)


@pytest.mark.parametrize(
    ("asset", "expected"),
    [
        ("tests/assets/pizza.jpg", "pizza"),
        ("tests/assets/banana.jpg", "banana"),
        ("tests/assets/apple.jpg", "apple"),
    ],
)
def test_clasificacion_real_coherente_con_deteccion(asset, expected):
    from app.core.config import Settings
    from app.models.yolo_food_detector import YoloFoodDetector

    path = Path(asset)
    if not path.exists():
        pytest.skip(f"asset {asset} no disponible")

    detector = YoloFoodDetector(Settings(model_path=DET_MODEL, confidence_threshold=0.35))
    detector.load()
    classifier = DetectorBasedClassifier(detector)

    with Image.open(path) as img:
        detections = detector.detect(img.convert("RGB"))
        results = classifier.classify(img.convert("RGB"), detections)

    assert len(detections) >= 1
    matches = [r for r in results if r is not None and r.name == expected]
    assert len(matches) >= 1, f"{asset}: {expected} no clasificado (resultados={results})"
    assert 0.0 < matches[0].confidence <= 1.0