"""Tests de integración real: YOLO detección → crop → CLIP → identidad.

Requieren el modelo CLIP en cache (openai/clip-vit-base-patch32). Si no está
disponible, se saltan (unit tests con scorer fake cubren la lógica pura).
"""

from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.models.yolo_food_detector import YoloFoodDetector
from app.models.zero_shot_classifier import ZeroShotFoodClassifier

CLIP_MODEL = "openai/clip-vit-base-patch32"

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch no instalado")
    or not Path("weights/yolo11n.pt").exists(),
    reason="torch o modelos no disponibles",
)


@pytest.fixture(scope="module")
def classifier():
    clip = ZeroShotFoodClassifier(
        model_name=CLIP_MODEL,
        device="cpu",
        threshold=0.22,
        crop_padding=0.05,
    )
    clip.load()
    return clip


def _detect(classifier, asset: str) -> tuple[Image.Image, list]:
    detector = YoloFoodDetector(Settings(model_path="weights/yolo11n.pt"))
    detector.load()
    with Image.open(asset) as img:
        image = img.convert("RGB")
        detections = detector.detect(image)
    return image, detections


@pytest.mark.parametrize(
    ("asset", "expected"),
    [
        ("tests/assets/pizza.jpg", "pizza"),
        ("tests/assets/banana.jpg", "banana"),
        ("tests/assets/apple.jpg", "apple"),
    ],
)
def test_clip_sobre_crop_detectado_por_yolo(classifier, asset, expected):
    image, detections = _detect(classifier, asset)
    if not detections:
        pytest.skip("sin detecciones de YOLO")

    results = classifier.classify(image, detections)

    assert results[0] is not None
    assert results[0].name == expected, f"CLIP no reconoció {expected}: {results[0]}"
    assert 0.0 <= results[0].confidence <= 1.0


def test_clip_candidatos_top_k(classifier):
    image, detections = _detect(classifier, "tests/assets/pizza.jpg")
    if not detections:
        pytest.skip("sin detecciones")

    top = classifier.classify_with_candidates(image, detections[0])

    assert len(top) >= 3
    assert top[0].score >= top[1].score
    assert top[0].name == "pizza"