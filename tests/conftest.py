"""Contrato de detección/segmentación: fixture de imagen y app con fakes."""

import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeFoodDetector, FakeFoodSegmenter  # noqa: E402

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63fcccfc0f000506018096b308cb0000000049454e44ae426082"
)


def make_png_image(width: int = 64, height: int = 64) -> bytes:
    """Imagen PNG sintética (patrón) — válida para decodificar, sin contenido."""
    buf = BytesIO()
    Image.new("RGB", (width, height), (200, 120, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def app_with_detector():
    """App FastAPI con detector + segmentador + clasificador fake (sin YOLO)."""
    from fastapi import FastAPI

    from app.api.analyze import router as analyze_router
    from app.api.health import router as health_router
    from app.core.config import Settings
    from app.models.basic_portion_estimator import BasicPortionEstimator
    from app.models.detector_based_classifier import DetectorBasedClassifier

    detector = FakeFoodDetector()
    detector.load()
    segmenter = FakeFoodSegmenter()
    segmenter.load()
    classifier = DetectorBasedClassifier(detector)

    app = FastAPI()
    app.state.detector = detector
    app.state.segmenter = segmenter
    app.state.classifier = classifier
    app.state.portion_estimator = BasicPortionEstimator()
    app.state.settings = Settings(debug_images_dir="")
    app.state.logger = None
    app.include_router(health_router)
    app.include_router(analyze_router)
    return app, detector, segmenter, classifier


@pytest.fixture
def client(app_with_detector):
    from fastapi.testclient import TestClient

    app, detector, segmenter, classifier = app_with_detector
    with TestClient(app) as test_client:
        yield test_client, detector, segmenter, classifier