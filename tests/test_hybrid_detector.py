"""Tests del detector híbrido (FASE 11-13): sin doble contabilización.

- YOLO detecta → DINO NO se ejecuta (una sola fuente de regiones).
- YOLO vacío → DINO fallback.
"""

from PIL import Image

from app.models.base import IFoodDetector
from app.models.detection import BoundingBox, Detection
from app.models.hybrid_detector import HybridFoodDetector

IMAGE = Image.new("RGB", (200, 200), (0, 0, 0))
DETECTION = Detection("pizza", 0.9, BoundingBox(10, 10, 100, 100))


class FakeYolo(IFoodDetector):
    def __init__(self, detections: list[Detection]):
        self._detections = detections
        self.calls = 0

    def load(self) -> None:
        pass

    def detect(self, image: Image.Image) -> list[Detection]:
        self.calls += 1
        return list(self._detections)

    @property
    def model_version(self) -> str:
        return "fake-yolo"

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def supported_classes(self) -> set:
        return {"pizza"}


class FakeDino(IFoodDetector):
    def __init__(self):
        self.calls = 0

    def load(self) -> None:
        pass

    def detect(self, image: Image.Image) -> list[Detection]:
        self.calls += 1
        return [Detection("unknown", 0.3, BoundingBox(0, 0, 50, 50))]

    @property
    def model_version(self) -> str:
        return "fake-dino"

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def supported_classes(self) -> set:
        return {"*"}


def test_yolo_detecta_dino_no_se_ejecuta():
    yolo = FakeYolo([DETECTION])
    dino = FakeDino()
    hybrid = HybridFoodDetector(yolo, dino)

    detections = hybrid.detect(IMAGE)

    assert detections == [DETECTION]
    assert yolo.calls == 1
    assert dino.calls == 0  # sin doble contabilización


def test_yolo_vacio_dino_fallback():
    yolo = FakeYolo([])
    dino = FakeDino()
    hybrid = HybridFoodDetector(yolo, dino)

    detections = hybrid.detect(IMAGE)

    assert len(detections) == 1
    assert detections[0].name == "unknown"
    assert yolo.calls == 1
    assert dino.calls == 1