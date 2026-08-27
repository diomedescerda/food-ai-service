"""Detector fake para tests: sin modelo real, respuestas deterministas."""

from typing import Set

from PIL import Image

from app.models.base import IFoodDetector
from app.models.detection import BoundingBox, Detection


class FakeFoodDetector(IFoodDetector):
    """Detector determinista: 1 detección (pizza) o las que se configuren."""

    def __init__(self, detections: list[Detection] | None = None, version: str = "fake-v1"):
        self._detections = detections if detections is not None else [
            Detection(
                name="pizza",
                confidence=0.94,
                bounding_box=BoundingBox(x=120, y=80, width=300, height=180),
            )
        ]
        self._version = version
        self._loaded = False
        self.detect_calls = 0

    def load(self) -> None:
        self._loaded = True

    def detect(self, image: Image.Image) -> list[Detection]:
        self.detect_calls += 1
        return list(self._detections)

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def supported_classes(self) -> Set[str]:
        return {d.name for d in self._detections}