"""Detector y segmentador fake para tests: sin modelos reales, deterministas."""

from typing import Set

from PIL import Image

from app.models.base import IFoodDetector
from app.models.detection import BoundingBox, Detection
from app.models.segmenter_base import IFoodSegmenter, SegmentationResult


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


class FakeFoodSegmenter(IFoodSegmenter):
    """Segmentador determinista: máscara PNG pequeña por detección."""

    def __init__(self, version: str = "fake-seg-v1", area_pixels: int = 200):
        self._version = version
        self._loaded = False
        self.area_pixels = area_pixels

    def load(self) -> None:
        self._loaded = True

    def segment(self, image: Image.Image, detections: list[Detection]) -> list[SegmentationResult | None]:
        import base64
        from io import BytesIO

        from PIL import Image as PILImage

        buf = BytesIO()
        PILImage.new("L", (8, 8), 255).save(buf, format="PNG")
        mask = base64.b64encode(buf.getvalue()).decode("ascii")
        return [SegmentationResult(mask=mask, area_pixels=self.area_pixels) for _ in detections]

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def is_loaded(self) -> bool:
        return self._loaded