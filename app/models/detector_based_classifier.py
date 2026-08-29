"""Clasificador basado en el detector — SIN inferencia adicional.

YOLO11n es un detector+clasificador integrado: ya asigna clase y confidence a
cada caja. Esta implementación simplemente expone ese resultado a través de
IFoodClassifier para que el pipeline hable contra interfaces.

Cuando exista un clasificador dedicado (fine-tuning colombiano), se implementa
IFoodClassifier y se reemplaza aquí sin tocar analyze.py.
"""

from typing import Sequence

from PIL import Image

from app.models.base import IFoodDetector
from app.models.classifier_base import ClassificationResult, IFoodClassifier
from app.models.detection import Detection

VERSION = "detector-based-v1"


class DetectorBasedClassifier(IFoodClassifier):
    def __init__(self, detector: IFoodDetector):
        self._detector = detector

    def classify(
        self, image: Image.Image, detections: Sequence[Detection]
    ) -> list[ClassificationResult | None]:
        # La clase/confianza ya vienen del detector (YOLO). Sin inferencia extra.
        return [
            ClassificationResult(name=d.name, confidence=d.confidence) for d in detections
        ]

    @property
    def model_version(self) -> str:
        return VERSION

    @property
    def is_loaded(self) -> bool:
        return self._detector.is_loaded