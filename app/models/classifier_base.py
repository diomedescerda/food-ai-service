"""Abstracción del clasificador de alimentos.

Diferencia clave con la detección:
- Detection: ¿dónde hay un objeto? ¿qué clase parece? ¿confianza?
- Classification: ¿qué alimento es EXACTAMENTE este objeto?

La implementación actual (DetectorBasedClassifier) delega en el detector porque
YOLO ya integra clasificación por detección. Esta interfaz permite sustituirla
por un clasificador dedicado (p. ej. ColombianFoodClassifier con fine-tuning)
sin tocar el pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from app.models.detection import Detection


@dataclass(frozen=True)
class ClassificationResult:
    """Identificación final del alimento para una detección."""

    name: str
    confidence: float


class IFoodClassifier(ABC):
    """Clasificador: imagen + detecciones → identificación final por objeto."""

    @abstractmethod
    def classify(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[ClassificationResult | None]:
        """Identificación por detección (mismo orden). None = sin clasificar."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Versión del clasificador (se reporta en cada análisis)."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True si el clasificador está listo."""