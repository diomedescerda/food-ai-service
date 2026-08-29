"""Abstracción del segmentador de alimentos.

Separada de IFoodDetector: detección (bbox) y segmentación (máscara) son
interfaces independientes. El pipeline puede usar modelos distintos o el
mismo backend sin acoplarse.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class SegmentationResult:
    """Máscara de un alimento.

    mask: PNG en escala de grises (255 = alimento), recortado al bounding box,
          codificado en base64. El frontend la dibuja sobre el bbox.
    area_pixels: píxeles de alimento dentro de la máscara (> 0 siempre).
    """

    mask: str
    area_pixels: int


class IFoodSegmenter(ABC):
    """Segmentador de alimentos: imagen → máscaras por detección."""

    @abstractmethod
    def load(self) -> None:
        """Carga el modelo una sola vez (startup)."""

    @abstractmethod
    def segment(self, image: Image.Image, detections: list) -> list[SegmentationResult | None]:
        """Devuelve una máscara por detección (en el orden de `detections`).

        El segmentador decide el emparejamiento máscara↔detección (p. ej. por
        IoU del bbox). None = sin máscara para esa detección.
        """

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Versión del modelo de segmentación."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True si el modelo está cargado."""