"""Abstracción del estimador de porción.

Separa la estimación de porción del resto del pipeline (detector, segmentador,
clasificador, nutrición): podrá sustituirse por un estimador avanzado
(depth/volumen — FASE 7) sin tocar analyze.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from app.models.detection import Detection
from app.models.segmenter_base import SegmentationResult


@dataclass(frozen=True)
class PortionEstimate:
    """Estimación de porción. estimatedGrams != measuredGrams: es una
    aproximación basada en referencias documentadas, nunca un peso medido."""

    portion_size: str  # small | medium | large | unknown
    estimated_grams: int | None
    min_grams: int | None
    max_grams: int | None
    confidence: float  # 0..1 — separada de la confianza de detección
    method: str  # basic_reference | unavailable


class IPortionEstimator(ABC):
    """Estimador de porción: imagen + detecciones + máscaras → estimación."""

    @abstractmethod
    def estimate(
        self,
        image: Image.Image,
        detections: list[Detection],
        segmentations: list[SegmentationResult | None],
    ) -> list[PortionEstimate | None]:
        """Estimación por detección (mismo orden). None = sin estimación."""