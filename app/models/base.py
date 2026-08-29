"""Abstracción del detector de alimentos.

El resto del pipeline (analyze.py, tests) depende de IFoodDetector, nunca de
un modelo concreto. Cambiar YOLO por otro detector = nueva implementación,
sin tocar el endpoint.
"""

from abc import ABC, abstractmethod
from typing import Set

from PIL import Image

from app.models.detection import Detection


class IFoodDetector(ABC):
    """Detector de alimentos: imagen → detecciones con clase, confianza y bbox."""

    @abstractmethod
    def load(self) -> None:
        """Carga el modelo una sola vez (ciclo de vida: startup)."""

    @abstractmethod
    def detect(self, image: Image.Image) -> list[Detection]:
        """Ejecuta la inferencia sobre la imagen y devuelve detecciones."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Versión del modelo (para reportar en cada análisis)."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True si el modelo está cargado y listo."""

    @property
    @abstractmethod
    def supported_classes(self) -> Set[str]:
        """Clases de comida que el modelo puede detectar (nombres honestos)."""