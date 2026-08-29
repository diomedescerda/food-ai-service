"""Abstracción del estimador de profundidad.

La profundidad monocular es RELATIVA (0..1), no una medición física absoluta:
un mapa de profundidad NO puede convertirse directamente a gramos sin escala.
analyze.py habla contra IDepthEstimator, nunca contra el modelo concreto.
"""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class DepthMap:
    """Mapa de profundidad relativa normalizado 0..1 (1 = más cercano)."""

    def __init__(self, data: np.ndarray, width: int, height: int):
        self.data = data  # float32, shape (height, width), rango 0..1
        self.width = width
        self.height = height

    def stats(self) -> dict:
        """Estadísticas básicas del mapa (validación: no NaN/inf)."""
        finite = self.data[np.isfinite(self.data)]
        return {
            "min": float(finite.min()) if finite.size else 0.0,
            "max": float(finite.max()) if finite.size else 0.0,
            "mean": float(finite.mean()) if finite.size else 0.0,
            "median": float(np.median(finite)) if finite.size else 0.0,
            "has_nan": bool(np.isnan(self.data).any()),
            "has_inf": bool(np.isinf(self.data).any()),
        }


class IDepthEstimator(ABC):
    """Estimador de profundidad: imagen → mapa de profundidad relativa."""

    @abstractmethod
    def estimate(self, image: Image.Image) -> DepthMap:
        """Devuelve el mapa de profundidad (relativa 0..1)."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Versión del modelo de profundidad."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True si el modelo está cargado."""