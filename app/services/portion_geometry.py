"""Geometría relativa de porción: combina máscara + profundidad.

NO convierte a gramos: sin escala física (referencia conocida en la imagen o
calibración) el volumen relativo no puede convertirse a masa. Produce
características geométricas para el estimador de porción avanzado.
"""

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from app.models.depth_base import DepthMap


@dataclass(frozen=True)
class GeometryFeatures:
    mask_area_pixels: int
    mask_ratio: float  # mask_area / image_area
    depth_min: float
    depth_max: float
    depth_mean: float
    depth_median: float
    depth_p25: float
    depth_p75: float
    depth_contrast: float  # median(mask) - median(fondo); > 0 = alimento más cercano


def _mask_from_b64(mask_b64: str, target_size: tuple[int, int]) -> np.ndarray:
    mask_img = Image.open(BytesIO(base64.b64decode(mask_b64))).convert("L")
    mask_img = mask_img.resize(target_size, Image.NEAREST)
    return (np.asarray(mask_img) > 0).astype(np.uint8)


def _mask_from_pil(mask_img: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    mask_img = mask_img.resize(target_size, Image.NEAREST)
    return (np.asarray(mask_img.convert("L")) > 0).astype(np.uint8)


class PortionGeometryEstimator:
    """Estadísticas de profundidad dentro de la máscara (robustas: mediana y
    percentiles; min/max como rango). None si la máscara está vacía."""

    def compute(
        self,
        depth: DepthMap,
        mask: np.ndarray | None,
        image_area: int,
    ) -> GeometryFeatures | None:
        if mask is None or mask.sum() == 0:
            return None

        depth_data = depth.data
        if mask.shape != depth_data.shape:
            return None

        mask_bool = mask.astype(bool)
        inside = depth_data[mask_bool]
        outside = depth_data[~mask_bool]

        if inside.size == 0:
            return None

        return GeometryFeatures(
            mask_area_pixels=int(mask_bool.sum()),
            mask_ratio=int(mask_bool.sum()) / image_area if image_area > 0 else 0.0,
            depth_min=float(inside.min()),
            depth_max=float(inside.max()),
            depth_mean=float(inside.mean()),
            depth_median=float(np.median(inside)),
            depth_p25=float(np.percentile(inside, 25)),
            depth_p75=float(np.percentile(inside, 75)),
            depth_contrast=float(np.median(inside) - np.median(outside)) if outside.size else 0.0,
        )


def depth_from_mask_b64(mask_b64: str, width: int, height: int) -> np.ndarray | None:
    """Máscara PNG (b64, recortada al bbox) como array full-image de tamaño (h, w).
    None si no decodifica o está vacía."""
    try:
        mask_img = Image.open(BytesIO(base64.b64decode(mask_b64))).convert("L")
    except Exception:
        return None
    mask_img = mask_img.resize((width, height), Image.NEAREST)
    return (np.asarray(mask_img) > 0).astype(np.uint8)