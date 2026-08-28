"""Tests del estimador de profundidad y del estimador de porción avanzado.

Depth: dimensiones, valores válidos, sin NaN/inf, carga única.
Geometría: máscara + depth compatibles, máscara vacía, contraste.
Advanced: sin escala física → gramos nulos (nunca masa inventada).
"""

import numpy as np
import pytest
from PIL import Image

from app.models.advanced_portion_estimator import AdvancedPortionEstimator
from app.models.depth_base import DepthMap, IDepthEstimator
from app.models.detection import BoundingBox, Detection
from app.models.segmenter_base import SegmentationResult
from app.services.portion_geometry import PortionGeometryEstimator, depth_from_mask_b64

IMAGE_100 = Image.new("RGB", (100, 100), (0, 0, 0))


class FakeDepthEstimator(IDepthEstimator):
    """Depth determinista: gradiente lineal (izquierda cerca, derecha lejos)."""

    def __init__(self, loaded: bool = True):
        self._loaded = loaded
        x = np.linspace(1.0, 0.0, 100, dtype="float32")
        self._gradient = np.tile(x, (100, 1))

    def load(self) -> None:
        self._loaded = True

    def estimate(self, image: Image.Image) -> DepthMap:
        h, w = image.size[1], image.size[0]
        x = np.linspace(1.0, 0.0, w, dtype="float32")
        return DepthMap(data=np.tile(x, (h, 1)), width=w, height=h)

    @property
    def model_version(self) -> str:
        return "fake-depth-v1"

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _b64_mask(size: tuple[int, int], value: int = 255) -> str:
    import base64
    from io import BytesIO

    buf = BytesIO()
    Image.new("L", size, value).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_depth_map_stats_validas():
    depth = FakeDepthEstimator().estimate(IMAGE_100)
    stats = depth.stats()
    assert depth.data.shape == (100, 100)
    assert not stats["has_nan"] and not stats["has_inf"]
    assert 0.0 <= stats["min"] <= stats["max"] <= 1.0
    assert stats["mean"] > 0.0


def test_depth_estimator_no_cargado_avanza_a_unknown():
    estimator = AdvancedPortionEstimator(FakeDepthEstimator(loaded=False))
    detections = [Detection("banana", 0.9, BoundingBox(10, 10, 50, 50))]
    estimates = estimator.estimate(IMAGE_100, detections, [None])
    assert estimates == [None]


def test_geometry_stats_dentro_de_mascara():
    depth = FakeDepthEstimator().estimate(IMAGE_100)
    mask = depth_from_mask_b64(_b64_mask((100, 100)), 100, 100)
    geometry = PortionGeometryEstimator().compute(depth, mask, 100 * 100)

    assert geometry is not None
    assert geometry.depth_min <= geometry.depth_p25 <= geometry.depth_median <= geometry.depth_p75 <= geometry.depth_max
    assert geometry.depth_mean > 0.0
    assert geometry.depth_median > 0.0
    assert geometry.mask_area_pixels == 100 * 100


def test_geometry_mascara_vacia_devuelve_none():
    depth = FakeDepthEstimator().estimate(IMAGE_100)
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    geometry = PortionGeometryEstimator().compute(depth, empty_mask, 100 * 100)
    assert geometry is None


def test_geometry_sin_nan_ni_inf():
    depth = FakeDepthEstimator().estimate(IMAGE_100)
    mask = np.ones((100, 100), dtype=np.uint8)
    geometry = PortionGeometryEstimator().compute(depth, mask, 100 * 100)
    assert geometry is not None
    for value in (geometry.depth_min, geometry.depth_max, geometry.depth_mean,
                  geometry.depth_median, geometry.depth_p25, geometry.depth_p75):
        assert np.isfinite(value)


def test_advanced_sin_escala_no_inventa_gramos():
    """Conclusión experimental de la fase: depth relativa sin escala física
    → gramos NULOS (method advanced_depth_relative, confidence 0)."""
    depth = FakeDepthEstimator()
    estimator = AdvancedPortionEstimator(depth)
    detections = [Detection("banana", 0.9, BoundingBox(10, 10, 50, 50))]
    segmentations = [SegmentationResult(mask=_b64_mask((50, 50)), area_pixels=2500)]

    estimates = estimator.estimate(IMAGE_100, detections, segmentations)

    assert estimates[0] is not None
    assert estimates[0].estimated_grams is None
    assert estimates[0].min_grams is None
    assert estimates[0].max_grams is None
    assert estimates[0].confidence == 0.0
    assert estimates[0].method == "advanced_depth_relative"


def test_advanced_sin_mascara_devuelve_relativo_sin_gramos():
    depth = FakeDepthEstimator()
    estimator = AdvancedPortionEstimator(depth)
    detections = [Detection("banana", 0.9, BoundingBox(10, 10, 50, 50))]

    estimates = estimator.estimate(IMAGE_100, detections, [None])

    assert estimates[0] is not None
    assert estimates[0].estimated_grams is None
    assert estimates[0].method == "advanced_depth_relative"


def test_geometry_contraste_alimento_vs_fondo():
    """El alimento (izquierda, cerca) debe tener mediana de profundidad mayor
    que el fondo (derecha, lejos) con el gradiente del fake."""
    depth = FakeDepthEstimator().estimate(IMAGE_100)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :50] = 1  # mitad izquierda = alimento
    geometry = PortionGeometryEstimator().compute(depth, mask, 100 * 100)

    assert geometry is not None
    assert geometry.depth_contrast > 0.0  # alimento más cercano que fondo


def test_mask_b64_corrupta_devuelve_none():
    mask = depth_from_mask_b64("no-valid-b64", 100, 100)
    assert mask is None