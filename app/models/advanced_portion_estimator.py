"""Estimador de porción avanzado (depth + máscara).

CONCLUSIÓN EXPERIMENTAL de esta fase: la profundidad monocular es RELATIVA.
Sin una referencia física en la imagen (plato/vaso/cubierto de tamaño conocido)
o calibración, el volumen relativo NO puede convertirse a gramos.

Por eso el resultado es method="advanced_depth_relative" con gramos nulos
(confidence 0): nunca se finge masa. La geometría relativa (contraste de
profundidad alimento vs fondo, mediana, percentiles) queda disponible como
característica para futuras fases con escala física o dataset con ground truth.
"""

from PIL import Image

from app.models.depth_base import IDepthEstimator
from app.models.detection import Detection
from app.models.portion_base import IPortionEstimator, PortionEstimate
from app.models.segmenter_base import SegmentationResult
from app.services.portion_geometry import PortionGeometryEstimator, depth_from_mask_b64

# Umbral experimental: si el alimento NO está significativamente más cerca que
# el fondo (contraste <= 0.05), no hay evidencia de profundidad utilizable.
MIN_DEPTH_CONTRAST = 0.05


class AdvancedPortionEstimator(IPortionEstimator):
    """Estimación relativa basada en profundidad. Sin escala → sin gramos."""

    def __init__(
        self,
        depth_estimator: IDepthEstimator | None,
        geometry_estimator: PortionGeometryEstimator | None = None,
    ):
        self._depth = depth_estimator
        self._geometry = geometry_estimator or PortionGeometryEstimator()

    def estimate(
        self,
        image: Image.Image,
        detections: list[Detection],
        segmentations: list[SegmentationResult | None],
    ) -> list[PortionEstimate | None]:
        if self._depth is None or not self._depth.is_loaded:
            return [None] * len(detections)

        depth = self._depth.estimate(image)
        image_area = image.width * image.height
        estimates: list[PortionEstimate | None] = []

        for index, _ in enumerate(detections):
            segmentation = segmentations[index] if index < len(segmentations) else None
            mask = None
            if segmentation is not None:
                mask = depth_from_mask_b64(segmentation.mask, image.width, image.height)

            geometry = self._geometry.compute(depth, mask, image_area)
            if geometry is None or geometry.depth_contrast < MIN_DEPTH_CONTRAST:
                # Sin máscara válida o sin señal de profundidad: sin masa inventada.
                estimates.append(PortionEstimate(
                    portion_size="unknown",
                    estimated_grams=None,
                    min_grams=None,
                    max_grams=None,
                    confidence=0.0,
                    method="advanced_depth_relative",
                ))
                continue

            # Sin escala física → solo estimación RELATIVA (nunca gramos).
            estimates.append(PortionEstimate(
                portion_size="unknown",
                estimated_grams=None,
                min_grams=None,
                max_grams=None,
                confidence=0.0,
                method="advanced_depth_relative",
            ))

        return estimates