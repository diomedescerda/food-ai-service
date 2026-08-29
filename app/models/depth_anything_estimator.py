"""Estimador de profundidad con Depth Anything V2 (Small).

Licencia Apache-2.0. La profundidad devuelta es RELATIVA (0..1, 1 = cercano),
no absoluta: sin referencia física NO se puede convertir a gramos.

El modelo se carga UNA vez en el startup (lifespan), nunca por request.
"""

import numpy as np
from PIL import Image

from app.models.depth_base import DepthMap, IDepthEstimator

VERSION = "depth-anything-v2-small"


class DepthAnythingEstimator(IDepthEstimator):
    def __init__(self, model_name: str, device: str):
        self._model_name = model_name
        self._device = device
        self._pipeline = None

    def load(self) -> None:
        from transformers import pipeline

        self._pipeline = pipeline(
            "depth-estimation",
            model=self._model_name,
            device=-1 if self._device == "cpu" else 0,
        )

    @property
    def model_version(self) -> str:
        return VERSION

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def estimate(self, image: Image.Image) -> DepthMap:
        if self._pipeline is None:
            raise RuntimeError("Estimador de profundidad no cargado: llamar a load() en el startup.")

        result = self._pipeline(image)
        predicted = result["predicted_depth"]
        data = predicted.squeeze().cpu().numpy().astype("float32")

        # Normalizar a 0..1 (1 = más cercano). La profundidad es relativa.
        finite = data[np.isfinite(data)]
        if finite.size == 0 or (finite.max() - finite.min()) < 1e-6:
            data = np.zeros_like(data)
        else:
            data = (data - finite.min()) / (finite.max() - finite.min())

        return DepthMap(data=data, width=image.width, height=image.height)