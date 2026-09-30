"""Detectores de regiones de comida (FASE 11).

- GroundingDinoDetector: open-vocabulary (prompt genérico "food on a plate").
  Genera regiones para alimentos que YOLO-COCO no conoce. Lento en CPU
  (~10 s/imagen) → se usa como FALLBACK del híbrido, no en el camino rápido.
- HybridFoodDetector: YOLO (rápido) → si no detecta nada → DINO.
"""

import torch
from PIL import Image

from app.models.base import IFoodDetector
from app.models.detection import BoundingBox, Detection

UNKNOWN_CLASS = "unknown"


class GroundingDinoDetector(IFoodDetector):
    def __init__(
        self, model_name: str, prompt: str = "food on a plate", threshold: float = 0.15
    ):
        self._model_name = model_name
        self._prompt = prompt
        self._threshold = threshold
        # Umbral de coincidencia de tokens del prompt (post-proceso DINO).
        self._text_threshold = 0.25
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_name
        )
        self._processor = AutoProcessor.from_pretrained(self._model_name)

    @property
    def model_version(self) -> str:
        return "grounding-dino-tiny"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supported_classes(self) -> set:
        # Open-vocabulary: cualquier clase; la identidad la decide CLIP.
        return {"*"}

    def detect(self, image: Image.Image) -> list[Detection]:
        if self._model is None:
            raise RuntimeError(
                "Detector DINO no cargado: llamar a load() en el startup."
            )

        inputs = self._processor(images=image, text=self._prompt, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**inputs)
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        # transformers >= 4.45 exige input_ids y text_threshold de forma
        # explícita (antes el prompt era string opcional y se usaba el default
        # 0.3): pasarlos siempre mantiene la llamada estable entre versiones.
        results = self._processor.post_process_grounded_object_detection(
            out,
            inputs.input_ids,
            threshold=self._threshold,
            text_threshold=self._text_threshold,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for box, score in zip(results["boxes"].tolist(), results["scores"].tolist()):
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    name=UNKNOWN_CLASS,
                    confidence=round(float(score), 4),
                    bounding_box=BoundingBox(
                        x=int(x1),
                        y=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                    ),
                )
            )
        return detections


class HybridFoodDetector(IFoodDetector):
    """YOLO (rápido, clasifica) → si 0 detecciones → DINO (open-vocabulary).
    Las regiones de DINO llevan name=unknown: la identidad la decide CLIP."""

    def __init__(self, yolo: IFoodDetector, dino: IFoodDetector):
        self._yolo = yolo
        self._dino = dino
        self._used_dino_fallback = False

    @property
    def used_dino_fallback(self) -> bool:
        """FASE 17: telemetría — True si la última llamada a detect() usó DINO."""
        return self._used_dino_fallback

    def load(self) -> None:
        self._yolo.load()
        self._dino.load()

    @property
    def model_version(self) -> str:
        return f"{self._yolo.model_version}+{self._dino.model_version}"

    @property
    def is_loaded(self) -> bool:
        return self._yolo.is_loaded and self._dino.is_loaded

    @property
    def supported_classes(self) -> set:
        return self._yolo.supported_classes | {"*"}

    def detect(self, image: Image.Image) -> list[Detection]:
        detections = self._yolo.detect(image)
        if detections:
            self._used_dino_fallback = False
            return detections
        # Fallback open-vocabulary solo cuando YOLO no encontró nada.
        self._used_dino_fallback = True
        regions = self._dino.detect(image)
        return nms_regions(regions)


def nms_regions(
    detections: list[Detection], iou_threshold: float = 0.5
) -> list[Detection]:
    """NMS simple por IoU (FASE 15): elimina regiones muy solapadas de DINO
    (el mismo alimento detectado en regiones casi idénticas → doble conteo)."""
    if len(detections) <= 1:
        return detections

    def _iou(a: Detection, b: Detection) -> float:
        x1 = max(a.bounding_box.x, b.bounding_box.x)
        y1 = max(a.bounding_box.y, b.bounding_box.y)
        x2 = min(
            a.bounding_box.x + a.bounding_box.width,
            b.bounding_box.x + b.bounding_box.width,
        )
        y2 = min(
            a.bounding_box.y + a.bounding_box.height,
            b.bounding_box.y + b.bounding_box.height,
        )
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (
            a.bounding_box.width * a.bounding_box.height
            + b.bounding_box.width * b.bounding_box.height
            - inter
        )
        return inter / union if union > 0 else 0.0

    ordered = sorted(detections, key=lambda d: -d.confidence)
    kept: list[Detection] = []
    for detection in ordered:
        if all(_iou(detection, other) < iou_threshold for other in kept):
            kept.append(detection)
    return kept
