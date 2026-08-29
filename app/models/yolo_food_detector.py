"""Detector YOLO (Ultralytics) especializado en alimentos.

YOLO11n preentrenado sobre COCO, filtrado a clases de comida. Carga única en
startup (no por request). Coordenadas convertidas a x/y/width/height en
píxeles. Modelo: yolov11n.pt — ultralytics lo descarga automáticamente al
primer uso si no existe localmente.
"""

from typing import Set

from PIL import Image

from app.core.config import Settings
from app.models.base import IFoodDetector
from app.models.detection import BoundingBox, Detection, FOOD_CLASS_IDS, FOOD_CLASS_NAMES


class YoloFoodDetector(IFoodDetector):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._class_names: dict[int, str] = {}

    def load(self) -> None:
        from ultralytics import YOLO  # import tardío: peso de import solo al cargar

        self._model = YOLO(self._settings.model_path)
        self._class_names = self._model.names

    @property
    def model_version(self) -> str:
        return self._settings.model_version

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supported_classes(self) -> Set[str]:
        return set(FOOD_CLASS_NAMES.values())

    def detect(self, image: Image.Image) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("Detector no cargado: llamar a load() en el startup.")

        results = self._model.predict(
            source=image,
            conf=self._settings.confidence_threshold,
            imgsz=self._settings.image_size,
            device=self._settings.device,
            max_det=self._settings.max_detections,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id not in FOOD_CLASS_IDS:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    name=self._class_names.get(class_id, f"class-{class_id}"),
                    confidence=round(float(box.conf[0]), 4),
                    bounding_box=BoundingBox(
                        x=round(x1),
                        y=round(y1),
                        width=round(x2 - x1),
                        height=round(y2 - y1),
                    ),
                )
            )
        return detections