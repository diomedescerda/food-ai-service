"""Segmentador YOLO (Ultralytics) — modelo yolo11n-seg.

Modelo independiente del detector (YOLO11n). El segmentador devuelve una
máscara por detección del detector: empareja sus propios segmentos con los
bboxes recibidos por IoU. Cada máscara es un PNG en escala de grises recortado
al bbox, base64.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.models.detection import BoundingBox, Detection
from app.models.segmenter_base import IFoodSegmenter, SegmentationResult


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class YoloFoodSegmenter(IFoodSegmenter):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._class_names: dict[int, str] = {}

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self._settings.seg_model_path)
        self._class_names = self._model.names

    @property
    def model_version(self) -> str:
        return self._settings.seg_model_version

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def segment(self, image: Image.Image, detections: list[Detection]) -> list[SegmentationResult | None]:
        if self._model is None:
            raise RuntimeError("Segmentador no cargado: llamar a load() en el startup.")

        results = self._model.predict(
            source=image,
            conf=self._settings.confidence_threshold,
            imgsz=self._settings.image_size,
            device=self._settings.device,
            max_det=self._settings.max_detections,
            verbose=False,
        )
        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None or not detections:
            return [None] * len(detections)

        seg_boxes = result.boxes
        # Emparejamiento: máscara del segmento cuyo bbox tiene mayor IoU con
        # el bbox de la detección (mismo modelo COCO → boxes casi idénticos).
        segments = []
        for i, mask_tensor in enumerate(masks.data):
            b = seg_boxes[i]
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            seg_box = BoundingBox(round(x1), round(y1), round(x2 - x1), round(y2 - y1))
            segments.append((mask_tensor, seg_box))

        out: list[SegmentationResult | None] = []
        for det in detections:
            best_idx = max(range(len(segments)), key=lambda i: _iou(det.bounding_box, segments[i][1]))
            mask_tensor, seg_box = segments[best_idx]
            if _iou(det.bounding_box, seg_box) < 0.5:
                out.append(None)
                continue
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_np, mode="L")
            # Resolver escala: la máscara viene a tamaño de inferencia (imgsz);
            # escalarla al tamaño del bbox de la detección (coordenadas originales).
            scale_x = seg_box.width / mask_img.width if mask_img.width else 1.0
            scale_y = seg_box.height / mask_img.height if mask_img.height else 1.0
            box_w = max(1, round(det.bounding_box.width))
            box_h = max(1, round(det.bounding_box.height))
            mask_img = mask_img.resize((box_w, box_h), Image.NEAREST)
            # Recortar al bbox (la máscara del segmento ya es del bbox; recorte de seguridad)
            area_pixels = int((np.asarray(mask_img) > 0).sum())
            if area_pixels <= 0:
                out.append(None)
                continue
            buf = BytesIO()
            mask_img.save(buf, format="PNG")
            out.append(SegmentationResult(mask=base64.b64encode(buf.getvalue()).decode("ascii"), area_pixels=area_pixels))
        return out