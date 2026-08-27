"""Tests del segmentador YOLO real (yolo11n-seg) con imágenes reales.

Validan: pizza/banana/apple → máscara con área > 0, dimensiones coherentes
con el bbox, asociada a la detección correcta y no igual al bbox completo.
"""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.models.yolo_food_detector import YoloFoodDetector
from app.models.yolo_food_segmenter import YoloFoodSegmenter

DET_MODEL = "weights/yolo11n.pt"
SEG_MODEL = "weights/yolo11n-seg.pt"

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("ultralytics", reason="ultralytics no instalado")
    or not Path(DET_MODEL).exists()
    or not Path(SEG_MODEL).exists(),
    reason="ultralytics o modelos no disponibles",
)


@pytest.fixture(scope="module")
def models():
    settings = Settings(
        model_path=DET_MODEL,
        seg_model_path=SEG_MODEL,
        confidence_threshold=0.35,
        image_size=640,
        device="cpu",
    )
    detector = YoloFoodDetector(settings)
    detector.load()
    segmenter = YoloFoodSegmenter(settings)
    segmenter.load()
    return detector, segmenter


ASSETS = {
    "pizza": "tests/assets/pizza.jpg",
    "banana": "tests/assets/banana.jpg",
    "apple": "tests/assets/apple.jpg",
}


@pytest.mark.parametrize("name", ["pizza", "banana", "apple"])
def test_segmentacion_real_de_alimento(models, name):
    detector, segmenter = models
    asset = Path(ASSETS[name])
    if not asset.exists():
        pytest.skip(f"asset {asset} no disponible")

    with Image.open(asset) as img:
        image = img.convert("RGB")
        detections = detector.detect(image)
        assert len(detections) >= 1, f"{name}: sin detecciones"

        segmentations = segmenter.segment(image, detections)
        for det, seg in zip(detections, segmentations):
            if seg is None:
                continue
            # Máscara decodificable PNG
            mask_img = Image.open(BytesIO(base64.b64decode(seg.mask))).convert("L")
            assert mask_img.size == (det.bounding_box.width, det.bounding_box.height), (
                f"{name}: máscara {mask_img.size} != bbox {det.bounding_box.width}x{det.bounding_box.height}"
            )
            # Área > 0
            assert seg.area_pixels > 0, f"{name}: área 0"
            # No es el bbox completo (área < área del bbox)
            box_area = det.bounding_box.width * det.bounding_box.height
            assert seg.area_pixels < box_area, f"{name}: máscara = bbox completo"
            # Fracción razonable del bbox
            assert seg.area_pixels > box_area * 0.2, f"{name}: máscara demasiado pequeña"
            assert seg.area_pixels == int((np.asarray(mask_img) > 0).sum())

        assert any(s is not None for s in segmentations), f"{name}: sin máscara válida"