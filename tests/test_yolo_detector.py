"""Tests del wrapper YOLO real (no fake).

Requieren ultralytics y el modelo preentrenado (yolov11n.pt se descarga
automáticamente en el primer uso). Si no hay red/modelo, se saltan.

Cubren: carga única, detecciones válidas, confidence y bboxes en píxeles,
imagen sin comida (sintética), y que no se cargue por request.
"""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.models.yolo_food_detector import YoloFoodDetector

MODEL_PATH = "weights/yolo11n.pt"

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("ultralytics", reason="ultralytics no instalado")
    or not Path(MODEL_PATH).exists(),
    reason="ultralytics o modelo no disponibles",
)


@pytest.fixture(scope="module")
def detector():
    settings = Settings(
        model_path=MODEL_PATH,
        confidence_threshold=0.35,
        image_size=640,
        device="cpu",
    )
    detector = YoloFoodDetector(settings)
    detector.load()
    return detector


def _synthetic_image(width=640, height=480):
    buf = BytesIO()
    Image.new("RGB", (width, height), (150, 150, 150)).save(buf, format="JPEG")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_modelo_se_carga_y_expone_version(detector):
    assert detector.is_loaded
    assert detector.model_version == "food-detector-v1"
    assert len(detector.supported_classes) > 0
    assert "pizza" in detector.supported_classes


def test_imagen_sintetica_no_genera_falsos_positivos(detector):
    """Imagen gris plana: sin comida detectable (0 detecciones esperadas)."""
    detections = detector.detect(_synthetic_image())

    assert detections == []


def test_detecciones_validas_formato_pixeles(detector):
    """Con el modelo real solo verificamos invariantes del contrato
    (imagen sintética → sin detecciones; el formato se valida con fake en
    test_analyze). Aquí: detect() nunca lanza y devuelve lista."""
    detections = detector.detect(_synthetic_image())

    assert isinstance(detections, list)
    for d in detections:
        assert 0.0 <= d.confidence <= 1.0
        assert d.bounding_box.width > 0
        assert d.bounding_box.height > 0
        assert d.name
        assert d.name in detector.supported_classes


def test_detector_no_se_recarga_por_request(detector):
    """El wrapper no expone recarga: load() es explícito en startup.
    Verificamos que detect() múltiple no reintancia el modelo (misma ref)."""
    model_ref = detector._model
    detector.detect(_synthetic_image())
    detector.detect(_synthetic_image())

    assert detector._model is model_ref


def test_imagen_real_de_pizza_detecta_comida(detector):
    """Smoke test de regresión con imagen real (tests/assets/pizza.jpg)."""
    asset = Path("tests/assets/pizza.jpg")
    if not asset.exists():
        pytest.skip("asset tests/assets/pizza.jpg no disponible")
    with Image.open(asset) as img:
        detections = detector.detect(img.convert("RGB"))

    pizzas = [d for d in detections if d.name == "pizza"]
    assert len(pizzas) >= 1, f"pizza no detectada en imagen real: {detections}"
    assert pizzas[0].confidence >= 0.5