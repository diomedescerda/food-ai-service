"""Tests del contrato /analyze con detector fake (sin modelo real).

Cubren: imagen válida, sin comida, múltiples detecciones, corrupta, demasiado
grande, MIME inválido y estado del modelo en /health.
"""

from pathlib import Path

from app.models.detection import BoundingBox, Detection

from .conftest import make_png_image
from .fakes import FakeFoodDetector


def _multipart(data, filename="plato.png", content_type="image/png", analysis_id="3f3f0f0f-1111-2222-3333-444444444444"):
    return {"image": (filename, data, content_type)}, {"analysis_id": analysis_id}


def test_analyze_imagen_valida_devuelve_detecciones_y_mascara(client):
    c, detector, segmenter, classifier = client
    files, form = _multipart(make_png_image())

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["model_version"] == detector.model_version
    assert body["seg_model_version"] == segmenter.model_version
    assert body["classifier_version"] == "detector-based-v1"
    assert isinstance(body["inference_time_ms"], int)
    assert len(body["foods"]) == 1
    food = body["foods"][0]
    assert food["name"] == "pizza"
    assert 0.0 <= food["confidence"] <= 1.0
    box = food["bounding_box"]
    assert box["x"] == 120 and box["y"] == 80
    assert box["width"] == 300 and box["height"] == 180
    assert food["segmentation"] is not None
    assert food["segmentation"]["area_pixels"] > 0
    assert food["segmentation"]["mask"]


def test_analyze_sin_segmentador_mascara_nula(client):
    c, _, segmenter, _ = client
    segmenter._loaded = False
    files, form = _multipart(make_png_image())

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 200
    body = response.json()
    assert body["seg_model_version"] == "none"
    assert body["foods"][0]["segmentation"] is None


def test_analyze_imagen_sin_comida_devuelve_foods_vacio(client):
    c, detector, _, _ = client
    detector._detections = []
    files, form = _multipart(make_png_image())

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["foods"] == []


def test_analyze_multiples_detecciones(client):
    c, detector, _, _ = client
    detector._detections = [
        Detection("rice", 0.94, BoundingBox(120, 80, 300, 180)),
        Detection("chicken", 0.91, BoundingBox(450, 120, 180, 220)),
        Detection("banana", 0.61, BoundingBox(30, 400, 90, 140)),
    ]
    files, form = _multipart(make_png_image())

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 200
    body = response.json()
    assert [f["name"] for f in body["foods"]] == ["rice", "chicken", "banana"]
    assert body["foods"][1]["bounding_box"]["x"] == 450


def test_analyze_archivo_corrupto_responde_400(client):
    c, _, _, _ = client
    files, form = _multipart(b"no soy una imagen")

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "CORRUPT_FILE"


def test_analyze_archivo_vacio_responde_400(client):
    c, _, _, _ = client
    files, form = _multipart(b"")

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "EMPTY_FILE"


def test_analyze_archivo_demasiado_grande_responde_400(client):
    c, _, _, _ = client
    files, form = _multipart(b"x" * (10 * 1024 * 1024 + 1))

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "IMAGE_TOO_LARGE"


def test_analyze_mime_invalido_responde_400(client):
    c, _, _, _ = client
    files, form = _multipart(make_png_image(), content_type="application/pdf")

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "INVALID_IMAGE"


def test_analyze_extension_invalida_responde_400(client):
    c, _, _, _ = client
    files, form = _multipart(make_png_image(), filename="plato.exe")

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "INVALID_IMAGE"


def test_analyze_analysis_id_invalido_responde_400(client):
    c, _, _, _ = client
    files, form = _multipart(make_png_image(), analysis_id="no-uuid")

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "INVALID_ANALYSIS_ID"


def test_analyze_detector_no_cargado_responde_503(client):
    c, detector, _, _ = client
    detector._loaded = False
    files, form = _multipart(make_png_image())

    response = c.post("/analyze", files=files, data=form)

    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "MODEL_NOT_READY"


def test_health_incluye_estado_del_modelo(client):
    c, detector, segmenter, classifier = client

    response = c.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model"] == {"loaded": True, "version": detector.model_version}
    assert body["segmentation_model"] == {"loaded": True, "version": segmenter.model_version}
    assert body["classifier_model"] == {"loaded": True, "version": classifier.model_version}


def test_detector_se_carga_una_sola_vez_en_startup(app_with_detector):
    """El detector fake confirma ciclo de vida: load() en startup, detect() por request."""
    app, detector, _, _ = app_with_detector

    assert detector.is_loaded
    assert detector.detect_calls == 0


def test_debug_overlay_genera_imagen_anotada(tmp_path, client):
    from PIL import Image as PILImage

    from app.utils.debug import draw_detections, save_debug_image

    detector = FakeFoodDetector()
    detector.load()
    image = PILImage.new("RGB", (640, 480), (255, 255, 255))

    path = save_debug_image(str(tmp_path), "test-1", image, detector.detect(image))

    assert path is not None
    assert Path(path).exists()
    annotated = PILImage.open(path)
    assert annotated.size == (640, 480)
    annotated_img = draw_detections(image, [])
    assert annotated_img.size == (640, 480)