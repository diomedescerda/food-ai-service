from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63fcccfc0f000506018096b308cb0000000049454e44ae426082"
)


def _multipart(analysis_id="3f3f0f0f-1111-2222-3333-444444444444", data=None, filename="plato.png", content_type="image/png"):
    files = {"image": (filename, data if data is not None else PNG_1X1, content_type)}
    form = {"analysis_id": analysis_id}
    return files, form


def test_analyze_imagen_valida_responde_received():
    files, form = _multipart()

    response = client.post("/analyze", files=files, data=form)

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"] == "3f3f0f0f-1111-2222-3333-444444444444"
    assert body["status"] == "received"


def test_analyze_analysis_id_invalido_responde_400():
    files, form = _multipart(analysis_id="no-es-uuid")

    response = client.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "INVALID_ANALYSIS_ID"


def test_analyze_archivo_vacio_responde_400():
    files, form = _multipart(data=b"")

    response = client.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "EMPTY_FILE"


def test_analyze_content_type_no_permitido_responde_400():
    files, form = _multipart(content_type="application/pdf")

    response = client.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "INVALID_IMAGE"


def test_analyze_extension_no_permitida_responde_400():
    files, form = _multipart(filename="plato.exe")

    response = client.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "INVALID_IMAGE"


def test_analyze_archivo_corrupto_responde_400():
    files, form = _multipart(data=b"no soy una imagen real")

    response = client.post("/analyze", files=files, data=form)

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "CORRUPT_FILE"


def test_analyze_sin_analysis_id_responde_422():
    files, _ = _multipart()

    response = client.post("/analyze", files=files)

    assert response.status_code == 422