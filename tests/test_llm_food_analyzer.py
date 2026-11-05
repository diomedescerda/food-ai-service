"""Contrato LLM bilingüe y fallback: HTTP simulado, sin credenciales reales."""

import base64
import json
import logging
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.models.llm_food_analyzer import LlmFoodAnalyzer
from app.models.llm_food_analyzer import LlmAnalyzerChain, ProviderCircuit
from app.schemas.analyze import AnalyzeResponse
from .conftest import make_png_image


@pytest.fixture
def anyio_backend():
    return "asyncio"


def food_payload():
    return {"is_food": True, "foods": [{
        "name_es": "Sushi de salmón", "name_en": "Salmon sushi", "nutrition_query": "salmon sushi",
        "confidence": 0.86, "portion": {"estimated_grams": 200, "min_grams": 150, "max_grams": 250, "confidence": 0.7},
        "estimated_nutrients_per_100g": {"calories": 150.0, "protein_g": 10.0, "carbs_g": 20.0, "fat_g": 3.0, "fiber_g": 1.0},
    }]}


def envelope(provider, payload):
    if provider == "anthropic":
        return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "identify_food", "input": payload}]}
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}


@pytest.fixture
def http_mock(monkeypatch):
    original = httpx.AsyncClient
    requests = []

    def install(responder):
        def handle(request):
            requests.append(request)
            return responder(request, len(requests))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
        return requests
    return install


def config(provider="anthropic", **kwargs):
    return Settings(_env_file=None, provider="llm", llm_provider=provider, llm_api_key="test-only-placeholder", **kwargs)


@pytest.mark.anyio
@pytest.mark.parametrize("provider,model", [("anthropic", "claude-sonnet-4-6"), ("openai", "gpt-4o")])
async def test_contrato_bilingue_y_redimensionado(http_mock, provider, model):
    requests = http_mock(lambda req, n: httpx.Response(200, json=envelope(provider, food_payload())))
    result = await LlmFoodAnalyzer(config(provider)).analyze(Image.new("RGB", (3200, 1600)), "id")
    food = result.foods[0]
    assert (food.name, food.name_en, food.nutrition_query) == ("Sushi de salmón", "Salmon sushi", "salmon sushi")
    assert food.segmentation is None
    assert food.portion.method == "llm_estimate"
    assert food.portion.min_grams <= food.portion.estimated_grams <= food.portion.max_grams
    assert food.estimated_nutrients_per_100g.calories == 150
    assert result.provider == "llm" and result.model_version == model
    sent = json.loads(requests[0].content)
    content = sent["messages"][0]["content"]
    encoded = content[0]["source"]["data"] if provider == "anthropic" else content[1]["image_url"]["url"].split(",", 1)[1]
    resized = Image.open(BytesIO(base64.b64decode(encoded)))
    assert resized.size == (1568, 784) and resized.format == "JPEG"
    assert requests[0].extensions["timeout"]["read"] == 30
    if provider == "anthropic":
        assert sent["tool_choice"]["name"] == "identify_food"
        assert requests[0].headers["anthropic-version"] == "2023-06-01"
    else:
        assert sent["response_format"]["json_schema"]["strict"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_reintenta_una_vez_errores_transitorios(http_mock, status):
    requests = http_mock(lambda req, n: httpx.Response(status) if n == 1 else httpx.Response(200, json=envelope("anthropic", food_payload())))
    assert (await LlmFoodAnalyzer(config()).analyze(Image.new("RGB", (1, 1)), "id")).provider == "llm"
    assert len(requests) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("status,count", [(429, 2), (500, 2), (401, 1), (400, 1)])
async def test_reintentos_acotados(http_mock, status, count):
    requests = http_mock(lambda req, n: httpx.Response(status))
    with pytest.raises(httpx.HTTPStatusError):
        await LlmFoodAnalyzer(config()).analyze(Image.new("RGB", (1, 1)), "id")
    assert len(requests) == count


@pytest.mark.anyio
async def test_timeout_no_reintenta(http_mock):
    def timeout(req, n):
        raise httpx.ReadTimeout("timeout", request=req)
    requests = http_mock(timeout)
    with pytest.raises(httpx.ReadTimeout):
        await LlmFoodAnalyzer(config()).analyze(Image.new("RGB", (1, 1)), "id")
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["json", "timeout", "schema", "range", "nutrients", "refusal", "missing_key", "429"])
def test_routing_fallback_observable(app_with_detector, http_mock, caplog, failure):
    app, detector, _, _ = app_with_detector
    app.state.settings = config()
    app.state.metrics = {}
    payload = food_payload()
    if failure == "schema":
        payload["foods"][0]["confidence"] = 1.1
    if failure == "range":
        payload["foods"][0]["portion"]["min_grams"] = 300
    if failure == "nutrients":
        payload["foods"][0]["calories"] = 123
    if failure == "missing_key":
        app.state.settings = Settings(_env_file=None, provider="llm")
    def respond(req, n):
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=req)
        if failure == "json":
            return httpx.Response(200, text="invalid json")
        if failure == "429":
            return httpx.Response(429)
        body = envelope("anthropic", payload)
        if failure == "refusal":
            body["stop_reason"] = "refusal"
        return httpx.Response(200, json=body)
    http_mock(respond)
    with caplog.at_level(logging.WARNING), TestClient(app) as client:
        response = client.post("/analyze", data={"analysis_id": "3f3f0f0f-1111-2222-3333-444444444444"}, files={"image": ("food.png", make_png_image(), "image/png")})
    assert response.status_code == 200
    assert response.json()["provider"] == "legacy"
    assert response.json()["foods"][0]["name"] == "pizza"
    assert detector.detect_calls == 1
    assert app.state.metrics["llm_calls"] == app.state.metrics["llm_fallbacks"] == 1
    assert "fallback" in caplog.text and "test-only-placeholder" not in caplog.text


@pytest.mark.parametrize("payload", [food_payload(), {"is_food": False, "foods": []}])
def test_routing_llm_no_necesita_detector(app_with_detector, http_mock, payload):
    app, detector, _, _ = app_with_detector
    app.state.settings = config()
    detector._loaded = False
    http_mock(lambda req, n: httpx.Response(200, json=envelope("anthropic", payload)))
    with TestClient(app) as client:
        response = client.post("/analyze", data={"analysis_id": "3f3f0f0f-1111-2222-3333-444444444444"}, files={"image": ("food.png", make_png_image(), "image/png")})
    assert response.status_code == 200 and response.json()["provider"] == "llm"
    assert len(response.json()["foods"]) == len(payload["foods"])
    assert detector.detect_calls == 0


def test_contrato_legacy_campos_nuevos_opcionales():
    result = AnalyzeResponse.model_validate({
        "analysis_id": "id", "status": "completed", "model_version": "legacy", "seg_model_version": "none",
        "inference_time_ms": 1, "foods": [{"name": "pizza", "confidence": 0.8, "bounding_box": {"x": 0, "y": 0, "width": 1, "height": 1}}],
    })
    assert result.provider is None
    assert result.foods[0].nutrition_query is result.foods[0].name_en is None


def test_defaults_y_secreto_no_visible():
    settings = Settings(_env_file=None)
    assert settings.provider == "legacy" and settings.llm_provider == "anthropic"
    assert settings.llm_timeout_seconds == 30 and settings.llm_retries == 1
    assert "test-only-placeholder" not in repr(config())


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["json", "refusal", "truncated"])
async def test_openai_rechaza_salida_invalida(http_mock, failure):
    body = envelope("openai", food_payload())
    if failure == "json":
        body["choices"][0]["message"]["content"] = "```invalid json```"
    elif failure == "refusal":
        body["choices"][0]["message"]["refusal"] = "rechazado"
    else:
        body["choices"][0]["finish_reason"] = "length"
    http_mock(lambda req, n: httpx.Response(200, json=body))
    with pytest.raises(ValueError):
        await LlmFoodAnalyzer(config("openai")).analyze(Image.new("RGB", (1, 1)), "id")


def test_readiness_y_metrics_exponen_proveedor(app_with_detector):
    from app.main import create_app
    app, _, _, _ = app_with_detector
    app.state.settings = config()
    with TestClient(app) as client:
        assert client.get("/health/readiness").json()["provider"] == "llm"
    metrics_app = create_app()
    metrics_app.state.settings = config()
    metrics_app.state.metrics = {"llm_calls": 3, "llm_fallbacks": 1}
    # Sin lifespan: solo prueba de contrato, sin cargar pesos reales.
    client = TestClient(metrics_app)
    metrics = client.get("/metrics").json()
    assert metrics["provider"] == "llm" and metrics["llm_calls"] == 3 and metrics["llm_fallbacks"] == 1
    assert metrics["llm_circuit_breakers"]["anthropic"]["state"] == "closed"


def test_doble_fallo_conserva_error_explicito(app_with_detector, http_mock):
    app, detector, _, _ = app_with_detector
    app.state.settings = Settings(_env_file=None, provider="llm")
    detector._loaded = False
    http_mock(lambda req, n: httpx.Response(500))
    with TestClient(app) as client:
        response = client.post("/analyze", data={"analysis_id": "3f3f0f0f-1111-2222-3333-444444444444"}, files={"image": ("food.png", make_png_image(), "image/png")})
    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "MODEL_NOT_READY"


@pytest.mark.anyio
async def test_anthropic_caido_openai_responde_y_circuito_se_recupera(http_mock):
    now = [100.0]
    def respond(req, n):
        if req.url.host == "api.anthropic.com" and now[0] < 500:
            return httpx.Response(503)
        provider = "anthropic" if req.url.host == "api.anthropic.com" else "openai"
        return httpx.Response(200, json=envelope(provider, food_payload()))
    requests = http_mock(respond)
    chain = LlmAnalyzerChain(config(llm_api_key_openai="test-only-openai", llm_circuit_failure_threshold=2, llm_retries=0), logging.getLogger(__name__))
    chain.circuits = {name: ProviderCircuit(2, 300, lambda: now[0]) for name in chain.circuits}
    for _ in range(3):
        result = await chain.analyze(Image.new("RGB", (1, 1)), "id")
        assert result.model_version == "gpt-4o" and result.provider == "llm"
    status = chain.snapshot()
    assert status["anthropic"]["state"] == "open"
    assert status["anthropic"]["calls"] == status["anthropic"]["failures"] == 2
    assert status["anthropic"]["skipped"] == 1
    assert [r.url.host for r in requests] == ["api.anthropic.com", "api.openai.com", "api.anthropic.com", "api.openai.com", "api.openai.com"]
    now[0] = 500
    assert (await chain.analyze(Image.new("RGB", (1, 1)), "id")).model_version == "claude-sonnet-4-6"
    assert chain.snapshot()["anthropic"]["state"] == "closed"
    assert chain.snapshot()["anthropic"]["consecutive_failures"] == 0


def test_circuito_solo_admite_una_prueba_y_reabre_si_falla():
    now = [100.0]
    circuit = ProviderCircuit(1, 300, lambda: now[0])
    assert circuit.allow()
    circuit.fail()
    assert not circuit.allow()
    now[0] = 401
    assert circuit.snapshot()["state"] == "half_open"
    assert circuit.allow() and not circuit.allow()
    circuit.fail()
    assert circuit.snapshot()["state"] == "open"
    assert not circuit.allow()


def test_ambos_proveedores_caidos_usan_legacy(app_with_detector, http_mock):
    app, detector, _, _ = app_with_detector
    app.state.settings = config(llm_api_key_openai="test-only-openai", llm_retries=0, llm_circuit_failure_threshold=1)
    app.state.metrics = {}
    requests = http_mock(lambda req, n: httpx.Response(503))
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post("/analyze", data={"analysis_id": "3f3f0f0f-1111-2222-3333-444444444444"}, files={"image": ("food.png", make_png_image(), "image/png")})
            assert response.status_code == 200 and response.json()["provider"] == "legacy"
        availability = client.get("/health/readiness").json()["llm_circuit_breakers"]
    assert len(requests) == 2 and detector.detect_calls == 2
    assert app.state.metrics["llm_fallbacks"] == 2
    assert all(state["state"] == "open" and state["skipped"] == 1 for state in availability.values())


@pytest.mark.anyio
@pytest.mark.parametrize("missing_or_invalid", ["missing", "negative"])
async def test_exige_macros_estimados_validos_por_alimento(http_mock, missing_or_invalid):
    payload = food_payload()
    if missing_or_invalid == "missing":
        del payload["foods"][0]["estimated_nutrients_per_100g"]
    else:
        payload["foods"][0]["estimated_nutrients_per_100g"]["calories"] = -1
    http_mock(lambda req, n: httpx.Response(200, json=envelope("anthropic", payload)))
    with pytest.raises(ValueError):
        await LlmFoodAnalyzer(config()).analyze(Image.new("RGB", (1, 1)), "id")
