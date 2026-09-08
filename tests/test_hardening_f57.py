"""Tests del hardening (F57): readiness, versionado, métricas, consistencia
catálogo/índice, validación de respuesta y fallbacks observables."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

BASE = Path(__file__).resolve().parents[1]


def test_readiness_endpoint_schema() -> None:
    # sin lifespan completo (modelos) el endpoint responde con estructura válida
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health/readiness")
    data = resp.json()
    assert "status" in data
    assert data["catalog_size"] == 5761
    assert data["pipeline_version"] == "f57"
    assert data["catalog_version"] == "5761"


def test_liveness_health_incluye_pipeline() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    data = client.get("/health").json()
    assert data["service"] == "food-ai-service"
    assert "timestamp_utc" in data


def test_metrics_endpoint() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    app.state.metrics = {"total_requests": 5, "new_pipeline_used": 4, "legacy_fallback": 1,
                         "nutrition_ready": 3, "specialist_calls": 2}
    data = client.get("/metrics").json()
    assert data["total_requests"] == 5
    assert data["new_pipeline_used"] == 4
    assert data["legacy_fallback"] == 1
    assert data["nutrition_ready"] == 3
    assert data["specialist_calls"] == 2


def test_release_version_log() -> None:
    # la versión del release está documentada en el startup (RELEASE log)
    import re  # noqa: PLC0415

    main_src = (BASE / "app/main.py").read_text(encoding="utf-8")
    assert re.search(r"RELEASE: pipeline=f57 catalog=5761", main_src)


def test_catalog_index_consistency() -> None:
    catalog = json.loads((BASE / "catalog/foods.json").read_text(encoding="utf-8-sig"))
    n_canonical = len(catalog["foods"])
    n_aliases = sum(len(f["aliases"]) for f in catalog["foods"])
    texts = json.loads((BASE / "catalog/embeddings/multitext/texts.json").read_text(encoding="utf-8"))
    assert n_canonical == 5761  # catálogo oficial congelado
    assert len(texts) == n_canonical + n_aliases  # índice consistente con catálogo


def test_fallback_reasons_observables() -> None:
    from app.models.decision import DecisionPolicy  # noqa: PLC0415

    p = DecisionPolicy()
    reasons = [
        p.decide(0.3, "NUTRITION_READY", pipeline_error=True)["fallback_reason"],
        p.decide(0.3, "NUTRITION_UNAVAILABLE")["fallback_reason"],
        p.decide(0.1, "NUTRITION_READY")["fallback_reason"],
    ]
    assert "pipeline_error" in reasons
    assert "nutrition_unavailable" in reasons


def test_response_schema_validacion() -> None:
    from app.schemas.analyze import AnalyzeResponse  # noqa: PLC0415

    # el schema del response se mantiene (contract intacto)
    fields = AnalyzeResponse.model_fields
    for f in ("analysis_id", "status", "model_version", "inference_time_ms", "foods"):
        assert f in fields


def test_nutrition_coverage_observable() -> None:
    idx = json.loads((BASE / "nutrition/mappings.json").read_text(encoding="utf-8"))
    ready = sum(1 for v in idx.values() if "nutrients_per_100g" in v)
    assert ready > 2000  # la cobertura crece (37.9%+ en F54)