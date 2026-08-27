from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "food-ai-service"
    assert body["version"] == "0.1.0"


def test_health_timestamp_utc():
    response = client.get("/health")

    body = response.json()
    assert body["timestamp_utc"].endswith("+00:00") or body["timestamp_utc"].endswith("Z")