"""Local E2E boundaries: real FastAPI routes; no model weights or paid network.

This module is never imported by the production entry point.
"""
import logging
import asyncio
import json

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.main import create_app
from app.models.detector_based_classifier import DetectorBasedClassifier
from tests.fakes import FakeFoodDetector, FakeFoodSegmenter
from app.models.detection import BoundingBox, Detection
from app.models.portion_base import PortionEstimate


class SyntheticPortionEstimator:
    def estimate(self, image, detections, segmentations):
        return [PortionEstimate("medium", 100, 80, 120, 0.5, "basic_reference") for _ in detections]


class SyntheticDetector(FakeFoodDetector):
    def detect(self, image):
        self.last_size = list(image.size)
        self.last_info_keys = list(image.info)
        return super().detect(image)


class LocalOnlyProviderTransport(httpx.AsyncBaseTransport):
    """Real localhost HTTP boundary: impossible to open a vendor connection."""
    async def handle_async_request(self, request):
        vendor = {"api.anthropic.com": "anthropic", "api.openai.com": "openai"}[request.url.host]
        # Parent may override when launching an independent service instance.
        import os
        port = int(os.environ.get("FOODAI_E2E_PORT", "8011"))
        url = f"http://127.0.0.1:{port}/__e2e/provider/{vendor}"
        async with httpx.AsyncClient() as client:
            result = await client.post(url, content=request.content, headers={"content-type": "application/json"})
            return httpx.Response(result.status_code, content=result.content, headers=result.headers)


def local_client_factory(**kwargs):
    return httpx.AsyncClient(transport=LocalOnlyProviderTransport(), **kwargs)


detector = SyntheticDetector([Detection("banana", 0.94, BoundingBox(0, 0, 8, 8))])
detector.load()
segmenter = FakeFoodSegmenter()
segmenter.load()
config = Settings(_env_file=None, debug_images_dir="", internal_service_key="foodai-e2e-secret")
app = create_app(config=config, runtime={"detector": detector, "segmenter": segmenter,
    "classifier": DetectorBasedClassifier(detector), "portion_estimator": SyntheticPortionEstimator(),
    "llm_http_client_factory": local_client_factory})
app.state.scenario = "legacy"
app.state.provider_requests = []


@app.post("/__e2e/scenario")
async def scenario(request: Request):
    name = (await request.json())["name"]
    allowed = {"legacy", "anthropic", "backup", "both_fail", "analyze_failure", "analyze_timeout", "no_food", "malformed_vision", "explain_selection", "explain_backup", "explain_openai", "invalid_claim", "unknown_claim", "explain_timeout"}
    if name not in allowed:
        return JSONResponse({"error": "unknown scenario"}, 400)
    app.state.scenario = name
    detector._loaded = name != "analyze_failure"
    app.state.provider_requests = []
    app.state.llm_analyzer_chain = None
    app.state.plan_explanation_service = None
    app.state.settings = config.model_copy(update={"provider": "llm" if name not in {"legacy", "explain_selection", "explain_backup", "explain_openai", "invalid_claim", "unknown_claim", "explain_timeout"} else "legacy",
        "llm_api_key": config.internal_service_key, "llm_api_key_openai": config.internal_service_key,
        "llm_retries": 0, "llm_circuit_failure_threshold": 1,
        "explain_llm_enabled": name.startswith("explain_") or name in {"invalid_claim", "unknown_claim"}, "explain_timeout_seconds": 0.3})
    if name == "explain_openai":
        from pydantic import SecretStr
        app.state.settings = app.state.settings.model_copy(update={"llm_provider": "openai", "llm_api_key": SecretStr("")})
    return {"scenario": name}


@app.get("/__e2e/state")
def state():
    return {"detector_calls": detector.detect_calls, "image_size": getattr(detector, "last_size", None),
            "image_info_keys": getattr(detector, "last_info_keys", None), "provider_requests": app.state.provider_requests}


@app.post("/__e2e/provider/{vendor}")
async def scripted_provider(vendor: str, request: Request):
    body = await request.json()
    app.state.provider_requests.append({"vendor": vendor, "payload": body})
    name = app.state.scenario
    if name in {"both_fail", "analyze_failure"} or (name in {"backup", "explain_backup"} and vendor == "anthropic"):
        return JSONResponse({"error": "scripted unavailability"}, 503)
    if name == "explain_timeout":
        await asyncio.sleep(1)
    if name == "analyze_timeout":
        await asyncio.sleep(3)
    tool = body.get("tools", [{}])[0].get("name", "")
    is_explanation = tool == "select_claims" or body.get("response_format", {}).get("json_schema", {}).get("name") == "permitted_claims"
    if is_explanation:
        content = body["messages"][0]["content"]
        permitted = json.loads(content.split("\n", 1)[1])["allowed_claims"]
        payload = {"selected_ids": [item["id"] for item in permitted]}
        if name == "invalid_claim":
            payload = {"selected_ids": ["metric_diff:invented-999"], "text": "Eat 999 grams, allergen free"}
        if name == "unknown_claim":
            payload = {"selected_ids": ["metric_diff:invented-999"]}
    else:
        payload = {"is_food": name != "no_food", "foods": [] if name == "no_food" else [{
            "name_es": "Banano", "name_en": "Banana", "nutrition_query": "banana", "confidence": 0.86,
            "portion": {"estimated_grams": 100, "min_grams": 80, "max_grams": 120, "confidence": 0.7},
            "estimated_nutrients_per_100g": {"calories": 89.0, "protein_g": 1.1, "carbs_g": 22.8, "fat_g": 0.3, "fiber_g": 2.6}}]}
        if name == "malformed_vision":
            payload["foods"][0]["hidden_ingredients"] = ["invented"]
    if vendor == "anthropic":
        return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "select_claims" if is_explanation else "identify_food", "input": payload}]}
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}
