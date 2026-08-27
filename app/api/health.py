from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.core.config import settings
from app.schemas.health import HealthModel, HealthResponse

router = APIRouter()

SERVICE_NAME = "food-ai-service"


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(request: Request) -> HealthResponse:
    detector = getattr(request.app.state, "detector", None)
    model = None
    if detector is not None:
        model = HealthModel(
            loaded=detector.is_loaded,
            version=detector.model_version,
        )
    return HealthResponse(
        status="healthy" if model is not None and model.loaded else "degraded",
        service=SERVICE_NAME,
        version=settings.api_version,
        timestamp_utc=datetime.now(timezone.utc),
        model=model,
    )