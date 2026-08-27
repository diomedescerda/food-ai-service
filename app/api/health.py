from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter()

SERVICE_NAME = "food-ai-service"


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=settings.api_version,
        timestamp_utc=datetime.now(timezone.utc),
    )