from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.core.config import settings
from app.schemas.health import HealthModel, HealthResponse

router = APIRouter()

SERVICE_NAME = "food-ai-service"


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(request: Request) -> HealthResponse:
    detector = getattr(request.app.state, "detector", None)
    segmenter = getattr(request.app.state, "segmenter", None)
    classifier = getattr(request.app.state, "classifier", None)
    depth = getattr(request.app.state, "depth_estimator", None)
    model = None
    seg_model = None
    classifier_model = None
    depth_model = None
    if detector is not None:
        model = HealthModel(
            loaded=detector.is_loaded,
            version=detector.model_version,
        )
    if segmenter is not None:
        seg_model = HealthModel(
            loaded=segmenter.is_loaded,
            version=segmenter.model_version,
        )
    if classifier is not None:
        classifier_model = HealthModel(
            loaded=classifier.is_loaded,
            version=classifier.model_version,
        )
    if depth is not None:
        depth_model = HealthModel(
            loaded=depth.is_loaded,
            version=depth.model_version,
        )
    healthy = (
        (model is not None and model.loaded)
        and (seg_model is not None and seg_model.loaded)
        and (classifier_model is not None and classifier_model.loaded)
    )
    return HealthResponse(
        status="healthy" if healthy else "degraded",
        service=SERVICE_NAME,
        version=settings.api_version,
        timestamp_utc=datetime.now(timezone.utc),
        model=model,
        segmentation_model=seg_model,
        classifier_model=classifier_model,
        depth_model=depth_model,
    )