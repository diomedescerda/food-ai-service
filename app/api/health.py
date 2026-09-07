from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.core.config import settings
from app.schemas.health import HealthModel, HealthResponse, ReadinessResponse

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
    # F57: pipeline 5.761 + nutrición en el liveness
    pipeline = getattr(request.app.state, "food_pipeline", None)
    nutrition = getattr(request.app.state, "nutrition_service", None)
    pipeline_model = None
    nutrition_model = None
    if pipeline is not None:
        pipeline_model = HealthModel(
            loaded=pipeline.available(),
            version="multitext-v1/reranker-general-v1/specialist-dino-base-v1",
        )
    if nutrition is not None:
        nutrition_model = HealthModel(
            loaded=nutrition.available(),
            version="mapping-v1",
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
        pipeline=pipeline_model,
        nutrition=nutrition_model,
    )


@router.get("/health/readiness", response_model=ReadinessResponse, tags=["health"])
def readiness(request: Request) -> ReadinessResponse:
    """Readiness: componentes necesarios para procesar requests. El pipeline
    y la nutrición son OPTATIVOS (fallback legacy seguro) — no se bloquea el
    readiness por ellos; sí por los modelos obligatorios."""
    detector = getattr(request.app.state, "detector", None)
    classifier = getattr(request.app.state, "classifier", None)
    pipeline = getattr(request.app.state, "food_pipeline", None)
    nutrition = getattr(request.app.state, "nutrition_service", None)
    index_loaded = pipeline is not None and pipeline.index is not None
    dino_loaded = pipeline is not None and pipeline._dino is not None
    clip_loaded = classifier is not None and classifier.is_loaded
    nutrition_loaded = nutrition is not None and nutrition.available()
    mandatory_ok = (
        detector is not None and detector.is_loaded
        and classifier is not None and classifier.is_loaded
    )
    return ReadinessResponse(
        status="ready" if mandatory_ok else "not_ready",
        catalog_size=5761,
        index_loaded=index_loaded,
        clip_loaded=clip_loaded,
        dino_loaded=dino_loaded,
        nutrition_loaded=nutrition_loaded,
        pipeline_version="f57",
        catalog_version="5761",
        timestamp_utc=datetime.now(timezone.utc),
    )