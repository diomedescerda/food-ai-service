import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.analyze import router as analyze_router
from app.api.health import router as health_router
from app.core.config import settings
from app.models.yolo_food_detector import YoloFoodDetector
from app.models.yolo_food_segmenter import YoloFoodSegmenter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida: carga detector y segmentador UNA vez al arrancar."""
    detector = YoloFoodDetector(settings)
    detector.load()
    app.state.detector = detector
    app.state.logger = logging.getLogger("uvicorn.error")
    app.state.logger.info(
        "Food detector cargado: version=%s, path=%s, classes=%s",
        detector.model_version,
        settings.model_path,
        sorted(detector.supported_classes),
    )

    segmenter = YoloFoodSegmenter(settings)
    segmenter.load()
    app.state.segmenter = segmenter
    app.state.logger.info(
        "Food segmenter cargado: version=%s, path=%s",
        segmenter.model_version,
        settings.seg_model_path,
    )

    app.state.settings = settings
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Servicio de IA de FoodAI: detección, segmentación, clasificación, porción y nutrición.",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(analyze_router)
    return app


app = create_app()