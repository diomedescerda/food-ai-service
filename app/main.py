import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.analyze import router as analyze_router
from app.api.health import router as health_router
from app.core.config import settings
from app.models.yolo_food_detector import YoloFoodDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida: carga el modelo UNA vez al arrancar (no por request)."""
    detector = YoloFoodDetector(settings)
    detector.load()
    app.state.detector = detector
    app.state.settings = settings
    app.state.logger = logging.getLogger("uvicorn.error")
    app.state.logger.info(
        "Food detector cargado: version=%s, path=%s, classes=%s",
        detector.model_version,
        settings.model_path,
        sorted(detector.supported_classes),
    )
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