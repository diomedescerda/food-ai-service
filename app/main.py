from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Servicio de IA de FoodAI: detección, segmentación, clasificación, porción y nutrición.",
    )
    app.include_router(health_router)
    return app


app = create_app()