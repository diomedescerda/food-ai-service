from datetime import datetime

from pydantic import BaseModel


class HealthModel(BaseModel):
    loaded: bool
    version: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp_utc: datetime
    model: HealthModel | None = None
    segmentation_model: HealthModel | None = None
    classifier_model: HealthModel | None = None
    depth_model: HealthModel | None = None
    pipeline: HealthModel | None = None
    nutrition: HealthModel | None = None


class ReadinessResponse(BaseModel):
    status: str
    catalog_size: int
    index_loaded: bool
    clip_loaded: bool
    dino_loaded: bool
    nutrition_loaded: bool
    pipeline_version: str
    catalog_version: str
    timestamp_utc: datetime