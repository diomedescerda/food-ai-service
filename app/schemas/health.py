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