from pydantic import BaseModel
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp_utc: datetime