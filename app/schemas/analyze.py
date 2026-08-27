from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectedFood(BaseModel):
    name: str
    confidence: float
    bounding_box: BoundingBox


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: str
    model_version: str
    inference_time_ms: int
    foods: list[DetectedFood]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail