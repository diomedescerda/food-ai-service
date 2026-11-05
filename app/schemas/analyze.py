from pydantic import BaseModel, ConfigDict, Field


class EstimatedNutrients(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    calories: float = Field(ge=0, le=1000)
    protein_g: float = Field(ge=0, le=100)
    carbs_g: float = Field(ge=0, le=100)
    fat_g: float = Field(ge=0, le=100)
    fiber_g: float = Field(ge=0, le=100)


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Segmentation(BaseModel):
    mask: str
    area_pixels: int


class PortionEstimate(BaseModel):
    portion_size: str
    estimated_grams: int | None
    min_grams: int | None
    max_grams: int | None
    confidence: float
    method: str


class DetectedFood(BaseModel):
    name: str
    nutrition_query: str | None = None
    name_en: str | None = None
    estimated_nutrients_per_100g: EstimatedNutrients | None = None
    confidence: float
    bounding_box: BoundingBox
    segmentation: Segmentation | None = None
    portion: PortionEstimate | None = None


class AnalyzeResponse(BaseModel):
    analysis_id: str
    provider: str | None = None
    status: str
    model_version: str
    seg_model_version: str
    classifier_version: str = "detector-based-v1"
    inference_time_ms: int
    foods: list[DetectedFood]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
