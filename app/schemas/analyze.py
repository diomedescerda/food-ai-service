from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail