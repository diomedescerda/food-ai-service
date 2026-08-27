import os
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.analyze import AnalyzeResponse

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

# Firmas mágicas: JPEG (FF D8 FF), PNG (89 50 4E 47), WEBP ("RIFF" .... "WEBP")
MAGIC_BYTES = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89\x50\x4e\x47",
    "image/webp": b"RIFF",
}


def _error(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"success": False, "error": {"code": code, "message": message}},
    )


def _has_valid_magic(content_type: str, head: bytes) -> bool:
    expected = MAGIC_BYTES.get(content_type)
    if expected is None:
        return False
    return head.startswith(expected)


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze(
    analysis_id: str = Form(...),
    image: UploadFile = File(...),
) -> AnalyzeResponse:
    """Recibe una imagen de comida (ingesta). Sin análisis todavía: responde
    status "received". La validación de contenido ocurre en el backend .NET;
    aquí solo se re-verifica el contrato mínimo."""
    try:
        UUID(analysis_id)
    except ValueError:
        raise _error("INVALID_ANALYSIS_ID", "analysis_id debe ser un UUID válido.")

    if image.size is None or image.size <= 0:
        raise _error("EMPTY_FILE", "El archivo está vacío.")

    if image.size > MAX_IMAGE_SIZE_BYTES:
        raise _error("IMAGE_TOO_LARGE", "La imagen supera el tamaño máximo.")

    content_type = (image.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise _error("INVALID_IMAGE", f"Tipo de contenido no permitido: {content_type or 'desconocido'}.")

    filename = image.filename or ""
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise _error("INVALID_IMAGE", f"Extensión no permitida: {extension}.")

    head = await image.read(12)
    await image.seek(0)
    if not _has_valid_magic(content_type, head):
        raise _error("CORRUPT_FILE", "El archivo no es una imagen válida.")

    return AnalyzeResponse(analysis_id=analysis_id, status="received")