"""Backend-to-service authentication, separate from patient authentication."""
import secrets

from fastapi import HTTPException, Request

from app.core.config import settings


def _validate_service_key(request: Request, *, required: bool) -> None:
    configured = getattr(request.app.state, "settings", settings).internal_service_key.get_secret_value()
    if not configured:
        if required:
            raise HTTPException(503, detail={"code": "SERVICE_AUTH_NOT_CONFIGURED"})
        return
    supplied = request.headers.get("X-FoodAI-Service-Key", "")
    if len(supplied) > 512 or not secrets.compare_digest(supplied.encode(), configured.encode()):
        raise HTTPException(401, detail={"code": "INVALID_SERVICE_KEY"})


def require_internal_key(request: Request) -> None:
    _validate_service_key(request, required=True)


def require_service_key(request: Request) -> None:
    _validate_service_key(request, required=False)
