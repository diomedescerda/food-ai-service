import logging

from fastapi import APIRouter, Depends, Request

from app.core.auth import require_internal_key
from app.schemas.explain import ExplainRequest, ExplainResponse
from app.services.plan_explanation import PlanExplanationService

router = APIRouter()


@router.post("/internal/foodai/explain", response_model=ExplainResponse, dependencies=[Depends(require_internal_key)], tags=["internal"])
async def explain(request: Request, payload: ExplainRequest) -> ExplainResponse:
    service = getattr(request.app.state, "plan_explanation_service", None)
    if service is None:
        service = PlanExplanationService(request.app.state.settings, getattr(request.app.state, "logger", None) or logging.getLogger(__name__), client_factory=getattr(request.app.state, "llm_http_client_factory", None))
        request.app.state.plan_explanation_service = service
    return await service.explain(payload)
