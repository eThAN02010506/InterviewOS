"""Human confirmation endpoints for researched entity identities."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import (
    EntityResolutionRequest,
    FactCardDecisionRequest,
    WorkflowResponse,
)
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]


@router.patch("/{session_id}/entities/{resolution_id}", response_model=WorkflowResponse)
async def resolve_entity(
    session_id: str,
    resolution_id: UUID,
    payload: EntityResolutionRequest,
    service: Service,
):
    state = await service.resolve_entity_candidate(
        session_id,
        resolution_id,
        accept=payload.accept,
        proposed_name=payload.proposed_name,
    )
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


@router.patch("/{session_id}/facts/{card_id}", response_model=WorkflowResponse)
async def decide_fact_card(
    session_id: str,
    card_id: UUID,
    payload: FactCardDecisionRequest,
    service: Service,
):
    state = await service.decide_fact_card(
        session_id,
        card_id,
        action=payload.action,
        note=payload.note,
    )
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))
