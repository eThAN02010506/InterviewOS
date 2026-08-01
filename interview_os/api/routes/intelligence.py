"""Human confirmation endpoints for researched entity identities."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import EntityResolutionRequest, WorkflowResponse
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
        session_id, resolution_id, accept=payload.accept
    )
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))
