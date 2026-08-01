"""Evidence aggregation and final evaluation endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import WorkflowResponse
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]


@router.post("/{session_id}", response_model=WorkflowResponse)
async def generate_evaluation(session_id: str, service: Service):
    state = await service.run_evaluation(session_id)
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))
