"""Controlled autonomous workflow routes."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import AutopilotRequest, WorkflowResponse
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]


@router.post("/{session_id}/run", response_model=WorkflowResponse)
async def run_autopilot(session_id: str, req: AutopilotRequest, service: Service):
    state = await service.run_autopilot(
        session_id,
        role=req.role,
        resume_text=req.resume_text,
        job_description=req.job_description,
        company_name=req.company_name,
        company_context=req.company_context,
        interviewer_name=req.interviewer_name,
        interviewer_position=req.interviewer_position,
        authorized_public_research=req.authorized_public_research,
    )
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))
