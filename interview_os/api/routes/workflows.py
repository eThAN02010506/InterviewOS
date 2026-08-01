"""End-to-end interview intelligence workflows."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import (
    CandidatePrepRequest,
    EnterpriseDesignRequest,
    WorkflowResponse,
)
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]


@router.post("/candidate-prep", response_model=WorkflowResponse)
async def candidate_prep(req: CandidatePrepRequest, service: Service):
    state = await service.run_candidate_prep(
        req.session_id,
        resume_text=req.resume_text,
        job_description=req.job_description,
        company_name=req.company_name,
        company_context=req.company_context,
        interviewer_name=req.interviewer_name,
        interviewer_position=req.interviewer_position,
        interviewer_public_info=req.interviewer_public_info,
    )
    return WorkflowResponse(session_id=req.session_id, state=state.model_dump(mode="json"))


@router.post("/enterprise-design", response_model=WorkflowResponse)
async def enterprise_design(req: EnterpriseDesignRequest, service: Service):
    state = await service.run_enterprise_design(
        req.session_id,
        resume_text=req.resume_text,
        job_description=req.job_description,
        company_name=req.company_name,
        company_context=req.company_context,
    )
    return WorkflowResponse(session_id=req.session_id, state=state.model_dump(mode="json"))
