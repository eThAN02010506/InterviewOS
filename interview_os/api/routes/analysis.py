"""Analysis routes - resume, JD, interviewer analysis."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import (
    AnalysisResponse,
    CompanyAnalysisRequest,
    InterviewerAnalysisRequest,
    JobAnalysisRequest,
    ResumeAnalysisRequest,
)
from interview_os.services.interview_service import InterviewService

logger = logging.getLogger(__name__)
router = APIRouter()


Service = Annotated[InterviewService, Depends(get_interview_service)]


async def _response(session_id: str, message, service: InterviewService) -> AnalysisResponse:
    state = await service.get_state(session_id)
    return AnalysisResponse(
        session_id=session_id,
        agent=message.sender,
        output=message.content,
        state=state.model_dump(mode="json"),
    )


@router.post("/resume", response_model=AnalysisResponse)
async def analyze_resume(req: ResumeAnalysisRequest, service: Service):
    message = await service.analyze_resume(req.session_id, req.text)
    return await _response(req.session_id, message, service)


@router.post("/job", response_model=AnalysisResponse)
async def analyze_job(req: JobAnalysisRequest, service: Service):
    message = await service.analyze_job(req.session_id, req.text)
    return await _response(req.session_id, message, service)


@router.post("/company", response_model=AnalysisResponse)
async def analyze_company(req: CompanyAnalysisRequest, service: Service):
    message = await service.analyze_company(req.session_id, req.name, req.context)
    return await _response(req.session_id, message, service)


@router.post("/interviewer", response_model=AnalysisResponse)
async def analyze_interviewer(req: InterviewerAnalysisRequest, service: Service):
    message = await service.analyze_interviewer(
        req.session_id, req.name, req.position, req.company, req.public_info
    )
    return await _response(req.session_id, message, service)
