"""Interview management routes."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import (
    InterviewSessionResponse,
    StartSessionRequest,
)
from interview_os.services.interview_service import InterviewService

logger = logging.getLogger(__name__)
router = APIRouter()

Service = Annotated[InterviewService, Depends(get_interview_service)]


@router.post("/sessions", response_model=InterviewSessionResponse)
async def start_session(req: StartSessionRequest, service: Service):
    session_id, state = await service.create_session(
        req.candidate_name, req.job_title, req.company_name
    )
    return InterviewSessionResponse(
        id=session_id,
        candidate_name=state.candidate.name,
        job_title=state.job.title,
        status="created",
        state=state.model_dump(mode="json"),
    )


@router.get("/sessions/{session_id}", response_model=InterviewSessionResponse)
async def get_session(session_id: str, service: Service):
    state = await service.get_state(session_id)
    return InterviewSessionResponse(
        id=session_id,
        candidate_name=state.candidate.name,
        job_title=state.job.title,
        status="active",
        state=state.model_dump(mode="json"),
    )
