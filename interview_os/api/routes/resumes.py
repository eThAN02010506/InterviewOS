"""Resume upload and human-in-the-loop review routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import ResumeClaimUpdateRequest, WorkflowResponse
from interview_os.services.interview_service import InterviewService
from interview_os.services.resume_service import MAX_RESUME_BYTES, ResumeProcessingError

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]


@router.post("/{session_id}/upload", response_model=WorkflowResponse)
async def upload_resume(session_id: str, service: Service, file: Annotated[UploadFile, File()]):
    content = await file.read(MAX_RESUME_BYTES + 1)
    state = await service.upload_resume(session_id, file.filename or "resume", content)
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


@router.patch("/{session_id}/claims/{claim_id}", response_model=WorkflowResponse)
async def update_claim(
    session_id: str, claim_id: UUID, req: ResumeClaimUpdateRequest, service: Service
):
    state = await service.update_resume_claim(session_id, claim_id, req.status, req.note)
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


__all__ = ["ResumeProcessingError"]
