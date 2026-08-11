"""Evidence aggregation and final evaluation endpoints."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import (
    AnswerEvaluationReviewRequest,
    TranscriptImportRequest,
    WorkflowResponse,
)
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]


@router.post("/{session_id}", response_model=WorkflowResponse)
async def generate_evaluation(session_id: str, service: Service):
    state = await service.run_evaluation(session_id)
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


@router.post("/{session_id}/transcript", response_model=WorkflowResponse)
async def import_transcript(session_id: str, payload: TranscriptImportRequest, service: Service):
    state = await service.import_interview_transcript(
        session_id,
        [entry.model_dump() for entry in payload.entries],
        auto_evaluate=payload.auto_evaluate,
    )
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


@router.patch("/{session_id}/answers/{record_id}/review", response_model=WorkflowResponse)
async def review_answer_evaluation(
    session_id: str,
    record_id: UUID,
    payload: AnswerEvaluationReviewRequest,
    service: Service,
):
    state = await service.review_answer_evaluation(
        session_id,
        record_id,
        **payload.model_dump(),
    )
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))
