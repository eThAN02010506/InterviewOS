"""Interactive mock interview endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import MockAnswerRequest, MockSessionResponse
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]


def _response(session_id: str, state, service: InterviewService) -> MockSessionResponse:
    question = service.current_mock_question(state)
    return MockSessionResponse(
        session_id=session_id,
        mock_session=state.mock_session.model_dump(mode="json"),
        current_question=question.model_dump(mode="json") if question else None,
    )


@router.post("/{session_id}/start", response_model=MockSessionResponse)
async def start_mock_interview(session_id: str, service: Service):
    state = await service.start_mock_interview(session_id)
    return _response(session_id, state, service)


@router.get("/{session_id}", response_model=MockSessionResponse)
async def get_mock_interview(session_id: str, service: Service):
    state = await service.get_state(session_id)
    return _response(session_id, state, service)


@router.post("/{session_id}/answers", response_model=MockSessionResponse)
async def submit_mock_answer(session_id: str, req: MockAnswerRequest, service: Service):
    state = await service.submit_mock_answer(session_id, req.question_id, req.answer)
    return _response(session_id, state, service)
