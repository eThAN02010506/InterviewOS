"""Interactive mock interview endpoints."""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import MockAnswerRequest, MockSessionResponse
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]

MAX_MOCK_AUDIO_BYTES = 25 * 1024 * 1024


def _response(session_id: str, state, service: InterviewService) -> MockSessionResponse:
    question = service.current_mock_question(state)
    session = state.mock_session
    questions = state.mock_interview.questions
    index = session.current_question_index
    pending = max(0, len(questions) - index)
    retry_available = False
    if session.status.value == "active" and question is not None:
        retry_available = any(
            item.question_id == question.id and item.question == question.question
            for item in session.responses
        )
    return MockSessionResponse(
        session_id=session_id,
        mock_session=session.model_dump(mode="json"),
        current_question=question.model_dump(mode="json") if question else None,
        pending_questions=pending,
        answered_questions=len(session.responses),
        retry_available=retry_available,
        refill_in_flight=session.refill_in_flight,
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
    state = await service.submit_mock_answer(
        session_id,
        req.question_id,
        req.answer,
        retry=req.retry,
        retry_response_id=req.retry_response_id,
    )
    return _response(session_id, state, service)


@router.post("/{session_id}/next", response_model=MockSessionResponse)
async def next_mock_question(session_id: str, service: Service):
    state = await service.advance_mock_interview(session_id)
    return _response(session_id, state, service)


@router.post("/{session_id}/previous", response_model=MockSessionResponse)
async def previous_mock_question(session_id: str, service: Service):
    state = await service.previous_mock_question(session_id)
    return _response(session_id, state, service)


@router.post("/{session_id}/finish", response_model=MockSessionResponse)
async def finish_mock_interview(session_id: str, service: Service):
    state = await service.finish_mock_interview(session_id)
    return _response(session_id, state, service)


@router.post("/{session_id}/transcribe")
async def transcribe_mock_answer(session_id: str, service: Service, file: Annotated[UploadFile, File()]):
    # get_state enforces ownership (404 for a foreign session).
    await service.get_state(session_id)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="音频为空")
    if len(content) > MAX_MOCK_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="音频不超过 25 MB")
    text = await service.transcribe_mock_spoken_answer(
        session_id, content, file.filename or "answer.webm"
    )
    speech_feedback = await service.analyze_mock_speech_delivery(
        session_id,
        content,
        text,
        content_type=file.content_type or "application/octet-stream",
    )
    return {"text": text, "speech_feedback": speech_feedback.model_dump(mode="json")}


@router.post("/{session_id}/questions/{question_id}/speech")
async def speak_mock_question(
    session_id: str,
    question_id: str,
    service: Service,
    response_id: UUID | None = None,
):
    try:
        parsed_id = UUID(question_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="问题 ID 无效") from exc
    content, content_type = await service.synthesize_mock_question(
        session_id, parsed_id, response_id=response_id
    )
    return Response(content=content, media_type=content_type)
