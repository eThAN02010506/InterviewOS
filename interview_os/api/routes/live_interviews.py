"""Interviewer-facing live transcript and next-question endpoints."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import (
    LiveInterviewStartRequest,
    LiveInterviewStatusRequest,
    LiveSuggestionDecisionRequest,
    LiveTranscriptRequest,
    WorkflowResponse,
)
from interview_os.core.state import (
    LiveInterviewStatus,
    QuestionSuggestionStatus,
    TranscriptSpeaker,
)
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]


def response(session_id: str, state) -> WorkflowResponse:
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


@router.get("/{session_id}", response_model=WorkflowResponse)
async def get_live_interview(session_id: str, service: Service):
    return response(session_id, await service.get_state(session_id))


@router.post("/{session_id}/start", response_model=WorkflowResponse)
async def start_live_interview(
    session_id: str, payload: LiveInterviewStartRequest, service: Service
):
    state = await service.start_live_interview(
        session_id, consent_confirmed=payload.consent_confirmed
    )
    return response(session_id, state)


@router.post("/{session_id}/status", response_model=WorkflowResponse)
async def set_live_status(
    session_id: str, payload: LiveInterviewStatusRequest, service: Service
):
    state = await service.set_live_interview_status(
        session_id, LiveInterviewStatus(payload.status)
    )
    return response(session_id, state)


@router.post("/{session_id}/segments", response_model=WorkflowResponse)
async def append_segment(session_id: str, payload: LiveTranscriptRequest, service: Service):
    state = await service.append_live_transcript(
        session_id,
        text=payload.text,
        speaker=TranscriptSpeaker(payload.speaker),
    )
    return response(session_id, state)


@router.post("/{session_id}/audio", response_model=WorkflowResponse)
async def transcribe_audio(
    session_id: str,
    service: Service,
    file: Annotated[UploadFile, File()],
    speaker: Annotated[str, Form()] = "candidate",
    language: Annotated[str, Form()] = "zh",
):
    try:
        parsed_speaker = TranscriptSpeaker(speaker)
    except ValueError:
        parsed_speaker = TranscriptSpeaker.UNKNOWN
    content = await file.read()
    state, _ = await service.transcribe_live_audio(
        session_id,
        content=content,
        filename=file.filename or "audio.webm",
        content_type=file.content_type or "application/octet-stream",
        speaker=parsed_speaker,
        language=language,
    )
    return response(session_id, state)


@router.post("/{session_id}/suggestions", response_model=WorkflowResponse)
async def plan_next_question(session_id: str, service: Service):
    return response(session_id, await service.plan_live_next_question(session_id))


@router.patch("/{session_id}/suggestions/{suggestion_id}", response_model=WorkflowResponse)
async def decide_suggestion(
    session_id: str,
    suggestion_id: UUID,
    payload: LiveSuggestionDecisionRequest,
    service: Service,
):
    state = await service.decide_live_suggestion(
        session_id,
        suggestion_id,
        status=QuestionSuggestionStatus(payload.status),
        final_question=payload.final_question,
    )
    return response(session_id, state)
