"""Interviewer-facing live transcript and next-question endpoints."""

from __future__ import annotations

import json
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import (
    ASRPreviewResponse,
    LiveEvidenceBatchConfirmationRequest,
    LiveEvidenceConfirmationRequest,
    LiveEvidenceMergeConfirmationRequest,
    LiveEvidenceReevaluationRequest,
    LiveInterviewStartRequest,
    LiveInterviewStatusRequest,
    LiveSuggestionDecisionRequest,
    LiveTranscriptRequest,
    LiveTranscriptUpdateRequest,
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


@router.patch("/{session_id}/segments/{segment_id}", response_model=WorkflowResponse)
async def update_segment(
    session_id: str,
    segment_id: UUID,
    payload: LiveTranscriptUpdateRequest,
    service: Service,
):
    state = await service.update_live_transcript_segment(
        session_id,
        segment_id,
        text=payload.text,
        speaker=TranscriptSpeaker(payload.speaker) if payload.speaker is not None else None,
    )
    return response(session_id, state)


@router.post("/{session_id}/segments/{segment_id}/evidence", response_model=WorkflowResponse)
async def confirm_segment_evidence(
    session_id: str,
    segment_id: UUID,
    payload: LiveEvidenceConfirmationRequest,
    service: Service,
):
    state = await service.confirm_live_answer(
        session_id,
        segment_id,
        question_segment_id=payload.question_segment_id,
        question=payload.question,
        competency=payload.competency,
    )
    return response(session_id, state)


@router.post("/{session_id}/evidence/merge", response_model=WorkflowResponse)
async def confirm_merged_evidence(
    session_id: str,
    payload: LiveEvidenceMergeConfirmationRequest,
    service: Service,
):
    state = await service.confirm_live_answer_segments(
        session_id,
        payload.segment_ids,
        question_segment_id=payload.question_segment_id,
        question=payload.question,
        competency=payload.competency,
    )
    return response(session_id, state)


@router.post("/{session_id}/evidence/batch", response_model=WorkflowResponse)
async def confirm_pending_evidence(
    session_id: str,
    payload: LiveEvidenceBatchConfirmationRequest,
    service: Service,
):
    state = await service.confirm_pending_live_answers(
        session_id,
        competency=payload.competency,
    )
    return response(session_id, state)


@router.delete("/{session_id}/evidence/{record_id}", response_model=WorkflowResponse)
async def revoke_evidence(session_id: str, record_id: UUID, service: Service):
    return response(session_id, await service.revoke_live_evidence(session_id, record_id))


@router.patch("/{session_id}/evidence/{record_id}/reevaluate", response_model=WorkflowResponse)
async def reevaluate_evidence(
    session_id: str,
    record_id: UUID,
    payload: LiveEvidenceReevaluationRequest,
    service: Service,
):
    state = await service.reevaluate_live_evidence(
        session_id,
        record_id,
        question=payload.question,
        competency=payload.competency,
    )
    return response(session_id, state)


@router.post("/{session_id}/audio", response_model=WorkflowResponse)
async def transcribe_audio(
    session_id: str,
    service: Service,
    file: Annotated[UploadFile, File()],
    speaker: Annotated[str, Form()] = "candidate",
    language: Annotated[str, Form()] = "zh",
    mode: Annotated[Literal["single", "dialogue"], Form()] = "single",
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
        mode=mode,
    )
    return response(session_id, state)


@router.post("/{session_id}/audio/preview", response_model=ASRPreviewResponse)
async def preview_audio_transcription(
    session_id: str,
    service: Service,
    file: Annotated[UploadFile, File()],
    language: Annotated[str, Form()] = "zh",
):
    content = await file.read()
    text = await service.preview_audio_transcription(
        session_id,
        content=content,
        filename=file.filename or "preview.wav",
        content_type=file.content_type or "application/octet-stream",
        language=language,
    )
    return ASRPreviewResponse(text=text)


@router.post("/{session_id}/suggestions", response_model=WorkflowResponse)
async def plan_next_question(session_id: str, service: Service):
    return response(session_id, await service.plan_live_next_question(session_id))


@router.post("/{session_id}/suggestions/stream")
async def stream_next_question(session_id: str, service: Service):
    """Stream the next-question suggestion token by token (text/event-stream)."""
    # Guard before streaming so a non-active session returns 409 instead of an
    # uncaught generator error mid-stream.
    state = await service.get_state(session_id)
    service.assert_live_active(state)

    async def event_stream():
        async for event in service.stream_live_suggestion(session_id):
            # JSON encoding keeps newlines inside one SSE data record and lets the
            # browser replace partial text after an upstream stream failure.
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
