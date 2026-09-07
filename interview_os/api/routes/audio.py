"""Whole-session audio upload + download routes."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Annotated, BinaryIO
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import LiveAudioDeleteRequest, WorkflowResponse
from interview_os.api.upload_utils import read_upload_bounded
from interview_os.services.interview_service import InterviewService

logger = logging.getLogger(__name__)
router = APIRouter()

Service = Annotated[InterviewService, Depends(get_interview_service)]

# The session id feeds a filesystem path, so it must be a strict UUID.
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# The browser rotates the continuous archive at 64 MiB. One extra MiB absorbs
# MediaRecorder's asynchronous stop boundary without reopening the old 512 MiB
# peak caused by accepting and joining a 256 MiB part in memory.
MAX_SESSION_AUDIO_BYTES = 65 * 1024 * 1024


def _audio_download_response(opened: tuple[BinaryIO, str]) -> StreamingResponse:
    stream, filename = opened

    def chunks():
        try:
            while block := stream.read(1024 * 1024):
                yield block
        finally:
            stream.close()

    suffix = Path(filename).suffix
    media_type = {
        ".webm": "audio/webm",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
    }[suffix]
    return StreamingResponse(
        chunks(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(os.fstat(stream.fileno()).st_size),
        },
    )


@router.post("/{session_id}/audio/final", response_model=WorkflowResponse)
async def save_session_audio(
    session_id: str,
    service: Service,
    file: Annotated[UploadFile, File()],
    recording_id: Annotated[UUID, Form()],
    expected_audio_revision: Annotated[int, Form()],
):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id")
    # Reject missing or foreign sessions before materializing a large body in
    # the application heap. Authentication middleware has already identified
    # the owner used by this lookup.
    await service.get_state(session_id)
    content = await read_upload_bounded(
        file,
        max_bytes=MAX_SESSION_AUDIO_BYTES,
        too_large_detail="Session audio exceeds the 65 MB limit",
    )
    if not content:
        raise HTTPException(status_code=422, detail="Audio payload is empty")
    extension = Path(file.filename or "session.wav").suffix.lstrip(".") or "wav"
    state = await service.save_live_audio(
        session_id,
        content,
        extension=extension,
        recording_id=recording_id,
        expected_audio_revision=expected_audio_revision,
    )
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


@router.get("/{session_id}/audio")
async def get_session_audio(session_id: str, service: Service):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id")
    opened = await service.open_live_audio_file(session_id)
    if opened is None:
        raise HTTPException(status_code=404, detail="No recording for this session")
    return _audio_download_response(opened)


@router.delete("/{session_id}/audio", response_model=WorkflowResponse)
async def delete_session_audio(
    session_id: str,
    payload: LiveAudioDeleteRequest,
    service: Service,
):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id")
    state = await service.delete_live_audio(
        session_id,
        expected_audio_revision=payload.expected_audio_revision,
        operation_id=payload.operation_id,
    )
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


@router.get("/{session_id}/audio/{recording_id}")
async def get_session_audio_part(
    session_id: str, recording_id: UUID, service: Service
):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id")
    opened = await service.open_live_audio_file(
        session_id, recording_id=recording_id
    )
    if opened is None:
        raise HTTPException(status_code=404, detail="Recording part is unavailable")
    return _audio_download_response(opened)
