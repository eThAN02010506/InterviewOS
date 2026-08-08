"""Whole-session audio upload + download routes."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from interview_os.api.dependencies import get_interview_service
from interview_os.api.schemas.interview import WorkflowResponse
from interview_os.services.interview_service import InterviewService

logger = logging.getLogger(__name__)
router = APIRouter()

Service = Annotated[InterviewService, Depends(get_interview_service)]

# The session id feeds a filesystem path, so it must be a strict UUID.
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
MAX_SESSION_AUDIO_BYTES = 256 * 1024 * 1024  # 256 MB


@router.post("/{session_id}/audio/final", response_model=WorkflowResponse)
async def save_session_audio(
    session_id: str,
    service: Service,
    file: Annotated[UploadFile, File()],
):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id")
    content = await file.read()
    if len(content) > MAX_SESSION_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Session audio exceeds the 256 MB limit")
    if not content:
        raise HTTPException(status_code=422, detail="Audio payload is empty")
    extension = Path(file.filename or "session.wav").suffix.lstrip(".") or "wav"
    state = await service.save_live_audio(session_id, content, extension=extension)
    return WorkflowResponse(session_id=session_id, state=state.model_dump(mode="json"))


@router.get("/{session_id}/audio")
async def get_session_audio(session_id: str, service: Service):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id")
    # Ownership is enforced by get_state (404 for a foreign session).
    state = await service.get_state(session_id)
    path = service.get_live_audio_path(session_id, state.live_interview.audio_file)
    if path is None:
        raise HTTPException(status_code=404, detail="No recording for this session")
    media_type = {
        ".webm": "audio/webm",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
    }[path.suffix]
    return FileResponse(path, media_type=media_type, filename=path.name)
