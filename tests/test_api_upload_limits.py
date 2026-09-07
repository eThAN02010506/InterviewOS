"""API boundary tests for bounded multipart audio reads."""

from __future__ import annotations

import asyncio
import re
from typing import cast
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.api.request_limits import BodyLimit, MultipartBodyLimitMiddleware
from interview_os.api.routes import audio as audio_routes
from interview_os.api.routes import live_interviews as live_routes
from interview_os.api.routes import mock_interviews as mock_routes
from interview_os.api.routes import resumes as resume_routes
from interview_os.api.upload_utils import read_upload_bounded
from interview_os.database.storage import Storage
from interview_os.services.interview_service import SessionNotFoundError
from interview_os.services.settings_service import LocalSettingsStore


class _TrackingUpload:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        end = len(self.payload) if size < 0 else self.position + size
        chunk = self.payload[self.position : end]
        self.position += len(chunk)
        return chunk


def test_wire_body_limit_rejects_declared_oversize_without_calling_application():
    called = False
    sent = []

    async def application(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = MultipartBodyLimitMiddleware(
        application,
        limits=(BodyLimit(re.compile(r"/upload"), 5, "too large"),),
    )
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/upload",
                "headers": [(b"content-length", b"6")],
            },
            receive,
            send,
        )
    )

    assert called is False
    assert sent[0]["status"] == 413
    assert b"too large" in sent[1]["body"]


def test_wire_body_limit_stops_chunked_request_at_first_excess_byte():
    chunks = iter(
        (
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": True},
            {"type": "http.request", "body": b"789", "more_body": False},
        )
    )
    reads = 0
    sent = []

    async def application(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body", False):
                return

    async def receive():
        nonlocal reads
        reads += 1
        return next(chunks)

    async def send(message):
        sent.append(message)

    middleware = MultipartBodyLimitMiddleware(
        application,
        limits=(BodyLimit(re.compile(r"/upload"), 5, "too large"),),
    )
    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "path": "/upload", "headers": []},
            receive,
            send,
        )
    )

    assert reads == 2
    assert sent[0]["status"] == 413


def _make_app(tmp_path):
    return create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'upload-limits.db'}"),
        llm_client=None,
        configure_llm=False,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
    )


def test_bounded_upload_reads_multiple_chunks_and_stops_at_limit_plus_one():
    reader = _TrackingUpload(b"0123456789")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            read_upload_bounded(
                cast(UploadFile, reader),
                max_bytes=5,
                too_large_detail="too large",
                chunk_bytes=2,
            )
        )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "too large"
    assert reader.read_sizes == [2, 2, 2]
    assert reader.position == 6


def test_bounded_upload_accepts_exact_limit_without_unbounded_reads():
    reader = _TrackingUpload(b"12345")

    content = asyncio.run(
        read_upload_bounded(
            cast(UploadFile, reader),
            max_bytes=5,
            too_large_detail="too large",
            chunk_bytes=2,
        )
    )

    assert content == b"12345"
    assert reader.position == 5
    assert reader.read_sizes
    assert all(0 < size <= 2 for size in reader.read_sizes)


@pytest.mark.parametrize(
    ("route_module", "limit_name", "path_template", "filename", "detail"),
    [
        (
            audio_routes,
            "MAX_SESSION_AUDIO_BYTES",
            "/api/live-interviews/{session_id}/audio/final",
            "session.wav",
            "Session audio exceeds the 65 MB limit",
        ),
        (
            live_routes,
            "MAX_LIVE_AUDIO_BYTES",
            "/api/live-interviews/{session_id}/audio",
            "chunk.webm",
            "Audio chunk exceeds the 25 MB limit",
        ),
        (
            live_routes,
            "MAX_LIVE_PREVIEW_AUDIO_BYTES",
            "/api/live-interviews/{session_id}/audio/preview",
            "preview.webm",
            "ASR preview exceeds the 10 MB limit",
        ),
        (
            mock_routes,
            "MAX_MOCK_AUDIO_BYTES",
            "/api/mock-interviews/{session_id}/transcribe",
            "answer.wav",
            "音频不超过 25 MB",
        ),
        (
            resume_routes,
            "MAX_RESUME_BYTES",
            "/api/resumes/{session_id}/upload",
            "resume.pdf",
            "简历不能超过 10 MB",
        ),
    ],
)
def test_audio_upload_routes_reject_oversized_payloads_with_413(
    tmp_path,
    monkeypatch,
    route_module,
    limit_name,
    path_template,
    filename,
    detail,
):
    monkeypatch.setattr(route_module, limit_name, 8)
    with TestClient(_make_app(tmp_path)) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        form_data = {}
        path = path_template.format(session_id=session_id)
        if path.endswith("/audio/final"):
            form_data = {
                "recording_id": str(uuid4()),
                "expected_audio_revision": "0",
            }
        elif path.endswith("/audio"):
            form_data = {"recording_id": str(uuid4())}
        result = client.post(
            path,
            data=form_data,
            files={"file": (filename, b"123456789", "application/octet-stream")},
        )

    assert result.status_code == 413
    assert result.json()["detail"] == detail


def test_wire_limit_rejection_keeps_cors_headers_for_desktop_renderer(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        result = client.post(
            f"/api/live-interviews/{uuid4()}/audio/final",
            content=b"",
            headers={
                "Origin": "http://127.0.0.1:8000",
                "Content-Length": str(67 * 1024 * 1024),
            },
        )

    assert result.status_code == 413
    assert result.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"
    assert result.json()["detail"] == "Session audio request exceeds the 66 MB wire limit"


def test_multipart_type_error_with_upload_object_remains_a_serializable_422(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        result = client.post(
            f"/api/live-interviews/{uuid4()}/audio/final",
            files={
                "file": ("session.webm", b"audio", "audio/webm"),
                "recording_id": ("id.txt", str(uuid4()).encode(), "text/plain"),
                "expected_audio_revision": (None, "0"),
            },
        )

    assert result.status_code == 422
    assert isinstance(result.json()["detail"], list)


def test_missing_resume_session_is_rejected_before_upload_is_read():
    upload = _TrackingUpload(b"private resume bytes")

    class MissingSessionService:
        async def get_state(self, session_id):
            raise SessionNotFoundError(session_id)

    with pytest.raises(SessionNotFoundError):
        asyncio.run(
            resume_routes.upload_resume(
                "00000000-0000-0000-0000-000000000000",
                cast(object, MissingSessionService()),
                cast(UploadFile, upload),
                "rules",
            )
        )

    assert upload.read_sizes == []


def test_live_status_mutation_schema_requires_expected_revision(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        missing_start_revision = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True, "operation_id": str(uuid4())},
        )
        missing_status_revision = client.post(
            f"/api/live-interviews/{session_id}/status",
            json={"status": "paused"},
        )

    assert missing_start_revision.status_code == 422
    assert missing_status_revision.status_code == 422
    assert missing_start_revision.json()["detail"][0]["loc"][-1] == "expected_revision"
    assert missing_status_revision.json()["detail"][0]["loc"][-1] == "expected_revision"


def test_live_start_schema_requires_operation_id(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        result = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True, "expected_revision": 0},
        )

    assert result.status_code == 422
    assert result.json()["detail"][0]["loc"][-1] == "operation_id"
