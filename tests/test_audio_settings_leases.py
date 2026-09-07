"""Provider-generation fencing for non-live and background audio work."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import update

from interview_os.database.schema import InterviewSession
from interview_os.database.storage import Storage
from interview_os.services.interview_service import (
    InterviewService,
    LiveInterviewStateError,
)

_FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 20) + b"data\x00\x00\x00\x00"


@pytest.mark.parametrize("operation", ["preview", "mock"])
async def test_non_live_asr_operations_hold_audio_settings_lease(tmp_path, operation):
    class BlockingASR:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def transcribe(self, content, **kwargs):
            self.started.set()
            await self.release.wait()
            return "stable transcript"

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / f'{operation}-lease.db'}")
    await storage.init_db()
    asr = BlockingASR()
    service = InterviewService(storage, llm_client=None, asr_client=asr)
    sid, _ = await service.create_session()
    call = (
        service.preview_audio_transcription(
            sid,
            content=_FAKE_WAV,
            filename="preview.wav",
            content_type="audio/wav",
        )
        if operation == "preview"
        else service.transcribe_mock_spoken_answer(sid, _FAKE_WAV, "answer.wav")
    )
    task = asyncio.create_task(call)
    await asr.started.wait()

    async with service._settings_lock:
        with pytest.raises(LiveInterviewStateError, match="audio request"):
            await service.ensure_audio_settings_mutable()

    asr.release.set()
    assert await task == "stable transcript"
    async with service._settings_lock:
        await service.ensure_audio_settings_mutable()
    await storage.close()


async def test_background_speech_analysis_blocks_settings_and_drops_stale_result(
    tmp_path,
):
    class BlockingOmni:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def analyze_speaking_style(self, content, **kwargs):
            self.started.set()
            await self.release.wait()
            return {
                "pace": "provider-specific result",
                "clarity": "provider-specific result",
            }

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'speech-lease.db'}")
    await storage.init_db()
    omni = BlockingOmni()
    service = InterviewService(storage, llm_client=None, omni_client=omni)
    sid, _ = await service.create_session()
    recording_id = uuid4()
    fallback = await service.schedule_mock_speech_delivery_analysis(
        sid,
        recording_id,
        _FAKE_WAV,
        "我先说明结论，再解释行动和结果。",
        content_type="audio/wav",
    )
    await omni.started.wait()

    async with service._settings_lock:
        with pytest.raises(LiveInterviewStateError, match="audio request"):
            await service.ensure_audio_settings_mutable()

    # Simulate an out-of-band recovery rotation. Even then, a result produced
    # by the retired provider generation must not be published.
    service.refresh_audio_settings_etag()
    omni.release.set()
    await service._background.flush()
    status, feedback = await service.get_mock_speech_delivery_status(
        sid, recording_id
    )

    assert status == "completed"
    assert feedback == fallback
    assert feedback.source == "text_fallback"
    async with service._settings_lock:
        await service.ensure_audio_settings_mutable()
    await storage.close()


@pytest.mark.parametrize(
    "live_state",
    [
        [],
        {"pending_audio_capture_ids": 5},
        {"pending_audio_capture_registered_at": []},
        {"audio_capture_guard_ids": {}},
        {"pending_audio_capture_settings_etags": []},
        {"pending_audio_capture_epochs": []},
    ],
)
async def test_malformed_stored_audio_lease_shape_fails_closed(
    tmp_path, monkeypatch, live_state
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'malformed-lease.db'}")
    await storage.init_db()
    service = InterviewService(storage, llm_client=None)

    async def malformed_records(**kwargs):
        assert kwargs == {"strict": True}
        return [({"live_interview": live_state}, None)]

    monkeypatch.setattr(storage, "list_all_session_state_records", malformed_records)

    with pytest.raises(LiveInterviewStateError, match="stored live state is invalid"):
        await service.ensure_audio_settings_mutable()

    await storage.close()


@pytest.mark.parametrize("invalid_state", ["{broken", "[]"])
async def test_invalid_persisted_session_json_blocks_audio_settings_change(
    tmp_path, invalid_state
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'invalid-state.db'}")
    await storage.init_db()
    service = InterviewService(storage, llm_client=None)
    sid, _ = await service.create_session()
    async with storage.session_factory() as session:
        await session.execute(
            update(InterviewSession)
            .where(InterviewSession.id == sid)
            .values(state_json=invalid_state)
        )
        await session.commit()
    service._runtimes.clear()

    with pytest.raises(LiveInterviewStateError, match="stored session state is invalid"):
        await service.ensure_audio_settings_mutable()

    await storage.close()
