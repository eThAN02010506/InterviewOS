"""Whole-session audio recording tests."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.core.state import (
    LiveAudioCaptureReceipt,
    LiveInterviewStatus,
    TranscriptSpeaker,
)
from interview_os.database.storage import Storage
from interview_os.services.interview_service import InterviewService, LiveInterviewStateError
from interview_os.services.media_service import MediaServiceMixin
from interview_os.services.settings_service import LocalSettingsStore
from interview_os.tools.asr import ASRClient


class _WorkflowLLM:
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "structured candidate profile" in prompt:
            return '{"name":"Ada","skills":["Python"]}'
        if "job description" in prompt:
            return '{"title":"Engineer","competencies":["System Design"]}'
        if "company dna" in prompt:
            return '{"name":"Example","dna":"Engineering culture"}'
        if "fuse three inputs" in prompt:
            return '{"summary":"Show impact","key_risks":["Scope"]}'
        return '{"name":"Ada","skills":["Python"]}'

    async def embed(self, text):
        return []


def _register(client: TestClient, username: str) -> str:
    resp = client.post(
        "/api/auth/register", json={"username": username, "password": "pw-123456"}
    )
    return resp.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _make_app(tmp_path, recordings_dir):
    return create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'audio.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
        recordings_dir=recordings_dir,
    )


_FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 20) + b"data\x00\x00\x00\x00"


def _start_consented_session(client: TestClient, token: str) -> str:
    sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
    resp = client.post(
        f"/api/live-interviews/{sid}/start",
        json={"consent_confirmed": True, "expected_revision": 0, "operation_id": str(uuid4())},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    return sid


def _capture_payload(
    client: TestClient, token: str, *, capture_epoch: int = 1, **updates
) -> dict:
    settings = client.get("/api/settings", headers=_auth(token)).json()
    return {
        "expected_capture_epoch": capture_epoch,
        "expected_settings_revision": settings["audio_settings_revision"],
        "expected_settings_etag": settings["audio_settings_etag"],
        **updates,
    }


def test_upload_and_download_session_audio(tmp_path):
    recordings_dir = tmp_path / "recordings"
    with TestClient(_make_app(tmp_path, recordings_dir)) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        recording_id = uuid4()
        resp = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(recording_id), "expected_audio_revision": "0"},
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        live = resp.json()["state"]["live_interview"]
        assert live["audio_file"] == f"{sid}-{recording_id}.wav"
        assert live["audio_size_bytes"] == len(_FAKE_WAV)
        # File exists on disk under the recordings dir
        assert (recordings_dir / f"{sid}-{recording_id}.wav").exists()
        if os.name == "posix":
            assert recordings_dir.stat().st_mode & 0o777 == 0o700
            assert (
                recordings_dir / f"{sid}-{recording_id}.wav"
            ).stat().st_mode & 0o777 == 0o600
        # Download returns the bytes
        dl = client.get(f"/api/live-interviews/{sid}/audio", headers=_auth(token))
        assert dl.status_code == 200
        assert dl.content == _FAKE_WAV
        assert dl.headers["content-type"] == "audio/wav"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_service_repairs_existing_recording_directory_permissions(tmp_path):
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir(mode=0o755)
    existing = recordings_dir / "existing.webm"
    existing.write_bytes(b"private voice")
    os.chmod(recordings_dir, 0o755)
    os.chmod(existing, 0o644)

    with TestClient(_make_app(tmp_path, recordings_dir)):
        pass

    assert recordings_dir.stat().st_mode & 0o777 == 0o700
    assert existing.stat().st_mode & 0o777 == 0o600


def test_atomic_audio_replace_has_no_fallible_permission_step_after_commit(
    tmp_path, monkeypatch
):
    temp = tmp_path / ".recording.tmp"
    target = tmp_path / "recording.webm"

    def reject_path_chmod(*args, **kwargs):
        raise OSError("chmod unavailable after rename")

    monkeypatch.setattr(os, "chmod", reject_path_chmod)

    MediaServiceMixin._write_then_replace(temp, target, b"VOICE")

    assert target.read_bytes() == b"VOICE"
    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == 0o600
    assert not temp.exists()


def test_audio_archive_mutations_require_generation_and_operation_contracts(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        headers = _auth(token)
        missing_recording_id = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"expected_audio_revision": "0"},
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=headers,
        )
        missing_revision = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(uuid4())},
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=headers,
        )
        delete_missing_operation = client.request(
            "DELETE",
            f"/api/live-interviews/{sid}/audio",
            json={"expected_audio_revision": 0},
            headers=headers,
        )
        delete_missing_revision = client.request(
            "DELETE",
            f"/api/live-interviews/{sid}/audio",
            json={"operation_id": str(uuid4())},
            headers=headers,
        )

    assert {
        missing_recording_id.status_code,
        missing_revision.status_code,
        delete_missing_operation.status_code,
        delete_missing_revision.status_code,
    } == {422}


def test_audio_cross_account_404(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        alice = _register(client, "alice")
        bob = _register(client, "bob")
        sid = _start_consented_session(client, alice)
        recording_id = uuid4()
        resp = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(recording_id), "expected_audio_revision": "0"},
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(alice),
        )
        assert resp.status_code == 200
        # Bob cannot upload to nor download Alice's session
        assert (
            client.post(
                f"/api/live-interviews/{sid}/audio/final",
                data={
                    "recording_id": str(uuid4()),
                    "expected_audio_revision": "0",
                },
                files={"file": ("s.wav", _FAKE_WAV, "audio/wav")},
                headers=_auth(bob),
            ).status_code
            == 404
        )
        assert client.get(f"/api/live-interviews/{sid}/audio", headers=_auth(bob)).status_code == 404


def test_audio_requires_consent(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        token = _register(client, "alice")
        sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
        client.post(
            f"/api/live-interviews/{sid}/start",
            json={"consent_confirmed": False, "expected_revision": 0, "operation_id": str(uuid4())},
            headers=_auth(token),
        )
        resp = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(uuid4()), "expected_audio_revision": "0"},
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        assert resp.status_code == 409


def test_upload_webm_preserves_encoding(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        recording_id = uuid4()
        resp = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(recording_id), "expected_audio_revision": "0"},
            files={"file": ("session.webm", b"WEBM-FAKE", "audio/webm")},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        live = resp.json()["state"]["live_interview"]
        assert live["audio_file"] == f"{sid}-{recording_id}.webm"
        dl = client.get(f"/api/live-interviews/{sid}/audio", headers=_auth(token))
        assert dl.status_code == 200
        assert dl.content == b"WEBM-FAKE"
        assert dl.headers["content-type"] == "audio/webm"


def test_latest_audio_download_never_serves_an_older_part_encoding(tmp_path):
    recordings_dir = tmp_path / "recordings"
    with TestClient(_make_app(tmp_path, recordings_dir)) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        first_id = uuid4()
        first = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(first_id), "expected_audio_revision": "0"},
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        assert first.status_code == 200

        replacement = b"WEBM-NEWEST"
        second_id = uuid4()
        second = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(second_id), "expected_audio_revision": "0"},
            files={"file": ("session.webm", replacement, "audio/webm")},
            headers=_auth(token),
        )
        assert second.status_code == 200
        assert (recordings_dir / f"{sid}-{first_id}.wav").exists()
        assert (recordings_dir / f"{sid}-{second_id}.webm").exists()

        downloaded = client.get(f"/api/live-interviews/{sid}/audio", headers=_auth(token))
        assert downloaded.content == replacement
        assert downloaded.headers["content-type"] == "audio/webm"


def test_segmented_audio_is_idempotent_downloadable_and_deletable(tmp_path):
    recordings_dir = tmp_path / "recordings"
    with TestClient(_make_app(tmp_path, recordings_dir)) as client:
        token = _register(client, "alice")
        headers = _auth(token)
        sid = _start_consented_session(client, token)
        first_id = uuid4()
        second_id = uuid4()

        first = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(first_id), "expected_audio_revision": "0"},
            files={"file": ("session.webm", b"FIRST-PART", "audio/webm")},
            headers=headers,
        )
        assert first.status_code == 200
        first_state = first.json()["state"]["live_interview"]
        assert [part["id"] for part in first_state["audio_parts"]] == [str(first_id)]
        assert len(first_state["audio_parts"][0]["payload_sha256"]) == 64

        second = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(second_id), "expected_audio_revision": "0"},
            files={"file": ("session.m4a", b"SECOND-PART", "audio/mp4")},
            headers=headers,
        )
        assert second.status_code == 200
        parts = second.json()["state"]["live_interview"]["audio_parts"]
        assert [part["id"] for part in parts] == [str(first_id), str(second_id)]
        assert all((recordings_dir / part["audio_file"]).is_file() for part in parts)

        # A lost HTTP response can be retried with the same immutable part ID.
        retry = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(first_id), "expected_audio_revision": "0"},
            files={"file": ("session.webm", b"FIRST-PART", "audio/webm")},
            headers=headers,
        )
        assert retry.status_code == 200
        assert len(retry.json()["state"]["live_interview"]["audio_parts"]) == 2

        first_download = client.get(
            f"/api/live-interviews/{sid}/audio/{first_id}", headers=headers
        )
        assert first_download.status_code == 200
        assert first_download.content == b"FIRST-PART"
        latest_download = client.get(f"/api/live-interviews/{sid}/audio", headers=headers)
        assert latest_download.content == b"SECOND-PART"

        conflict = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(first_id), "expected_audio_revision": "0"},
            files={"file": ("session.webm", b"OTHER-PART", "audio/webm")},
            headers=headers,
        )
        assert conflict.status_code == 409

        deleted = client.request(
            "DELETE",
            f"/api/live-interviews/{sid}/audio",
            json={"expected_audio_revision": 0, "operation_id": str(uuid4())},
            headers=headers,
        )
        assert deleted.status_code == 200
        deleted_live = deleted.json()["state"]["live_interview"]
        assert deleted_live["audio_parts"] == []
        assert deleted_live["audio_file"] == ""
        assert not list(recordings_dir.glob(f"{sid}*"))


def test_lost_delete_response_retry_cannot_erase_a_newer_recording_part(tmp_path):
    recordings_dir = tmp_path / "recordings"
    with TestClient(_make_app(tmp_path, recordings_dir)) as client:
        token = _register(client, "alice")
        headers = _auth(token)
        sid = _start_consented_session(client, token)
        first_id = uuid4()
        second_id = uuid4()
        delete_operation = uuid4()

        first = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(first_id), "expected_audio_revision": "0"},
            files={"file": ("first.webm", b"FIRST", "audio/webm")},
            headers=headers,
        )
        deleted = client.request(
            "DELETE",
            f"/api/live-interviews/{sid}/audio",
            json={
                "expected_audio_revision": 0,
                "operation_id": str(delete_operation),
            },
            headers=headers,
        )
        second = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(second_id), "expected_audio_revision": "1"},
            files={"file": ("second.webm", b"SECOND", "audio/webm")},
            headers=headers,
        )
        retry = client.request(
            "DELETE",
            f"/api/live-interviews/{sid}/audio",
            json={
                "expected_audio_revision": 0,
                "operation_id": str(delete_operation),
            },
            headers=headers,
        )
        stale_new_delete = client.request(
            "DELETE",
            f"/api/live-interviews/{sid}/audio",
            json={
                "expected_audio_revision": 0,
                "operation_id": str(uuid4()),
            },
            headers=headers,
        )

        assert first.status_code == deleted.status_code == second.status_code == 200
        assert retry.status_code == 200
        assert stale_new_delete.status_code == 409
        live = retry.json()["state"]["live_interview"]
        assert live["audio_archive_revision"] == 1
        assert [part["id"] for part in live["audio_parts"]] == [str(second_id)]
        assert (recordings_dir / f"{sid}-{second_id}.webm").read_bytes() == b"SECOND"


def test_finalized_utterance_retry_is_idempotent_and_rejects_conflicts(tmp_path):
    calls = 0

    def asr_handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"text": "候选人给出了具体行动和量化结果。"})

    asr = ASRClient(
        base_url="http://asr.test:8003/v1",
        transport=httpx.MockTransport(asr_handler),
    )
    app = create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'utterance.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        asr_client=asr,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
        recordings_dir=tmp_path / "recordings",
    )
    with TestClient(app) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        recording_id = uuid4()
        registered = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{recording_id}",
            json=_capture_payload(client, token),
            headers=_auth(token),
        )
        assert registered.status_code == 200
        request = {
            "data": {
                "speaker": "candidate",
                "language": "zh",
                "recording_id": str(recording_id),
            },
            "files": {"file": ("answer.webm", b"SAME-AUDIO", "audio/webm")},
            "headers": _auth(token),
        }

        first = client.post(f"/api/live-interviews/{sid}/audio", **request)
        retry = client.post(f"/api/live-interviews/{sid}/audio", **request)
        conflict = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={**request["data"], "recording_id": str(recording_id)},
            files={"file": ("answer.webm", b"DIFFERENT-AUDIO", "audio/webm")},
            headers=_auth(token),
        )

        assert first.status_code == 200
        assert retry.status_code == 200
        assert conflict.status_code == 409
        live = retry.json()["state"]["live_interview"]
        assert calls == 1
        assert len(live["segments"]) == 1
        assert [item["id"] for item in live["audio_capture_receipts"]] == [
            str(recording_id)
        ]
        registered_again = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{recording_id}",
            json=_capture_payload(client, token),
            headers=_auth(token),
        )
        assert registered_again.status_code == 200
        assert registered_again.json()["state"]["live_interview"][
            "pending_audio_capture_ids"
        ] == []


@pytest.mark.parametrize("cancelled_caller", ["leader", "follower"])
async def test_live_audio_singleflight_survives_waiter_cancellation(
    tmp_path, cancelled_caller
):
    class BlockingASR:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def transcribe(self, content, **kwargs):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return "共享语音任务完成后的唯一转写。"

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'singleflight.db'}")
    await storage.init_db()
    asr = BlockingASR()
    service = InterviewService(
        storage,
        llm_client=None,
        asr_client=asr,
        recordings_dir=tmp_path / "recordings",
    )
    sid, _ = await service.create_session()
    await service.start_live_interview(
        sid,
        consent_confirmed=True,
        expected_revision=0,
        operation_id=uuid4(),
    )
    recording_id = uuid4()
    await service.register_live_audio_capture(
        sid,
        recording_id,
        expected_capture_epoch=1,
        expected_settings_revision=service.audio_settings_revision,
        expected_settings_etag=service.audio_settings_etag,
    )
    call = {
        "content": b"SAME-AUDIO",
        "filename": "answer.webm",
        "content_type": "audio/webm",
        "speaker": TranscriptSpeaker.CANDIDATE,
        "recording_id": recording_id,
    }
    leader = asyncio.create_task(service.transcribe_live_audio(sid, **call))
    await asr.started.wait()
    follower = asyncio.create_task(service.transcribe_live_audio(sid, **call))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    victim = leader if cancelled_caller == "leader" else follower
    survivor = follower if cancelled_caller == "leader" else leader
    victim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await victim

    assert asr.calls == 1
    assert service._active_audio_settings_leases == 1
    assert len(service._live_audio_in_flight) == 1
    asr.release.set()
    state, transcript = await survivor
    await asyncio.sleep(0)

    assert transcript == "共享语音任务完成后的唯一转写。"
    assert asr.calls == 1
    assert service._active_audio_settings_leases == 0
    assert service._live_audio_in_flight == {}
    assert [segment.text for segment in state.live_interview.segments] == [transcript]
    assert [receipt.id for receipt in state.live_interview.audio_capture_receipts] == [
        recording_id
    ]
    await storage.close()


async def test_service_shutdown_cancels_detached_live_audio_before_clients_close(
    tmp_path,
):
    class BlockingASR:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def transcribe(self, content, **kwargs):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'shutdown-audio.db'}")
    await storage.init_db()
    asr = BlockingASR()
    service = InterviewService(
        storage,
        llm_client=None,
        asr_client=asr,
        recordings_dir=tmp_path / "recordings",
    )
    sid, _ = await service.create_session()
    await service.start_live_interview(
        sid, consent_confirmed=True, expected_revision=0, operation_id=uuid4()
    )
    recording_id = uuid4()
    await service.register_live_audio_capture(
        sid,
        recording_id,
        expected_capture_epoch=1,
        expected_settings_revision=service.audio_settings_revision,
        expected_settings_etag=service.audio_settings_etag,
    )
    caller = asyncio.create_task(
        service.transcribe_live_audio(
            sid,
            content=b"AUDIO",
            filename="answer.webm",
            content_type="audio/webm",
            speaker=TranscriptSpeaker.CANDIDATE,
            recording_id=recording_id,
        )
    )
    await asr.started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert service._active_audio_settings_leases == 1
    await service.shutdown_runtime_tasks()

    assert asr.cancelled.is_set()
    assert service._live_audio_in_flight == {}
    assert service._active_audio_settings_leases == 0
    await storage.close()


def test_cancelled_capture_cannot_be_revived_by_late_registration_or_upload(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:8003/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"text": "不应写入的延迟音频"})
        ),
    )
    app = create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'cancelled-capture.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        asr_client=asr,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
        recordings_dir=tmp_path / "recordings",
    )
    with TestClient(app) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        recording_id = uuid4()

        cancelled = client.delete(
            f"/api/live-interviews/{sid}/audio/captures/{recording_id}",
            headers=_auth(token),
        )
        late_registration = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{recording_id}",
            json=_capture_payload(client, token),
            headers=_auth(token),
        )
        late_upload = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={
                "speaker": "candidate",
                "language": "zh",
                "recording_id": str(recording_id),
            },
            files={"file": ("answer.webm", b"LATE-AUDIO", "audio/webm")},
            headers=_auth(token),
        )

        assert cancelled.status_code == 200
        assert late_registration.status_code == 409
        assert late_upload.status_code == 409
        live = client.get(
            f"/api/live-interviews/{sid}", headers=_auth(token)
        ).json()["state"]["live_interview"]
        assert live["segments"] == []
        assert live["pending_audio_capture_ids"] == []


def test_finalized_utterance_can_drain_after_another_client_completes_session(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:8003/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"text": "结束前录到的最后一句。"})
        ),
    )
    app = create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'completed-drain.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        asr_client=asr,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
        recordings_dir=tmp_path / "recordings",
    )
    with TestClient(app) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        recording_id = uuid4()
        registered = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{recording_id}",
            json=_capture_payload(client, token),
            headers=_auth(token),
        )
        assert registered.status_code == 200
        client.post(
            f"/api/live-interviews/{sid}/status",
            json={"status": "completed", "expected_revision": 1},
            headers=_auth(token),
        )

        legacy = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={"speaker": "candidate", "language": "zh"},
            files={"file": ("answer.webm", b"LATE-AUDIO", "audio/webm")},
            headers=_auth(token),
        )
        late_registration = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{uuid4()}",
            json=_capture_payload(client, token),
            headers=_auth(token),
        )
        unregistered = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={
                "speaker": "candidate",
                "language": "zh",
                "recording_id": str(uuid4()),
            },
            files={"file": ("answer.webm", b"LATE-AUDIO", "audio/webm")},
            headers=_auth(token),
        )
        finalized = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={
                "speaker": "candidate",
                "language": "zh",
                "recording_id": str(recording_id),
            },
            files={"file": ("answer.webm", b"LATE-AUDIO", "audio/webm")},
            headers=_auth(token),
        )

        assert legacy.status_code == 422
        assert late_registration.status_code == 409
        assert unregistered.status_code == 409
        assert finalized.status_code == 200
        live = finalized.json()["state"]["live_interview"]
        assert live["status"] == "completed"
        assert live["pending_audio_capture_ids"] == []
        assert live["segments"][-1]["text"] == "结束前录到的最后一句。"


def test_paused_session_accepts_only_a_pre_registered_utterance(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:8003/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"text": "暂停前的最后一句。"})
        ),
    )
    app = create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'paused-drain.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        asr_client=asr,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
        recordings_dir=tmp_path / "recordings",
    )
    with TestClient(app) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        recording_id = uuid4()
        assert (
            client.post(
                f"/api/live-interviews/{sid}/audio/captures/{recording_id}",
                json=_capture_payload(client, token),
                headers=_auth(token),
            ).status_code
            == 200
        )
        assert (
            client.post(
                f"/api/live-interviews/{sid}/status",
                json={"status": "paused", "expected_revision": 1},
                headers=_auth(token),
            ).status_code
            == 200
        )

        missing_id = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={"speaker": "candidate", "language": "zh"},
            files={"file": ("answer.webm", b"NEW-AUDIO", "audio/webm")},
            headers=_auth(token),
        )
        unknown_id = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={
                "speaker": "candidate",
                "language": "zh",
                "recording_id": str(uuid4()),
            },
            files={"file": ("answer.webm", b"NEW-AUDIO", "audio/webm")},
            headers=_auth(token),
        )
        registered = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={
                "speaker": "candidate",
                "language": "zh",
                "recording_id": str(recording_id),
            },
            files={"file": ("answer.webm", b"FINAL-AUDIO", "audio/webm")},
            headers=_auth(token),
        )

        assert missing_id.status_code == 422
        assert unknown_id.status_code == 409
        assert registered.status_code == 200
        assert registered.json()["state"]["live_interview"]["segments"][-1][
            "text"
        ] == "暂停前的最后一句。"


async def test_segment_retry_heals_failed_state_persistence(tmp_path, monkeypatch):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'audio-persist.db'}")
    await storage.init_db()
    recordings_dir = tmp_path / "recordings"
    service = InterviewService(storage, llm_client=None, recordings_dir=recordings_dir)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    original_persist = service._persist
    attempts = 0

    async def fail_once(session_id, state):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated database write failure")
        await original_persist(session_id, state)

    monkeypatch.setattr(service, "_persist", fail_once)
    part_id = uuid4()
    with pytest.raises(OSError, match="simulated"):
        await service.save_live_audio(
            sid,
            b"RECOVERABLE",
            extension="webm",
            recording_id=part_id,
            expected_audio_revision=0,
        )

    state = await service.save_live_audio(
        sid,
        b"RECOVERABLE",
        extension="webm",
        recording_id=part_id,
        expected_audio_revision=0,
    )
    assert attempts == 2
    assert [part.id for part in state.live_interview.audio_parts] == [part_id]
    persisted = await storage.get_session_state(sid)
    assert persisted is not None
    assert persisted["live_interview"]["audio_parts"][0]["id"] == str(part_id)
    await storage.close()


async def test_audio_delete_revision_fences_a_delayed_whole_session_upload(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'audio-delete-fence.db'}")
    await storage.init_db()
    recordings_dir = tmp_path / "recordings"
    service = InterviewService(storage, llm_client=None, recordings_dir=recordings_dir)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    stale_part = uuid4()

    deleted = await service.delete_live_audio(
        sid, expected_audio_revision=0, operation_id=uuid4()
    )
    assert deleted.live_interview.audio_archive_revision == 1
    with pytest.raises(LiveInterviewStateError, match="另一窗口变更"):
        await service.save_live_audio(
            sid,
            b"STALE-PART",
            extension="webm",
            recording_id=stale_part,
            expected_audio_revision=0,
        )
    assert not list(recordings_dir.glob(f"{sid}*"))

    fresh_part = uuid4()
    saved = await service.save_live_audio(
        sid,
        b"FRESH-PART",
        extension="webm",
        recording_id=fresh_part,
        expected_audio_revision=1,
    )
    assert [part.id for part in saved.live_interview.audio_parts] == [fresh_part]
    await storage.close()


def test_guard_cannot_chain_or_renew_after_live_session_closes(tmp_path):
    app = create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'guard-drain.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        asr_client=ASRClient(
            base_url="http://asr.test:8003/v1",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"text": "结束竞态中的回答。"})
            ),
        ),
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
        recordings_dir=tmp_path / "recordings",
    )
    with TestClient(app) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        guard_id = uuid4()
        guard = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{guard_id}",
            json=_capture_payload(client, token, capture_role="guard"),
            headers=_auth(token),
        )
        assert guard.status_code == 200
        registered_at = guard.json()["state"]["live_interview"][
            "pending_audio_capture_registered_at"
        ][str(guard_id)]
        completed = client.post(
            f"/api/live-interviews/{sid}/status",
            json={"status": "completed", "expected_revision": 1},
            headers=_auth(token),
        )
        assert completed.status_code == 200

        renewed = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{guard_id}",
            json=_capture_payload(client, token, capture_role="guard"),
            headers=_auth(token),
        )
        assert renewed.status_code == 200
        assert renewed.json()["state"]["live_interview"][
            "pending_audio_capture_registered_at"
        ][str(guard_id)] == registered_at

        chained_guard = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{uuid4()}",
            json=_capture_payload(
                client,
                token,
                capture_role="guard",
                parent_recording_id=str(guard_id),
            ),
            headers=_auth(token),
        )
        assert chained_guard.status_code == 409

        child_id = uuid4()
        child = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{child_id}",
            json=_capture_payload(
                client,
                token,
                capture_role="utterance",
                parent_recording_id=str(guard_id),
            ),
            headers=_auth(token),
        )
        assert child.status_code == 200
        uploaded = client.post(
            f"/api/live-interviews/{sid}/audio",
            data={
                "speaker": "candidate",
                "language": "zh",
                "recording_id": str(child_id),
            },
            files={"file": ("answer.webm", b"RACE-AUDIO", "audio/webm")},
            headers=_auth(token),
        )
        assert uploaded.status_code == 200


async def test_expired_guard_does_not_authorize_or_block_audio_settings(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'expired-guard.db'}")
    await storage.init_db()
    service = InterviewService(storage, llm_client=None)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    guard_id = uuid4()
    await service.register_live_audio_capture(
        sid, guard_id, expected_capture_epoch=1, capture_role="guard"
    )
    await service.set_live_interview_status(sid, LiveInterviewStatus.PAUSED)
    runtime = await service._get_runtime(sid)
    runtime.state.live_interview.capture_closed_at = datetime.now(timezone.utc) - timedelta(
        minutes=1
    )
    await service._persist(sid, runtime.state)

    with pytest.raises(LiveInterviewStateError, match="must be active"):
        await service.register_live_audio_capture(
            sid,
            uuid4(),
            expected_capture_epoch=1,
            parent_recording_id=guard_id,
            capture_role="utterance",
        )
    await service.ensure_audio_settings_mutable()
    current = await service.get_state(sid)
    assert guard_id not in current.live_interview.pending_audio_capture_ids
    await storage.close()


def test_capture_epoch_rejects_a_delayed_guard_from_an_earlier_active_run(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        old_guard_id = uuid4()
        registered = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{old_guard_id}",
            json=_capture_payload(client, token, capture_role="guard"),
            headers=_auth(token),
        )
        assert registered.status_code == 200
        paused = client.post(
            f"/api/live-interviews/{sid}/status",
            json={"status": "paused", "expected_revision": 1},
            headers=_auth(token),
        ).json()
        resumed = client.post(
            f"/api/live-interviews/{sid}/status",
            json={
                "status": "active",
                "expected_revision": paused["state"]["live_interview"]["status_revision"],
                "operation_id": str(uuid4()),
            },
            headers=_auth(token),
        )
        assert resumed.json()["state"]["live_interview"]["capture_epoch"] == 2

        delayed = client.post(
            f"/api/live-interviews/{sid}/audio/captures/{uuid4()}",
            json=_capture_payload(client, token, capture_role="guard"),
            headers=_auth(token),
        )
        assert delayed.status_code == 409
        live = client.get(
            f"/api/live-interviews/{sid}", headers=_auth(token)
        ).json()["state"]["live_interview"]
        assert old_guard_id.hex not in live["audio_capture_guard_ids"]
        assert str(old_guard_id) not in live["audio_capture_guard_ids"]


async def test_live_audio_delete_restores_files_when_manifest_persist_fails(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'delete-rollback.db'}")
    await storage.init_db()
    recordings_dir = tmp_path / "recordings"
    service = InterviewService(storage, llm_client=None, recordings_dir=recordings_dir)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    part_id = uuid4()
    await service.save_live_audio(
        sid,
        b"RECOVERABLE-AUDIO",
        extension="webm",
        recording_id=part_id,
        expected_audio_revision=0,
    )
    original = recordings_dir / f"{sid}-{part_id}.webm"
    original_persist = service._persist

    async def fail_delete_persist(session_id, state):
        if not state.live_interview.audio_parts:
            raise OSError("simulated database failure")
        await original_persist(session_id, state)

    monkeypatch.setattr(service, "_persist", fail_delete_persist)
    with pytest.raises(OSError, match="simulated"):
        await service.delete_live_audio(
            sid, expected_audio_revision=0, operation_id=uuid4()
        )

    assert original.read_bytes() == b"RECOVERABLE-AUDIO"
    restored = await service.get_state(sid)
    assert [part.id for part in restored.live_interview.audio_parts] == [part_id]
    assert restored.live_interview.audio_archive_revision == 0
    assert not list(recordings_dir.glob("*.delete-pending"))
    await storage.close()


async def test_live_audio_delete_completes_when_persist_committed_before_error(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'delete-committed.db'}")
    await storage.init_db()
    recordings_dir = tmp_path / "recordings"
    service = InterviewService(storage, llm_client=None, recordings_dir=recordings_dir)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    part_id = uuid4()
    await service.save_live_audio(
        sid,
        b"DELETE-ME",
        extension="webm",
        recording_id=part_id,
        expected_audio_revision=0,
    )
    original_persist = service._persist

    async def commit_then_fail(session_id, state):
        await original_persist(session_id, state)
        if not state.live_interview.audio_parts:
            raise OSError("response lost after commit")

    monkeypatch.setattr(service, "_persist", commit_then_fail)
    deleted = await service.delete_live_audio(
        sid, expected_audio_revision=0, operation_id=uuid4()
    )

    assert deleted.live_interview.audio_archive_revision == 1
    assert deleted.live_interview.audio_parts == []
    assert not list(recordings_dir.iterdir())
    await storage.close()


async def test_segmented_audio_write_removes_orphan_when_manifest_persist_fails(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'part-rollback.db'}")
    await storage.init_db()
    recordings_dir = tmp_path / "recordings"
    service = InterviewService(storage, llm_client=None, recordings_dir=recordings_dir)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)

    async def fail_persist(session_id, state):
        raise OSError("manifest unavailable")

    monkeypatch.setattr(service, "_persist", fail_persist)
    with pytest.raises(OSError, match="manifest unavailable"):
        await service.save_live_audio(
            sid,
            b"UNCOMMITTED-PART",
            extension="webm",
            recording_id=uuid4(),
            expected_audio_revision=0,
        )

    assert not list(recordings_dir.iterdir())
    assert (await service.get_state(sid)).live_interview.audio_parts == []
    await storage.close()


async def test_legacy_audio_overwrite_restores_original_on_manifest_failure(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'legacy-rollback.db'}")
    await storage.init_db()
    recordings_dir = tmp_path / "recordings"
    service = InterviewService(storage, llm_client=None, recordings_dir=recordings_dir)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    await service.save_live_audio(
        sid, b"ORIGINAL", extension="wav", expected_audio_revision=0
    )

    async def fail_persist(session_id, state):
        raise OSError("manifest unavailable")

    monkeypatch.setattr(service, "_persist", fail_persist)
    with pytest.raises(OSError, match="manifest unavailable"):
        await service.save_live_audio(
            sid, b"REPLACEMENT", extension="wav", expected_audio_revision=0
        )

    assert (recordings_dir / f"{sid}.wav").read_bytes() == b"ORIGINAL"
    assert not list(recordings_dir.glob(".*.replace-pending"))
    await storage.close()


async def test_audio_capture_registration_failure_does_not_publish_ghost_state(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'register-cow.db'}")
    await storage.init_db()
    service = InterviewService(storage, llm_client=None)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    recording_id = uuid4()

    async def fail_persist(session_id, state):
        raise OSError("registration unavailable")

    monkeypatch.setattr(service, "_persist", fail_persist)
    with pytest.raises(OSError, match="registration unavailable"):
        await service.register_live_audio_capture(
            sid, recording_id, expected_capture_epoch=1
        )

    current = await service.get_state(sid)
    assert recording_id not in current.live_interview.pending_audio_capture_ids
    await storage.close()


async def test_audio_capture_cancel_failure_keeps_durable_pending_lease(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'cancel-cow.db'}")
    await storage.init_db()
    service = InterviewService(storage, llm_client=None)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    recording_id = uuid4()
    await service.register_live_audio_capture(
        sid, recording_id, expected_capture_epoch=1
    )

    async def fail_persist(session_id, state):
        raise OSError("cancellation unavailable")

    monkeypatch.setattr(service, "_persist", fail_persist)
    with pytest.raises(OSError, match="cancellation unavailable"):
        await service.cancel_live_audio_capture(sid, recording_id)

    current = await service.get_state(sid)
    assert recording_id in current.live_interview.pending_audio_capture_ids
    assert recording_id not in current.live_interview.cancelled_audio_capture_ids
    await storage.close()


async def test_audio_capture_commit_failure_keeps_transcript_and_receipt_unpublished(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'commit-cow.db'}")
    await storage.init_db()
    service = InterviewService(storage, llm_client=None)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    recording_id = uuid4()
    await service.register_live_audio_capture(
        sid, recording_id, expected_capture_epoch=1
    )

    async def fail_persist(session_id, state):
        raise OSError("transcript unavailable")

    monkeypatch.setattr(service, "_persist", fail_persist)
    with pytest.raises(OSError, match="transcript unavailable"):
        await service._commit_live_audio_capture(
            sid,
            await service._get_runtime(sid),
            segments=[(TranscriptSpeaker.CANDIDATE, "尚未提交的回答", "asr")],
            receipt=LiveAudioCaptureReceipt(
                id=recording_id, payload_sha256="f" * 64
            ),
        )

    current = await service.get_state(sid)
    assert current.live_interview.segments == []
    assert current.live_interview.audio_capture_receipts == []
    assert recording_id in current.live_interview.pending_audio_capture_ids
    await storage.close()


async def test_capture_arriving_before_pause_waits_through_drain_lock(
    tmp_path, monkeypatch
):
    import interview_os.services.media_service as media_service_module

    monkeypatch.setattr(
        media_service_module, "LIVE_AUDIO_CAPTURE_DRAIN_GRACE_SECONDS", 0.05
    )
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'arrival-drain.db'}")
    await storage.init_db()
    service = InterviewService(storage, llm_client=None)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    guard_id = uuid4()
    await service.register_live_audio_capture(
        sid, guard_id, expected_capture_epoch=1, capture_role="guard"
    )
    pause_persist_started = asyncio.Event()
    release_pause_persist = asyncio.Event()
    original_persist = service._persist
    blocked_once = False

    async def block_pause_persist(session_id, state):
        nonlocal blocked_once
        if state.live_interview.status == LiveInterviewStatus.PAUSED and not blocked_once:
            blocked_once = True
            pause_persist_started.set()
            await release_pause_persist.wait()
        await original_persist(session_id, state)

    monkeypatch.setattr(service, "_persist", block_pause_persist)
    pause_task = asyncio.create_task(
        service.set_live_interview_status(sid, LiveInterviewStatus.PAUSED)
    )
    await pause_persist_started.wait()
    child_id = uuid4()
    child_task = asyncio.create_task(
        service.register_live_audio_capture(
            sid,
            child_id,
            expected_capture_epoch=1,
            parent_recording_id=guard_id,
        )
    )
    await asyncio.sleep(0.08)
    release_pause_persist.set()
    await pause_task
    registered = await child_task

    assert child_id in registered.live_interview.pending_audio_capture_ids
    await storage.close()


async def test_stale_audio_generation_leases_are_pruned_before_pending_cap(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'stale-generation.db'}")
    await storage.init_db()
    service = InterviewService(storage, llm_client=None)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    runtime = await service._get_runtime(sid)
    live = runtime.state.live_interview
    now = datetime.now(timezone.utc)
    for _ in range(256):
        capture_id = uuid4()
        live.pending_audio_capture_ids.append(capture_id)
        live.pending_audio_capture_registered_at[str(capture_id)] = now
        live.pending_audio_capture_settings_etags[str(capture_id)] = "retired-etag"
        live.pending_audio_capture_epochs[str(capture_id)] = live.capture_epoch
    await service._persist(sid, runtime.state)

    fresh_id = uuid4()
    state = await service.register_live_audio_capture(
        sid,
        fresh_id,
        expected_capture_epoch=1,
        expected_settings_etag=service.audio_settings_etag,
    )

    assert state.live_interview.pending_audio_capture_ids == [fresh_id]
    await storage.close()


async def test_runtime_recovery_reconciles_uncertain_audio_replacement(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'journal-recovery.db'}")
    await storage.init_db()
    recordings_dir = tmp_path / "recordings"
    service = InterviewService(storage, llm_client=None, recordings_dir=recordings_dir)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    await service.save_live_audio(
        sid, b"DURABLE-OLD", extension="wav", expected_audio_revision=0
    )
    original_get = storage.get_session_state
    recovery_read_pending = True

    async def fail_persist(session_id, state):
        raise OSError("commit result unknown")

    async def fail_first_recovery_read(*args, **kwargs):
        nonlocal recovery_read_pending
        if recovery_read_pending:
            recovery_read_pending = False
            raise OSError("database temporarily unavailable")
        return await original_get(*args, **kwargs)

    monkeypatch.setattr(service, "_persist", fail_persist)
    monkeypatch.setattr(storage, "get_session_state", fail_first_recovery_read)
    with pytest.raises(OSError, match="commit result unknown"):
        await service.save_live_audio(
            sid, b"UNCOMMITTED-NEW", extension="wav", expected_audio_revision=0
        )

    current = await service.get_state(sid)
    assert current.live_interview.audio_payload_sha256
    assert (recordings_dir / f"{sid}.wav").read_bytes() == b"DURABLE-OLD"
    assert not list(recordings_dir.glob(".*.replace-pending"))
    await storage.close()


async def test_runtime_recovery_restores_uncertain_audio_delete(tmp_path, monkeypatch):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'delete-journal.db'}")
    await storage.init_db()
    recordings_dir = tmp_path / "recordings"
    service = InterviewService(storage, llm_client=None, recordings_dir=recordings_dir)
    sid, _ = await service.create_session()
    await service.start_live_interview(sid, consent_confirmed=True)
    part_id = uuid4()
    await service.save_live_audio(
        sid,
        b"DURABLE-PART",
        extension="webm",
        recording_id=part_id,
        expected_audio_revision=0,
    )
    original_get = storage.get_session_state
    recovery_read_pending = True

    async def fail_persist(session_id, state):
        raise OSError("delete result unknown")

    async def fail_first_recovery_read(*args, **kwargs):
        nonlocal recovery_read_pending
        if recovery_read_pending:
            recovery_read_pending = False
            raise OSError("database temporarily unavailable")
        return await original_get(*args, **kwargs)

    monkeypatch.setattr(service, "_persist", fail_persist)
    monkeypatch.setattr(storage, "get_session_state", fail_first_recovery_read)
    with pytest.raises(OSError, match="delete result unknown"):
        await service.delete_live_audio(
            sid, expected_audio_revision=0, operation_id=uuid4()
        )

    current = await service.get_state(sid)
    assert [part.id for part in current.live_interview.audio_parts] == [part_id]
    assert (recordings_dir / f"{sid}-{part_id}.webm").read_bytes() == b"DURABLE-PART"
    assert not list(recordings_dir.glob(".*.delete-pending"))
    await storage.close()


def test_m4a_download_uses_mp4_audio_media_type(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        recording_id = uuid4()
        uploaded = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            data={"recording_id": str(recording_id), "expected_audio_revision": "0"},
            files={"file": ("session.m4a", b"M4A-FAKE", "audio/mp4")},
            headers=_auth(token),
        )
        assert uploaded.status_code == 200
        downloaded = client.get(f"/api/live-interviews/{sid}/audio", headers=_auth(token))
        assert downloaded.content == b"M4A-FAKE"
        assert downloaded.headers["content-type"] == "audio/mp4"


def test_audio_invalid_session_id_422(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        token = _register(client, "alice")
        # A non-UUID session id reaches the route and is rejected by the regex.
        resp = client.post(
            "/api/live-interviews/not-a-uuid/audio/final",
            data={"recording_id": str(uuid4()), "expected_audio_revision": "0"},
            files={"file": ("s.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        assert resp.status_code == 422
