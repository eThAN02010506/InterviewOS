"""Whole-session audio recording tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.database.storage import Storage
from interview_os.services.settings_service import LocalSettingsStore


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
        json={"consent_confirmed": True},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    return sid


def test_upload_and_download_session_audio(tmp_path):
    recordings_dir = tmp_path / "recordings"
    with TestClient(_make_app(tmp_path, recordings_dir)) as client:
        token = _register(client, "alice")
        sid = _start_consented_session(client, token)
        resp = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        live = resp.json()["state"]["live_interview"]
        assert live["audio_file"] == f"{sid}.wav"
        assert live["audio_size_bytes"] == len(_FAKE_WAV)
        # File exists on disk under the recordings dir
        assert (recordings_dir / f"{sid}.wav").exists()
        # Download returns the bytes
        dl = client.get(f"/api/live-interviews/{sid}/audio", headers=_auth(token))
        assert dl.status_code == 200
        assert dl.content == _FAKE_WAV
        assert dl.headers["content-type"] == "audio/wav"


def test_audio_cross_account_404(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        alice = _register(client, "alice")
        bob = _register(client, "bob")
        sid = _start_consented_session(client, alice)
        resp = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(alice),
        )
        assert resp.status_code == 200
        # Bob cannot upload to nor download Alice's session
        assert (
            client.post(
                f"/api/live-interviews/{sid}/audio/final",
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
            json={"consent_confirmed": False},
            headers=_auth(token),
        )
        resp = client.post(
            f"/api/live-interviews/{sid}/audio/final",
            files={"file": ("session.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        assert resp.status_code == 409


def test_audio_invalid_session_id_422(tmp_path):
    with TestClient(_make_app(tmp_path, tmp_path / "recordings")) as client:
        token = _register(client, "alice")
        # A non-UUID session id reaches the route and is rejected by the regex.
        resp = client.post(
            "/api/live-interviews/not-a-uuid/audio/final",
            files={"file": ("s.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        assert resp.status_code == 422
