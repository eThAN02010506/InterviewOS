"""Mock interview spoken-answer transcription endpoint tests."""
from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.database.storage import Storage
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


def _register(client: TestClient, username: str = "alice") -> str:
    resp = client.post(
        "/api/auth/register", json={"username": username, "password": "pw-123456"}
    )
    return resp.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _make_app(tmp_path, asr):
    return create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'mock-voice.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        asr_client=asr,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
    )


_FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + (b"\x00" * 20) + b"data\x00\x00\x00\x00"


def test_mock_transcribe_returns_text(tmp_path):
    def asr_handler(request):
        assert request.url.path == "/v1/audio/transcriptions"
        return httpx.Response(200, json={"text": "我平时用压测比较两个方案，P95 降低 40%。"})

    asr = ASRClient(
        base_url="http://asr.test:9001", transport=httpx.MockTransport(asr_handler)
    )
    with TestClient(_make_app(tmp_path, asr)) as client:
        token = _register(client)
        sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
        # Start the mock interview so the session is usable.
        client.post(f"/api/mock-interviews/{sid}/start", headers=_auth(token))
        resp = client.post(
            f"/api/mock-interviews/{sid}/transcribe",
            files={"file": ("answer.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        assert "压测" in resp.json()["text"]
        # Pure transcription: no answer was submitted, transcription is not stored.
        state = client.get(f"/api/interviews/sessions/{sid}", headers=_auth(token)).json()["state"]
        assert not (state.get("mock_interview") or {}).get("answers")


def test_mock_transcribe_empty_audio_422(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:9001", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"text": ""}))
    )
    with TestClient(_make_app(tmp_path, asr)) as client:
        token = _register(client)
        sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
        resp = client.post(
            f"/api/mock-interviews/{sid}/transcribe",
            files={"file": ("answer.wav", b"", "audio/wav")},
            headers=_auth(token),
        )
        assert resp.status_code == 422


def test_mock_transcribe_cross_account_404(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:9001", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"text": "x"}))
    )
    with TestClient(_make_app(tmp_path, asr)) as client:
        alice = _register(client, "alice")
        bob = _register(client, "bob")
        sid = client.post("/api/interviews/sessions", json={}, headers=_auth(alice)).json()["id"]
        resp = client.post(
            f"/api/mock-interviews/{sid}/transcribe",
            files={"file": ("answer.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(bob),
        )
        assert resp.status_code == 404


def test_mock_transcribe_anonymous_uses_local_owner(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:9001", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"text": "x"}))
    )
    with TestClient(_make_app(tmp_path, asr)) as client:
        # Anonymous requests fall back to the legacy 'local' owner: it can create
        # and transcribe its own session, but cannot touch another account's.
        sid = client.post("/api/interviews/sessions", json={}).json()["id"]
        resp = client.post(
            f"/api/mock-interviews/{sid}/transcribe",
            files={"file": ("answer.wav", _FAKE_WAV, "audio/wav")},
        )
        assert resp.status_code == 200
        # A real account's session is invisible to anonymous.
        token = _register(client, "bob")
        bob_sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
        assert (
            client.post(
                f"/api/mock-interviews/{bob_sid}/transcribe",
                files={"file": ("answer.wav", _FAKE_WAV, "audio/wav")},
            ).status_code
            == 404
        )