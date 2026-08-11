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


class _OmniFallback:
    mode = "asr_text"

    async def analyze_speaking_style(self, content, **kwargs):
        return {}

    def status(self):
        return {"enabled": True, "mode": self.mode}

    def secret_snapshot(self):
        return {"mode": self.mode}

    def configure(self, **kwargs):
        self.mode = kwargs.get("mode", self.mode)

    async def close(self):
        return None


class _OmniProhibited(_OmniFallback):
    mode = "audio"

    async def analyze_speaking_style(self, content, **kwargs):
        return {
            "pace": "语速平稳",
            "clarity": "可以判断候选人的口音和性格",
            "improvements": ["根据年龄调整表达方式"],
        }


class _FakeTTS:
    def __init__(self):
        self.spoken = []

    async def synthesize(self, text):
        self.spoken.append(text)
        return b"RIFF-fake-question-audio", "audio/wav"

    def status(self):
        return {"enabled": True, "model": "fake-tts"}

    def secret_snapshot(self):
        return {"model": "fake-tts", "api_key": ""}

    def configure(self, **kwargs):
        return None

    async def close(self):
        return None


def _register(client: TestClient, username: str = "alice") -> str:
    resp = client.post(
        "/api/auth/register", json={"username": username, "password": "pw-123456"}
    )
    return resp.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _make_app(tmp_path, asr, *, tts=None):
    return create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'mock-voice.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        asr_client=asr,
        omni_client=_OmniFallback(),
        tts_client=tts,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
    )


def _make_app_with_omni(tmp_path, asr, omni):
    return create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'mock-voice-omni.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        asr_client=asr,
        omni_client=omni,
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
        assert resp.json()["speech_feedback"]["source"] == "text_fallback"
        assert "不进入" in resp.json()["speech_feedback"]["disclaimer"]
        # Pure transcription: no answer was submitted, transcription is not stored.
        state = client.get(f"/api/interviews/sessions/{sid}", headers=_auth(token)).json()["state"]
        assert not (state.get("mock_interview") or {}).get("answers")


def test_mock_transcribe_rejects_prohibited_audio_model_inferences(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:9001",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"text": "我先说明结论，再解释行动。"})
        ),
    )
    with TestClient(_make_app_with_omni(tmp_path, asr, _OmniProhibited())) as client:
        token = _register(client)
        sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
        resp = client.post(
            f"/api/mock-interviews/{sid}/transcribe",
            files={"file": ("answer.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )

    assert resp.status_code == 200
    feedback = resp.json()["speech_feedback"]
    assert feedback["source"] == "text_fallback"
    assert "口音" not in str(feedback)
    assert "性格" not in str(feedback)
    assert "年龄" not in str(feedback)


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


def test_asr_preview_is_provisional_and_does_not_mutate_transcript(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:9001",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"text": "这是实时草稿"})
        ),
    )
    with TestClient(_make_app(tmp_path, asr)) as client:
        token = _register(client)
        sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
        preview = client.post(
            f"/api/live-interviews/{sid}/audio/preview",
            files={"file": ("preview.wav", _FAKE_WAV, "audio/wav")},
            headers=_auth(token),
        )
        state = client.get(f"/api/interviews/sessions/{sid}", headers=_auth(token)).json()["state"]
    assert preview.status_code == 200
    assert preview.json()["text"] == "这是实时草稿"
    assert state["live_interview"]["segments"] == []


def test_current_mock_question_can_be_synthesized_but_other_id_cannot(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:9001",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"text": "x"})),
    )
    tts = _FakeTTS()
    with TestClient(_make_app(tmp_path, asr, tts=tts)) as client:
        token = _register(client)
        sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
        prepared = client.post(
            "/api/workflows/candidate-prep",
            json={
                "session_id": sid,
                "resume_text": "Python engineer",
                "job_description": "Platform Engineer\n岗位职责：设计分布式平台\n任职要求：熟悉 Python",
                "company_name": "Example",
            },
            headers=_auth(token),
        )
        assert prepared.status_code == 200
        started = client.post(f"/api/mock-interviews/{sid}/start", headers=_auth(token)).json()
        question = started["current_question"]
        spoken = client.post(
            f"/api/mock-interviews/{sid}/questions/{question['id']}/speech",
            headers=_auth(token),
        )
        denied = client.post(
            f"/api/mock-interviews/{sid}/questions/00000000-0000-0000-0000-000000000000/speech",
            headers=_auth(token),
        )
    assert spoken.status_code == 200
    assert spoken.headers["content-type"].startswith("audio/wav")
    assert tts.spoken == [question["question"]]
    assert denied.status_code == 409


def test_answered_follow_up_speech_uses_response_question(tmp_path):
    asr = ASRClient(
        base_url="http://asr.test:9001",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"text": "x"})
        ),
    )
    tts = _FakeTTS()
    with TestClient(_make_app(tmp_path, asr, tts=tts)) as client:
        token = _register(client)
        headers = _auth(token)
        sid = client.post("/api/interviews/sessions", json={}, headers=headers).json()["id"]
        client.post(
            "/api/workflows/candidate-prep",
            json={
                "session_id": sid,
                "resume_text": "Python engineer",
                "job_description": "Platform Engineer\n岗位职责：设计分布式平台\n任职要求：熟悉 Python",
                "company_name": "Example",
            },
            headers=headers,
        )
        started = client.post(f"/api/mock-interviews/{sid}/start", headers=headers).json()
        question = started["current_question"]
        client.post(
            f"/api/mock-interviews/{sid}/answers",
            json={"question_id": question["id"], "answer": "I compared two designs."},
            headers=headers,
        )
        advanced = client.post(f"/api/mock-interviews/{sid}/next", headers=headers).json()
        follow_up = advanced["current_question"]["question"]
        pending_spoken = client.post(
            f"/api/mock-interviews/{sid}/questions/{question['id']}/speech",
            headers=headers,
        )
        answered = client.post(
            f"/api/mock-interviews/{sid}/answers",
            json={"question_id": question["id"], "answer": "At ten times traffic."},
            headers=headers,
        ).json()
        response = answered["mock_session"]["responses"][-1]
        spoken = client.post(
            f"/api/mock-interviews/{sid}/questions/{question['id']}/speech",
            params={"response_id": response["id"]},
            headers=headers,
        )

    assert response["is_follow_up"] is True
    assert response["question"] == follow_up
    assert pending_spoken.status_code == 200
    assert spoken.status_code == 200
    assert tts.spoken == [follow_up, follow_up]
