"""Transcript export endpoint tests."""
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


def _register(client: TestClient) -> str:
    resp = client.post(
        "/api/auth/register", json={"username": "alice", "password": "pw-123456"}
    )
    return resp.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _make_app(tmp_path):
    return create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'transcript.db'}"),
        llm_client=_WorkflowLLM(),
        configure_llm=False,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
    )


def _seed_live_session(client: TestClient, token: str) -> str:
    sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
    client.post(
        f"/api/live-interviews/{sid}/start", json={"consent_confirmed": True}, headers=_auth(token)
    )
    client.post(
        f"/api/live-interviews/{sid}/segments",
        json={"text": "请讲一次系统设计权衡。", "speaker": "interviewer"},
        headers=_auth(token),
    )
    answer = client.post(
        f"/api/live-interviews/{sid}/segments",
        json={"text": "我对比了缓存与数据库方案，用压测验证性能提升。", "speaker": "candidate"},
        headers=_auth(token),
    )
    answer_segment = answer.json()["state"]["live_interview"]["segments"][-1]
    client.post(
        f"/api/live-interviews/{sid}/segments/{answer_segment['id']}/evidence",
        json={"competency": "System Design"},
        headers=_auth(token),
    )
    return sid


def test_transcript_contains_full_record(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        token = _register(client)
        sid = _seed_live_session(client, token)
        resp = client.get(
            f"/api/interviews/sessions/{sid}/transcript", headers=_auth(token)
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        text = resp.text
        assert "InterviewOS 面试记录" in text
        assert "候选人" in text
        assert "请讲一次系统设计权衡。" in text
        assert "我对比了缓存与数据库方案" in text
        assert "System Design" in text


def test_transcript_cross_account_404(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        alice = _register(client)
        bob_token = client.post(
            "/api/auth/register", json={"username": "bob", "password": "pw-123456"}
        ).json()["token"]
        sid = _seed_live_session(client, alice)
        resp = client.get(
            f"/api/interviews/sessions/{sid}/transcript", headers=_auth(bob_token)
        )
        assert resp.status_code == 404


def test_transcript_empty_session_has_header(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        token = _register(client)
        sid = client.post("/api/interviews/sessions", json={}, headers=_auth(token)).json()["id"]
        resp = client.get(
            f"/api/interviews/sessions/{sid}/transcript", headers=_auth(token)
        )
        assert resp.status_code == 200
        assert "InterviewOS 面试记录" in resp.text
