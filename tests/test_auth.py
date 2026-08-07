"""Auth + per-account isolation tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.database.storage import Storage, hash_password, verify_password
from interview_os.services.settings_service import LocalSettingsStore


def _make_app(tmp_path, llm_client=None):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}")
    app = create_app(
        storage=storage,
        llm_client=llm_client,
        configure_llm=False,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
    )
    return app


def _register(client: TestClient, username: str, password: str = "pw-123456") -> str:
    resp = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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


def test_password_hash_roundtrip():
    stored = hash_password("correct-horse")
    assert verify_password("correct-horse", stored)
    assert not verify_password("wrong-password", stored)


def test_register_login_me_logout(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        token = _register(client, "alice")
        me = client.get("/api/auth/me", headers=_auth(token))
        assert me.status_code == 200
        assert me.json()["username"] == "alice"
        # Duplicate username rejected
        dup = client.post(
            "/api/auth/register", json={"username": "alice", "password": "x"}
        )
        assert dup.status_code == 409
        # Wrong password rejected
        bad = client.post(
            "/api/auth/login", json={"username": "alice", "password": "wrong"}
        )
        assert bad.status_code == 401
        # Correct login issues a fresh token
        login = client.post(
            "/api/auth/login", json={"username": "alice", "password": "pw-123456"}
        )
        assert login.status_code == 200
        assert login.json()["token"]
        # Logout invalidates the token
        out = client.post("/api/auth/logout", headers=_auth(token))
        assert out.status_code == 200
        assert client.get("/api/auth/me", headers=_auth(token)).status_code == 401
        # No token -> 401
        assert client.get("/api/auth/me").status_code == 401


def test_sessions_isolated_between_accounts(tmp_path):
    with TestClient(_make_app(tmp_path, _WorkflowLLM())) as client:
        alice = _register(client, "alice")
        bob = _register(client, "bob")
        # Alice creates a session
        created = client.post(
            "/api/interviews/sessions", json={}, headers=_auth(alice)
        )
        assert created.status_code == 200
        sid = created.json()["id"]
        # Alice lists it
        alice_list = client.get("/api/interviews/sessions", headers=_auth(alice))
        assert [s["id"] for s in alice_list.json()] == [sid]
        # Bob cannot see it (list) nor fetch it (404, not 403)
        bob_list = client.get("/api/interviews/sessions", headers=_auth(bob))
        assert bob_list.json() == []
        assert (
            client.get(f"/api/interviews/sessions/{sid}", headers=_auth(bob)).status_code
            == 404
        )
        # Alice can fetch it
        assert (
            client.get(f"/api/interviews/sessions/{sid}", headers=_auth(alice)).status_code
            == 200
        )
        # Anonymous (legacy 'local' owner) cannot see Alice's session
        assert client.get(f"/api/interviews/sessions/{sid}").status_code == 404


def test_invalid_token_rejected(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        assert (
            client.get(
                "/api/interviews/sessions", headers=_auth("not-a-real-token")
            ).status_code
            == 401
        )
