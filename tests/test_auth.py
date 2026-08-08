"""Auth + per-account isolation tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.core.debug import DebugEvent
from interview_os.database.storage import (
    Storage,
    UsernameAlreadyExistsError,
    hash_password,
    verify_password,
)
from interview_os.services.settings_service import LocalSettingsStore


def _make_app(tmp_path, llm_client=None):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}")
    app = create_app(
        storage=storage,
        llm_client=llm_client,
        configure_llm=False,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
        require_auth=True,
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


def test_registration_maps_database_uniqueness_race_to_conflict(tmp_path, monkeypatch):
    with TestClient(_make_app(tmp_path)) as client:
        async def lose_concurrent_insert(username, password):
            raise UsernameAlreadyExistsError(username)

        monkeypatch.setattr(client.app.state.storage, "create_user", lose_concurrent_insert)
        response = client.post(
            "/api/auth/register", json={"username": "racing-user", "password": "pw-123456"}
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "用户名已存在"


@pytest.mark.asyncio
async def test_storage_translates_duplicate_username_constraint(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'duplicate-user.db'}")
    await storage.init_db()
    await storage.create_user("alice", "pw-123456")
    with pytest.raises(UsernameAlreadyExistsError):
        await storage.create_user("alice", "different-password")
    await storage.close()


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
        # Production API rejects anonymous access before resolving any owner.
        assert client.get(f"/api/interviews/sessions/{sid}").status_code == 401
        # Debug is local-only but still account-scoped: Alice sees her session,
        # while Bob cannot enumerate or inspect it.
        debug = client.get("/api/debug/sessions", headers=_auth(alice))
        assert [s["id"] for s in debug.json()["sessions"]] == [sid]
        assert client.get("/api/debug/sessions", headers=_auth(bob)).json()["sessions"] == []
        assert (
            client.get(f"/api/debug/sessions/{sid}", headers=_auth(bob)).status_code
            == 404
        )
        bob_sid = client.post(
            "/api/interviews/sessions", json={}, headers=_auth(bob)
        ).json()["id"]
        bob_user_id = client.get("/api/auth/me", headers=_auth(bob)).json()["id"]
        client.app.state.debug_events.record(
            DebugEvent(
                category="search",
                action="provider_request",
                owner_id=bob_user_id,
                detail="bob-private-query",
            )
        )
        alice_events = client.get("/api/debug/events", headers=_auth(alice)).json()["events"]
        assert any(event["session_id"] == sid for event in alice_events)
        assert all(event["session_id"] != bob_sid for event in alice_events)
        assert all(event["detail"] != "bob-private-query" for event in alice_events)


def test_invalid_token_rejected(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        assert (
            client.get(
                "/api/interviews/sessions", headers=_auth("not-a-real-token")
            ).status_code
            == 401
        )


def test_business_settings_and_debug_routes_require_login(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        assert client.get("/api/interviews/sessions").status_code == 401
        assert client.get("/api/settings").status_code == 401
        assert client.get("/api/debug/status").status_code == 401
        assert client.get("/health").status_code == 200
        token = _register(client, "alice")
        assert client.get("/api/interviews/sessions", headers=_auth(token)).status_code == 200
        assert client.get("/api/settings", headers=_auth(token)).status_code == 200
        assert client.get("/api/debug/status", headers=_auth(token)).status_code == 200


@pytest.mark.asyncio
async def test_first_real_account_claims_legacy_sessions(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'legacy-claim.db'}")
    await storage.init_db()
    await storage.save_session("legacy-session", {}, owner_id="local")
    alice = await storage.create_user("alice", "pw-123456")
    assert await storage.get_session_state("legacy-session", owner_id=alice.id) == {}
    bob = await storage.create_user("bob", "pw-123456")
    await storage.save_session("another-legacy", {}, owner_id="local")
    assert await storage.get_session_state("another-legacy", owner_id=bob.id) is None
    await storage.close()


def test_reserved_local_username_cannot_capture_legacy_identity(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        response = client.post(
            "/api/auth/register",
            json={"username": "local", "password": "pw-123456"},
        )
        assert response.status_code == 422
