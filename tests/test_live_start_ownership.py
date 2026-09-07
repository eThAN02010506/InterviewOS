"""Concurrent live-start ownership and persistence contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.core.state import LiveInterviewSession
from interview_os.database.storage import Storage
from interview_os.services.settings_service import LocalSettingsStore


def _make_app(tmp_path, database_url: str):
    return create_app(
        storage=Storage(database_url),
        llm_client=None,
        configure_llm=False,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
    )


def test_concurrent_live_starts_persist_only_the_winning_operation(tmp_path):
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'start-owner.db'}"
    operations = (uuid4(), uuid4())

    with TestClient(_make_app(tmp_path, database_url)) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        gate = Barrier(2)

        def start(operation_id: UUID):
            gate.wait(timeout=5)
            return operation_id, client.post(
                f"/api/live-interviews/{session_id}/start",
                json={
                    "consent_confirmed": True,
                    "expected_revision": 0,
                    "operation_id": str(operation_id),
                },
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(start, operations))

        assert sorted(result.status_code for _, result in results) == [200, 409]
        winning_operation, winning_response = next(
            item for item in results if item[1].status_code == 200
        )
        losing_operation, _ = next(
            item for item in results if item[1].status_code == 409
        )
        winning_live = winning_response.json()["state"]["live_interview"]
        current_live = client.get(
            f"/api/live-interviews/{session_id}"
        ).json()["state"]["live_interview"]

        assert winning_live["active_start_operation_id"] == str(winning_operation)
        assert current_live["active_start_operation_id"] == str(winning_operation)
        assert current_live["active_start_operation_id"] != str(losing_operation)
        assert current_live["status_revision"] == 1
        assert current_live["capture_epoch"] == 1

        # Retrying the winning logical operation confirms the committed result
        # even with its original revision, without opening a second capture epoch.
        confirmed = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={
                "consent_confirmed": True,
                "expected_revision": 0,
                "operation_id": str(winning_operation),
            },
        )
        rejected_loser = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={
                "consent_confirmed": True,
                "expected_revision": 0,
                "operation_id": str(losing_operation),
            },
        )

        assert confirmed.status_code == 200
        assert confirmed.json()["state"]["live_interview"]["status_revision"] == 1
        assert confirmed.json()["state"]["live_interview"]["capture_epoch"] == 1
        assert rejected_loser.status_code == 409

    # The ownership token is part of the persisted state, not process memory.
    with TestClient(_make_app(tmp_path, database_url)) as restarted_client:
        restored = restarted_client.get(
            f"/api/live-interviews/{session_id}"
        ).json()["state"]["live_interview"]
        assert restored["active_start_operation_id"] == str(winning_operation)


def test_pausing_live_session_retires_start_operation_ownership(tmp_path):
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'retire-start-owner.db'}"
    operation_id = uuid4()
    with TestClient(_make_app(tmp_path, database_url)) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        started = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={
                "consent_confirmed": True,
                "expected_revision": 0,
                "operation_id": str(operation_id),
            },
        )
        paused = client.post(
            f"/api/live-interviews/{session_id}/status",
            json={
                "status": "paused",
                "expected_revision": started.json()["state"]["live_interview"][
                    "status_revision"
                ],
            },
        )

    assert paused.status_code == 200
    assert paused.json()["state"]["live_interview"]["active_start_operation_id"] is None


def test_concurrent_live_resumes_persist_only_the_winning_operation(tmp_path):
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'resume-owner.db'}"
    operations = (uuid4(), uuid4())

    with TestClient(_make_app(tmp_path, database_url)) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        started = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={
                "consent_confirmed": True,
                "expected_revision": 0,
                "operation_id": str(uuid4()),
            },
        )
        paused = client.post(
            f"/api/live-interviews/{session_id}/status",
            json={
                "status": "paused",
                "expected_revision": started.json()["state"]["live_interview"][
                    "status_revision"
                ],
            },
        )
        paused_revision = paused.json()["state"]["live_interview"]["status_revision"]
        gate = Barrier(2)

        def resume(operation_id: UUID):
            gate.wait(timeout=5)
            return operation_id, client.post(
                f"/api/live-interviews/{session_id}/status",
                json={
                    "status": "active",
                    "expected_revision": paused_revision,
                    "operation_id": str(operation_id),
                },
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(resume, operations))

        assert sorted(result.status_code for _, result in results) == [200, 409]
        winning_operation, winning_response = next(
            item for item in results if item[1].status_code == 200
        )
        losing_operation, _ = next(
            item for item in results if item[1].status_code == 409
        )
        winning_live = winning_response.json()["state"]["live_interview"]
        assert winning_live["active_start_operation_id"] == str(winning_operation)
        assert winning_live["status_revision"] == paused_revision + 1
        assert winning_live["capture_epoch"] == 2

        confirmed = client.post(
            f"/api/live-interviews/{session_id}/status",
            json={
                "status": "active",
                "expected_revision": paused_revision,
                "operation_id": str(winning_operation),
            },
        )
        rejected_loser = client.post(
            f"/api/live-interviews/{session_id}/status",
            json={
                "status": "active",
                "expected_revision": paused_revision,
                "operation_id": str(losing_operation),
            },
        )

        assert confirmed.status_code == 200
        assert confirmed.json()["state"]["live_interview"]["status_revision"] == (
            paused_revision + 1
        )
        assert confirmed.json()["state"]["live_interview"]["capture_epoch"] == 2
        assert rejected_loser.status_code == 409


def test_live_resume_requires_operation_id_but_pause_does_not(tmp_path):
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'resume-schema.db'}"
    with TestClient(_make_app(tmp_path, database_url)) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        started = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={
                "consent_confirmed": True,
                "expected_revision": 0,
                "operation_id": str(uuid4()),
            },
        )
        paused = client.post(
            f"/api/live-interviews/{session_id}/status",
            json={"status": "paused", "expected_revision": 1},
        )
        missing_operation = client.post(
            f"/api/live-interviews/{session_id}/status",
            json={
                "status": "active",
                "expected_revision": paused.json()["state"]["live_interview"][
                    "status_revision"
                ],
            },
        )

    assert started.status_code == 200
    assert paused.status_code == 200
    assert missing_operation.status_code == 422
    assert "operation_id is required" in str(missing_operation.json()["detail"])


def test_live_state_without_start_operation_id_remains_loadable():
    legacy = LiveInterviewSession.model_validate(
        {"status": "active", "status_revision": 3, "capture_epoch": 1}
    )

    assert legacy.active_start_operation_id is None
