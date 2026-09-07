import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from interview_os.api.routes.debug import require_local_request
from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel
from interview_os.core.request_context import _owner_ctx


def test_debug_event_store_is_bounded_and_newest_first():
    store = DebugEventStore(capacity=2)
    store.record(DebugEvent(category="test", action="first"))
    store.record(DebugEvent(category="test", action="second", level=DebugLevel.WARNING))
    store.record(DebugEvent(category="test", action="third"))

    assert [event.action for event in store.list_events()] == ["third", "second"]
    assert [event.action for event in store.list_events(level=DebugLevel.WARNING)] == ["second"]


def test_debug_event_store_filters_by_session():
    store = DebugEventStore(capacity=10)
    store.record(DebugEvent(category="test", action="a", session_id="one"))
    store.record(DebugEvent(category="test", action="b", session_id="two"))
    assert [event.action for event in store.list_events(session_id="one")] == ["a"]


def test_debug_store_captures_request_owner_for_sessionless_events():
    store = DebugEventStore()
    token = _owner_ctx.set("account-a")
    try:
        store.record(DebugEvent(category="search", action="provider_request"))
    finally:
        _owner_ctx.reset(token)

    assert store.list_events()[0].owner_id == "account-a"


def test_debug_owner_filter_is_applied_before_result_limit():
    store = DebugEventStore(capacity=10)
    store.record(DebugEvent(category="test", action="alice-old", owner_id="alice"))
    for index in range(5):
        store.record(
            DebugEvent(category="test", action=f"bob-{index}", owner_id="bob")
        )

    events = store.list_events(limit=1, owner_id="alice")
    assert [event.action for event in events] == ["alice-old"]


def test_debug_persistence_failure_is_fail_open(monkeypatch, tmp_path):
    store = DebugEventStore(path=tmp_path / "events.json")

    def fail_persistence():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_persist_locked", fail_persistence)
    store.record(DebugEvent(category="service", action="committed"))

    assert [event.action for event in store.list_events()] == ["committed"]


def test_cyclic_debug_metadata_is_dropped_without_affecting_business_flow():
    store = DebugEventStore()
    event = DebugEvent(category="service", action="must-not-break")
    event.metadata["cycle"] = event.metadata

    store.record(event)

    assert store.list_events() == []


def test_legacy_unredacted_debug_file_is_sanitized_at_runtime_startup_boundary(tmp_path):
    path = tmp_path / "debug_events.json"
    event = DebugEvent(
        category="legacy",
        action="loaded",
        detail="api_key=very-secret user@example.com 13800138000",
        metadata={"authorization": "Bearer secret-token"},
    )
    path.write_text(
        json.dumps({"events": [event.model_dump(mode="json")]}),
        encoding="utf-8",
    )

    store = DebugEventStore(path=path)
    loaded = store.list_events()[0]

    assert "very-secret" not in loaded.detail
    assert "user@example.com" not in loaded.detail
    assert "13800138000" not in loaded.detail
    assert loaded.metadata["authorization"] == "[REDACTED_SECRET]"
    # Construction/import is read-only. The app lifespan invokes this explicit
    # startup boundary before it starts accepting requests.
    assert "very-secret" in path.read_text(encoding="utf-8")
    store.rewrite_sanitized_file()
    assert "very-secret" not in path.read_text(encoding="utf-8")


def test_bearer_credentials_are_fully_redacted_in_memory_and_on_disk(tmp_path):
    path = tmp_path / "debug_events.json"
    store = DebugEventStore(path=path)

    store.record(
        DebugEvent(
            category="provider",
            action="request",
            detail="Authorization: Bearer supersecret token: Bearer secondsecret",
        )
    )

    detail = store.list_events()[0].detail
    assert "supersecret" not in detail
    assert "secondsecret" not in detail
    persisted = path.read_text(encoding="utf-8")
    assert "supersecret" not in persisted
    assert "secondsecret" not in persisted


def test_debug_console_rejects_remote_clients():
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/debug/status",
            "headers": [],
            "client": ("192.168.1.50", 12345),
            "server": ("127.0.0.1", 8000),
            "scheme": "http",
            "query_string": b"",
        }
    )
    with pytest.raises(HTTPException) as exc_info:
        require_local_request(request)
    assert exc_info.value.status_code == 403
