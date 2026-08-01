import pytest
from fastapi import HTTPException
from starlette.requests import Request

from interview_os.api.routes.debug import require_local_request
from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel


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
