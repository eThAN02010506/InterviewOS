"""Background task lifecycle and diagnostic redaction tests."""

from __future__ import annotations

import inspect

import pytest

from interview_os.core.debug import DebugEventStore
from interview_os.services.background import BackgroundTaskManager


@pytest.mark.asyncio
async def test_schedule_after_close_closes_rejected_coroutine():
    manager = BackgroundTaskManager()
    await manager.close()

    async def work():
        return None

    coroutine = work()
    assert manager.schedule(coroutine) is None
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_background_failure_records_only_exception_type():
    events = DebugEventStore()
    manager = BackgroundTaskManager(debug_events=events)

    async def failing_work():
        raise RuntimeError("private transcript and secret backend address")

    manager.schedule(failing_work())
    await manager.flush()

    event = events.list_events()[0]
    assert event.action == "background_task_failed"
    assert event.detail == "error_type=RuntimeError"
    assert "private transcript" not in event.detail
    await manager.close()
