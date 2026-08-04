"""Minimal in-process background task scheduler for the live copilot.

Live evidence scoring must not block the next-question suggestion, so the
service schedules it here and lets the suggestion path return first. Tasks run
in the process event loop; only short critical sections should take the session
lock when they write results back.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any

from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Tracks scheduled tasks to keep them from being garbage-collected."""

    def __init__(self, debug_events: DebugEventStore | None = None) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._debug_events = debug_events
        self._closed = False

    def schedule(self, awaitable: Awaitable[Any]) -> asyncio.Task:
        """Run ``awaitable`` in the background. The LLM call must happen inside
        the awaitable without holding the session lock."""
        if self._closed:
            logger.warning("Background task manager is closed; dropping task")
            return None  # type: ignore[return-value]
        task = asyncio.create_task(self._run(awaitable))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - background boundary, log and continue
            logger.error("Background task failed: %s", exc)
            if self._debug_events is not None:
                self._debug_events.record(
                    DebugEvent(
                        level=DebugLevel.ERROR,
                        category="service",
                        action="background_scoring_failed",
                        detail=str(exc)[:1000],
                    )
                )

    async def flush(self) -> None:
        """Await all currently scheduled tasks (used by tests and shutdown)."""
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        """Cancel all in-flight tasks and wait for them to settle."""
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
