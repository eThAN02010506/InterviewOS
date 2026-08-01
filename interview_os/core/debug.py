"""Bounded, read-only observability primitives for the local debug console."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DebugLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DebugEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: DebugLevel = DebugLevel.INFO
    category: str
    action: str
    session_id: str = ""
    agent: str = ""
    duration_ms: float | None = None
    detail: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DebugEventStore:
    """A process-local ring buffer with predictable O(capacity) memory."""

    def __init__(self, capacity: int = 500) -> None:
        if capacity < 1:
            raise ValueError("Debug event capacity must be positive")
        self.capacity = capacity
        self._events: deque[DebugEvent] = deque(maxlen=capacity)
        self._lock = Lock()

    def record(self, event: DebugEvent) -> None:
        with self._lock:
            self._events.append(event)

    def list_events(
        self,
        *,
        limit: int = 100,
        level: DebugLevel | None = None,
        session_id: str = "",
    ) -> list[DebugEvent]:
        bounded_limit = max(1, min(limit, self.capacity))
        with self._lock:
            events = list(self._events)
        filtered = (
            event
            for event in reversed(events)
            if (level is None or event.level == level)
            and (not session_id or event.session_id == session_id)
        )
        result: list[DebugEvent] = []
        for event in filtered:
            result.append(event)
            if len(result) == bounded_limit:
                break
        return result
