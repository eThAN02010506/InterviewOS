"""Bounded, read-only observability primitives for the local debug console."""

from __future__ import annotations

import re
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
        event = event.model_copy(
            update={
                "detail": _redact_text(event.detail),
                "metadata": _redact_value(event.metadata),
            }
        )
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


_SECRET_PATTERN = re.compile(
    r"(?i)(?:tvly-[\w-]{8,}|(?:api[_ -]?key|authorization|token)\s*[:=]\s*[^\s,;}]+)"
)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")


def _redact_text(value: str) -> str:
    value = _SECRET_PATTERN.sub("[REDACTED_SECRET]", value)
    value = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
    return _PHONE_PATTERN.sub("[REDACTED_PHONE]", value)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {
            key: "[REDACTED_SECRET]"
            if any(marker in key.lower() for marker in ("key", "token", "authorization", "secret"))
            else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value
