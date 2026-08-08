"""Bounded, read-only observability primitives for the local debug console."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from interview_os.core.request_context import current_owner


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
    owner_id: str = ""
    session_id: str = ""
    agent: str = ""
    duration_ms: float | None = None
    detail: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DebugEventStore:
    """A process-local ring buffer with predictable O(capacity) memory."""

    def __init__(self, capacity: int = 500, path: str | Path | None = None) -> None:
        if capacity < 1:
            raise ValueError("Debug event capacity must be positive")
        self.capacity = capacity
        self._events: deque[DebugEvent] = deque(maxlen=capacity)
        self._lock = Lock()
        self._path = Path(path).expanduser() if path is not None else None
        self._load()

    @property
    def persistent(self) -> bool:
        return self._path is not None

    def record(self, event: DebugEvent) -> None:
        event = event.model_copy(
            update={
                "owner_id": event.owner_id or current_owner(),
                "detail": _redact_text(event.detail),
                "metadata": _redact_value(event.metadata),
            }
        )
        with self._lock:
            self._events.append(event)
            self._persist_locked()

    def list_events(
        self,
        *,
        limit: int = 100,
        level: DebugLevel | None = None,
        session_id: str = "",
        owner_id: str | None = None,
    ) -> list[DebugEvent]:
        bounded_limit = max(1, min(limit, self.capacity))
        with self._lock:
            events = list(self._events)
        filtered = (
            event
            for event in reversed(events)
            if (level is None or event.level == level)
            and (not session_id or event.session_id == session_id)
            and (owner_id is None or event.owner_id == owner_id)
        )
        result: list[DebugEvent] = []
        for event in filtered:
            result.append(event)
            if len(result) == bounded_limit:
                break
        return result

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            events = payload.get("events", []) if isinstance(payload, dict) else []
            self._events.extend(DebugEvent.model_validate(item) for item in events)
        except (OSError, ValueError, TypeError):
            self._events.clear()

    def _persist_locked(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {"events": [item.model_dump(mode="json") for item in self._events]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self._path)
        os.chmod(self._path, 0o600)


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
