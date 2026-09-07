"""Bounded, read-only observability primitives for the local debug console."""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _sanitize_event(event: DebugEvent) -> DebugEvent:
        """Return a copy safe for both the local API and persistent buffer."""

        return event.model_copy(
            update={
                "owner_id": event.owner_id or current_owner(),
                "detail": _redact_text(event.detail),
                "metadata": _redact_value(event.metadata),
            }
        )

    def record(self, event: DebugEvent) -> None:
        try:
            event = self._sanitize_event(event)
        except Exception:  # noqa: BLE001 - diagnostics must never break domain work
            # Never fall back to the unredacted event: malformed/cyclic metadata
            # is less important than preserving both the business mutation and
            # the confidentiality boundary.
            logger.warning("Dropped an invalid local debug event")
            return
        with self._lock:
            self._events.append(event)
            try:
                self._persist_locked()
            except Exception:  # noqa: BLE001 - diagnostics are strictly best-effort
                # Observability is best-effort. The in-memory event remains
                # available, but a full/read-only disk must never change the
                # outcome of the business mutation being observed.
                logger.warning("Could not persist the local debug-event buffer")

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

    def rewrite_sanitized_file(self) -> None:
        """Persist the sanitized load result at runtime startup, never at import time."""

        if self._path is None or not self._path.exists():
            return
        with self._lock:
            try:
                self._persist_locked()
            except Exception:  # noqa: BLE001 - diagnostics cannot block startup
                logger.warning("Could not rewrite the sanitized debug-event buffer")

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            events = payload.get("events", []) if isinstance(payload, dict) else []
            sanitized = []
            dropped = 0
            for item in events:
                try:
                    sanitized.append(
                        self._sanitize_event(DebugEvent.model_validate(item))
                    )
                except Exception:  # noqa: BLE001 - discard unsafe diagnostic rows
                    dropped += 1
            self._events.extend(sanitized)
            if dropped:
                logger.warning("Dropped %d invalid persisted debug events", dropped)
        except Exception:  # noqa: BLE001 - corrupted diagnostics cannot block startup
            self._events.clear()

    def _persist_locked(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        encoded = json.dumps(
            {"events": [item.model_dump(mode="json") for item in self._events]},
            ensure_ascii=False,
        ).encode("utf-8")
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        descriptor: int | None = None
        replaced = False
        try:
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            replaced = True
            os.chmod(self._path, 0o600)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if not replaced:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


_SECRET_PATTERN = re.compile(
    r"(?i)(?:tvly-[\w-]{8,}|"
    r"(?:api[_ -]?key|authorization|token)\s*[:=]\s*(?:bearer\s+)?[^\s,;}]+|"
    r"bearer\s+[A-Za-z0-9._~+/=-]{4,})"
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
            str(key): "[REDACTED_SECRET]"
            if any(
                marker in str(key).lower()
                for marker in ("key", "token", "authorization", "secret")
            )
            else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact_value(item) for item in value]
    return value
