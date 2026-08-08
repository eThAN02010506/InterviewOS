"""Request-scoped identity shared by services, agents, tools, and observability."""

from __future__ import annotations

from contextvars import ContextVar

_owner_ctx: ContextVar[str] = ContextVar("interview_owner", default="local")


def current_owner() -> str:
    return _owner_ctx.get()
