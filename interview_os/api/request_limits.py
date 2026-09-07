"""ASGI request-body limits for multipart upload endpoints.

Starlette spools multipart files before a route handler receives ``UploadFile``.
The route-level bounded reader still protects application heap materialization,
while this middleware bounds the earlier HTTP body stream (file plus multipart
headers) so chunked requests cannot grow the spool without limit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MULTIPART_OVERHEAD_ALLOWANCE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class BodyLimit:
    pattern: re.Pattern[str]
    max_bytes: int
    detail: str


class _RequestBodyTooLarge(Exception):
    pass


class MultipartBodyLimitMiddleware:
    """Reject configured upload requests once their wire body exceeds a limit."""

    def __init__(self, app: ASGIApp, *, limits: tuple[BodyLimit, ...]) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self._limit_for(str(scope.get("path", "")))
        if limit is None or str(scope.get("method", "GET")).upper() not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        content_length = self._content_length(scope)
        if content_length is not None and content_length > limit.max_bytes:
            await self._reject(send, limit.detail)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        response_started = False

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            # Request parsing precedes endpoint response generation. Keep the
            # guard explicit in case a future downstream ASGI app streams early.
            if response_started:
                raise
            await self._reject(send, limit.detail)

    def _limit_for(self, path: str) -> BodyLimit | None:
        return next((item for item in self.limits if item.pattern.fullmatch(path)), None)

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return max(0, parsed)
        return None

    @staticmethod
    async def _reject(send: Send, detail: str) -> None:
        response = JSONResponse(status_code=413, content={"detail": detail})
        await response({"type": "http"}, _empty_receive, send)


async def _empty_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


def upload_body_limits() -> tuple[BodyLimit, ...]:
    """Return route-specific wire limits with room for multipart metadata."""

    mib = 1024 * 1024
    overhead = MULTIPART_OVERHEAD_ALLOWANCE_BYTES
    return (
        BodyLimit(
            re.compile(r"/api/live-interviews/[^/]+/audio/final"),
            65 * mib + overhead,
            "Session audio request exceeds the 66 MB wire limit",
        ),
        BodyLimit(
            re.compile(r"/api/live-interviews/[^/]+/audio/preview"),
            10 * mib + overhead,
            "ASR preview request exceeds the 11 MB wire limit",
        ),
        BodyLimit(
            re.compile(r"/api/live-interviews/[^/]+/audio"),
            25 * mib + overhead,
            "Audio request exceeds the 26 MB wire limit",
        ),
        BodyLimit(
            re.compile(r"/api/mock-interviews/[^/]+/transcribe"),
            25 * mib + overhead,
            "Mock audio request exceeds the 26 MB wire limit",
        ),
        BodyLimit(
            re.compile(r"/api/resumes/[^/]+/upload"),
            10 * mib + overhead,
            "Resume request exceeds the 11 MB wire limit",
        ),
    )


__all__ = [
    "MULTIPART_OVERHEAD_ALLOWANCE_BYTES",
    "BodyLimit",
    "MultipartBodyLimitMiddleware",
    "upload_body_limits",
]
