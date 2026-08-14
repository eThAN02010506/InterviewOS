"""Configurable LAN speech-to-text client.

The adapter intentionally targets the OpenAI Whisper multipart contract while
accepting a few common response envelopes used by self-hosted ASR services.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class ASRError(RuntimeError):
    """Raised when the configured transcription service cannot produce text."""


class ASRClient:
    def __init__(
        self,
        base_url: str = "http://192.168.1.97:8007",
        api_key: str = "",
        model: str = "whisper-1",
        transcription_path: str = "/v1/audio/transcriptions",
        timeout_seconds: float = 90.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self.configure(
            base_url=base_url,
            api_key=api_key,
            model=model,
            transcription_path=transcription_path,
            timeout_seconds=timeout_seconds,
        )

    def configure(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        transcription_path: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if base_url is not None:
            value = base_url.strip().rstrip("/")
            if not value.startswith(("http://", "https://")):
                raise ValueError("ASR base_url must use http:// or https://")
            self.base_url = value
        if api_key is not None and api_key.strip():
            self.api_key = api_key.strip()
        elif not hasattr(self, "api_key"):
            self.api_key = ""
        if model is not None:
            self.model = model.strip() or "whisper-1"
        if transcription_path is not None:
            path = transcription_path.strip() or "/v1/audio/transcriptions"
            self.transcription_path = path if path.startswith("/") else f"/{path}"
        if timeout_seconds is not None:
            if not 1 <= timeout_seconds <= 600:
                raise ValueError("ASR timeout_seconds must be between 1 and 600")
            self.timeout_seconds = float(timeout_seconds)

    async def transcribe(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str = "application/octet-stream",
        language: str = "zh",
    ) -> str:
        if not content:
            raise ASRError("Audio payload is empty")
        client = self._get_client()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = {"model": self.model}
        if language.strip():
            data["language"] = language.strip()
        try:
            response = await client.post(
                f"{self.base_url}{self.transcription_path}",
                headers=headers,
                data=data,
                files={"file": (filename, content, content_type)},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            raise ASRError(f"ASR request failed: {exc}") from exc
        try:
            payload: Any = response.json()
        except ValueError as exc:
            text = response.text.strip()
            if text:
                return text
            raise ASRError("ASR returned an empty non-JSON response") from exc
        text = self._extract_text(payload)
        if not text:
            raise ASRError("ASR response did not contain transcript text")
        return text

    async def probe(self) -> dict[str, Any]:
        client = self._get_client()
        attempts = ("/health", "/v1/models", "/")
        last_error = ""
        for path in attempts:
            try:
                response = await client.get(f"{self.base_url}{path}", timeout=3.0)
                if 200 <= response.status_code < 400:
                    return {"ok": True, "path": path, "status_code": response.status_code}
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
        return {"ok": False, "error": last_error or "ASR service unavailable"}

    def status(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "transcription_path": self.transcription_path,
            "timeout_seconds": self.timeout_seconds,
            "api_key_configured": bool(self.api_key),
        }

    def secret_snapshot(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "transcription_path": self.transcription_path,
            "timeout_seconds": self.timeout_seconds,
        }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    @classmethod
    def _extract_text(cls, payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if not isinstance(payload, dict):
            return ""
        for key in ("text", "transcript", "transcription"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("result", "data"):
            nested = cls._extract_text(payload.get(key))
            if nested:
                return nested
        return ""
