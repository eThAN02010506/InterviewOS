"""OpenAI-compatible text-to-speech client for mock interview questions."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_TTS_BASE_URL = "http://192.168.1.97:8002/v1"
DEFAULT_TTS_MODEL = "qwen3-tts"


class TTSError(RuntimeError):
    """A safe, user-facing TTS failure."""


class TTSClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        speech_path: str | None = None,
        timeout_seconds: float = 90,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("TTS_BASE_URL") or DEFAULT_TTS_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("TTS_API_KEY") or ""
        self.model = model or os.getenv("TTS_MODEL") or DEFAULT_TTS_MODEL
        self.voice = voice or os.getenv("TTS_VOICE") or "温和、专业、清晰的中文声音"
        self.speech_path = speech_path or "/audio/speech"
        self.timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(timeout=timeout_seconds, transport=transport)

    def configure(self, **values: Any) -> None:
        if values.get("base_url") is not None:
            base_url = str(values["base_url"]).strip().rstrip("/")
            if not base_url.startswith(("http://", "https://")):
                raise ValueError("TTS base_url must use http:// or https://")
            self.base_url = base_url
        for name in ("model", "voice"):
            value = values.get(name)
            if value is not None and str(value).strip():
                setattr(self, name, str(value).strip())
        if values.get("speech_path") is not None:
            path = str(values["speech_path"]).strip()
            self.speech_path = path if path.startswith("/") else f"/{path}"
        if values.get("api_key") is not None and str(values["api_key"]).strip():
            self.api_key = str(values["api_key"]).strip()
        if values.get("timeout_seconds") is not None:
            self.timeout_seconds = float(values["timeout_seconds"])

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        clean_text = text.strip()
        if not clean_text:
            raise TTSError("待朗读的问题为空")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = await self._client.post(
                f"{self.base_url}{self.speech_path}",
                json={
                    "model": self.model,
                    "input": clean_text,
                    "voice": self.voice,
                    "response_format": "wav",
                },
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TTSError("问题语音生成服务暂时不可用") from exc
        content_type = response.headers.get("content-type", "audio/wav").split(";")[0]
        if not response.content or not content_type.startswith("audio/"):
            raise TTSError("语音服务未返回可播放音频")
        return response.content, content_type

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "base_url": self.base_url,
            "model": self.model,
            "voice": self.voice,
            "speech_path": self.speech_path,
            "timeout_seconds": self.timeout_seconds,
            "api_key_configured": bool(self.api_key),
        }

    def secret_snapshot(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "voice": self.voice,
            "speech_path": self.speech_path,
            "timeout_seconds": self.timeout_seconds,
        }

    async def close(self) -> None:
        await self._client.aclose()
