"""Omni-modal audio-direct client for the live copilot.

Sends candidate audio directly to an OpenAI-compatible multimodal model
(e.g. Qwen3-Omni on 8004), skipping ASR, and returns a next-question suggestion.
Supports streaming so the first token appears in ~1s instead of after the full
generation. This is an alternative live-audio path alongside ASR + text LLM.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OMNI_BASE_URL = "http://192.168.1.97:8004/v1"
DEFAULT_OMNI_MODEL = (
    "C:\\llama.cpp\\models\\steampunque\\Qwen3-Omni-30B-A3B-Instruct-MP\\"
    "Qwen3-Omni-30B-A3B-Instruct.Q4_K_H.gguf"
)


def _read_file(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()

# A short built-in Chinese utterance used to verify the endpoint actually
# understands audio end to end, not just that it responds.
_PROBE_AUDIO_BASE64 = (
    "UklGRnwAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YVgAAAB+fn5+fn5+fn5+"
    "fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+"
    "fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+"
)


class OmniAudioClient:
    """Calls an OpenAI-compatible multimodal chat endpoint with audio content."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        name: str = "音频直连",
        mode: str = "asr_text",
    ) -> None:
        self.base_url: str = base_url or os.getenv("OMNI_BASE_URL", DEFAULT_OMNI_BASE_URL).rstrip("/")
        self.api_key: str = api_key or os.getenv("OMNI_API_KEY") or ""
        self.model: str = model or os.getenv("OMNI_MODEL") or DEFAULT_OMNI_MODEL
        self.name: str = name
        self.mode: str = mode if mode in {"asr_text", "audio_direct"} else "asr_text"
        self._client = httpx.AsyncClient(timeout=120.0)
        self._capability: dict[str, Any] = {}

    def configure(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        if base_url is not None:
            value = base_url.strip().rstrip("/")
            if not value.startswith(("http://", "https://")):
                raise ValueError("Omni base_url must use http:// or https://")
            self.base_url = value
        if api_key is not None and api_key.strip():
            self.api_key = api_key.strip()
        if model is not None and model.strip():
            self.model = model.strip()
        if name is not None and name.strip():
            self.name = name.strip()
        if mode is not None and mode in {"asr_text", "audio_direct"}:
            self.mode = mode

    @property
    def enabled(self) -> bool:
        return True

    @staticmethod
    def _audio_message(audio_base64: str, fmt: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_base64, "format": fmt},
                }
            ],
        }

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 200,
        stream: bool = False,
    ) -> Any:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        started = perf_counter()
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response, perf_counter() - started
        except httpx.HTTPError as exc:
            logger.error("Omni chat failed: %s", exc)
            raise

    async def suggest_next_question(
        self,
        audio_bytes: bytes,
        *,
        content_type: str = "audio/wav",
        context: str = "",
        stream: bool = True,
    ) -> AsyncIterator[str] | str:
        """Ask the omni model for a next question given candidate audio.

        With ``stream=True`` returns an async iterator of text chunks; otherwise
        returns the full text.
        """
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        fmt = "wav" if "wav" in content_type else "mp3" if "mp3" in content_type else "webm"
        system = (
            "你是面试官助手。你会听到候选人的一段回答音频，请根据这段回答给出一个"
            "聚焦证据缺口的下一问追问建议。用中文，简洁，直接给出问题本身。"
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if context:
            messages.append({"role": "user", "content": f"面试背景：{context}"})
        messages.append(self._audio_message(audio_base64, fmt))
        if not stream:
            response, _ = await self._chat(messages, max_tokens=200, stream=False)
            data = response.json()
            return data["choices"][0]["message"]["content"]
        response, _ = await self._chat(messages, max_tokens=200, stream=True)

        async def chunks() -> AsyncIterator[str]:
            content = ""
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        content += piece
                        yield piece
                except ValueError:
                    continue
            if not content:
                yield "[音频直连模型未返回可用建议]"

        return chunks()

    @staticmethod
    async def _read_probe_audio(path: str) -> bytes:
        import asyncio

        return await asyncio.to_thread(_read_file, path)

    async def probe_capability(self) -> dict[str, Any]:
        """Send an audio sample and check for a usable suggestion.

        Uses a configured probe audio file when present (e.g. a real Chinese
        utterance), else a built-in placeholder that at least exercises the
        endpoint. The result is cached in ``self._capability``.
        """
        started = perf_counter()
        try:
            probe_path = os.getenv("OMNI_PROBE_AUDIO")
            audio_bytes = (
                base64.b64decode(_PROBE_AUDIO_BASE64)
                if not probe_path or not os.path.exists(probe_path)
                else await self._read_probe_audio(probe_path)
            )
            result = await self.suggest_next_question(
                audio_bytes,
                content_type="audio/wav",
                context="候选人刚回答了一个关于系统设计的问题",
                stream=False,
            )
            text = result if isinstance(result, str) else ""
            ok = bool(text.strip()) and not text.startswith("[")
            self._capability = {
                "ok": ok,
                "latency_ms": round((perf_counter() - started) * 1000, 1),
                "sample": text[:80] if text else "",
                "probe_audio": os.path.basename(probe_path) if probe_path else "builtin",
            }
        except Exception as exc:  # noqa: BLE001 - probe reports failure, does not raise
            logger.warning("Omni capability probe failed: %s", exc)
            self._capability = {
                "ok": False,
                "latency_ms": round((perf_counter() - started) * 1000, 1),
                "sample": "",
                "error": str(exc)[:200],
            }
        return self._capability

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.mode,
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_configured": bool(self.api_key),
            "capability": self._capability,
        }

    def secret_snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
        }

    async def close(self) -> None:
        await self._client.aclose()
