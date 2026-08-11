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


def _parse_diarized_response(raw: str) -> list[dict[str, str]]:
    """Parse the diarize JSON into speaker-labelled segments."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    segments: list[dict[str, str]] = []
    for entry in payload.get("segments", []) if isinstance(payload, dict) else []:
        if not isinstance(entry, dict):
            continue
        speaker_raw = str(entry.get("speaker") or "").strip()
        utterance = str(entry.get("text") or "").strip()
        if not utterance:
            continue
        speaker = (
            "candidate"
            if "候选" in speaker_raw or "candidate" in speaker_raw.lower()
            else "interviewer"
            if "面试" in speaker_raw or "interview" in speaker_raw.lower()
            else "unknown"
        )
        segments.append({"speaker": speaker, "text": utterance})
    return segments


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}

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
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url: str = base_url or os.getenv("OMNI_BASE_URL", DEFAULT_OMNI_BASE_URL).rstrip("/")
        self.api_key: str = api_key or os.getenv("OMNI_API_KEY") or ""
        self.model: str = model or os.getenv("OMNI_MODEL") or DEFAULT_OMNI_MODEL
        self.name: str = name
        self.mode: str = mode if mode in {"asr_text", "audio_direct"} else "asr_text"
        self._client = httpx.AsyncClient(timeout=120.0, transport=transport)
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
    ) -> Any:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
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
            logger.error("Omni chat failed (%s)", type(exc).__name__)
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
            response, _ = await self._chat(messages, max_tokens=200)
            data = response.json()
            return data["choices"][0]["message"]["content"]

        async def chunks() -> AsyncIterator[str]:
            content = ""
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 200,
                "stream": True,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        event = line[5:].strip()
                        if event == "[DONE]":
                            break
                        try:
                            chunk = json.loads(event)
                            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                            piece = delta.get("content") or ""
                            if piece:
                                content += piece
                                yield piece
                        except ValueError:
                            continue
            except httpx.HTTPError as exc:
                logger.error("Omni streaming chat failed (%s)", type(exc).__name__)
            if not content:
                yield "[音频直连模型未返回可用建议]"

        return chunks()

    async def transcribe_diarize(
        self,
        audio_bytes: bytes,
        *,
        content_type: str = "audio/wav",
    ) -> list[dict[str, str]]:
        """Ask the omni model to split the audio by speaker.

        The model hears a two-person dialog and returns per-utterance speaker
        labels. Returns a list of ``{"speaker": "interviewer"|"candidate",
        "text": ...}``; an empty list on any failure so the caller falls back.
        """
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        fmt = "wav" if "wav" in content_type else "mp3" if "mp3" in content_type else "webm"
        system = (
            "你是面试助手。这段对话里第一个说话的是面试官，另一个是候选人。"
            "请逐句标注说话人。只输出 JSON："
            '{"segments":[{"speaker":"面试官或候选人","text":"..."}]}'
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            self._audio_message(audio_base64, fmt),
        ]
        try:
            response, _ = await self._chat(messages, max_tokens=600)
            raw = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - report and fall back
            logger.warning("Omni diarize failed (%s)", type(exc).__name__)
            return []
        segments = _parse_diarized_response(raw)
        if not segments:
            logger.warning("Omni diarize returned no usable segments")
        return segments

    async def analyze_speaking_style(
        self,
        audio_bytes: bytes,
        *,
        transcript: str,
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        """Analyze delivery for candidate coaching, never for hiring evidence."""
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        fmt = "wav" if "wav" in content_type else "mp3" if "mp3" in content_type else "webm"
        system = (
            "你是面试表达辅导员，只分析可改变的发言特征：语速、停顿、填充词、"
            "音量稳定性、语调变化和清晰度。禁止评价口音、性格、情绪状态、健康、"
            "年龄、性别、族裔或其他个人特征；禁止给出录用判断。证据不足时明确说无法判断。"
            "只输出 JSON，字段：pace, pauses, fillers, volume, intonation, clarity, "
            "strengths(字符串数组), improvements(字符串数组)。每个结论必须具体可行动。"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"转写文本（仅用于辅助核对）：{transcript[:4000]}"},
            self._audio_message(audio_base64, fmt),
        ]
        try:
            response, _ = await self._chat(messages, max_tokens=700)
            raw = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - caller has a text-only fallback
            logger.warning("Omni speaking-style analysis failed: %s", type(exc).__name__)
            return {}
        return _parse_json_object(raw)

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
            logger.warning("Omni capability probe failed (%s)", type(exc).__name__)
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
