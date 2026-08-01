"""Local LLM adapter via OpenAI-compatible API.

Supports Ollama / vLLM / LM Studio backends.
"""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any

import httpx

from interview_os.models.llm_interface import LLMClient

logger = logging.getLogger(__name__)


class LocalLLMClient(LLMClient):
    """Local LLM client using OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.base_url: str = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
        self.api_key: str = api_key or os.getenv("LLM_API_KEY") or "ollama"
        self.model: str = model or os.getenv("LLM_MODEL") or "qwen2.5:14b"
        self.embedding_model: str = (
            embedding_model or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-zh-v1.5"
        )
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        self._metrics = {
            "requests": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency_ms": 0.0,
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        started = perf_counter()
        self._metrics["requests"] += 1
        try:
            resp = await self._client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            self._metrics["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self._metrics["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            self._metrics["failures"] += 1
            logger.error("LLM chat failed: %s", exc)
            return f"[LLM Error: {exc}]"
        finally:
            self._metrics["total_latency_ms"] += (perf_counter() - started) * 1000

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "input": text}
        try:
            resp = await self._client.post(
                "/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPError as exc:
            logger.error("LLM embed failed: %s", exc)
            return []

    async def close(self) -> None:
        await self._client.aclose()

    async def reconfigure(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Apply UI settings without replacing the object held by existing agents."""
        if base_url is not None and base_url.rstrip("/") != self.base_url.rstrip("/"):
            await self._client.aclose()
            self.base_url = base_url.rstrip("/")
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        if api_key is not None:
            self.api_key = api_key
        if model is not None:
            self.model = model
        if embedding_model is not None:
            self.embedding_model = embedding_model

    def settings_status(self) -> dict[str, Any]:
        requests = int(self._metrics["requests"])
        return {
            "base_url": self.base_url,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "api_key_configured": bool(self.api_key),
            "metrics": {
                **self._metrics,
                "average_latency_ms": round(float(self._metrics["total_latency_ms"]) / requests, 2)
                if requests
                else 0.0,
            },
        }

    def secret_snapshot(self) -> dict[str, str]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "embedding_model": self.embedding_model,
        }

    async def probe(self) -> dict[str, Any]:
        """Check the OpenAI-compatible model endpoint without generating text."""
        started = perf_counter()
        response = await self._client.get(
            "/models", headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        data = response.json()
        models = data.get("data") or data.get("models") or []
        return {
            "ok": True,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "models": [item.get("id") or item.get("name") for item in models[:10]],
        }
