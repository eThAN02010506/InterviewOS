"""Local LLM adapter via OpenAI-compatible API.

Supports Ollama / vLLM / LM Studio backends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from interview_os.core.provider_config import (
    normalize_provider_api_key,
    normalize_provider_endpoint,
    provider_api_key_from_env,
    provider_endpoint_from_env,
)
from interview_os.models.llm_interface import LLMClient
from interview_os.services.settings_service import PermissionRestrictedJsonStore

logger = logging.getLogger(__name__)
MAX_COST_PER_MILLION_USD = 1_000_000.0
MAX_PERSISTED_METRIC = 10**18


class LLMStreamError(RuntimeError):
    """Raised after a streaming transport fails, without exposing it as model text."""


class LLMResponseError(ValueError):
    """Raised when a successful provider response lacks usable model content."""


class _BorrowedAsyncTransport(httpx.AsyncBaseTransport):
    """Share an injected transport without individual HTTP clients closing it."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        return None


def _approx_tokens(text: str) -> int:
    """Rough token estimate for streaming (no server usage count available)."""
    return max(1, len(text) // 4)


def _safe_token_count(value: Any) -> int:
    """Normalize untrusted provider usage without letting telemetry fail a request."""

    if isinstance(value, bool):
        return 0
    try:
        count = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return min(MAX_PERSISTED_METRIC, max(0, count))


class LocalLLMClient(LLMClient):
    """Local LLM client using OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        metrics_path: str | Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = provider_endpoint_from_env(
            base_url,
            env_name="LLM_BASE_URL",
            default="http://localhost:11434/v1",
            label="LLM base_url",
            logger=logger,
        )
        self.api_key = provider_api_key_from_env(
            api_key,
            env_name="LLM_API_KEY",
            default="ollama",
            logger=logger,
        )
        self.model: str = model or os.getenv("LLM_MODEL") or "qwen2.5:14b"
        self.embedding_model: str = (
            embedding_model or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-zh-v1.5"
        )
        self.input_cost_per_million = self._safe_cost(input_cost_per_million)
        self.output_cost_per_million = self._safe_cost(output_cost_per_million)
        self._injected_transport = transport
        self._client = self._new_http_client(self.base_url)
        self._active_client_requests: dict[httpx.AsyncClient, int] = {}
        self._retired_clients: set[httpx.AsyncClient] = set()
        self._client_close_tasks: dict[asyncio.Task[None], httpx.AsyncClient] = {}
        self._clients_drained = asyncio.Event()
        self._clients_drained.set()
        self._closed = False
        self._configuration_generation = 0
        self._metrics_store = (
            PermissionRestrictedJsonStore(metrics_path) if metrics_path is not None else None
        )
        loaded = self._metrics_store.load() if self._metrics_store else {}
        has_accrued_cost = "accrued_cost_usd" in loaded
        defaults = {
            "requests": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency_ms": 0.0,
            "accrued_cost_usd": 0.0,
            "unpriced_prompt_tokens": 0,
            "unpriced_completion_tokens": 0,
        }
        self._metrics: dict[str, int | float] = {}
        for key, default in defaults.items():
            value = loaded.get(key, default)
            try:
                normalized = float(value) if isinstance(default, float) else int(value)
            except (TypeError, ValueError, OverflowError):
                normalized = default
            if isinstance(normalized, float) and not math.isfinite(normalized):
                normalized = default
            self._metrics[key] = min(MAX_PERSISTED_METRIC, max(0, normalized))
        if not has_accrued_cost:
            # Legacy files retained token totals but not the unit prices that
            # applied at request time. Treat that history as explicitly
            # unpriced instead of retroactively applying today's UI price.
            self._metrics["unpriced_prompt_tokens"] = self._metrics["prompt_tokens"]
            self._metrics["unpriced_completion_tokens"] = self._metrics[
                "completion_tokens"
            ]
        self._response_format_supported: bool | None = None

    def _new_http_client(self, base_url: str) -> httpx.AsyncClient:
        transport = (
            _BorrowedAsyncTransport(self._injected_transport)
            if self._injected_transport is not None
            else None
        )
        return httpx.AsyncClient(base_url=base_url, timeout=120.0, transport=transport)

    @staticmethod
    def _safe_cost(value: float) -> float:
        try:
            cost = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(cost):
            return 0.0
        return min(MAX_COST_PER_MILLION_USD, max(0.0, cost))

    def _acquire_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("LLM client is closed")
        client = self._client
        self._active_client_requests[client] = self._active_client_requests.get(client, 0) + 1
        self._clients_drained.clear()
        return client

    def _schedule_client_close(self, client: httpx.AsyncClient) -> None:
        if client in self._client_close_tasks.values():
            return
        task = asyncio.create_task(client.aclose())
        # Keep both the task and its target strongly referenced until cleanup
        # reaches a terminal state, even if a settings request is cancelled.
        self._client_close_tasks[task] = client

        def complete(done: asyncio.Task[None]) -> None:
            self._client_close_tasks.pop(done, None)
            try:
                done.result()
            except asyncio.CancelledError:
                logger.warning("Retired LLM transport cleanup was cancelled")
            except Exception as exc:  # noqa: BLE001 - cleanup must not roll back config
                logger.warning(
                    "Retired LLM transport cleanup failed (%s)", type(exc).__name__
                )

        task.add_done_callback(complete)

    def _retire_client(self, client: httpx.AsyncClient) -> None:
        self._retired_clients.add(client)
        if self._active_client_requests.get(client, 0) == 0:
            self._retired_clients.discard(client)
            self._schedule_client_close(client)

    def _release_client(self, client: httpx.AsyncClient) -> None:
        active = self._active_client_requests.get(client, 0)
        if active <= 1:
            self._active_client_requests.pop(client, None)
            if client in self._retired_clients:
                self._retired_clients.discard(client)
                self._schedule_client_close(client)
        else:
            self._active_client_requests[client] = active - 1
        if not self._active_client_requests:
            self._clients_drained.set()

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
        api_key = self.api_key
        request_generation = self._configuration_generation
        if self._response_format_supported is False:
            payload.pop("response_format", None)
        # gpt-oss can spend the entire completion budget in reasoning and return
        # an empty `content` field at its default effort. llama.cpp's compatible
        # endpoint accepts this standard hint and still allows an explicit caller
        # override through kwargs.
        if "gpt-oss" in self.model.lower():
            template_kwargs = dict(payload.get("chat_template_kwargs") or {})
            # llama.cpp reads GPT-OSS effort from the Jinja template kwargs,
            # not from the OpenAI top-level compatibility key.
            requested_effort = payload.pop("reasoning_effort", "low")
            template_kwargs.setdefault("reasoning_effort", requested_effort)
            payload["chat_template_kwargs"] = template_kwargs
        started = perf_counter()
        request_input_cost = self.input_cost_per_million
        request_output_cost = self.output_cost_per_million
        client = self._acquire_client()
        try:
            for attempt in range(2):
                self._metrics["requests"] += 1
                try:
                    resp = await client.post(
                        "/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices") if isinstance(data, dict) else None
                    if not isinstance(choices, list) or not choices:
                        raise LLMResponseError("Missing completion choices")
                    message = choices[0].get("message") if isinstance(choices[0], dict) else None
                    content = message.get("content") if isinstance(message, dict) else None
                    if not isinstance(content, str) or not content.strip():
                        raise LLMResponseError("Missing completion content")
                    usage = data.get("usage")
                    if not isinstance(usage, dict):
                        usage = {}
                    self._record_usage(
                        _safe_token_count(usage.get("prompt_tokens")),
                        _safe_token_count(usage.get("completion_tokens")),
                        input_cost_per_million=request_input_cost,
                        output_cost_per_million=request_output_cost,
                    )
                    if (
                        "response_format" in payload
                        and client is self._client
                        and request_generation == self._configuration_generation
                    ):
                        self._response_format_supported = True
                    return content
                except httpx.HTTPError as exc:
                    self._metrics["failures"] += 1
                    unsupported_response_format = (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code in {400, 404, 422}
                        and "response_format" in payload
                    )
                    if attempt == 0 and unsupported_response_format:
                        # OpenAI-compatible local servers vary in JSON Schema
                        # support. Remember the capability and retry this request
                        # without the unsupported transport hint; prompt-level
                        # validation and targeted repair still apply.
                        if (
                            client is self._client
                            and request_generation == self._configuration_generation
                        ):
                            self._response_format_supported = False
                        payload.pop("response_format", None)
                        logger.warning("LLM endpoint rejected response_format; retrying without it")
                        continue
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                        exc.response.status_code == 429 or exc.response.status_code >= 500
                    )
                    if attempt == 0 and retryable:
                        logger.warning(
                            "LLM chat request failed; retrying once (%s)",
                            type(exc).__name__,
                        )
                        await asyncio.sleep(0.25)
                        continue
                    logger.error("LLM chat failed after retry (%s)", type(exc).__name__)
                    return "[LLM Error: request failed]"
                except (ValueError, TypeError) as exc:
                    self._metrics["failures"] += 1
                    if attempt == 0:
                        logger.warning(
                            "LLM chat returned an invalid response; retrying once (%s)",
                            type(exc).__name__,
                        )
                        await asyncio.sleep(0.25)
                        continue
                    logger.error("LLM chat returned invalid responses (%s)", type(exc).__name__)
                    return "[LLM Error: invalid response]"
            return "[LLM Error: request failed]"
        finally:
            self._release_client(client)
            self._metrics["total_latency_ms"] += (perf_counter() - started) * 1000
            self._persist_metrics()

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from the OpenAI-compatible endpoint as SSE lines.

        Yields each ``delta.content`` piece as it arrives; the caller can relay
        them to the browser so a suggestion renders incrementally instead of
        after the full generation.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }
        api_key = self.api_key
        if "gpt-oss" in self.model.lower():
            template_kwargs = dict(payload.get("chat_template_kwargs") or {})
            requested_effort = payload.pop("reasoning_effort", "low")
            template_kwargs.setdefault("reasoning_effort", requested_effort)
            payload["chat_template_kwargs"] = template_kwargs
        started = perf_counter()
        request_input_cost = self.input_cost_per_million
        request_output_cost = self.output_cost_per_million
        client = self._acquire_client()
        try:
            self._metrics["requests"] += 1
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                resp.raise_for_status()
                # OpenAI-compatible streaming servers commonly omit usage.
                # Count the accepted prompt once and estimate from the exact
                # serialized message payload used for this request.
                prompt_text = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                prompt_tokens = _approx_tokens(prompt_text)
                content = ""
                try:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        piece = line[5:].strip()
                        if piece == "[DONE]":
                            break
                        try:
                            chunk = json.loads(piece)
                            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                            text = delta.get("content") or ""
                        except (ValueError, TypeError):
                            continue
                        if text:
                            content += text
                            yield text
                finally:
                    # Once the provider accepted the request, account for the
                    # accepted prompt and any partial completion even if the
                    # browser disconnects and closes this async generator.
                    self._record_usage(
                        prompt_tokens,
                        _approx_tokens(content) if content else 0,
                        input_cost_per_million=request_input_cost,
                        output_cost_per_million=request_output_cost,
                    )
        except httpx.HTTPError as exc:
            self._metrics["failures"] += 1
            logger.error("LLM chat_stream failed (%s)", type(exc).__name__)
            raise LLMStreamError("Local model stream failed") from None
        finally:
            self._release_client(client)
            self._metrics["total_latency_ms"] += (perf_counter() - started) * 1000
            self._persist_metrics()

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "input": text}
        api_key = self.api_key
        client = self._acquire_client()
        try:
            resp = await client.post(
                "/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPError as exc:
            logger.error("LLM embed failed (%s)", type(exc).__name__)
            return []
        finally:
            self._release_client(client)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._retire_client(self._client)
        await self._clients_drained.wait()
        pending = list(self._client_close_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._injected_transport is not None:
            await self._injected_transport.aclose()
            self._injected_transport = None

    async def reconfigure(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        """Apply UI settings without replacing the object held by existing agents."""
        self.reconfigure_now(
            base_url=base_url,
            api_key=api_key,
            model=model,
            embedding_model=embedding_model,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )

    def reconfigure_now(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        """Atomically publish settings before any event-loop scheduling point."""

        if self._closed:
            raise RuntimeError("LLM client is closed")
        next_api_key = (
            normalize_provider_api_key(api_key, label="LLM API key")
            if api_key is not None
            else None
        )
        next_base_url = (
            normalize_provider_endpoint(base_url, label="LLM base_url")
            if base_url is not None
            else None
        )
        capability_generation_changed = False
        if next_base_url is not None and next_base_url != self.base_url:
            next_client = self._new_http_client(next_base_url)
            previous_client = self._client
            self.base_url = next_base_url
            self._client = next_client
            self._response_format_supported = None
            capability_generation_changed = True
            self._retire_client(previous_client)
        if next_api_key is not None:
            self.api_key = next_api_key
        if model is not None:
            if model != self.model:
                self._response_format_supported = None
                capability_generation_changed = True
            self.model = model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        if input_cost_per_million is not None:
            self.input_cost_per_million = self._safe_cost(input_cost_per_million)
        if output_cost_per_million is not None:
            self.output_cost_per_million = self._safe_cost(output_cost_per_million)
        if capability_generation_changed:
            self._configuration_generation += 1

    def settings_status(self) -> dict[str, Any]:
        requests = int(self._metrics["requests"])
        return {
            "base_url": self.base_url,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "api_key_configured": bool(self.api_key),
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "metrics": {
                **self._metrics,
                "average_latency_ms": round(float(self._metrics["total_latency_ms"]) / requests, 2)
                if requests
                else 0.0,
                "estimated_cost_usd": round(float(self._metrics["accrued_cost_usd"]), 6),
                "persistent": self._metrics_store is not None,
            },
        }

    def secret_snapshot(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
        }

    def clone_with_settings(self, **updates: Any) -> LocalLLMClient:
        """Create a separately configurable client on the same injected adapter."""

        config = self.secret_snapshot()
        config.update({key: value for key, value in updates.items() if value is not None})
        transport = (
            _BorrowedAsyncTransport(self._injected_transport)
            if self._injected_transport is not None
            else None
        )
        return type(self)(**config, transport=transport)

    def _record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        input_cost_per_million: float,
        output_cost_per_million: float,
    ) -> None:
        prompt_tokens = min(MAX_PERSISTED_METRIC, max(0, prompt_tokens))
        completion_tokens = min(MAX_PERSISTED_METRIC, max(0, completion_tokens))
        self._metrics["prompt_tokens"] = min(
            MAX_PERSISTED_METRIC,
            int(self._metrics["prompt_tokens"]) + prompt_tokens,
        )
        self._metrics["completion_tokens"] = min(
            MAX_PERSISTED_METRIC,
            int(self._metrics["completion_tokens"]) + completion_tokens,
        )
        request_cost = (
            prompt_tokens * input_cost_per_million
            + completion_tokens * output_cost_per_million
        ) / 1_000_000
        self._metrics["accrued_cost_usd"] = min(
            MAX_PERSISTED_METRIC,
            float(self._metrics["accrued_cost_usd"]) + request_cost,
        )

    def _persist_metrics(self) -> None:
        if self._metrics_store is None:
            return
        try:
            self._metrics_store.save(self._metrics)
        except Exception as exc:  # noqa: BLE001 - telemetry must never break inference
            logger.warning("Unable to persist LLM metrics (%s)", type(exc).__name__)

    async def probe(self) -> dict[str, Any]:
        """Check the OpenAI-compatible model endpoint without generating text."""
        started = perf_counter()
        api_key = self.api_key
        client = self._acquire_client()
        try:
            response = await client.get(
                "/models", headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data") or data.get("models") or []
            return {
                "ok": True,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "models": [item.get("id") or item.get("name") for item in models[:10]],
            }
        finally:
            self._release_client(client)
