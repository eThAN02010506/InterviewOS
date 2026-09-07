"""Runtime settings API and local settings UI."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, FiniteFloat

from interview_os.core.provider_config import (
    normalize_provider_api_key,
    normalize_provider_endpoint,
)
from interview_os.models.local_llm import LocalLLMClient
from interview_os.models.omni_client import OmniAudioClient
from interview_os.services.service_contracts import LiveInterviewStateError
from interview_os.services.settings_service import (
    LocalSettingsStore,
    audio_settings_fingerprint,
    is_valid_http_endpoint,
)
from interview_os.tools.asr import ASRClient
from interview_os.tools.tts import TTSClient
from interview_os.tools.web_search import SearchProviderManager

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_SEARCH_REQUEST_COST_USD = 10_000.0
MAX_MODEL_COST_PER_MILLION_USD = 1_000_000.0


class SearchSettingsUpdate(BaseModel):
    provider: Literal["none", "tavily", "searxng", "brave"]
    tavily_api_key: str | None = None
    searxng_base_url: str | None = None
    brave_api_key: str | None = None
    search_request_cost_usd: FiniteFloat | None = Field(
        default=None, ge=0, le=MAX_SEARCH_REQUEST_COST_USD
    )


class LLMSettingsUpdate(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    embedding_model: str | None = None
    input_cost_per_million: FiniteFloat | None = Field(
        default=None, ge=0, le=MAX_MODEL_COST_PER_MILLION_USD
    )
    output_cost_per_million: FiniteFloat | None = Field(
        default=None, ge=0, le=MAX_MODEL_COST_PER_MILLION_USD
    )


class ASRSettingsUpdate(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    transcription_path: str | None = None
    timeout_seconds: float | None = Field(default=None, ge=1, le=600)


class LiveAudioSettingsUpdate(BaseModel):
    mode: Literal["asr_text", "audio_direct"] | None = None
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class ResumeLLMSettingsUpdate(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class TTSSettingsUpdate(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    voice: str | None = None
    speech_path: str | None = None
    timeout_seconds: float | None = Field(default=None, ge=1, le=600)


class SettingsUpdate(BaseModel):
    search: SearchSettingsUpdate | None = None
    llm: LLMSettingsUpdate | None = None
    asr: ASRSettingsUpdate | None = None
    live_audio: LiveAudioSettingsUpdate | None = None
    resume_llm: ResumeLLMSettingsUpdate | None = None
    tts: TTSSettingsUpdate | None = None
    persist: bool = True


def _updated_snapshot(current: dict, update: BaseModel) -> dict:
    """Merge non-null UI fields without changing the live client."""
    proposed = dict(current)
    proposed.update(
        {key: value for key, value in update.model_dump().items() if value is not None}
    )
    return proposed


def _validate_update_values(payload: SettingsUpdate) -> None:
    """Validate URL and credential values before mutating any live client."""

    endpoints = (
        ("LLM base_url", payload.llm.base_url if payload.llm else None, False),
        ("ASR base_url", payload.asr.base_url if payload.asr else None, False),
        (
            "Live audio base_url",
            payload.live_audio.base_url if payload.live_audio else None,
            False,
        ),
        (
            "Resume LLM base_url",
            payload.resume_llm.base_url if payload.resume_llm else None,
            False,
        ),
        ("TTS base_url", payload.tts.base_url if payload.tts else None, False),
        (
            "SearXNG base_url",
            payload.search.searxng_base_url if payload.search else None,
            True,
        ),
    )
    for label, endpoint, allow_empty in endpoints:
        if endpoint is not None and not is_valid_http_endpoint(
            endpoint, allow_empty=allow_empty
        ):
            raise ValueError(f"{label} must be a valid http:// or https:// URL")

    credentials = (
        (
            "Tavily API key",
            payload.search.tavily_api_key if payload.search else None,
        ),
        (
            "Brave API key",
            payload.search.brave_api_key if payload.search else None,
        ),
        ("LLM API key", payload.llm.api_key if payload.llm else None),
        ("ASR API key", payload.asr.api_key if payload.asr else None),
        (
            "Live audio API key",
            payload.live_audio.api_key if payload.live_audio else None,
        ),
        (
            "Resume LLM API key",
            payload.resume_llm.api_key if payload.resume_llm else None,
        ),
        ("TTS API key", payload.tts.api_key if payload.tts else None),
    )
    for label, credential in credentials:
        if credential is not None:
            normalize_provider_api_key(credential, label=label)


def _audio_processing_settings_changed(
    payload: SettingsUpdate,
    asr: ASRClient,
    omni: OmniAudioClient | None,
) -> bool:
    """Compare effective audio values, ignoring UI-only fields and no-op forms."""

    current_asr = asr.secret_snapshot()
    proposed_asr = dict(current_asr)
    if payload.asr is not None:
        update = payload.asr.model_dump(exclude_none=True)
        if "base_url" in update:
            proposed_asr["base_url"] = normalize_provider_endpoint(
                update["base_url"], label="ASR base_url"
            )
        if "api_key" in update:
            proposed_asr["api_key"] = normalize_provider_api_key(
                update["api_key"], label="ASR API key"
            )
        if "model" in update:
            proposed_asr["model"] = update["model"].strip() or "whisper-1"
        if "transcription_path" in update:
            path = update["transcription_path"].strip() or "/v1/audio/transcriptions"
            proposed_asr["transcription_path"] = (
                path if path.startswith("/") else f"/{path}"
            )
        if "timeout_seconds" in update:
            proposed_asr["timeout_seconds"] = float(update["timeout_seconds"])
    if proposed_asr != current_asr:
        return True

    if payload.live_audio is None:
        return False
    if omni is None:
        return True
    current_live = omni.secret_snapshot()
    proposed_live = dict(current_live)
    update = payload.live_audio.model_dump(exclude_none=True)
    if "base_url" in update:
        proposed_live["base_url"] = normalize_provider_endpoint(
            update["base_url"], label="Live audio base_url"
        )
    if "api_key" in update:
        proposed_live["api_key"] = normalize_provider_api_key(
            update["api_key"], label="Live audio API key"
        )
    if "model" in update and update["model"].strip():
        proposed_live["model"] = update["model"].strip()
    if "mode" in update:
        proposed_live["mode"] = update["mode"]
    # ``name`` changes only presentation and cannot alter transcription output.
    return any(
        proposed_live[name] != current_live[name]
        for name in ("base_url", "api_key", "model", "mode")
    )


def _restore_runtime_settings_now(request: Request, snapshots: dict) -> list[object]:
    """Synchronously republish the last committed settings after a failed update."""
    search: SearchProviderManager = request.app.state.search_manager
    llm = request.app.state.llm_client
    asr: ASRClient = request.app.state.asr_client
    omni = getattr(request.app.state, "omni_client", None)
    tts: TTSClient | None = getattr(request.app.state, "tts_client", None)
    service = request.app.state.interview_service
    original_resume = snapshots["resume_client"]
    current_resume = getattr(request.app.state, "resume_llm_client", None)
    cleanup_clients: list[object] = []

    # Pointer restoration comes first and has no scheduling point. A normal
    # inference task therefore cannot acquire an uncommitted split client in
    # the exception-to-rollback event-loop gap.
    if current_resume is not original_resume:
        request.app.state.resume_llm_client = original_resume
        service.resume_llm_client = original_resume
        if current_resume is not None and hasattr(current_resume, "close"):
            cleanup_clients.append(current_resume)

    search.configure(**snapshots["search"])
    if isinstance(llm, LocalLLMClient):
        llm.reconfigure_now(**snapshots["llm"])
    asr.configure(**snapshots["asr"])
    asr.api_key = snapshots["asr"]["api_key"]
    if omni is not None and snapshots["live_audio"] is not None:
        omni.configure(**snapshots["live_audio"])
        omni.api_key = snapshots["live_audio"]["api_key"]
        service.set_live_audio_mode(snapshots["live_mode"])
        omni.mode = snapshots["live_mode"]
    if tts is not None and snapshots["tts"] is not None:
        tts.configure(**snapshots["tts"])
        tts.api_key = snapshots["tts"]["api_key"]
    if isinstance(original_resume, LocalLLMClient) and original_resume is not llm:
        original_resume.reconfigure_now(**snapshots["resume_llm"])
    return cleanup_clients


async def _close_uncommitted_clients(clients: list[object]) -> None:
    """Drain uncommitted split clients after committed pointers are restored."""

    for client in clients:
        if hasattr(client, "close"):
            try:
                await client.close()
            except BaseException as exc:  # noqa: BLE001 - pointers are already coherent
                logger.warning(
                    "Uncommitted resume client cleanup failed (%s)",
                    type(exc).__name__,
                )


def _settings_snapshot(request: Request) -> dict:
    """Read one coherent in-memory settings snapshot while the caller holds the lock."""

    search: SearchProviderManager = request.app.state.search_manager
    llm = request.app.state.llm_client
    asr: ASRClient = request.app.state.asr_client
    omni = getattr(request.app.state, "omni_client", None)
    tts = getattr(request.app.state, "tts_client", None)
    resume_llm = getattr(request.app.state, "resume_llm_client", None)
    return {
        "search": search.status(),
        "llm": llm.settings_status() if isinstance(llm, LocalLLMClient) else {"managed": True},
        "asr": asr.status(),
        "live_audio": omni.status() if omni is not None else {"enabled": False},
        "tts": tts.status() if tts is not None else {"enabled": False},
        "resume_llm": (
            resume_llm.settings_status() if isinstance(resume_llm, LocalLLMClient) else {"managed": True}
        ),
        "persistence": "local_permission_restricted",
        "persisted_locally": request.app.state.settings_store.path.exists(),
        "audio_settings_revision": request.app.state.interview_service.audio_settings_revision,
        "audio_settings_etag": request.app.state.interview_service.audio_settings_etag,
    }


@router.get("")
async def get_settings(request: Request):
    async with request.app.state.settings_lock:
        return _settings_snapshot(request)


@router.put("")
async def update_settings(payload: SettingsUpdate, request: Request):
    try:
        _validate_update_values(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    search: SearchProviderManager = request.app.state.search_manager
    llm = request.app.state.llm_client
    asr: ASRClient = request.app.state.asr_client
    store: LocalSettingsStore = request.app.state.settings_store
    omni = getattr(request.app.state, "omni_client", None)
    tts = getattr(request.app.state, "tts_client", None)
    async with request.app.state.settings_lock:
        service = request.app.state.interview_service
        audio_update_requested = _audio_processing_settings_changed(
            payload, asr, omni
        )
        previous_audio_settings_etag = service.audio_settings_etag
        previous_audio_settings_revision = service.audio_settings_revision
        next_audio_settings_etag = (
            uuid4().hex if audio_update_requested else previous_audio_settings_etag
        )
        next_audio_settings_revision = (
            previous_audio_settings_revision + 1
            if audio_update_requested
            else previous_audio_settings_revision
        )
        if audio_update_requested:
            try:
                await service.ensure_audio_settings_mutable()
            except LiveInterviewStateError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        resume_llm = getattr(request.app.state, "resume_llm_client", None)
        snapshots = {
            "search": search.secret_snapshot(),
            "llm": llm.secret_snapshot() if isinstance(llm, LocalLLMClient) else None,
            "asr": asr.secret_snapshot(),
            "live_audio": omni.secret_snapshot() if omni is not None else None,
            "live_mode": request.app.state.interview_service.live_audio_mode,
            "tts": tts.secret_snapshot() if tts is not None else None,
            "resume_client": resume_llm,
            "resume_llm": (
                resume_llm.secret_snapshot()
                if isinstance(resume_llm, LocalLLMClient)
                else None
            ),
        }
        try:
            if payload.search:
                search.configure(**payload.search.model_dump())
            if payload.llm:
                if not isinstance(llm, LocalLLMClient):
                    raise ValueError("The injected LLM client cannot be configured from the UI")
                await llm.reconfigure(**payload.llm.model_dump())
            if payload.asr:
                asr.configure(**payload.asr.model_dump())
            if payload.live_audio:
                if omni is None:
                    raise ValueError("Live audio direct is not configured")
                omni.configure(**payload.live_audio.model_dump())
                if payload.live_audio.mode is not None:
                    service.set_live_audio_mode(payload.live_audio.mode)
                    omni.mode = payload.live_audio.mode
            if payload.tts:
                if tts is None:
                    raise ValueError("TTS client is not configured")
                tts.configure(**payload.tts.model_dump())
            if payload.resume_llm:
                resume_llm = getattr(request.app.state, "resume_llm_client", None)
                if not isinstance(resume_llm, LocalLLMClient):
                    raise ValueError("Resume LLM client cannot be configured from the UI")
                if resume_llm is llm:
                    # The startup default aliases resume structuring to the main
                    # model. A dedicated UI update must split that alias instead
                    # of silently moving the primary reasoning endpoint too.
                    config = _updated_snapshot({}, payload.resume_llm)
                    resume_llm = llm.clone_with_settings(**config)
                    request.app.state.resume_llm_client = resume_llm
                else:
                    await resume_llm.reconfigure(**payload.resume_llm.model_dump())
                service.resume_llm_client = resume_llm
            if payload.persist:
                persisted_metadata = store.load().get("_meta")
                metadata = (
                    dict(persisted_metadata)
                    if isinstance(persisted_metadata, dict)
                    else {}
                )
                metadata["audio_settings_etag"] = next_audio_settings_etag
                metadata["audio_settings_revision"] = next_audio_settings_revision
                metadata["audio_settings_fingerprint"] = audio_settings_fingerprint(
                    asr, omni
                )
                saved = {
                    "_meta": metadata,
                    "search": search.secret_snapshot(),
                }
                if isinstance(llm, LocalLLMClient):
                    saved["llm"] = llm.secret_snapshot()
                saved["asr"] = asr.secret_snapshot()
                if omni is not None:
                    saved["live_audio"] = omni.secret_snapshot()
                if tts is not None:
                    saved["tts"] = tts.secret_snapshot()
                resume_llm = getattr(request.app.state, "resume_llm_client", None)
                if isinstance(resume_llm, LocalLLMClient):
                    saved["resume_llm"] = resume_llm.secret_snapshot()
                store.save(saved)
            if audio_update_requested:
                service.audio_settings_revision = next_audio_settings_revision
                # Publish only after the transaction succeeds. Persistent
                # updates have already committed this exact generation beside
                # the effective provider settings in the same atomic file.
                service.audio_settings_etag = next_audio_settings_etag
        except BaseException as exc:
            rollback_failed = False
            cancelled_while_restoring = False
            cleanup_clients: list[object] = []
            try:
                cleanup_clients = _restore_runtime_settings_now(request, snapshots)
            except BaseException as restore_exc:  # noqa: BLE001
                logger.error(
                    "Settings rollback failed (%s)", type(restore_exc).__name__
                )
                rollback_failed = True
            rollback_task = (
                asyncio.create_task(_close_uncommitted_clients(cleanup_clients))
                if cleanup_clients
                else None
            )
            while rollback_task is not None:
                try:
                    await asyncio.shield(rollback_task)
                    break
                except asyncio.CancelledError:
                    # A second client disconnect must not release settings_lock
                    # while the shielded rollback is still mutating providers.
                    cancelled_while_restoring = True
                    if rollback_task.done():
                        try:
                            rollback_task.result()
                        except BaseException as restore_exc:  # noqa: BLE001
                            logger.error(
                                "Settings rollback failed (%s)",
                                type(restore_exc).__name__,
                            )
                            rollback_failed = True
                        break
                    continue
                except BaseException as restore_exc:  # noqa: BLE001
                    logger.error(
                        "Settings rollback failed (%s)", type(restore_exc).__name__
                    )
                    rollback_failed = True
                    break
            # If rollback was partial, stale tabs must not keep registering
            # against a configuration whose effective values are uncertain.
            if audio_update_requested and rollback_failed:
                service.refresh_audio_settings_etag()
                service.audio_settings_revision = previous_audio_settings_revision + 1
            else:
                service.audio_settings_etag = previous_audio_settings_etag
                service.audio_settings_revision = previous_audio_settings_revision
            if cancelled_while_restoring and not isinstance(
                exc, asyncio.CancelledError
            ):
                raise asyncio.CancelledError from exc
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if isinstance(exc, OSError):
                raise HTTPException(
                    status_code=500, detail="设置保存失败，运行时配置未更改"
                ) from exc
            raise
        # Build the response before releasing the transaction lock so another
        # settings request cannot make this response describe a later commit.
        return _settings_snapshot(request)


@router.post("/probe-live-audio")
async def probe_live_audio(request: Request):
    async with request.app.state.settings_lock:
        omni = getattr(request.app.state, "omni_client", None)
        if omni is None:
            raise HTTPException(status_code=400, detail="Live audio direct is not configured")
        # Probe an immutable client snapshot outside the settings transaction.
        # A slow local multimodal model must not block capture registration or
        # consume the short post-pause audio drain window.
        settings_etag = request.app.state.interview_service.audio_settings_etag
        isolated_probe = isinstance(omni, OmniAudioClient)
        probe_client = omni.clone_for_probe() if isolated_probe else omni
    try:
        capability = await probe_client.probe_capability()
    finally:
        if isolated_probe:
            await probe_client.close()
    if isolated_probe:
        async with request.app.state.settings_lock:
            service = request.app.state.interview_service
            if (
                request.app.state.omni_client is omni
                and service.audio_settings_etag == settings_etag
            ):
                omni.publish_capability(capability)
    return {"capability": capability}
