"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import hmac
import logging
import math
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from interview_os.api import auth
from interview_os.api.request_limits import (
    MultipartBodyLimitMiddleware,
    upload_body_limits,
)
from interview_os.api.routes import (
    analysis,
    audio,
    autopilot,
    debug,
    evaluations,
    intelligence,
    interviews,
    live_interviews,
    mock_interviews,
    resumes,
    settings,
    workflows,
)
from interview_os.core.debug import DebugEventStore
from interview_os.core.provider_config import normalize_provider_api_key
from interview_os.database.storage import Storage
from interview_os.models.local_llm import LocalLLMClient
from interview_os.models.omni_client import OmniAudioClient
from interview_os.runtime import resolve_runtime_paths
from interview_os.services.interview_service import (
    CandidateSessionStateError,
    EvaluationStateError,
    InterviewService,
    LiveInterviewStateError,
    MockInterviewStateError,
    ResumeReviewStateError,
    SessionNotFoundError,
    WorkflowExecutionError,
)
from interview_os.services.resume_service import ResumeProcessingError
from interview_os.services.settings_service import (
    LocalSettingsStore,
    audio_settings_fingerprint,
    is_valid_http_endpoint,
)
from interview_os.tools.asr import ASRClient
from interview_os.tools.tts import TTSClient
from interview_os.tools.web_search import SearchProvider, SearchProviderManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _bootstrap_token_matches(supplied: str, expected: str) -> bool:
    """Compare arbitrary HTTP header text without compare_digest type errors."""

    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _json_safe_validation_detail(
    value: Any, _seen: set[int] | None = None
) -> Any:
    """Keep validation responses serializable when the JSON parser produced inf."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (dict, list, tuple)):
        seen = _seen if _seen is not None else set()
        identity = id(value)
        if identity in seen:
            return "<recursive>"
        seen.add(identity)
        try:
            if isinstance(value, dict):
                return {
                    key if isinstance(key, str) else str(key): _json_safe_validation_detail(
                        item, seen
                    )
                    for key, item in value.items()
                }
            return [_json_safe_validation_detail(item, seen) for item in value]
        finally:
            seen.remove(identity)
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - validation reporting must remain serializable
        return f"<{type(value).__name__}>"


def _validated_persisted_settings(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Validate untrusted local settings before passing sections to constructors."""

    section_models: dict[str, type[BaseModel]] = {
        "search": settings.SearchSettingsUpdate,
        "llm": settings.LLMSettingsUpdate,
        "asr": settings.ASRSettingsUpdate,
        "live_audio": settings.LiveAudioSettingsUpdate,
        "tts": settings.TTSSettingsUpdate,
        "resume_llm": settings.ResumeLLMSettingsUpdate,
    }
    # Older builds persisted resume-model fields inherited from the general LLM
    # form. They were never consumed by the resume client and can be discarded
    # during the next normal save without treating an otherwise valid file as
    # potentially hostile or corrupt.
    known_legacy_fields = {
        "resume_llm": {
            "embedding_model",
            "input_cost_per_million",
            "output_cost_per_million",
        }
    }
    validated: dict[str, Any] = {}
    invalid = bool(set(payload).difference({"_meta", *section_models}))
    if "_meta" in payload:
        if isinstance(payload["_meta"], dict):
            validated["_meta"] = payload["_meta"]
        else:
            invalid = True
    for name, model in section_models.items():
        if name not in payload:
            continue
        raw = payload[name]
        if not isinstance(raw, dict):
            invalid = True
            continue
        unknown = set(raw).difference(model.model_fields)
        if unknown.difference(known_legacy_fields.get(name, set())):
            invalid = True
        try:
            section = model.model_validate(raw).model_dump(exclude_none=True)
        except (TypeError, ValueError):
            invalid = True
            continue
        endpoint = section.get("base_url")
        searxng_endpoint = section.get("searxng_base_url")
        if (
            isinstance(endpoint, str)
            and not is_valid_http_endpoint(endpoint)
        ) or (
            isinstance(searxng_endpoint, str)
            and not is_valid_http_endpoint(searxng_endpoint, allow_empty=True)
        ):
            invalid = True
            continue
        credential_names = (
            ("tavily_api_key", "Tavily API key"),
            ("brave_api_key", "Brave API key"),
            ("api_key", f"{name} API key"),
        )
        try:
            for field_name, label in credential_names:
                credential = section.get(field_name)
                if credential is not None:
                    normalize_provider_api_key(credential, label=label)
        except ValueError:
            invalid = True
            continue
        validated[name] = section
    return validated, invalid


def create_app(
    *,
    storage: Storage | None = None,
    llm_client: Any = None,
    search_provider: SearchProvider | None = None,
    configure_llm: bool = True,
    settings_store: LocalSettingsStore | None = None,
    asr_client: ASRClient | None = None,
    omni_client: OmniAudioClient | None = None,
    tts_client: TTSClient | None = None,
    resume_llm_client: Any = None,
    recordings_dir: Path | None = None,
    require_auth: bool | None = None,
) -> FastAPI:
    runtime_paths = resolve_runtime_paths()
    uses_runtime_database = storage is None and not os.getenv("DATABASE_URL")
    storage = storage or Storage(os.getenv("DATABASE_URL", runtime_paths.database_url))
    if require_auth is None:
        require_auth = os.getenv("INTERVIEW_OS_REQUIRE_AUTH", "1") != "0"
    use_persistent_runtime = settings_store is not None or (llm_client is None and configure_llm)
    settings_store = settings_store or LocalSettingsStore(runtime_paths.settings_path)
    saved_settings, settings_load_status = settings_store.load_with_status()
    saved_settings, invalid_settings_shape = _validated_persisted_settings(
        saved_settings
    )
    if invalid_settings_shape and settings_load_status == "ok":
        settings_load_status = "invalid"
    if settings_load_status in {"invalid", "error"}:
        logger.warning(
            "Persistent settings could not be read (%s); leaving the source file untouched",
            settings_load_status,
        )
    saved_metadata = saved_settings.get("_meta")
    audio_settings_etag = (
        saved_metadata.get("audio_settings_etag")
        if isinstance(saved_metadata, dict)
        else None
    )
    audio_settings_revision = (
        saved_metadata.get("audio_settings_revision")
        if isinstance(saved_metadata, dict)
        else 0
    )
    if (
        not isinstance(audio_settings_revision, int)
        or isinstance(audio_settings_revision, bool)
        or audio_settings_revision < 0
    ):
        audio_settings_revision = 0
    valid_audio_settings_etag = isinstance(audio_settings_etag, str) and bool(
        re.fullmatch(r"[0-9a-f]{32}", audio_settings_etag)
    )
    if not valid_audio_settings_etag:
        audio_settings_etag = uuid4().hex
    runtime_dir = settings_store.path.parent
    if llm_client is None and configure_llm:
        llm_client = LocalLLMClient(
            **saved_settings.get("llm", {}), metrics_path=runtime_dir / "llm_metrics.json"
        )
    debug_events = DebugEventStore(
        capacity=int(os.getenv("DEBUG_EVENT_CAPACITY", "500")),
        path=runtime_dir / "debug_events.json" if use_persistent_runtime else None,
    )
    search_manager = SearchProviderManager(
        debug_events=debug_events,
        cache_path=runtime_dir / "search_cache.json" if use_persistent_runtime else None,
        metrics_path=runtime_dir / "search_metrics.json" if use_persistent_runtime else None,
    )
    saved_search = saved_settings.get("search")
    if isinstance(saved_search, dict):
        try:
            search_manager.configure(**saved_search)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid persisted search settings")
    active_search_provider = search_provider or search_manager
    asr_client = asr_client or ASRClient(**saved_settings.get("asr", {}))
    tts_client = tts_client or TTSClient(**saved_settings.get("tts", {}))
    saved_live_audio = saved_settings.get("live_audio")
    if omni_client is None:
        omni_client = (
            OmniAudioClient(**saved_live_audio)
            if isinstance(saved_live_audio, dict) and saved_live_audio
            else OmniAudioClient()
        )
    saved_mode = (
        saved_live_audio.get("mode", "asr_text")
        if isinstance(saved_live_audio, dict)
        else "asr_text"
    )
    if isinstance(saved_mode, str) and saved_mode in {"asr_text", "audio_direct"}:
        omni_client.mode = saved_mode
    effective_audio_fingerprint = audio_settings_fingerprint(asr_client, omni_client)
    stored_audio_fingerprint = (
        saved_metadata.get("audio_settings_fingerprint")
        if isinstance(saved_metadata, dict)
        else None
    )
    if valid_audio_settings_etag and stored_audio_fingerprint != effective_audio_fingerprint:
        # A new default/environment endpoint or a schema upgrade must fence
        # captures recorded under the prior effective configuration.
        audio_settings_etag = uuid4().hex
        audio_settings_revision += 1
    audio_metadata_needs_persist = settings_load_status in {"ok", "missing"} and (
        not valid_audio_settings_etag
        or stored_audio_fingerprint != effective_audio_fingerprint
    )
    saved_resume_llm = saved_settings.get("resume_llm")
    if resume_llm_client is None:
        resume_llm_client = (
            llm_client.clone_with_settings(**saved_resume_llm)
            if isinstance(llm_client, LocalLLMClient)
            and isinstance(saved_resume_llm, dict)
            and saved_resume_llm.get("base_url")
            else LocalLLMClient(**saved_resume_llm)
            if isinstance(saved_resume_llm, dict) and saved_resume_llm.get("base_url")
            else llm_client
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal audio_metadata_needs_persist

        async def close_runtime_resources() -> None:
            # Background scorers and refill tasks can still be using model clients.
            # Stop them before closing any shared transport, then close each distinct
            # client exactly once (resume_llm commonly aliases llm_client).
            service = getattr(application.state, "interview_service", None)
            if service is not None:
                try:
                    await service.shutdown_runtime_tasks()
                except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                    logger.error("Service task shutdown failed (%s)", type(exc).__name__)
            closeables = [
                llm_client,
                getattr(application.state, "resume_llm_client", resume_llm_client),
                asr_client,
                omni_client,
                tts_client,
            ]
            seen: set[int] = set()
            for client in closeables:
                if client is None or id(client) in seen or not hasattr(client, "close"):
                    continue
                seen.add(id(client))
                try:
                    await client.close()
                except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                    logger.error("Client shutdown failed (%s)", type(exc).__name__)
            try:
                await storage.close()
            except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                logger.error("Storage shutdown failed (%s)", type(exc).__name__)

        try:
            debug_events.rewrite_sanitized_file()
            if audio_metadata_needs_persist:
                current_settings, current_status = settings_store.load_with_status()
                current_settings, current_invalid = _validated_persisted_settings(
                    current_settings
                )
                if current_status in {"ok", "missing"} and not current_invalid:
                    current_metadata = current_settings.get("_meta")
                    metadata = (
                        dict(current_metadata) if isinstance(current_metadata, dict) else {}
                    )
                    metadata["audio_settings_etag"] = audio_settings_etag
                    metadata["audio_settings_revision"] = audio_settings_revision
                    metadata["audio_settings_fingerprint"] = effective_audio_fingerprint
                    settings_to_persist = dict(current_settings)
                    settings_to_persist["_meta"] = metadata
                    try:
                        settings_store.save(settings_to_persist)
                        audio_metadata_needs_persist = False
                    except (OSError, ValueError):
                        # Runtime-only operation remains available; a later successful UI
                        # save persists the generation together with provider settings.
                        logger.warning(
                            "Could not initialize persistent audio settings generation"
                        )
            if uses_runtime_database:
                runtime_paths.database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            await storage.init_db()
            application.state.interview_service = InterviewService(
                storage,
                llm_client,
                active_search_provider,
                debug_events,
                asr_client,
                omni_client=omni_client,
                tts_client=tts_client,
                resume_llm_client=resume_llm_client,
                recordings_dir=recordings_dir or runtime_paths.recordings_dir,
                audio_settings_etag=audio_settings_etag,
                audio_settings_revision=audio_settings_revision,
            )
            recovered_live_audio, removed_live_audio = (
                await application.state.interview_service.cleanup_orphaned_live_audio()
            )
            if recovered_live_audio or removed_live_audio:
                logger.info(
                    "Reconciled live audio transactions: recovered=%d removed=%d",
                    recovered_live_audio,
                    removed_live_audio,
                )
            removed_mock_audio = (
                await application.state.interview_service.cleanup_orphaned_mock_audio()
            )
            if removed_mock_audio:
                logger.info("Removed %d expired unbound mock recording(s)", removed_mock_audio)
            application.state.storage = storage
            if isinstance(saved_mode, str):
                try:
                    application.state.interview_service.set_live_audio_mode(saved_mode)
                    omni_client.mode = saved_mode
                except ValueError:
                    logger.warning("Ignoring invalid persisted live audio mode: %s", saved_mode)
            application.state.search_manager = search_manager
            application.state.llm_client = llm_client
            application.state.debug_events = debug_events
            application.state.settings_store = settings_store
            application.state.asr_client = asr_client
            application.state.omni_client = omni_client
            application.state.tts_client = tts_client
            application.state.resume_llm_client = resume_llm_client
            application.state.settings_lock = application.state.interview_service._settings_lock
            yield
        finally:
            # Cleanup has its own ownership. A host cancellation is remembered
            # and re-raised only after every shared runtime resource has closed.
            cleanup_task = asyncio.create_task(
                close_runtime_resources(), name="interview-os-runtime-shutdown"
            )
            shutdown_cancelled = False
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    shutdown_cancelled = True
            try:
                cleanup_task.result()
            except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                logger.error("Runtime shutdown failed (%s)", type(exc).__name__)
            if shutdown_cancelled:
                raise asyncio.CancelledError()

    application = FastAPI(
        title="InterviewOS",
        description="A Local LLM-powered Interview Intelligence Operating System",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.runtime_paths = runtime_paths
    bootstrap_token = os.getenv("INTERVIEW_OS_BOOTSTRAP_TOKEN", "").strip()

    if bootstrap_token:

        @application.middleware("http")
        async def require_desktop_bootstrap(request: Request, call_next):
            """Keep an authenticated desktop sidecar private to its launcher."""

            protected_path = request.url.path == "/health" or request.url.path.startswith("/api/")
            if request.method != "OPTIONS" and protected_path:
                supplied = request.headers.get("X-InterviewOS-Bootstrap", "")
                if not _bootstrap_token_matches(supplied, bootstrap_token):
                    return JSONResponse(status_code=403, content={"detail": "Invalid app bootstrap"})
            return await call_next(request)

    default_origins = "http://127.0.0.1:8000,http://localhost:8000"
    if runtime_paths.mode == "desktop":
        default_origins += ",tauri://localhost,http://tauri.localhost,https://tauri.localhost"
    allowed_origins = os.getenv("CORS_ORIGINS", default_origins).split(",")
    # Starlette wraps later registrations around earlier ones. Keep CORS on the
    # outside so even an early request-body rejection remains readable by the
    # desktop renderer instead of surfacing as an opaque network failure.
    application.add_middleware(
        MultipartBodyLimitMiddleware,
        limits=upload_body_limits(),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    protected = [Depends(auth.get_current_user)] if require_auth else []
    application.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    application.include_router(
        interviews.router, prefix="/api/interviews", tags=["interviews"], dependencies=protected
    )
    application.include_router(
        analysis.router, prefix="/api/analysis", tags=["analysis"], dependencies=protected
    )
    application.include_router(
        autopilot.router, prefix="/api/autopilot", tags=["autopilot"], dependencies=protected
    )
    application.include_router(
        settings.router, prefix="/api/settings", tags=["settings"], dependencies=protected
    )
    application.include_router(
        workflows.router, prefix="/api/workflows", tags=["workflows"], dependencies=protected
    )
    application.include_router(
        mock_interviews.router,
        prefix="/api/mock-interviews",
        tags=["mock-interviews"],
        dependencies=protected,
    )
    application.include_router(
        debug.router, prefix="/api/debug", tags=["debug"], dependencies=protected
    )
    application.include_router(
        resumes.router, prefix="/api/resumes", tags=["resumes"], dependencies=protected
    )
    application.include_router(
        evaluations.router,
        prefix="/api/evaluations",
        tags=["evaluations"],
        dependencies=protected,
    )
    application.include_router(
        intelligence.router,
        prefix="/api/intelligence",
        tags=["intelligence"],
        dependencies=protected,
    )
    application.include_router(
        live_interviews.router,
        prefix="/api/live-interviews",
        tags=["live-interviews"],
        dependencies=protected,
    )
    application.include_router(
        audio.router,
        prefix="/api/live-interviews",
        tags=["live-interviews-audio"],
        dependencies=protected,
    )
    application.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    application.mount(
        "/modules", StaticFiles(directory=WEB_DIR / "modules"), name="web-modules"
    )

    @application.get("/app.js", include_in_schema=False)
    async def web_application_script():
        return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")

    @application.get("/styles.css", include_in_schema=False)
    async def web_application_styles():
        return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

    @application.exception_handler(SessionNotFoundError)
    async def session_not_found(_: Request, exc: SessionNotFoundError):
        return JSONResponse(status_code=404, content={"detail": f"Session '{exc}' not found"})

    @application.exception_handler(RequestValidationError)
    async def invalid_request_payload(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": _json_safe_validation_detail(exc.errors())},
        )

    @application.exception_handler(WorkflowExecutionError)
    async def workflow_failed(_: Request, exc: WorkflowExecutionError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @application.exception_handler(MockInterviewStateError)
    async def invalid_mock_state(_: Request, exc: MockInterviewStateError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.exception_handler(EvaluationStateError)
    async def invalid_evaluation_state(_: Request, exc: EvaluationStateError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.exception_handler(ResumeProcessingError)
    async def invalid_resume(_: Request, exc: ResumeProcessingError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.exception_handler(ResumeReviewStateError)
    async def invalid_resume_review(_: Request, exc: ResumeReviewStateError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(CandidateSessionStateError)
    async def invalid_candidate_session(_: Request, exc: CandidateSessionStateError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.exception_handler(LiveInterviewStateError)
    async def invalid_live_interview(_: Request, exc: LiveInterviewStateError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.get("/")
    async def root(request: Request):
        accept = request.headers.get("accept", "")
        if "application/json" in accept and "text/html" not in accept:
            return JSONResponse({"name": "InterviewOS", "version": "0.2.0", "status": "running"})
        # Inject a version derived from all entry and module mtimes so browsers
        # invalidate the frontend whenever an extracted ES module changes,
        # without a manual version bump.
        version = 0
        assets = [WEB_DIR / "app.js", WEB_DIR / "styles.css"]
        assets.extend((WEB_DIR / "modules").glob("*.js"))
        for asset in assets:
            try:
                version = max(version, int(os.stat(asset).st_mtime_ns))
            except OSError:
                continue
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        html = re.sub(r"\?v=STATIC_VERSION", f"?v={version}", html)
        return HTMLResponse(html)

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    return application


app = create_app()
