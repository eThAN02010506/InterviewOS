"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from interview_os.api.routes import (
    analysis,
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
from interview_os.database.storage import Storage
from interview_os.models.local_llm import LocalLLMClient
from interview_os.services.interview_service import (
    EvaluationStateError,
    InterviewService,
    LiveInterviewStateError,
    MockInterviewStateError,
    ResumeReviewStateError,
    SessionNotFoundError,
    WorkflowExecutionError,
)
from interview_os.services.resume_service import ResumeProcessingError
from interview_os.services.settings_service import LocalSettingsStore
from interview_os.tools.asr import ASRClient
from interview_os.tools.web_search import SearchProvider, SearchProviderManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(
    *,
    storage: Storage | None = None,
    llm_client: Any = None,
    search_provider: SearchProvider | None = None,
    configure_llm: bool = True,
    settings_store: LocalSettingsStore | None = None,
    asr_client: ASRClient | None = None,
) -> FastAPI:
    storage = storage or Storage(os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./interview_os.db"))
    use_persistent_runtime = settings_store is not None or (llm_client is None and configure_llm)
    settings_store = settings_store or LocalSettingsStore()
    saved_settings = settings_store.load()
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

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await storage.init_db()
        application.state.interview_service = InterviewService(
            storage, llm_client, active_search_provider, debug_events, asr_client
        )
        application.state.search_manager = search_manager
        application.state.llm_client = llm_client
        application.state.debug_events = debug_events
        application.state.settings_store = settings_store
        application.state.asr_client = asr_client
        yield
        if llm_client is not None and hasattr(llm_client, "close"):
            await llm_client.close()
        await application.state.interview_service._background.close()
        await asr_client.close()
        await storage.close()

    application = FastAPI(
        title="InterviewOS",
        description="A Local LLM-powered Interview Intelligence Operating System",
        version="0.2.0",
        lifespan=lifespan,
    )
    allowed_origins = os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(interviews.router, prefix="/api/interviews", tags=["interviews"])
    application.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
    application.include_router(autopilot.router, prefix="/api/autopilot", tags=["autopilot"])
    application.include_router(settings.router, prefix="/api/settings", tags=["settings"])
    application.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])
    application.include_router(
        mock_interviews.router, prefix="/api/mock-interviews", tags=["mock-interviews"]
    )
    application.include_router(debug.router, prefix="/api/debug", tags=["debug"])
    application.include_router(resumes.router, prefix="/api/resumes", tags=["resumes"])
    application.include_router(evaluations.router, prefix="/api/evaluations", tags=["evaluations"])
    application.include_router(
        intelligence.router, prefix="/api/intelligence", tags=["intelligence"]
    )
    application.include_router(
        live_interviews.router, prefix="/api/live-interviews", tags=["live-interviews"]
    )
    application.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @application.exception_handler(SessionNotFoundError)
    async def session_not_found(_: Request, exc: SessionNotFoundError):
        return JSONResponse(status_code=404, content={"detail": f"Session '{exc}' not found"})

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

    @application.exception_handler(LiveInterviewStateError)
    async def invalid_live_interview(_: Request, exc: LiveInterviewStateError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.get("/")
    async def root(request: Request):
        accept = request.headers.get("accept", "")
        if "application/json" in accept and "text/html" not in accept:
            return JSONResponse({"name": "InterviewOS", "version": "0.2.0", "status": "running"})
        return FileResponse(WEB_DIR / "index.html")

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    return application


app = create_app()
