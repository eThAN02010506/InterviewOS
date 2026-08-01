"""FastAPI application entry point."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from interview_os.api.routes import analysis, interviews, mock_interviews, settings, workflows
from interview_os.database.storage import Storage
from interview_os.models.local_llm import LocalLLMClient
from interview_os.services.interview_service import (
    InterviewService,
    MockInterviewStateError,
    SessionNotFoundError,
    WorkflowExecutionError,
)
from interview_os.tools.web_search import SearchProvider, SearchProviderManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(
    *,
    storage: Storage | None = None,
    llm_client: Any = None,
    search_provider: SearchProvider | None = None,
    configure_llm: bool = True,
) -> FastAPI:
    storage = storage or Storage(os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./interview_os.db"))
    if llm_client is None and configure_llm:
        llm_client = LocalLLMClient()
    search_manager = SearchProviderManager()
    active_search_provider = search_provider or search_manager

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await storage.init_db()
        application.state.interview_service = InterviewService(
            storage, llm_client, active_search_provider
        )
        application.state.search_manager = search_manager
        application.state.llm_client = llm_client
        yield
        if llm_client is not None and hasattr(llm_client, "close"):
            await llm_client.close()
        await storage.close()

    application = FastAPI(
        title="InterviewOS",
        description="A Local LLM-powered Interview Intelligence Operating System",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(interviews.router, prefix="/api/interviews", tags=["interviews"])
    application.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
    application.include_router(settings.router, prefix="/api/settings", tags=["settings"])
    application.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])
    application.include_router(
        mock_interviews.router, prefix="/api/mock-interviews", tags=["mock-interviews"]
    )

    @application.exception_handler(SessionNotFoundError)
    async def session_not_found(_: Request, exc: SessionNotFoundError):
        return JSONResponse(status_code=404, content={"detail": f"Session '{exc}' not found"})

    @application.exception_handler(WorkflowExecutionError)
    async def workflow_failed(_: Request, exc: WorkflowExecutionError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @application.exception_handler(MockInterviewStateError)
    async def invalid_mock_state(_: Request, exc: MockInterviewStateError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.get("/")
    async def root():
        return {"name": "InterviewOS", "version": "0.2.0", "status": "running"}

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    return application


app = create_app()
