"""Local-only, read-only operational debug console API."""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from interview_os.api.dependencies import get_interview_service
from interview_os.core.debug import DebugEventStore, DebugLevel
from interview_os.core.request_context import current_owner
from interview_os.core.state import InterviewState
from interview_os.models.local_llm import LocalLLMClient
from interview_os.services.interview_service import InterviewService

router = APIRouter()
Service = Annotated[InterviewService, Depends(get_interview_service)]
LOCAL_CLIENTS = {"127.0.0.1", "::1", "testclient"}


def require_local_request(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in LOCAL_CLIENTS:
        raise HTTPException(status_code=403, detail="Debug console is restricted to localhost")


@router.get("/status", dependencies=[Depends(require_local_request)])
async def debug_status(request: Request):
    search = request.app.state.search_manager.status()
    llm = request.app.state.llm_client
    return {
        "application": {"status": "running", "version": "0.2.0"},
        "llm": llm.settings_status() if isinstance(llm, LocalLLMClient) else {"managed": True},
        "search": search,
        "asr": request.app.state.asr_client.status(),
        "tts": request.app.state.tts_client.status(),
        "event_capacity": request.app.state.debug_events.capacity,
        "events_persistent": request.app.state.debug_events.persistent,
    }


@router.get("/events", dependencies=[Depends(require_local_request)])
async def debug_events(
    request: Request,
    service: Service,
    limit: int = Query(default=100, ge=1, le=500),
    level: DebugLevel | None = None,
    session_id: str = "",
):
    store: DebugEventStore = request.app.state.debug_events
    owner = current_owner()
    return {
        "events": [
            event.model_dump(mode="json")
            for event in store.list_events(
                limit=limit,
                level=level,
                session_id=session_id,
                owner_id=owner,
            )
        ]
    }


@router.get("/sessions", dependencies=[Depends(require_local_request)])
async def debug_sessions(service: Service, limit: int = Query(default=50, ge=1, le=200)):
    return {
        "sessions": await service.storage.list_sessions(limit, owner_id=current_owner())
    }


@router.get("/sessions/{session_id}", dependencies=[Depends(require_local_request)])
async def debug_session(session_id: str, service: Service):
    owner = current_owner()
    raw = await service.storage.get_session_state(session_id, owner_id=owner)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    state = InterviewState.model_validate(raw)
    runtime = service.get_cached_runtime(session_id, owner_id=owner)
    messages = runtime.get_message_log() if runtime else []
    return {
        "state": {
            "session_id": session_id,
            "stage": state.current_stage.value,
            "autopilot": state.autopilot.status.value,
            "evidence_count": len(state.evidence),
            "fact_card_count": len(state.fact_cards),
            "resume_claim_counts": {
                status: sum(
                    1 for claim in state.resume_review.claims if claim.status.value == status
                )
                for status in (
                    "unverified",
                    "confirmed",
                    "modified",
                    "needs_documents",
                    "disputed",
                    "ignored",
                )
            },
        },
        "messages": [
            {
                "id": str(message.id),
                "type": message.type.value,
                "sender": message.sender,
                "recipient": message.recipient,
                "content_chars": len(message.content),
                "timestamp": message.timestamp.isoformat(),
            }
            for message in messages[-100:]
        ],
    }


@router.post("/probes/llm", dependencies=[Depends(require_local_request)])
async def probe_llm(request: Request):
    llm = request.app.state.llm_client
    if not isinstance(llm, LocalLLMClient):
        return {"ok": True, "managed": True}
    try:
        return await llm.probe()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="LLM probe failed") from exc


@router.post("/probes/asr", dependencies=[Depends(require_local_request)])
async def probe_asr(request: Request):
    result = await request.app.state.asr_client.probe()
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "ASR probe failed"))
    return result
