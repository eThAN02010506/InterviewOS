"""Runtime settings API and local settings UI."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from interview_os.models.local_llm import LocalLLMClient
from interview_os.services.settings_service import LocalSettingsStore
from interview_os.tools.asr import ASRClient
from interview_os.tools.tts import TTSClient
from interview_os.tools.web_search import SearchProviderManager

router = APIRouter()
logger = logging.getLogger(__name__)


class SearchSettingsUpdate(BaseModel):
    provider: Literal["none", "tavily", "searxng", "brave"]
    tavily_api_key: str | None = None
    searxng_base_url: str | None = None
    brave_api_key: str | None = None
    search_request_cost_usd: float | None = Field(default=None, ge=0)


class LLMSettingsUpdate(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    embedding_model: str | None = None
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)


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


async def _restore_runtime_settings(request: Request, snapshots: dict) -> None:
    """Best-effort rollback for a settings transaction that did not commit."""
    search: SearchProviderManager = request.app.state.search_manager
    llm = request.app.state.llm_client
    asr: ASRClient = request.app.state.asr_client
    omni = getattr(request.app.state, "omni_client", None)
    tts: TTSClient | None = getattr(request.app.state, "tts_client", None)
    service = request.app.state.interview_service
    original_resume = snapshots["resume_client"]
    current_resume = getattr(request.app.state, "resume_llm_client", None)

    search.configure(**snapshots["search"])
    if isinstance(llm, LocalLLMClient):
        await llm.reconfigure(**snapshots["llm"])
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
    if current_resume is not original_resume:
        if current_resume is not None and hasattr(current_resume, "close"):
            await current_resume.close()
        request.app.state.resume_llm_client = original_resume
        service.resume_llm_client = original_resume
    elif isinstance(original_resume, LocalLLMClient) and original_resume is not llm:
        await original_resume.reconfigure(**snapshots["resume_llm"])


@router.get("")
async def get_settings(request: Request):
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
    }


@router.put("")
async def update_settings(payload: SettingsUpdate, request: Request):
    search: SearchProviderManager = request.app.state.search_manager
    llm = request.app.state.llm_client
    asr: ASRClient = request.app.state.asr_client
    store: LocalSettingsStore = request.app.state.settings_store
    omni = getattr(request.app.state, "omni_client", None)
    tts = getattr(request.app.state, "tts_client", None)
    async with request.app.state.settings_lock:
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
                service = request.app.state.interview_service
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
                service = request.app.state.interview_service
                if resume_llm is llm:
                    # The startup default aliases resume structuring to the main
                    # model. A dedicated UI update must split that alias instead
                    # of silently moving the primary reasoning endpoint too.
                    config = _updated_snapshot(llm.secret_snapshot(), payload.resume_llm)
                    resume_llm = LocalLLMClient(**config)
                    request.app.state.resume_llm_client = resume_llm
                else:
                    await resume_llm.reconfigure(**payload.resume_llm.model_dump())
                service.resume_llm_client = resume_llm
            if payload.persist:
                saved = {"search": search.secret_snapshot()}
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
        except Exception as exc:
            try:
                await _restore_runtime_settings(request, snapshots)
            except Exception as restore_exc:  # noqa: BLE001 - preserve original error
                logger.error("Settings rollback failed (%s)", type(restore_exc).__name__)
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
        return await get_settings(request)


@router.post("/probe-live-audio")
async def probe_live_audio(request: Request):
    omni = getattr(request.app.state, "omni_client", None)
    if omni is None:
        raise HTTPException(status_code=400, detail="Live audio direct is not configured")
    capability = await omni.probe_capability()
    return {"capability": capability}


@router.get("/ui", response_class=HTMLResponse)
async def settings_ui():
    return HTMLResponse(SETTINGS_HTML)


SETTINGS_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>InterviewOS 设置</title><style>
body{font:15px system-ui;max-width:720px;margin:40px auto;padding:0 20px;color:#17202a}
fieldset{border:1px solid #d9e0e7;border-radius:10px;margin:18px 0;padding:18px}label{display:block;margin:12px 0 5px}
input,select,button{box-sizing:border-box;width:100%;padding:10px;border:1px solid #b8c2cc;border-radius:7px}
button{margin-top:18px;background:#1f6feb;color:white;border:0;cursor:pointer}.hint{color:#65717e;font-size:13px}
</style></head><body><h1>InterviewOS 设置</h1>
<p class="hint">密钥不会由 API 回传、日志或 Debug Console 展示。设置会保存到仅当前用户可读的本地文件。</p>
<form id="form"><fieldset><legend>联网搜索</legend><label>Provider</label><select id="provider">
<option value="tavily">Tavily</option><option value="searxng">SearXNG</option><option value="brave">Brave</option><option value="none">关闭</option></select>
<label>Tavily API Key</label><input id="tavily" type="password" autocomplete="off" placeholder="已配置时可留空">
<label>SearXNG 地址</label><input id="searxng" placeholder="http://localhost:8080">
<label>Brave API Key</label><input id="brave" type="password" autocomplete="off" placeholder="已配置时可留空"></fieldset>
<fieldset><legend>Local LLM</legend><label>API 地址</label><input id="base_url"><label>模型</label><input id="model">
<label>API Key</label><input id="llm_key" type="password" autocomplete="off" placeholder="已配置时可留空"></fieldset>
<button>保存并立即应用</button><p id="status" class="hint"></p></form><script>
const val=id=>document.getElementById(id).value; const optional=id=>val(id)||null;
async function load(){const s=await (await fetch('/api/settings')).json();document.getElementById('provider').value=s.search.selected;document.getElementById('base_url').value=s.llm.base_url||'';document.getElementById('model').value=s.llm.model||''}
form.onsubmit=async e=>{e.preventDefault();const body={search:{provider:val('provider'),tavily_api_key:optional('tavily'),searxng_base_url:optional('searxng'),brave_api_key:optional('brave')},llm:{base_url:optional('base_url'),model:optional('model'),api_key:optional('llm_key')}};
const r=await fetch('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();status.textContent=r.ok?'设置已应用':(d.detail||'保存失败')};load();
</script></body></html>"""
