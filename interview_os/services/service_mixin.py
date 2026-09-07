"""Typed dependency contract shared by InterviewService domain mixins."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

from interview_os.core.debug import DebugEventStore, DebugLevel
from interview_os.core.message import Message
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import (
    InterviewerProfile,
    InterviewState,
    SpeechDeliveryFeedback,
    TranscriptSegment,
    TranscriptSpeaker,
)
from interview_os.database.storage import Storage
from interview_os.services.background import BackgroundTaskManager
from interview_os.services.resume_service import ResumeProcessor
from interview_os.tools.asr import ASRClient
from interview_os.tools.tts import TTSClient
from interview_os.tools.web_search import SearchProvider


class InterviewServiceMixin:
    """Declare facade-owned dependencies used by domain service modules.

    The concrete ``InterviewService`` supplies these members. Keeping the
    contract explicit lets type checking catch accidental cross-module calls
    while the facade preserves the historical public API.
    """

    storage: Storage
    llm_client: Any
    search_provider: SearchProvider | None
    debug_events: DebugEventStore | None
    asr_client: ASRClient | None
    omni_client: Any
    tts_client: TTSClient | None
    resume_llm_client: Any
    live_audio_mode: str
    audio_settings_revision: int
    audio_settings_etag: str

    def refresh_audio_settings_etag(self) -> str:
        raise NotImplementedError
    _background: BackgroundTaskManager
    _runtimes: dict[tuple[str, str], AgentRuntime]
    _locks: dict[tuple[str, str], Any]
    _raw_locks: dict[tuple[str, str], asyncio.Lock]
    _runtime_load_locks: dict[tuple[str, str], asyncio.Lock]
    _runtime_recovery_required: set[tuple[str, str]]
    _settings_lock: asyncio.Lock
    _active_audio_settings_leases: int
    _recordings_dir: Path
    _mock_speech_feedback: dict[tuple[str, str, UUID], tuple[str, SpeechDeliveryFeedback, float]]
    _live_audio_in_flight: dict[
        tuple[str, str, UUID],
        tuple[str, asyncio.Task[tuple[InterviewState, str]]],
    ]
    resume_processor: ResumeProcessor

    async def _get_runtime(self, session_id: str) -> AgentRuntime:
        raise NotImplementedError

    def _lock_for(self, session_id: str) -> Any:
        raise NotImplementedError

    async def _persist(self, session_id: str, state: InterviewState) -> None:
        raise NotImplementedError

    async def _recover_live_state_after_persist_error(
        self,
        session_id: str,
        runtime: AgentRuntime,
        previous_state: InterviewState,
    ) -> None:
        raise NotImplementedError

    def _reconcile_live_audio_for_state(
        self,
        session_id: str,
        state: InterviewState,
        *,
        strict: bool = False,
    ) -> tuple[int, int]:
        raise NotImplementedError

    def _record_debug(
        self,
        action: str,
        session_id: str,
        *,
        level: DebugLevel = DebugLevel.INFO,
        detail: str = "",
        duration_ms: float | None = None,
    ) -> None:
        raise NotImplementedError

    async def _run(self, session_id: str, agent_name: str, instruction: str) -> Message:
        raise NotImplementedError

    async def _execute_workflow(
        self,
        session_id: str,
        runtime: AgentRuntime,
        name: str,
        steps: list[tuple[str, str]],
        company_name: str | None = None,
        company_context: str | None = None,
        interviewer: InterviewerProfile | None = None,
        parallel_prefix: int = 0,
        authorized_public_research: bool = False,
    ) -> InterviewState:
        raise NotImplementedError

    def _invalidate_final_reports(self, state: InterviewState, next_action: str) -> bool:
        raise NotImplementedError

    def _sync_intelligence(self, state: InterviewState) -> None:
        raise NotImplementedError

    def _pending_mock_audio_path(self, session_id: str, recording_id: UUID) -> Path | None:
        raise NotImplementedError

    def current_mock_question(self, state: InterviewState) -> Any:
        raise NotImplementedError

    async def run_evaluation(self, session_id: str) -> InterviewState:
        raise NotImplementedError

    @staticmethod
    def assert_live_active(state: InterviewState) -> None:
        raise NotImplementedError

    async def confirm_live_answer(
        self,
        session_id: str,
        answer_segment_id: UUID,
        *,
        question_segment_id: UUID | None = None,
        question: str = "",
        competency: str = "",
    ) -> InterviewState:
        raise NotImplementedError

    @staticmethod
    def _resolve_live_question_segment(
        segments: list[TranscriptSegment],
        answer_segment: TranscriptSegment,
        question_segment_id: UUID | None,
    ) -> TranscriptSegment | None:
        raise NotImplementedError

    def _refresh_live_answer_boundaries(self, state: InterviewState) -> None:
        raise NotImplementedError

    def _refresh_live_rolling_summary(self, state: InterviewState) -> None:
        raise NotImplementedError

    async def _maybe_plan_live_next_question(self, session_id: str) -> None:
        raise NotImplementedError

    @staticmethod
    def _live_target_competencies(state: InterviewState) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def _infer_live_competency(state: InterviewState, question: str) -> str:
        raise NotImplementedError

    async def append_live_transcript(
        self,
        session_id: str,
        *,
        text: str,
        speaker: TranscriptSpeaker,
        source: str = "manual",
    ) -> InterviewState:
        raise NotImplementedError

    def _append_live_transcript_locked(
        self,
        state: InterviewState,
        *,
        text: str,
        speaker: TranscriptSpeaker,
        source: str,
        allow_completed_capture: bool = False,
    ) -> TranscriptSegment | None:
        raise NotImplementedError

    def _refresh_live_coverage_guidance(self, state: InterviewState) -> None:
        raise NotImplementedError

    def _refresh_live_question_usage(self, state: InterviewState) -> None:
        raise NotImplementedError

    def _refresh_live_action_card(self, state: InterviewState) -> None:
        raise NotImplementedError

    def _live_audio_context(self, state: InterviewState) -> str:
        raise NotImplementedError

    async def _inject_audio_direct_suggestion(
        self,
        session_id: str,
        runtime: AgentRuntime,
        suggestion_text: str,
    ) -> None:
        raise NotImplementedError
