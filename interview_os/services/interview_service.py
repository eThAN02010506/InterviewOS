"""Use-case layer coordinating runtimes and persistence."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import wave
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from interview_os.core.answer_feedback import apply_specific_feedback
from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel
from interview_os.core.evidence import Evidence, EvidencePolarity, EvidenceSource
from interview_os.core.factory import create_runtime
from interview_os.core.message import Message
from interview_os.core.question_understanding import deterministic_question_understanding
from interview_os.core.request_context import current_owner
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import (
    ACTION_CARD_SOURCE_REFS_MAX,
    ANSWER_BOUNDARY_SUGGESTIONS_MAX,
    BOUNDARY_BASE_SCORE,
    BOUNDARY_MAX_SCORE,
    BOUNDARY_MERGE_THRESHOLD,
    BOUNDARY_MIN_SCORE,
    COVERAGE_GUIDANCE_MAX_ITEMS,
    CROSS_VALIDATION_EVIDENCE_COUNT,
    LIVE_RECENT_SEGMENT_WINDOW,
    LIVE_SUMMARY_CHAR_LIMIT,
    MIN_ANSWER_BOUNDARY_SEGMENTS,
    MIN_COMPETENCY_COVERAGE,
    MIN_EVIDENCE_COUNT,
    MOCK_CACHE_MIN,
    OMNI_CONTEXT_CHAR_LIMIT,
    QUESTION_USAGE_MAX_ITEMS,
    WEAK_SIGNAL_THRESHOLD,
    AnswerEvaluation,
    AnswerReviewStatus,
    AnswerScoringSource,
    AutopilotState,
    AutopilotStatus,
    CandidateProfile,
    CompanyInfo,
    EntityResolution,
    EvaluationReport,
    FeedbackReport,
    InterviewBlueprint,
    InterviewerProfile,
    InterviewQuestion,
    InterviewStage,
    InterviewState,
    InterviewStrategy,
    JobDescription,
    JobDescriptionReview,
    LiveActionCard,
    LiveAnswerBoundarySuggestion,
    LiveCoverageGuidance,
    LiveInterviewRecord,
    LiveInterviewSession,
    LiveInterviewStatus,
    LiveQuestionUsage,
    MockAnswerRecord,
    MockInterviewPlan,
    MockInterviewSession,
    MockSessionStatus,
    QuestionSuggestion,
    QuestionSuggestionStatus,
    QuestionSuggestionType,
    RequirementOrigin,
    ResumeClaimStatus,
    ResumeReview,
    SpeechDeliveryFeedback,
    TranscriptSegment,
    TranscriptSpeaker,
    WorkflowProgress,
    WorkflowStatus,
)
from interview_os.database.storage import Storage
from interview_os.services.background import BackgroundTaskManager
from interview_os.services.intelligence_service import (
    build_fact_cards,
    decide_fact_card,
    resolve_entity,
    review_job_description,
    sync_entity_resolutions,
)
from interview_os.services.resume_llm import structure_resume_with_llm
from interview_os.services.resume_ocr import ResumeOCRError, ocr_scanned_pdf
from interview_os.services.resume_service import (
    ResumeProcessingError,
    ResumeProcessor,
    ScannedPDFError,
)
from interview_os.tools.asr import ASRClient, ASRError
from interview_os.tools.tts import TTSClient, TTSError
from interview_os.tools.web_search import (
    SearchProvider,
    filter_employer_business_results,
    filter_entity_results,
    format_employer_business_context,
    format_search_results,
    merge_search_results,
)

logger = logging.getLogger(__name__)

SPEECH_FEEDBACK_PROHIBITED_TERMS = (
    "口音",
    "方言",
    "性格",
    "情绪状态",
    "焦虑",
    "抑郁",
    "健康",
    "年龄",
    "性别",
    "族裔",
    "种族",
    "accent",
    "personality",
    "mental health",
    "ethnicity",
    "race",
    "gender",
    "age",
)


class SessionNotFoundError(LookupError):
    pass


class WorkflowExecutionError(RuntimeError):
    pass


class MockInterviewStateError(RuntimeError):
    pass


class EvaluationStateError(RuntimeError):
    pass


class ResumeReviewStateError(RuntimeError):
    pass


class CandidateSessionStateError(RuntimeError):
    pass


class LiveInterviewStateError(RuntimeError):
    pass


class InterviewService:
    """Owns session-scoped runtimes and serializes mutations per session."""

    def __init__(
        self,
        storage: Storage,
        llm_client: Any = None,
        search_provider: SearchProvider | None = None,
        debug_events: DebugEventStore | None = None,
        asr_client: ASRClient | None = None,
        background: BackgroundTaskManager | None = None,
        omni_client: Any = None,
        tts_client: TTSClient | None = None,
        resume_llm_client: Any = None,
        recordings_dir: Path | None = None,
    ) -> None:
        self.storage = storage
        self.llm_client = llm_client
        self.search_provider = search_provider
        self.debug_events = debug_events
        self.asr_client = asr_client
        self.omni_client = omni_client
        self.tts_client = tts_client
        self.resume_llm_client = resume_llm_client
        self.live_audio_mode = "asr_text"  # "asr_text" | "audio_direct"
        self._background = background or BackgroundTaskManager(debug_events=debug_events)
        self._runtimes: dict[tuple[str, str], AgentRuntime] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._recordings_dir = recordings_dir or Path("data/recordings")
        self._mock_speech_feedback: dict[
            tuple[str, str, UUID], tuple[str, SpeechDeliveryFeedback, float]
        ] = {}
        self.resume_processor = ResumeProcessor()

    def set_live_audio_mode(self, mode: str) -> None:
        if mode not in {"asr_text", "audio_direct"}:
            raise ValueError(f"Unsupported live audio mode: {mode}")
        if mode == "audio_direct" and self.omni_client is None:
            raise ValueError("Live audio direct requires an omni client")
        self.live_audio_mode = mode

    async def start_live_interview(
        self, session_id: str, *, consent_confirmed: bool
    ) -> InterviewState:
        if not consent_confirmed:
            raise LiveInterviewStateError("开始监听前必须确认候选人已知情并同意转写")
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if live.status == LiveInterviewStatus.COMPLETED:
                runtime.state.live_interview = LiveInterviewSession()
                live = runtime.state.live_interview
            live.status = LiveInterviewStatus.ACTIVE
            live.consent_confirmed = True
            live.started_at = live.started_at or datetime.now(timezone.utc)
            live.completed_at = None
            self._refresh_live_coverage_guidance(runtime.state)
            self._refresh_live_question_usage(runtime.state)
            runtime.state.next_action = "Listen to the interview and prepare the next question"
            await self._persist(session_id, runtime.state)
        self._record_debug("live_interview_started", session_id)
        return runtime.state

    async def set_live_interview_status(
        self, session_id: str, status: LiveInterviewStatus
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if not live.consent_confirmed:
                raise LiveInterviewStateError("Live interview has not been started with consent")
            if status not in {
                LiveInterviewStatus.ACTIVE,
                LiveInterviewStatus.PAUSED,
                LiveInterviewStatus.COMPLETED,
            }:
                raise LiveInterviewStateError("Unsupported live interview status")
            live.status = status
            if status == LiveInterviewStatus.COMPLETED:
                live.completed_at = datetime.now(timezone.utc)
                runtime.state.next_action = "Review the transcript before final evaluation"
            await self._persist(session_id, runtime.state)
        self._record_debug(f"live_interview_{status.value}", session_id)
        return runtime.state

    async def append_live_transcript(
        self,
        session_id: str,
        *,
        text: str,
        speaker: TranscriptSpeaker,
        source: str = "manual",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        clean_text = text.strip()
        if not clean_text:
            raise LiveInterviewStateError("Transcript text cannot be empty")
        duplicate_detail = ""
        duplicate_detected = False
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if live.status != LiveInterviewStatus.ACTIVE:
                raise LiveInterviewStateError(
                    "Live interview must be active before adding transcript"
                )
            duplicate = self._find_duplicate_live_segment(live.segments, clean_text, speaker)
            if duplicate is not None:
                live.duplicate_segments_dropped += 1
                live.last_duplicate_reason = (
                    f"重复片段已忽略：与 #{duplicate.sequence} "
                    f"{duplicate.speaker.value} 发言高度一致"
                )
                duplicate_detail = (
                    f"speaker={speaker.value}; source={source}; "
                    f"duplicate_of={duplicate.sequence}; chars={len(clean_text)}"
                )
                duplicate_detected = True
                await self._persist(session_id, runtime.state)
            else:
                live.current_speaker = speaker
                live.segments.append(
                    TranscriptSegment(
                        sequence=len(live.segments) + 1,
                        speaker=speaker,
                        text=clean_text,
                        source=source,
                    )
                )
                self._refresh_live_answer_boundaries(runtime.state)
                self._refresh_live_rolling_summary(runtime.state)
                self._refresh_live_coverage_guidance(runtime.state)
                await self._persist(session_id, runtime.state)
        if duplicate_detected:
            self._record_debug(
                "live_transcript_duplicate_dropped", session_id, detail=duplicate_detail
            )
            return runtime.state
        self._record_debug(
            "live_transcript_added",
            session_id,
            detail=f"speaker={speaker.value}; source={source}; chars={len(clean_text)}",
        )
        return runtime.state

    async def update_live_transcript_segment(
        self,
        session_id: str,
        segment_id: UUID,
        *,
        text: str | None = None,
        speaker: TranscriptSpeaker | None = None,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        clean_text = text.strip() if text is not None else None
        if text is not None and not clean_text:
            raise LiveInterviewStateError("Transcript text cannot be empty")
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if not live.consent_confirmed:
                raise LiveInterviewStateError("Live interview has not been started with consent")
            segment = next((item for item in live.segments if item.id == segment_id), None)
            if segment is None:
                raise LiveInterviewStateError("Transcript segment was not found")
            if any(
                segment_id in record.transcript_segment_ids
                for record in runtime.state.live_interview_records
            ):
                raise LiveInterviewStateError("Confirmed evidence segments cannot be edited")
            if clean_text is not None:
                segment.text = clean_text
            if speaker is not None:
                segment.speaker = speaker
                live.current_speaker = speaker
            segment.stable = True
            segment.confirmed = True
            self._refresh_live_answer_boundaries(runtime.state)
            self._refresh_live_rolling_summary(runtime.state)
            self._refresh_live_coverage_guidance(runtime.state)
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_transcript_updated",
            session_id,
            detail=(f"speaker={segment.speaker.value}; chars={len(segment.text)}"),
        )
        return runtime.state

    async def transcribe_live_audio(
        self,
        session_id: str,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        speaker: TranscriptSpeaker,
        language: str = "zh",
        mode: str = "single",
    ) -> tuple[InterviewState, str]:
        if len(content) > 25 * 1024 * 1024:
            raise LiveInterviewStateError("Audio chunk exceeds the 25 MB limit")
        runtime = await self._get_runtime(session_id)
        if runtime.state.live_interview.status != LiveInterviewStatus.ACTIVE:
            raise LiveInterviewStateError("Live interview must be active before transcribing audio")
        started = perf_counter()
        if self.live_audio_mode == "audio_direct":
            if self.omni_client is None:
                raise LiveInterviewStateError("Omni audio client is not configured")
            if mode == "dialogue":
                return await self._transcribe_dialogue(
                    session_id, runtime, content, content_type=content_type, started=started
                )
            if speaker != TranscriptSpeaker.CANDIDATE:
                raise LiveInterviewStateError("音频直连模式仅支持候选人回答")
            context = self._live_audio_context(runtime.state)
            result = await self.omni_client.suggest_next_question(
                content, content_type=content_type, context=context, stream=False
            )
            suggestion_text = result if isinstance(result, str) else ""
            await self._inject_audio_direct_suggestion(session_id, runtime, suggestion_text)
            self._record_debug(
                "audio_direct_suggestion",
                session_id,
                detail=f"chars={len(suggestion_text)}",
                duration_ms=(perf_counter() - started) * 1000,
            )
            return runtime.state, suggestion_text
        if self.asr_client is None:
            raise LiveInterviewStateError("ASR client is not configured")
        try:
            transcript = await self.asr_client.transcribe(
                content,
                filename=filename,
                content_type=content_type,
                language=language,
            )
        except ASRError as exc:
            self._record_debug(
                "asr_transcription_failed",
                session_id,
                level=DebugLevel.ERROR,
                detail=str(exc),
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise WorkflowExecutionError(str(exc)) from exc
        state = await self.append_live_transcript(
            session_id, text=transcript, speaker=speaker, source="asr"
        )
        self._record_debug(
            "asr_transcription_completed",
            session_id,
            detail=f"speaker={speaker.value}; bytes={len(content)}; chars={len(transcript)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return state, transcript

    async def preview_audio_transcription(
        self,
        session_id: str,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        language: str = "zh",
    ) -> str:
        """Transcribe a cumulative recording snapshot without persisting it.

        The browser calls this at a bounded interval while recording so users
        can see provisional words. Only the final audio request creates a stable
        transcript segment, preventing duplicate evidence and partial text from
        entering downstream agents.
        """
        await self._get_runtime(session_id)
        if len(content) > 10 * 1024 * 1024:
            raise LiveInterviewStateError("ASR preview exceeds the 10 MB limit")
        if not content:
            raise LiveInterviewStateError("ASR preview audio is empty")
        if self.asr_client is None:
            raise LiveInterviewStateError("ASR client is not configured")
        started = perf_counter()
        try:
            transcript = await self.asr_client.transcribe(
                content,
                filename=filename,
                content_type=content_type,
                language=language,
            )
        except ASRError as exc:
            self._record_debug(
                "asr_preview_failed",
                session_id,
                level=DebugLevel.ERROR,
                detail=str(exc),
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise WorkflowExecutionError(str(exc)) from exc
        self._record_debug(
            "asr_preview_completed",
            session_id,
            detail=f"bytes={len(content)}; chars={len(transcript)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return transcript

    async def transcribe_mock_spoken_answer(
        self, session_id: str, content: bytes, filename: str
    ) -> str:
        """Transcribe a spoken mock-interview answer to text.

        A pure transcription: it does not touch live state, does not persist the
        audio, and does not change mock state — the caller fills the answer into
        the mock answer field. Ownership is enforced by ``_get_runtime``.
        """
        await self._get_runtime(session_id)
        if self.asr_client is None:
            raise WorkflowExecutionError("ASR client is not configured")
        content_type = (
            "audio/wav"
            if filename.lower().endswith(".wav")
            else "audio/webm"
            if filename.lower().endswith(".webm")
            else "audio/mp4"
            if filename.lower().endswith(".m4a")
            else "application/octet-stream"
        )
        started = perf_counter()
        try:
            transcript = await self.asr_client.transcribe(
                content,
                filename=filename,
                content_type=content_type,
                language="zh",
            )
        except ASRError as exc:
            self._record_debug(
                "mock_asr_transcription_failed",
                session_id,
                level=DebugLevel.ERROR,
                detail=str(exc),
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise WorkflowExecutionError(str(exc)) from exc
        self._record_debug(
            "mock_asr_transcription_completed",
            session_id,
            detail=f"bytes={len(content)}; chars={len(transcript)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return transcript

    async def save_mock_answer_audio(
        self, session_id: str, recording_id: UUID, content: bytes, *, extension: str = "wav"
    ) -> str:
        """Persist a candidate-owned mock recording until it is linked to an answer."""
        runtime = await self._get_runtime(session_id)
        ext = extension.lstrip(".").lower()
        if ext not in {"wav", "webm", "m4a"}:
            ext = "wav"
        filename = f"mock-{session_id}-{recording_id}.{ext}"
        async with self._lock_for(session_id):
            self._recordings_dir.mkdir(parents=True, exist_ok=True)
            target = self._recordings_dir / filename
            temporary = target.with_suffix(f".{ext}.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, target)
            referenced = {
                item.audio_file
                for item in (
                    *runtime.state.mock_session.responses,
                    *runtime.state.mock_session.attempt_history,
                )
                if item.audio_file
            }
            cutoff = datetime.now(timezone.utc).timestamp() - 86400
            for candidate in self._recordings_dir.glob(f"mock-{session_id}-*"):
                try:
                    if (
                        candidate.name not in referenced
                        and candidate != target
                        and candidate.stat().st_mtime < cutoff
                    ):
                        candidate.unlink(missing_ok=True)
                except OSError:
                    continue
        self._record_debug(
            "mock_answer_audio_staged",
            session_id,
            detail=f"recording_id={recording_id}; bytes={len(content)}",
        )
        return filename

    def _pending_mock_audio_path(self, session_id: str, recording_id: UUID) -> Path | None:
        for ext in ("wav", "webm", "m4a"):
            candidate = self._recordings_dir / f"mock-{session_id}-{recording_id}.{ext}"
            if candidate.is_file():
                return candidate
        return None

    def get_mock_answer_audio_path(self, session_id: str, stored_filename: str) -> Path | None:
        pattern = re.compile(rf"^mock-{re.escape(session_id)}-[0-9a-f-]{{36}}\.(?:wav|webm|m4a)$")
        if not pattern.fullmatch(stored_filename):
            return None
        candidate = self._recordings_dir / stored_filename
        return candidate if candidate.is_file() else None

    async def cleanup_orphaned_mock_audio(self, *, max_age_seconds: int = 86400) -> int:
        """Remove expired staged files while retaining every persisted attempt."""
        referenced: set[str] = set()
        for state in await self.storage.list_all_session_states():
            mock = state.get("mock_session")
            if not isinstance(mock, dict):
                continue
            for bucket in ("responses", "attempt_history"):
                records = mock.get(bucket)
                if not isinstance(records, list):
                    continue
                referenced.update(
                    str(item.get("audio_file"))
                    for item in records
                    if isinstance(item, dict) and item.get("audio_file")
                )
        cutoff = datetime.now(timezone.utc).timestamp() - max(0, max_age_seconds)
        removed = 0
        if not self._recordings_dir.exists():
            return removed
        for candidate in self._recordings_dir.glob("mock-*"):
            try:
                if candidate.name not in referenced and candidate.stat().st_mtime < cutoff:
                    candidate.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        return removed

    async def delete_mock_answer_audio(self, session_id: str, response_id: UUID) -> bool:
        runtime = await self._get_runtime(session_id)
        filename = ""
        async with self._lock_for(session_id):
            records = [
                *runtime.state.mock_session.responses,
                *runtime.state.mock_session.attempt_history,
            ]
            record = next((item for item in records if item.id == response_id), None)
            if record is None:
                raise MockInterviewStateError("Mock answer was not found")
            filename = record.audio_file
            if not filename:
                return False
            path = self.get_mock_answer_audio_path(session_id, filename)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    raise WorkflowExecutionError(
                        "Unable to delete the local mock recording"
                    ) from exc
            record.audio_file = ""
            await self._persist(session_id, runtime.state)
        return True

    async def analyze_mock_speech_delivery(
        self,
        session_id: str,
        content: bytes,
        transcript: str,
        *,
        content_type: str = "audio/wav",
    ) -> SpeechDeliveryFeedback:
        """Return coaching-only delivery feedback with a safe local fallback."""
        await self._get_runtime(session_id)
        fallback = self._build_mock_speech_fallback(content, transcript, content_type)
        duration = self._wav_duration_seconds(content) if "wav" in content_type else 0.0
        clean = transcript.strip()
        if self.omni_client is None or not hasattr(self.omni_client, "analyze_speaking_style"):
            return fallback
        started = perf_counter()
        try:
            result = await asyncio.wait_for(
                self.omni_client.analyze_speaking_style(
                    content,
                    transcript=clean,
                    content_type=content_type,
                ),
                timeout=25.0,
            )
        except TimeoutError:
            self._record_debug(
                "mock_speech_delivery_fallback",
                session_id,
                detail="reason=audio_model_timeout; source=text_fallback",
                duration_ms=(perf_counter() - started) * 1000,
            )
            return fallback
        except Exception as exc:  # noqa: BLE001 - provider output must safely degrade
            self._record_debug(
                "mock_speech_delivery_fallback",
                session_id,
                detail=f"reason=audio_model_error; type={type(exc).__name__}",
                duration_ms=(perf_counter() - started) * 1000,
            )
            return fallback
        allowed_text = ("pace", "pauses", "fillers", "volume", "intonation", "clarity")
        if not isinstance(result, dict) or not any(
            str(result.get(key) or "").strip() for key in allowed_text
        ):
            return fallback
        if not self._speech_feedback_is_compliant(result):
            self._record_debug(
                "mock_speech_delivery_fallback",
                session_id,
                detail="reason=prohibited_inference; source=text_fallback",
                duration_ms=(perf_counter() - started) * 1000,
            )
            return fallback
        values = {key: str(result.get(key) or getattr(fallback, key))[:500] for key in allowed_text}
        raw_strengths = result.get("strengths")
        raw_improvements = result.get("improvements")
        feedback = SpeechDeliveryFeedback(
            source="audio_model",
            **values,
            strengths=[str(item)[:300] for item in raw_strengths[:4]]
            if isinstance(raw_strengths, list)
            else fallback.strengths,
            improvements=[str(item)[:300] for item in raw_improvements[:4]]
            if isinstance(raw_improvements, list)
            else fallback.improvements,
        )
        self._record_debug(
            "mock_speech_delivery_analyzed",
            session_id,
            detail=f"source=audio_model; duration_seconds={duration:.1f}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return feedback

    def _build_mock_speech_fallback(
        self, content: bytes, transcript: str, content_type: str
    ) -> SpeechDeliveryFeedback:
        duration = self._wav_duration_seconds(content) if "wav" in content_type else 0.0
        clean = transcript.strip()
        filler_count = sum(clean.count(word) for word in ("嗯", "啊", "然后", "就是", "那个"))
        chars_per_minute = round(len(clean) * 60 / duration) if duration > 0 else 0
        pace = (
            f"约 {chars_per_minute} 字/分钟，语速偏快，关键结论后可停顿 1–2 秒。"
            if chars_per_minute > 300
            else f"约 {chars_per_minute} 字/分钟，语速偏慢，可先说结论再补证据。"
            if 0 < chars_per_minute < 120
            else f"约 {chars_per_minute} 字/分钟，处于易跟随区间。"
            if chars_per_minute
            else "未获得可靠音频时长，无法计算语速。"
        )
        return SpeechDeliveryFeedback(
            pace=pace,
            fillers=(
                f"转写中识别到约 {filler_count} 处常见填充词，可用短停顿替代。"
                if filler_count
                else "转写中未识别到明显填充词。"
            ),
            clarity=(
                "转写文本过短，暂时无法判断表达清晰度。"
                if len(clean) < 20
                else "转写文本可读；请用“结论—行动—结果”的句式进一步提高清晰度。"
            ),
            strengths=["已完成可回放的口语回答，可对照转写文本自查。"],
            improvements=[
                "回放时只检查一个目标：每个关键结论后留出 1–2 秒停顿。",
                "下次开头先用一句话给出结论，再说行动和可量化结果。",
            ],
        )

    def schedule_mock_speech_delivery_analysis(
        self,
        session_id: str,
        recording_id: UUID,
        content: bytes,
        transcript: str,
        *,
        content_type: str,
    ) -> SpeechDeliveryFeedback:
        """Return text feedback immediately and enrich it from audio in background."""
        owner = current_owner()
        now = datetime.now(timezone.utc).timestamp()
        expired = [
            key
            for key, (_, _, created_at) in self._mock_speech_feedback.items()
            if created_at < now - 3600
        ]
        for key in expired:
            self._mock_speech_feedback.pop(key, None)
        fallback = self._build_mock_speech_fallback(content, transcript, content_type)
        key = (owner, session_id, recording_id)
        self._mock_speech_feedback[key] = ("analyzing", fallback, now)
        self._background.schedule(
            self._analyze_mock_speech_delivery_task(
                session_id,
                recording_id,
                content,
                transcript,
                content_type=content_type,
                owner=owner,
            )
        )
        return fallback

    async def _analyze_mock_speech_delivery_task(
        self,
        session_id: str,
        recording_id: UUID,
        content: bytes,
        transcript: str,
        *,
        content_type: str,
        owner: str,
    ) -> None:
        feedback = await self.analyze_mock_speech_delivery(
            session_id, content, transcript, content_type=content_type
        )
        key = (owner, session_id, recording_id)
        self._mock_speech_feedback[key] = (
            "completed",
            feedback,
            datetime.now(timezone.utc).timestamp(),
        )
        runtime = self._runtimes.get((owner, session_id))
        if runtime is None:
            return
        filename_prefix = f"mock-{session_id}-{recording_id}."
        async with self._lock_for(session_id):
            records = [
                *runtime.state.mock_session.responses,
                *runtime.state.mock_session.attempt_history,
            ]
            record = next(
                (item for item in records if item.audio_file.startswith(filename_prefix)),
                None,
            )
            if record is not None:
                record.speech_delivery = feedback
                await self._persist(session_id, runtime.state)

    async def get_mock_speech_delivery_status(
        self, session_id: str, recording_id: UUID
    ) -> tuple[str, SpeechDeliveryFeedback]:
        await self._get_runtime(session_id)
        value = self._mock_speech_feedback.get((current_owner(), session_id, recording_id))
        if value is None:
            raise MockInterviewStateError("Speech feedback was not found")
        return value[0], value[1]

    @staticmethod
    def _speech_feedback_is_compliant(result: dict[str, Any]) -> bool:
        """Reject model text that crosses the coaching-only policy boundary."""
        rendered = json.dumps(result, ensure_ascii=False).casefold()
        for term in SPEECH_FEEDBACK_PROHIBITED_TERMS:
            normalized = term.casefold()
            if normalized.isascii():
                if re.search(rf"\b{re.escape(normalized)}\b", rendered):
                    return False
            elif normalized in rendered:
                return False
        return True

    @staticmethod
    def _wav_duration_seconds(content: bytes) -> float:
        try:
            with wave.open(io.BytesIO(content), "rb") as audio:
                rate = audio.getframerate()
                return audio.getnframes() / rate if rate else 0.0
        except (wave.Error, EOFError):
            return 0.0

    async def synthesize_mock_question(
        self,
        session_id: str,
        question_id: UUID,
        *,
        response_id: UUID | None = None,
    ) -> tuple[bytes, str]:
        """Synthesize a question that belongs to this owned mock session."""
        runtime = await self._get_runtime(session_id)
        question = self.current_mock_question(runtime.state)
        if question is None or question.id != question_id:
            raise MockInterviewStateError("Question is not the current mock question")
        # Before a follow-up is answered it exists only in session state, not yet
        # as a response record. Prefer that server-owned pending text so automatic
        # narration speaks what the UI is actually asking.
        speech_text = runtime.state.mock_session.pending_follow_up or question.question
        if response_id is not None:
            response = next(
                (
                    item
                    for item in runtime.state.mock_session.responses
                    if item.id == response_id and item.question_id == question_id
                ),
                None,
            )
            if response is None:
                raise MockInterviewStateError("Response is not part of the current question")
            speech_text = response.question
        if self.tts_client is None:
            raise WorkflowExecutionError("TTS client is not configured")
        started = perf_counter()
        try:
            audio, content_type = await self.tts_client.synthesize(speech_text)
        except TTSError as exc:
            self._record_debug(
                "mock_question_tts_failed",
                session_id,
                level=DebugLevel.ERROR,
                detail="tts_provider_error",
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise WorkflowExecutionError(str(exc)) from exc
        self._record_debug(
            "mock_question_tts_completed",
            session_id,
            detail=f"question_id={question_id}; bytes={len(audio)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return audio, content_type

    async def plan_live_next_question(self, session_id: str) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if live.status != LiveInterviewStatus.ACTIVE:
                raise LiveInterviewStateError("Live interview must be active to plan a question")
            candidate_segments = [
                item
                for item in live.segments
                if item.speaker == TranscriptSpeaker.CANDIDATE and item.stable and item.confirmed
            ]
            if not candidate_segments:
                raise LiveInterviewStateError("需要至少一段候选人回答后才能准备下一问题")
            self._refresh_live_coverage_guidance(runtime.state)
            self._refresh_live_question_usage(runtime.state)
            await runtime.run(
                "live_interview_agent",
                "根据最新候选人回答准备一个问题；优先补齐关键证据，然后推进未覆盖能力。",
            )
            self._refresh_live_question_usage(runtime.state)
            runtime.state.next_action = "Interviewer reviews the suggested next question"
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_question_planned",
            session_id,
            detail=(
                f"summary_until={runtime.state.live_interview.summarized_until_sequence}; "
                f"recent_segments={min(12, len(runtime.state.live_interview.segments))}; "
                f"duplicates_dropped={runtime.state.live_interview.duplicate_segments_dropped}; "
                f"top_gap={runtime.state.live_interview.coverage_guidance[0].competency if runtime.state.live_interview.coverage_guidance else 'none'}; "
                f"pending_blueprint={sum(1 for item in runtime.state.live_interview.question_usage if item.status == 'pending')}; "
                f"top_boundary_confidence={runtime.state.live_interview.answer_boundary_suggestions[-1].confidence if runtime.state.live_interview.answer_boundary_suggestions else 0}; "
                f"action={runtime.state.live_interview.action_card.action_type}"
            ),
        )
        return runtime.state

    async def decide_live_suggestion(
        self,
        session_id: str,
        suggestion_id: UUID,
        *,
        status: QuestionSuggestionStatus,
        final_question: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            suggestion = next(
                (
                    item
                    for item in runtime.state.live_interview.suggestions
                    if item.id == suggestion_id
                ),
                None,
            )
            if suggestion is None:
                raise LiveInterviewStateError("Question suggestion was not found")
            if suggestion.status != QuestionSuggestionStatus.PENDING:
                raise LiveInterviewStateError("Question suggestion has already been decided")
            suggestion.status = status
            suggestion.final_question = (
                final_question.strip() or suggestion.suggested_question
                if status in {QuestionSuggestionStatus.ADOPTED, QuestionSuggestionStatus.EDITED}
                else ""
            )
            if status in {QuestionSuggestionStatus.ADOPTED, QuestionSuggestionStatus.EDITED}:
                source_id = suggestion.source_question_id.strip()
                if source_id and source_id not in runtime.state.live_interview.used_question_ids:
                    runtime.state.live_interview.used_question_ids.append(source_id)
                runtime.state.live_interview.segments.append(
                    TranscriptSegment(
                        sequence=len(runtime.state.live_interview.segments) + 1,
                        speaker=TranscriptSpeaker.INTERVIEWER,
                        text=suggestion.final_question,
                        source="copilot",
                    )
                )
            self._refresh_live_question_usage(runtime.state)
            await self._persist(session_id, runtime.state)
        self._record_debug("live_question_decided", session_id, detail=f"status={status.value}")
        return runtime.state

    async def confirm_live_answer(
        self,
        session_id: str,
        answer_segment_id: UUID,
        *,
        question_segment_id: UUID | None = None,
        question: str = "",
        competency: str = "",
    ) -> InterviewState:
        return await self.confirm_live_answer_segments(
            session_id,
            [answer_segment_id],
            question_segment_id=question_segment_id,
            question=question,
            competency=competency,
        )

    async def confirm_live_answer_segments(
        self,
        session_id: str,
        answer_segment_ids: list[UUID],
        *,
        question_segment_id: UUID | None = None,
        question: str = "",
        competency: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if not live.consent_confirmed:
                raise LiveInterviewStateError("Live interview has not been started with consent")
            unique_answer_ids = list(dict.fromkeys(answer_segment_ids))
            if not unique_answer_ids:
                raise LiveInterviewStateError("Candidate answer segment was not found")
            segment_by_id = {item.id: item for item in live.segments}
            answer_segments = [segment_by_id.get(segment_id) for segment_id in unique_answer_ids]
            if any(item is None for item in answer_segments):
                raise LiveInterviewStateError("Candidate answer segment was not found")
            ordered_answer_segments = [
                item for item in live.segments if item.id in set(unique_answer_ids)
            ]
            if len(ordered_answer_segments) != len(unique_answer_ids):
                raise LiveInterviewStateError("Candidate answer segment was not found")
            if any(item.speaker != TranscriptSpeaker.CANDIDATE for item in ordered_answer_segments):
                raise LiveInterviewStateError(
                    "Only candidate transcript segments can become evidence"
                )
            if any(not item.stable or not item.confirmed for item in ordered_answer_segments):
                raise LiveInterviewStateError("Transcript segment must be stable and confirmed")
            if any(
                segment_id in record.transcript_segment_ids
                for record in runtime.state.live_interview_records
                for segment_id in unique_answer_ids
            ):
                raise LiveInterviewStateError("This answer has already been confirmed as evidence")

            question_segment = self._resolve_live_question_segment(
                live.segments, ordered_answer_segments[0], question_segment_id
            )
            clean_question = question.strip() or (question_segment.text if question_segment else "")
            if not clean_question:
                clean_question = "现场面试问题"
            clean_competency = (
                competency.strip()
                or self._infer_live_competency(runtime.state, clean_question)
                or "综合能力"
            )
            answer_text = "\n".join(item.text for item in ordered_answer_segments)
            segment_ids = [item.id for item in ordered_answer_segments]
            if question_segment is not None:
                segment_ids.insert(0, question_segment.id)
            record = LiveInterviewRecord(
                question=clean_question,
                answer=answer_text,
                competency=clean_competency,
                evaluation=AnswerEvaluation(
                    content=0.0, technical_depth=0.0, structure=0.0, impact=0.0
                ),
                answer_modality=(
                    "typed"
                    if all(item.source == "manual" for item in ordered_answer_segments)
                    else "live_asr"
                ),
                source="live_interview",
                transcript_segment_ids=segment_ids,
                scoring_status="scoring",
            )
            placeholder = Evidence(
                competency=clean_competency,
                signal=answer_text[:200],
                confidence=0.0,
                source=EvidenceSource.LIVE_INTERVIEW,
                source_record_id=record.id,
                notes="评分中：确认后由后台评分回填置信度",
            )
            runtime.state.live_interview_records.append(record)
            runtime.state.evidence.append(placeholder)
            self._invalidate_final_reports(
                runtime.state, "New live evidence was confirmed; regenerate evaluation"
            )
            self._refresh_live_answer_boundaries(runtime.state)
            self._refresh_live_rolling_summary(runtime.state)
            self._refresh_live_coverage_guidance(runtime.state)
            runtime.state.next_action = "Review more live evidence or generate evaluation"
            await self._persist(session_id, runtime.state)
            self._background.schedule(self._score_live_record_task(session_id, record.id))
        self._record_debug(
            "live_answer_confirmed",
            session_id,
            detail=(
                f"competency={clean_competency}; "
                f"segments={len(ordered_answer_segments)}; chars={len(answer_text)}"
            ),
        )
        if runtime.state.live_interview.status == LiveInterviewStatus.ACTIVE:
            await self._maybe_plan_live_next_question(session_id)
        return runtime.state

    async def reevaluate_live_evidence(
        self,
        session_id: str,
        record_id: UUID,
        *,
        question: str = "",
        competency: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            record = next(
                (item for item in runtime.state.live_interview_records if item.id == record_id),
                None,
            )
            if record is None:
                raise LiveInterviewStateError("Live evidence record was not found")
            if record.source != "live_interview":
                raise LiveInterviewStateError(
                    "Only live interview evidence can be re-evaluated here"
                )
            if question.strip():
                record.question = question.strip()
            if competency.strip():
                record.competency = competency.strip()
            evaluation = await self._score_live_record(runtime, record)
            self._apply_live_scoring_result(runtime.state, record, evaluation)
            record.evaluation = evaluation
            record.scoring_status = "scored"
            record.scoring_error = ""
            self._invalidate_final_reports(
                runtime.state, "Live evidence was re-evaluated; regenerate evaluation"
            )
            self._refresh_live_coverage_guidance(runtime.state)
            runtime.state.next_action = "Review updated live evidence or generate evaluation"
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_evidence_reevaluated",
            session_id,
            detail=f"record={str(record_id)[:8]}; competency={record.competency}",
        )
        return runtime.state

    async def revoke_live_evidence(self, session_id: str, record_id: UUID) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            record = next(
                (item for item in runtime.state.live_interview_records if item.id == record_id),
                None,
            )
            if record is None:
                raise LiveInterviewStateError("Live evidence record was not found")
            if record.source != "live_interview":
                raise LiveInterviewStateError("Only live interview evidence can be revoked here")
            runtime.state.live_interview_records = [
                item for item in runtime.state.live_interview_records if item.id != record_id
            ]
            runtime.state.evidence = [
                item for item in runtime.state.evidence if item.source_record_id != record_id
            ]
            self._invalidate_final_reports(
                runtime.state, "Live evidence was revoked; regenerate evaluation"
            )
            self._refresh_live_answer_boundaries(runtime.state)
            self._refresh_live_rolling_summary(runtime.state)
            self._refresh_live_coverage_guidance(runtime.state)
            runtime.state.next_action = "Review corrected live evidence before evaluation"
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_evidence_revoked",
            session_id,
            detail=f"record={str(record_id)[:8]}; competency={record.competency}",
        )
        return runtime.state

    async def _score_live_record(
        self, runtime: AgentRuntime, record: LiveInterviewRecord
    ) -> AnswerEvaluation:
        message = await runtime.run(
            "coach_agent",
            json.dumps(
                {
                    "question": record.question,
                    "answer": record.answer,
                    "competency": record.competency,
                    "evidence_source": "live_interview",
                    "record_id": str(record.id),
                    "persist_evidence": False,
                    "answer_modality": record.answer_modality,
                },
                ensure_ascii=False,
            ),
        )
        try:
            evaluation = AnswerEvaluation.model_validate_json(message.content)
        except (ValueError, TypeError) as exc:
            raise EvaluationStateError("Live answer evaluation failed") from exc
        return evaluation

    @staticmethod
    def _apply_live_scoring_result(
        state: InterviewState,
        record: LiveInterviewRecord,
        evaluation: AnswerEvaluation,
    ) -> None:
        """Backfill the confirmed placeholder evidence from a scoring result.

        The placeholder is created at confirmation time so coverage guidance and
        the action card are accurate immediately. The coach backfills it in place
        (matching by record id); this runs as a second safety net and covers the
        reevaluate path where competency may have changed.
        """
        placeholder = next(
            (
                item
                for item in state.evidence
                if item.source == EvidenceSource.LIVE_INTERVIEW
                and item.source_record_id == record.id
            ),
            None,
        )
        if placeholder is not None:
            placeholder.competency = record.competency
            placeholder.signal = record.answer[:200]
            placeholder.confidence = evaluation.overall_score()
            # The signal has just been replaced by model output. Never carry a
            # human classification from the previous signal across that change.
            placeholder.polarity = EvidencePolarity.NEUTRAL
            placeholder.notes = "; ".join(evaluation.missing_signals)
            return
        state.evidence.append(
            Evidence(
                competency=record.competency,
                signal=record.answer[:200],
                confidence=evaluation.overall_score(),
                source=EvidenceSource.LIVE_INTERVIEW,
                source_record_id=record.id,
                polarity=EvidencePolarity.NEUTRAL,
                notes="; ".join(evaluation.missing_signals),
            )
        )

    async def _score_live_record_task(self, session_id: str, record_id: UUID) -> None:
        """Background coroutine: score a confirmed live answer and write it back."""
        runtime = self._runtimes.get((current_owner(), session_id))
        if runtime is None:
            logger.warning(
                "Dropping background live scoring for %s: runtime not cached", session_id
            )
            return
        record = next(
            (item for item in runtime.state.live_interview_records if item.id == record_id),
            None,
        )
        if record is None:
            return
        try:
            evaluation = await self._score_live_record(runtime, record)
        except Exception as exc:  # noqa: BLE001 - background boundary, record failure
            logger.error("Live scoring failed for %s: %s", record_id, exc)
            async with self._lock_for(session_id):
                current = next(
                    (item for item in runtime.state.live_interview_records if item.id == record_id),
                    None,
                )
                if current is None:
                    return
                if (
                    current.evaluation.scoring_source == AnswerScoringSource.HUMAN
                    and current.evaluation.review_status == AnswerReviewStatus.REVIEWED
                ):
                    # The same trust rule applies when the stale model request
                    # fails: it must not make a reviewed record provisional.
                    return
                current.scoring_status = "failed"
                current.scoring_error = str(exc)[:500]
                await self._persist(session_id, runtime.state)
            return
        async with self._lock_for(session_id):
            current = next(
                (item for item in runtime.state.live_interview_records if item.id == record_id),
                None,
            )
            if current is None:
                # The record was revoked while scoring ran; drop this task's
                # placeholder so no orphaned evidence survives.
                runtime.state.evidence = [
                    item
                    for item in runtime.state.evidence
                    if not (
                        item.source == EvidenceSource.LIVE_INTERVIEW
                        and item.source_record_id == record_id
                    )
                ]
                await self._persist(session_id, runtime.state)
                return
            if (
                current.evaluation.scoring_source == AnswerScoringSource.HUMAN
                and current.evaluation.review_status == AnswerReviewStatus.REVIEWED
            ):
                # A reviewer can finish while the original background model call is
                # still running. Human provenance wins that race permanently.
                return
            self._apply_live_scoring_result(runtime.state, current, evaluation)
            current.evaluation = evaluation
            current.scoring_status = "scored"
            current.scoring_error = ""
            self._invalidate_final_reports(
                runtime.state, "Live evidence scoring changed; regenerate evaluation"
            )
            self._refresh_live_coverage_guidance(runtime.state)
            await self._persist(session_id, runtime.state)

    async def _maybe_plan_live_next_question(self, session_id: str) -> None:
        """Auto-trigger next-question planning after evidence is confirmed.

        Fires only when the live session is active, no suggestion is still
        pending (so repeated confirms do not stack LLM calls), and there is
        something to plan against: pending candidate segments or live evidence.
        """
        runtime = await self._get_runtime(session_id)
        live = runtime.state.live_interview
        if live.status != LiveInterviewStatus.ACTIVE:
            return
        if any(item.status == QuestionSuggestionStatus.PENDING for item in live.suggestions):
            return
        live_evidence = [
            item for item in runtime.state.live_interview_records if item.source == "live_interview"
        ]
        recorded_segment_ids = {
            segment_id for record in live_evidence for segment_id in record.transcript_segment_ids
        }
        has_pending_candidate = any(
            item.speaker == TranscriptSpeaker.CANDIDATE
            and item.stable
            and item.confirmed
            and item.id not in recorded_segment_ids
            for item in live.segments
        )
        if not has_pending_candidate and not live_evidence:
            return
        await self.plan_live_next_question(session_id)

    @staticmethod
    def _live_audio_context(state: InterviewState) -> str:
        """Focused, priority-ordered context for the omni audio-direct model.

        The direct path exists to be fast, so it carries only what the model
        needs to turn a candidate utterance into a good follow-up: job/covered
        competencies (anchor), the recent stable conversation, and the most
        recent live evidence. Lower-value "advancement" info (blueprint backlog,
        historical summary, coverage guidance) is intentionally excluded to keep
        prefill small. If the focused blocks still exceed the limit, blocks are
        dropped lowest-priority first rather than truncating the tail (which
        would lose the anchor).
        """
        competencies = "、".join(state.job.competencies) or state.job.title or "目标岗位"
        covered = "、".join(
            dict.fromkeys(item.competency for item in state.evidence if item.competency.strip())
        )
        anchor = f"岗位能力：{competencies}"
        if covered:
            anchor += f"；已覆盖能力：{covered}"
        stable_segments = [
            item
            for item in state.live_interview.segments
            if item.stable
            and item.confirmed
            and item.speaker != TranscriptSpeaker.UNKNOWN
        ][-LIVE_RECENT_SEGMENT_WINDOW:]
        latest_candidate = next(
            (
                item
                for item in reversed(stable_segments)
                if item.speaker == TranscriptSpeaker.CANDIDATE
            ),
            None,
        )
        optional_parts: list[str] = []
        history_segments = [
            item
            for item in stable_segments
            if latest_candidate is None or item.id != latest_candidate.id
        ]
        if history_segments:
            speaker_labels = {
                TranscriptSpeaker.INTERVIEWER: "面试官",
                TranscriptSpeaker.CANDIDATE: "候选人",
                TranscriptSpeaker.UNKNOWN: "待确认",
            }
            optional_parts.append(
                "较早对话（只用于消歧，不得作为本轮追问主题）："
                + "；".join(
                    f"{speaker_labels.get(item.speaker, item.speaker.value)}：{item.text}"
                    for item in history_segments
                )
            )

        live_evidence = [
            f"{item.competency}：{item.signal}"
            for item in state.evidence
            if item.source == EvidenceSource.LIVE_INTERVIEW
        ][-5:]
        if live_evidence:
            optional_parts.append("已确认证据：" + "；".join(live_evidence))

        latest_block = (
            "本轮必须只围绕下方最新候选人回答提出直接追问；不得改问较早回答：\n"
            f"最新候选人回答：{latest_candidate.text}"
            if latest_candidate is not None
            else "本轮尚无已确认的候选人回答。"
        )
        parts = [anchor, *optional_parts, latest_block]

        # Drop lowest-priority optional blocks first. The competency anchor and
        # explicit latest-answer block are never displaced by older conversation.
        while len("\n".join(parts)) > OMNI_CONTEXT_CHAR_LIMIT and optional_parts:
            optional_parts.pop()
            parts = [anchor, *optional_parts, latest_block]
        if len("\n".join(parts)) > OMNI_CONTEXT_CHAR_LIMIT:
            available = max(200, OMNI_CONTEXT_CHAR_LIMIT - len(anchor) - 100)
            if len(latest_block) > available:
                head = max(80, int(available * 0.6))
                tail = max(80, available - head - 2)
                latest_block = f"{latest_block[:head]}…{latest_block[-tail:]}"
            parts = [anchor, latest_block]
        return "\n".join(parts)

    async def _inject_audio_direct_suggestion(
        self,
        session_id: str,
        runtime: AgentRuntime,
        suggestion_text: str,
    ) -> None:
        """Create a QuestionSuggestion from an audio-direct model answer."""
        clean = suggestion_text.strip()
        if not clean or clean.startswith("[LLM Error"):
            clean = "请再补充说明一下你刚才提到的方案权衡与结果。"
        competency = (
            runtime.state.job.competencies[0] if runtime.state.job.competencies else "综合能力"
        )
        suggestion = QuestionSuggestion(
            suggested_question=clean[:4000],
            question_type=QuestionSuggestionType.FOLLOW_UP,
            competency=competency,
            rationale="基于候选人实时语音回答生成的追问建议",
            evidence_gap="需要进一步验证回答中的行动、权衡与量化结果",
            expected_signals=["具体行动", "技术权衡", "可量化结果"],
            confidence=0.7,
        )
        async with self._lock_for(session_id):
            if runtime.state.live_interview.status != LiveInterviewStatus.ACTIVE:
                # A stream/audio request may finish after the interviewer ends or
                # pauses the session. Do not append a stale suggestion afterward.
                return
            runtime.state.live_interview.suggestions.append(suggestion)
            self._refresh_live_question_usage(runtime.state)
            self._refresh_live_action_card(runtime.state)
            runtime.state.next_action = "Interviewer reviews the audio-direct suggestion"
            await self._persist(session_id, runtime.state)

    async def stream_live_suggestion(self, session_id: str) -> AsyncIterator[dict[str, str]]:
        """Stream a next-question suggestion token by token.

        Yields each text chunk as the model generates it, then persists the
        completed suggestion into the review queue so the flow is the same as
        a non-streamed plan. Falls back to a static suggestion if streaming is
        not possible. Callers should guard with :meth:`assert_live_active`
        first so the 409 is raised before streaming begins.
        """
        runtime = await self._get_runtime(session_id)
        state = runtime.state
        self.assert_live_active(state)
        context = self._live_audio_context(state)
        text = ""
        stream_failed = False
        if self.llm_client is not None and hasattr(self.llm_client, "chat_stream"):
            prompt = (
                "你是面试官助手。根据面试上下文，给出一个聚焦证据缺口的下一问追问。"
                "必须直接追问上下文中明确标记的最新候选人回答，不得重新追问较早回答。"
                "直接输出问题本身，中文，简洁，不超过两句话。不要输出 JSON。\n\n"
                f"面试上下文：\n{context}"
            )
            messages = [
                {"role": "system", "content": "你是面试官助手，根据上下文给出下一问追问。"},
                {"role": "user", "content": prompt},
            ]
            try:
                async for piece in self.llm_client.chat_stream(
                    messages, temperature=0.4, max_tokens=200
                ):
                    text += piece
                    yield {"type": "append", "text": piece}
            except Exception as exc:  # noqa: BLE001 - model adapter boundary
                stream_failed = True
                logger.warning("Live suggestion stream degraded: %s", type(exc).__name__)
        if stream_failed or not text.strip():
            text = "请再补充说明一下你刚才提到的方案权衡与量化结果。"
            yield {"type": "replace", "text": text}
        await self._inject_audio_direct_suggestion(session_id, runtime, text)

    @staticmethod
    def assert_live_active(state: InterviewState) -> None:
        if state.live_interview.status != LiveInterviewStatus.ACTIVE:
            raise LiveInterviewStateError("Live interview must be active")

    @staticmethod
    def _merge_diarized_segments(
        segments: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Merge consecutive utterances from the same speaker into one segment.

        The diarize model returns one entry per sentence; a long candidate
        answer spanning several sentences should become one transcript segment
        so it can be confirmed as a single piece of evidence.
        """
        merged: list[dict[str, str]] = []
        for segment in segments:
            speaker = segment.get("speaker", "unknown")
            text = segment.get("text", "").strip()
            if not text:
                continue
            if merged and merged[-1]["speaker"] == speaker:
                merged[-1]["text"] = f"{merged[-1]['text']}\n{text}"
            else:
                merged.append({"speaker": speaker, "text": text})
        return merged

    async def _transcribe_dialogue(
        self,
        session_id: str,
        runtime: AgentRuntime,
        content: bytes,
        *,
        content_type: str,
        started: float,
    ) -> tuple[InterviewState, str]:
        """Transcribe a dialog, auto-splitting speakers, then append segments."""
        diarized = await self.omni_client.transcribe_diarize(content, content_type=content_type)
        merged = self._merge_diarized_segments(diarized)
        if not merged:
            raise LiveInterviewStateError("音频直连未能识别对话中的说话人，请重试或手动输入")
        summary_parts: list[str] = []
        for segment in merged:
            speaker = (
                TranscriptSpeaker.CANDIDATE
                if segment["speaker"] == "candidate"
                else TranscriptSpeaker.INTERVIEWER
                if segment["speaker"] == "interviewer"
                else TranscriptSpeaker.UNKNOWN
            )
            text = segment["text"]
            summary_parts.append(text)
            await self.append_live_transcript(
                session_id, text=text, speaker=speaker, source="audio_direct"
            )
        self._record_debug(
            "audio_direct_dialogue",
            session_id,
            detail=f"segments={len(merged)}; speakers={len({s['speaker'] for s in merged})}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        summary = " | ".join(summary_parts)
        return runtime.state, summary

    async def confirm_pending_live_answers(
        self, session_id: str, *, competency: str = ""
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        recorded_segment_ids = {
            segment_id
            for record in runtime.state.live_interview_records
            for segment_id in record.transcript_segment_ids
        }
        pending_ids = [
            item.id
            for item in runtime.state.live_interview.segments
            if item.speaker == TranscriptSpeaker.CANDIDATE
            and item.stable
            and item.confirmed
            and item.id not in recorded_segment_ids
        ]
        if not pending_ids:
            raise LiveInterviewStateError(
                "No pending candidate answers require evidence confirmation"
            )
        state = runtime.state
        for segment_id in pending_ids:
            state = await self.confirm_live_answer(
                session_id,
                segment_id,
                competency=competency,
            )
        self._record_debug(
            "live_answers_batch_confirmed",
            session_id,
            detail=f"count={len(pending_ids)}",
        )
        return state

    async def create_session(
        self, candidate_name: str = "", job_title: str = "", company_name: str = ""
    ) -> tuple[str, InterviewState]:
        owner = current_owner()
        session_id = str(uuid4())
        runtime = create_runtime(
            self.llm_client, self.search_provider, self.debug_events, session_id
        )
        runtime.state.candidate.name = candidate_name
        runtime.state.job.title = job_title
        runtime.state.company.name = company_name
        self._runtimes[(owner, session_id)] = runtime
        self._locks[(owner, session_id)] = asyncio.Lock()
        await self._persist(session_id, runtime.state)
        self._record_debug("session_created", session_id)
        return session_id, runtime.state

    async def get_state(self, session_id: str) -> InterviewState:
        return (await self._get_runtime(session_id)).state

    async def list_sessions(self) -> list[dict[str, Any]]:
        return await self.storage.list_sessions(owner_id=current_owner())

    async def save_live_audio(
        self, session_id: str, content: bytes, *, extension: str = "wav"
    ) -> InterviewState:
        """Persist a whole-session audio file for a live interview.

        Writes the bytes to ``data/recordings/{session_id}.{extension}`` (atomic
        temp+rename) and records the real filename on the session state, which
        is persisted with the rest of the blob. The extension reflects the
        uploaded encoding (wav from the normal path, webm from a best-effort
        pagehide upload) so the stored file's type matches its bytes.
        Ownership is enforced by ``_get_runtime`` (owner-scoped), so a foreign
        session raises 404.
        """
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if not live.consent_confirmed:
                raise LiveInterviewStateError("未确认知情同意前无法保存全场录音")
            ext = (extension or "wav").lstrip(".").lower()
            if ext not in {"wav", "webm", "m4a"}:
                ext = "wav"
            self._recordings_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{session_id}.{ext}"
            target = self._recordings_dir / filename
            temp = self._recordings_dir / f"{session_id}.{ext}.tmp"
            temp.write_bytes(content)
            os.replace(temp, target)
            live.audio_file = filename
            live.audio_size_bytes = len(content)
            live.audio_saved_at = datetime.now(timezone.utc)
            await self._persist(session_id, runtime.state)
            # A pagehide fallback can upload WebM after the normal WAV save.
            # Keep one authoritative recording per session so stale encodings
            # cannot be downloaded later or retained beyond the current copy.
            for old_ext in {"wav", "webm", "m4a"} - {ext}:
                stale = self._recordings_dir / f"{session_id}.{old_ext}"
                try:
                    stale.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Could not remove superseded session recording %s: %s",
                        stale.name,
                        type(exc).__name__,
                    )
        self._record_debug(
            "live_audio_saved",
            session_id,
            detail=f"bytes={len(content)}; file={live.audio_file}",
        )
        return runtime.state

    def get_live_audio_path(self, session_id: str, stored_filename: str = "") -> Path | None:
        """Return the on-disk path of a session's recording, if present.

        The caller (owner-scoped route) already validated ownership via
        ``get_state``. Only the filename persisted in that owner-scoped state is
        accepted; scanning by session id could serve a superseded recording.
        """
        allowed_names = {f"{session_id}.{ext}" for ext in ("wav", "webm", "m4a")}
        if stored_filename not in allowed_names:
            return None
        candidate = self._recordings_dir / stored_filename
        return candidate if candidate.is_file() else None

    async def analyze_resume(self, session_id: str, text: str) -> Message:
        runtime = await self._get_runtime(session_id)
        await self._prepare_resume_transition(session_id, runtime, text)
        return await self._run(session_id, "candidate_agent", text)

    async def upload_resume(
        self,
        session_id: str,
        filename: str,
        content: bytes,
        *,
        structure: str = "rules",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        # Reject an unsafe replacement before parsing the document or sending
        # its text to an optional structuring model. The second check below is
        # still required because interview activity may arrive while parsing.
        async with self._lock_for(session_id):
            self._assert_candidate_reset_allowed(runtime.state)
        # Parsing and optional LLM structuring are slow and do not touch session
        # state, so keep them outside the per-session mutation lock.
        structuring_client = self.resume_llm_client or self.llm_client
        try:
            text, review = await asyncio.to_thread(
                self.resume_processor.process, filename, content
            )
        except ScannedPDFError:
            try:
                ocr_text, pages, ocr_engine = await ocr_scanned_pdf(content, structuring_client)
            except ResumeOCRError as exc:
                self._record_debug(
                    "resume_ocr_failed",
                    session_id,
                    detail=f"{type(exc).__name__}; file_type=pdf",
                )
                raise ResumeProcessingError(
                    "扫描版 PDF 自动 OCR 失败，请确认 resume_llm 支持图片或上传可复制文字的文件"
                ) from None
            text, review = self.resume_processor.process_ocr_text(
                filename, content, ocr_text, pages
            )
            self._record_debug(
                "resume_ocr_completed",
                session_id,
                detail=f"pages={pages}; chars={len(text)}; provider={ocr_engine}",
            )
        if structure == "llm" and structuring_client is not None:
            sections = await structure_resume_with_llm(text, structuring_client)
            if sections:
                review.structured = sections
                review.structured_by = "llm"
        async with self._lock_for(session_id):
            self._assert_candidate_reset_allowed(runtime.state)
            candidate_name = runtime.state.candidate.name
            self._reset_candidate_outputs(runtime.state)
            # A newly uploaded document replaces all candidate-derived state.
            # Keeping the previous parsed profile would mix two resumes before
            # the candidate agent has a chance to analyze the new document.
            runtime.state.candidate = CandidateProfile(
                name=candidate_name,
                raw_resume_text=text,
            )
            runtime.state.resume_review = review
            runtime.state.past_employer_sources = []
            runtime.state.past_employer_research_status = "not_requested"
            runtime.state.past_employer_block = ""
            runtime.state.next_action = "Review resume checks, then continue the interview workflow"
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "resume_processed",
                session_id,
                detail=f"{review.metadata.file_type}, {review.metadata.character_count} chars, "
                f"{len(review.issues)} issues, {len(review.claims)} claims, "
                f"structured={review.structured_by}",
            )
            return runtime.state

    @staticmethod
    def _has_substantive_interview_activity(state: InterviewState) -> bool:
        """Whether resetting candidate inputs would discard real interview data."""
        return bool(
            state.mock_session.responses
            or state.live_interview.segments
            or state.live_interview_records
            or state.evidence
            or state.live_interview.audio_file
        )

    @classmethod
    def _assert_candidate_reset_allowed(cls, state: InterviewState) -> None:
        if cls._has_substantive_interview_activity(state):
            raise CandidateSessionStateError(
                "当前会话已有面试回答、转写、证据或录音；请新建会话后再更换简历或重新生成方案"
            )

    @staticmethod
    def _reset_candidate_outputs(state: InterviewState) -> None:
        """Invalidate every artifact derived from candidate input before regeneration."""
        state.entity_resolutions = []
        state.fact_cards = []
        state.past_employer_sources = []
        state.past_employer_research_status = "not_requested"
        state.past_employer_block = ""
        state.strategy = InterviewStrategy()
        state.blueprint = InterviewBlueprint()
        state.mock_interview = MockInterviewPlan()
        state.mock_session = MockInterviewSession()
        state.evaluation = EvaluationReport()
        state.feedback = FeedbackReport()
        state.live_interview_records = []
        state.live_interview = LiveInterviewSession()
        state.workflow = WorkflowProgress()
        state.autopilot = AutopilotState()
        state.evidence = []
        state.evaluated_competencies = {}
        state.missing_signals = []
        state.conversation_history = []
        state.current_stage = InterviewStage.NOT_STARTED
        state.next_action = "Regenerate candidate-dependent interview artifacts"

    @staticmethod
    def _invalidate_final_reports(state: InterviewState, next_action: str) -> bool:
        """Clear derived decisions when their underlying evidence changes.

        This is intentionally a no-op before any final report exists, so ordinary
        answer collection does not disturb the active interview stage. Once an
        evaluation has been produced, every evidence mutation returns the state
        to a retryable wrap-up phase and also reopens completed automation state.
        """
        has_derived_report = bool(
            state.evaluation.finalized_at
            or state.evaluation.competencies
            or state.feedback.overall
            or state.workflow.name == "evaluation"
            or state.current_stage == InterviewStage.COMPLETED
        )
        if not has_derived_report:
            return False
        state.evaluation = EvaluationReport()
        state.feedback = FeedbackReport()
        state.evaluated_competencies = {}
        state.current_stage = InterviewStage.WRAP_UP
        if state.workflow.name == "evaluation":
            state.workflow = WorkflowProgress()
        if state.autopilot.enabled and state.autopilot.status == AutopilotStatus.COMPLETED:
            state.autopilot.status = AutopilotStatus.WAITING_FOR_INPUT
            state.autopilot.phase = "evaluation_review"
            state.autopilot.pause_reason = "Evidence changed after final evaluation"
            state.autopilot.completed_actions = [
                action
                for action in state.autopilot.completed_actions
                if action not in {"final_evaluation", "feedback_generation"}
            ]
            state.autopilot.updated_at = datetime.now(timezone.utc)
        state.next_action = next_action
        return True

    async def _prepare_resume_transition(
        self, session_id: str, runtime: AgentRuntime, resume_text: str
    ) -> None:
        """Safely invalidate stale derived state before a workflow consumes a resume."""
        async with self._lock_for(session_id):
            self._assert_candidate_reset_allowed(runtime.state)
            candidate_name = runtime.state.candidate.name
            same_resume = runtime.state.candidate.raw_resume_text.strip() == resume_text.strip()
            self._reset_candidate_outputs(runtime.state)
            if not same_resume:
                runtime.state.candidate = CandidateProfile(
                    name=candidate_name,
                    raw_resume_text=resume_text,
                )
                runtime.state.resume_review = ResumeReview()
            else:
                runtime.state.candidate.raw_resume_text = resume_text
            await self._persist(session_id, runtime.state)

    async def update_resume_claim(
        self,
        session_id: str,
        claim_id: UUID,
        status: ResumeClaimStatus,
        note: str = "",
        statement: str | None = None,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            claim = next(
                (item for item in runtime.state.resume_review.claims if item.id == claim_id), None
            )
            if claim is None:
                raise ResumeReviewStateError("Resume claim not found")
            claim.status = status
            claim.note = note.strip()[:500]
            if statement is not None:
                revised = statement.strip()[:2000]
                if not revised:
                    raise ResumeReviewStateError("Resume claim statement cannot be empty")
                claim.original_statement = claim.original_statement or claim.statement
                claim.statement = revised
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "resume_claim_updated", session_id, detail=f"{claim.category}: {status}"
            )
            return runtime.state

    async def analyze_job(self, session_id: str, text: str) -> Message:
        return await self._run(session_id, "job_agent", text)

    async def update_job_requirement(
        self,
        session_id: str,
        index: int,
        *,
        action: str,
        text: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        clean_text = text.strip()
        async with self._lock_for(session_id):
            requirements = runtime.state.job_review.requirements
            if index < 0 or index >= len(requirements):
                raise ResumeReviewStateError("Job requirement not found")
            if action == "delete":
                requirements.pop(index)
            elif action == "confirm":
                requirements[index].origin = RequirementOrigin.EXPLICIT
            elif action == "edit":
                if not clean_text:
                    raise ResumeReviewStateError("Job requirement text cannot be empty")
                requirements[index].text = clean_text
                requirements[index].origin = RequirementOrigin.EXPLICIT
            else:
                raise ResumeReviewStateError("Unsupported job requirement action")
            explicit_count = sum(
                1 for item in requirements if item.origin == RequirementOrigin.EXPLICIT
            )
            if explicit_count:
                runtime.state.job_review.is_title_only = False
                runtime.state.job_review.completeness_score = max(
                    runtime.state.job_review.completeness_score,
                    min(1.0, 0.45 + explicit_count * 0.05),
                )
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "job_requirement_updated",
                session_id,
                detail=f"index={index}; action={action}",
            )
            return runtime.state

    async def analyze_company(self, session_id: str, name: str, context: str = "") -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            runtime.state.company.name = name
            message = await runtime.run("company_agent", context)
            self._sync_intelligence(runtime.state)
            await self._persist(session_id, runtime.state)
            return message

    def _recent_employers(self, state: InterviewState, limit: int = 2) -> list[str]:
        """Pick the past employers worth researching.

        Selection favors: (1) employers within the last ~5 years, (2) long
        tenures, and (3) names/summaries that plausibly relate to the current
        target company. Returns employer names in research priority order.
        """
        raw_lines = state.candidate.raw_resume_text.splitlines()

        def _date_for(company: str) -> str:
            """Find a date line for a company in the raw resume text."""
            for line in raw_lines:
                if company.lower() in line.lower() and re.search(
                    r"(?:19|20)\d{2}[-/.]\d{1,2}\s+(?:to|至)", line
                ):
                    return line.strip()
            return ""

        employers: list[dict[str, Any]] = []
        confirmed_context = "\n".join(state.confirmed_resume_facts()).lower()
        reviewed = bool(state.resume_review.claims)
        experience = state.candidate.experience or []
        if experience:
            for e in experience:
                company = str(e.get("company", "")).strip()
                if not company:
                    continue
                start = str(e.get("duration") or e.get("date_range") or "").strip()
                if not re.search(r"(?:19|20)\d{2}", start):
                    start = _date_for(company) or start
                employers.append(
                    {
                        "company": company,
                        "role": str(e.get("role", "")).strip(),
                        "start": start,
                        "summary": str(e.get("summary", "")).strip(),
                    }
                )
        if reviewed:
            employers = [
                entry for entry in employers if entry["company"].lower() in confirmed_context
            ]
        if not employers and not reviewed:
            # Fall back to parsing raw text lines "YYYY-MM to COMPANY".
            for line in raw_lines:
                match = re.search(
                    r"(?:19|20)\d{2}[-/.]\d{1,2}\s+(?:to|至)\s+([A-Za-z一-鿿][A-Za-z一-鿿0-9 &,.]+?)"
                    r"(?:\s+(?:19|20)\d{2}[-/.]\d{1,2})?$",
                    line.strip(),
                )
                if match:
                    company = match.group(1).strip()
                    if company and len(company) >= 2:
                        employers.append({"company": company, "start": line.strip()})
        # Filter: within ~5 years OR long tenure OR name overlap with target company.
        current = state.company.name or ""
        current_tokens = set(re.findall(r"[A-Za-z0-9]+", current.lower()))
        current_terms = set(re.findall(r"[一-鿿]{2,}", current))
        candidates: list[dict[str, Any]] = []
        for entry in employers:
            company = entry["company"]
            start = entry.get("start") or ""
            years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", start)]
            if not years:
                # A dateless entry may still be a recent employer (LLM wrote
                # "3 years" instead of a date range). Keep it, lowest priority.
                name_tokens = set(re.findall(r"[A-Za-z0-9]+", company.lower()))
                name_terms = set(re.findall(r"[一-鿿]{2,}", company))
                entry["_recent"] = False
                entry["_tenure"] = 0
                entry["_related"] = bool(
                    (name_tokens & current_tokens) or (name_terms & current_terms)
                )
                candidates.append(entry)
                continue
            recent = bool(max(years) >= (datetime.now(timezone.utc).year - 5))  # within ~5 yrs
            tenure_span = max(years) - min(years)
            long_tenure = tenure_span >= 3
            name_tokens = set(re.findall(r"[A-Za-z0-9]+", company.lower()))
            name_terms = set(re.findall(r"[一-鿿]{2,}", company))
            related = bool((name_tokens & current_tokens) or (name_terms & current_terms))
            if recent or long_tenure or related:
                entry["_recent"] = recent
                entry["_tenure"] = tenure_span
                entry["_related"] = related
                candidates.append(entry)

        # Priority: recent and related > long tenure > recency alone.
        def priority(entry: dict[str, Any]) -> tuple[int, int]:
            p = 0
            if entry["_recent"]:
                p += 4
            if entry["_related"]:
                p += 2
            if entry["_tenure"] >= 3:
                p += 1
            return (p, entry["_tenure"])

        candidates.sort(key=priority, reverse=True)
        return [entry["company"] for entry in candidates[:limit]]

    async def _research_employers_inline(self, state: InterviewState) -> str:
        """Research recent past employers and return a prompt block (no persist).

        Used inside a running workflow where the session lock is already held.
        Returns an empty string when research is not allowed or yields nothing.
        """
        # Past-employer research touches sensitive candidate data, so it requires
        # explicit public-research authorization (no implicit consent in
        # interactive mode).
        if not state.autopilot.authorized_public_research:
            return ""
        employers = self._recent_employers(state, limit=2)
        collected: list[dict[str, Any]] = []
        for company in employers:
            collected.extend(await self._search_employer(company))
        existing = filter_employer_business_results(
            state.past_employer_sources, entity=""
        )
        merged = merge_search_results(existing, collected, limit=8)
        if not merged:
            state.past_employer_sources = []
            state.past_employer_block = ""
            return ""
        state.past_employer_sources = merged
        return format_employer_business_context(merged)

    async def _search_employer(self, company: str) -> list[dict[str, Any]]:
        """Find business-background sources for one past employer.

        This path deliberately never queries current jobs or recruiting pages:
        those describe the employer's present vacancies, not the candidate's
        historical responsibilities.
        """
        provider = self.search_provider
        if provider is None:
            return []
        queries = [
            f'"{company}" 官方网站 公司简介 主要业务 产品 服务',
            f'"{company}" official website about products services business',
        ]
        results: list[dict[str, Any]] = []
        for query in queries:
            try:
                batch = await provider.search(query, limit=5, search_depth="basic")
            except Exception as exc:  # noqa: BLE001 - provider may be disabled
                logger.warning("Past-employer search failed: %s", exc)
                continue
            batch_dicts: list[dict[str, Any]] = []
            for item in batch:
                if isinstance(item, dict):
                    batch_dicts.append(item)
                else:
                    batch_dicts.append(item.model_dump(mode="json"))
            filtered = filter_entity_results(
                batch_dicts, entity=company
            )
            if filtered:
                results = merge_search_results(results, filtered, limit=8)
                if any(
                    item.get("is_official")
                    or item.get("source_quality") == "official"
                    for item in filtered
                ):
                    break
        return filter_employer_business_results(results, entity=company)

    async def research_recent_employers(self, session_id: str) -> InterviewState:
        """Search the candidate's recent/important past employers and persist sources.

        Runs inside the session lock; stores merged sources, sets the research
        status, rebuilds fact cards/entity resolutions, and persists. Search is
        gated by the same public-research consent as company research.
        """
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            state = runtime.state
            if not state.autopilot.authorized_public_research:
                state.past_employer_research_status = "consent_required"
                await self._persist(session_id, state)
                return state
            employers = self._recent_employers(state, limit=2)
            collected: list[dict[str, Any]] = []
            state.past_employer_sources = filter_employer_business_results(
                state.past_employer_sources, entity=""
            )
            for company in employers:
                if any(
                    company.lower() in str(source.get("title", "")).lower()
                    or company.lower() in str(source.get("snippet", "")).lower()
                    for source in state.past_employer_sources
                ):
                    continue  # already researched by the inline workflow path
                sources = await self._search_employer(company)
                collected.extend(sources)
            state.past_employer_sources = merge_search_results(
                state.past_employer_sources, collected, limit=8
            )
            state.past_employer_research_status = (
                "completed" if state.past_employer_sources else "no_results"
            )
            state.past_employer_block = format_employer_business_context(
                state.past_employer_sources
            )
            self._sync_intelligence(state)
            await self._persist(session_id, state)
        self._record_debug(
            "past_employer_research",
            session_id,
            detail=f"employer_count={len(employers)}; sources={len(state.past_employer_sources)}",
        )
        return runtime.state

    async def analyze_interviewer(
        self,
        session_id: str,
        name: str,
        position: str = "",
        company: str = "",
        public_info: str = "",
    ) -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            runtime.state.interviewer = InterviewerProfile(
                name=name,
                position=position,
                company=company,
                public_expressions=[{"text": public_info}] if public_info else [],
            )
            message = await runtime.run("interviewer_agent")
            self._sync_intelligence(runtime.state)
            await self._persist(session_id, runtime.state)
            return message

    async def run_candidate_prep(
        self,
        session_id: str,
        *,
        resume_text: str,
        job_description: str,
        company_name: str,
        company_context: str = "",
        interviewer_name: str = "",
        interviewer_position: str = "",
        interviewer_public_info: str = "",
        authorized_public_research: bool = False,
        prepare_resume_transition: bool = True,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        if prepare_resume_transition:
            await self._prepare_resume_transition(session_id, runtime, resume_text)
        job_input, job_sources, job_research_status = await self._enrich_title_only_job_input(
            job_description, company_name, authorized=authorized_public_research
        )
        steps: list[tuple[str, str]] = [
            ("candidate_agent", resume_text),
            ("job_agent", job_input),
            ("company_agent", company_context),
        ]
        interviewer = None
        if interviewer_name:
            interviewer = InterviewerProfile(
                name=interviewer_name,
                position=interviewer_position,
                company=company_name,
                public_expressions=(
                    [{"text": interviewer_public_info}] if interviewer_public_info else []
                ),
            )
            steps.append(("interviewer_agent", ""))
        steps.extend(
            [
                ("interview_strategy_agent", "Generate personalized interview strategy"),
                ("mock_interview_agent", "Generate personalized mock questions"),
            ]
        )
        try:
            state = await self._execute_workflow(
                session_id,
                runtime,
                "candidate_prep",
                steps,
                company_name=company_name,
                interviewer=interviewer,
                parallel_prefix=4 if interviewer else 3,
                authorized_public_research=authorized_public_research,
            )
        except Exception:
            # JobAgent may already have persisted the internal source-enriched
            # prompt before a later strategy/question step fails. Restore the
            # user's original title and provenance on the failure path too.
            await self._attach_job_research(
                session_id,
                runtime.state,
                original_input=job_description,
                sources=job_sources,
                status=job_research_status,
            )
            raise
        await self._attach_job_research(
            session_id,
            state,
            original_input=job_description,
            sources=job_sources,
            status=job_research_status,
        )
        return state

    async def run_enterprise_design(
        self,
        session_id: str,
        *,
        resume_text: str,
        job_description: str,
        company_name: str,
        company_context: str = "",
        authorized_public_research: bool = False,
        prepare_resume_transition: bool = True,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        if prepare_resume_transition:
            await self._prepare_resume_transition(session_id, runtime, resume_text)
        job_input, job_sources, job_research_status = await self._enrich_title_only_job_input(
            job_description, company_name, authorized=authorized_public_research
        )
        steps = [
            ("candidate_agent", resume_text),
            ("job_agent", job_input),
            ("company_agent", company_context),
            ("interview_design_agent", "Design an evidence-based interview blueprint"),
        ]
        try:
            state = await self._execute_workflow(
                session_id,
                runtime,
                "enterprise_design",
                steps,
                company_name=company_name,
                parallel_prefix=3,
                authorized_public_research=authorized_public_research,
            )
        except Exception:
            await self._attach_job_research(
                session_id,
                runtime.state,
                original_input=job_description,
                sources=job_sources,
                status=job_research_status,
            )
            raise
        await self._attach_job_research(
            session_id,
            state,
            original_input=job_description,
            sources=job_sources,
            status=job_research_status,
        )
        return state

    async def _enrich_title_only_job_input(
        self, job_description: str, company_name: str, *, authorized: bool
    ) -> tuple[str, list[dict[str, Any]], str]:
        """Resolve a bare title into source-bound public JD context before planning."""
        initial_review = review_job_description(job_description, [])
        if not initial_review.is_title_only:
            return job_description, [], "not_needed"
        if not authorized:
            return job_description, [], "consent_required"
        if self.search_provider is None:
            return job_description, [], "not_configured"
        title = job_description.strip()
        company = company_name.strip()
        query = f'"{title}"'
        if company:
            query += f' "{company}"'
        query += " 岗位职责 任职要求 招聘 JD"
        try:
            results = await self.search_provider.search(query, limit=6, search_depth="advanced")
        except Exception as exc:  # noqa: BLE001 - provider boundary
            logger.warning("Public JD research failed (%s)", type(exc).__name__)
            return job_description, [], "failed"
        title_tokens = {
            token.casefold() for token in re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", title)
        }
        relevant = []
        for result in results:
            payload = result.model_dump(mode="json")
            haystack = f"{result.title} {result.snippet}".casefold()
            title_match = any(token in haystack for token in title_tokens)
            if title_match:
                relevant.append(payload)
        sources = relevant[:5]
        if not sources:
            return job_description, [], "no_reliable_sources"
        context = format_search_results(sources)
        enriched = (
            f"用户输入的职位名称：{title}\n"
            f"目标公司：{company or '未指定'}\n\n"
            "以下是公开招聘结果的候选 JD 摘要，未经用户确认；只能用于生成待核验的岗位问题，"
            "不得表述为目标公司的明确要求：\n"
            f"{context}"
        )
        return enriched, sources, "completed"

    async def _attach_job_research(
        self,
        session_id: str,
        state: InterviewState,
        *,
        original_input: str,
        sources: list[dict[str, Any]],
        status: str,
    ) -> None:
        if status == "not_needed":
            return
        async with self._lock_for(session_id):
            state.job.raw_description = original_input
            state.job_review.is_title_only = True
            state.job_review.completeness_score = 0.55 if sources else 0.2
            state.job_review.public_sources = sources
            state.job_review.public_research_status = status
            state.job_review.researched_title = original_input.strip()
            state.job_review.requirements = [
                item.model_copy(update={"origin": RequirementOrigin.INFERRED})
                for item in state.job_review.requirements
            ]
            warning = (
                "已根据公开招聘来源补全候选 JD；职责与要求仍是待确认推测。"
                if sources
                else "需要允许公开检索后才能根据职位名称补全 JD。"
                if status == "consent_required"
                else "未找到可靠公开 JD；当前问题只能按职位名称生成通用准备方向。"
            )
            state.job_review.warnings = list(dict.fromkeys([warning, *state.job_review.warnings]))
            await self._persist(session_id, state)
        self._record_debug(
            "job_jd_research_completed",
            session_id,
            detail=f"status={status}; sources={len(sources)}",
        )

    async def start_mock_interview(self, session_id: str) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            if not runtime.state.mock_interview.questions:
                raise MockInterviewStateError(
                    "No mock questions are available; run candidate preparation first"
                )
            if runtime.state.mock_session.status == MockSessionStatus.ACTIVE:
                return runtime.state
            if runtime.state.mock_session.status == MockSessionStatus.COMPLETED:
                raise MockInterviewStateError("Mock interview is already completed")
            if runtime.state.mock_session.status != MockSessionStatus.IDLE:
                raise MockInterviewStateError("Mock interview cannot restart while evaluating")
            runtime.state.mock_session = MockInterviewSession(
                status=MockSessionStatus.ACTIVE,
                started_at=datetime.now(timezone.utc),
            )
            runtime.state.next_action = "Answer the current mock interview question"
            await self._persist(session_id, runtime.state)
            return runtime.state

    async def add_custom_mock_question(
        self,
        session_id: str,
        question_text: str,
        *,
        competency: str = "",
        practice_now: bool = True,
    ) -> InterviewState:
        """Interpret and persist a user-supplied question in the mock pipeline.

        The semantic model call runs outside the session mutation lock. A
        deterministic interpretation is always available, and the original
        wording stays untouched regardless of model output.
        """
        runtime = await self._get_runtime(session_id)
        normalized_text = " ".join(question_text.split()).strip()
        normalized_competency = " ".join(competency.split()).strip()
        if len(normalized_text) < 2:
            raise MockInterviewStateError("自定义问题至少需要 2 个字符")

        # Analyze from a stable snapshot without blocking answers/navigation.
        # The write phase below revalidates status and duplicates after the
        # optional model call completes.
        async with self._lock_for(session_id):
            state = runtime.state
            if state.mock_session.status in {
                MockSessionStatus.EVALUATING,
                MockSessionStatus.COMPLETED,
            }:
                raise MockInterviewStateError("本轮已结束，请新建练习会话后添加问题")
            state_snapshot = state.model_copy(deep=True)
        agent = runtime.get_agent("mock_interview_agent")
        if agent is not None and hasattr(agent, "analyze_custom_question"):
            understanding = await agent.analyze_custom_question(  # type: ignore[union-attr]
                state_snapshot,
                normalized_text,
                competency=normalized_competency,
            )
        else:
            understanding = deterministic_question_understanding(
                normalized_text, competency=normalized_competency
            )

        async with self._lock_for(session_id):
            state = runtime.state
            mock_session = state.mock_session
            if mock_session.status in {
                MockSessionStatus.EVALUATING,
                MockSessionStatus.COMPLETED,
            }:
                raise MockInterviewStateError("本轮已结束，请新建练习会话后添加问题")

            questions = state.mock_interview.questions
            comparison = normalized_text.casefold()
            existing_index = next(
                (
                    index
                    for index, item in enumerate(questions)
                    if " ".join(item.question.split()).strip().casefold() == comparison
                ),
                None,
            )
            if existing_index is None:
                selected_competency = normalized_competency or understanding.competency
                requirements = (
                    agent.question_requirements(normalized_text)  # type: ignore[union-attr]
                    if agent is not None and hasattr(agent, "question_requirements")
                    else []
                )
                framework = (
                    agent.deterministic_framework(  # type: ignore[union-attr]
                        state, selected_competency
                    )
                    if agent is not None and hasattr(agent, "deterministic_framework")
                    else "建议按：直接结论 → 真实案例 → 个人行动 → 可验证结果 → 复盘来组织。"
                )
                example = (
                    agent.teaching_example(  # type: ignore[union-attr]
                        selected_competency, normalized_text
                    )
                    if agent is not None and hasattr(agent, "teaching_example")
                    else ""
                )
                custom_question = InterviewQuestion(
                    question=normalized_text,
                    competency=selected_competency,
                    rationale="用户认为面试中可能出现的问题",
                    strong_signals=requirements,
                    follow_ups=understanding.likely_follow_ups[:3],
                    answer_framework=framework,
                    question_requirements=requirements,
                    example_answer=example,
                    understanding=understanding,
                    source="custom",
                )
                if practice_now:
                    insertion_index = (
                        mock_session.current_question_index + 1
                        if mock_session.status == MockSessionStatus.ACTIVE and questions
                        else 0
                    )
                    questions.insert(insertion_index, custom_question)
                    target_index = insertion_index
                else:
                    questions.append(custom_question)
                    target_index = len(questions) - 1
            else:
                target_index = existing_index

            if practice_now:
                if mock_session.status == MockSessionStatus.IDLE:
                    state.mock_session = MockInterviewSession(
                        status=MockSessionStatus.ACTIVE,
                        started_at=datetime.now(timezone.utc),
                        current_question_index=target_index,
                    )
                    mock_session = state.mock_session
                else:
                    mock_session.pending_follow_up = ""
                    mock_session.pending_parent_question_id = None
                    mock_session.current_question_index = target_index
                state.next_action = "Answer the current custom mock interview question"
            else:
                state.next_action = "Review the custom question in the mock interview pool"
            await self._persist(session_id, state)
            return state

    async def submit_mock_answer(
        self,
        session_id: str,
        question_id: UUID,
        answer: str,
        *,
        retry: bool = False,
        retry_response_id: UUID | None = None,
        recording_id: UUID | None = None,
    ) -> InterviewState:
        """Evaluate one mock answer. Does NOT advance the interview — the caller
        (frontend) chooses 重新来 / 下一题 / 结束面试 afterwards."""
        runtime = await self._get_runtime(session_id)
        staged_audio: Path | None = None
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            if recording_id is not None:
                staged_audio = self._pending_mock_audio_path(session_id, recording_id)
                if staged_audio is None:
                    raise MockInterviewStateError("Recording does not belong to this session")
            if staged_audio is not None and any(
                item.audio_file == staged_audio.name
                for item in (
                    *mock_session.responses,
                    *mock_session.attempt_history,
                )
            ):
                raise MockInterviewStateError("Recording has already been submitted")
            questions = runtime.state.mock_interview.questions
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            if mock_session.current_question_index >= len(questions):
                raise MockInterviewStateError("Mock interview has no remaining questions")
            question = questions[mock_session.current_question_index]
            if retry_response_id is not None and not retry:
                raise MockInterviewStateError("retry_response_id requires retry=true")
            retry_target = (
                next(
                    (
                        item
                        for item in mock_session.responses
                        if item.id == retry_response_id and item.question_id == question.id
                    ),
                    None,
                )
                if retry_response_id is not None
                else None
            )
            if retry_response_id is not None and retry_target is None:
                raise MockInterviewStateError(
                    "Retry response does not belong to the current question"
                )
            is_follow_up = (
                retry_target.is_follow_up
                if retry_target is not None
                else bool(mock_session.pending_follow_up)
            )
            expected_id = mock_session.pending_parent_question_id or question.id
            if expected_id != question_id:
                raise MockInterviewStateError("Answer does not match the current question")
            asked_question = (
                retry_target.question
                if retry_target is not None
                else mock_session.pending_follow_up or question.question
            )

            prior = retry_target or next(
                (
                    item
                    for item in reversed(mock_session.responses)
                    if item.question_id == question.id and item.question == asked_question
                ),
                None,
            )
            if prior is not None and not retry:
                raise MockInterviewStateError(
                    "This question was already answered; submit with retry=true to replace it"
                )

            replaced_records: list[MockAnswerRecord] = []
            if retry and prior is not None:
                # Identify the replacement set before model work, but do not
                # mutate live state until a valid new evaluation exists.
                replaced_records = [prior]
                if prior is not None and not prior.is_follow_up:
                    # Follow-up answers were elicited from the old main answer;
                    # retaining them would leave stale evidence in the report.
                    replaced_records.extend(
                        item
                        for item in mock_session.responses
                        if item.question_id == question.id and item.is_follow_up
                    )

            record = MockAnswerRecord(
                question_id=question.id,
                question=asked_question,
                competency=question.competency,
                answer=answer,
                evaluation=AnswerEvaluation(
                    content=0.0, technical_depth=0.0, structure=0.0, impact=0.0
                ),
                answer_modality="asr" if staged_audio is not None else "typed",
                is_follow_up=is_follow_up,
                audio_file=staged_audio.name if staged_audio is not None else "",
                speech_delivery=(
                    self._mock_speech_feedback[(current_owner(), session_id, recording_id)][1]
                    if recording_id is not None
                    and (current_owner(), session_id, recording_id) in self._mock_speech_feedback
                    else SpeechDeliveryFeedback()
                ),
            )
            payload = json.dumps(
                {
                    "question": asked_question,
                    "answer": answer,
                    "competency": question.competency or "Answer Quality",
                    "evidence_source": "mock_interview",
                    "record_id": str(record.id),
                    "persist_evidence": False,
                    "answer_modality": record.answer_modality,
                }
            )
            message = await runtime.run("coach_agent", payload)
            try:
                evaluation = AnswerEvaluation.model_validate_json(message.content)
            except (ValueError, TypeError) as exc:
                raise MockInterviewStateError(
                    "Coach agent did not return a valid answer evaluation"
                ) from exc
            record.evaluation = evaluation
            if replaced_records:
                replaced_ids = {item.id for item in replaced_records}
                mock_session.attempt_history.extend(
                    item.model_copy(deep=True) for item in replaced_records
                )
                mock_session.responses = [
                    item for item in mock_session.responses if item.id not in replaced_ids
                ]
                runtime.state.evidence = [
                    item
                    for item in runtime.state.evidence
                    if not (
                        item.source == EvidenceSource.MOCK_INTERVIEW
                        and item.source_record_id in replaced_ids
                    )
                ]
                if prior is not None and not prior.is_follow_up:
                    mock_session.pending_follow_up = ""
                    mock_session.pending_parent_question_id = None
            mock_session.responses.append(record)
            runtime.state.evidence.append(
                Evidence(
                    competency=record.competency or "Answer Quality",
                    signal=answer[:200],
                    confidence=evaluation.overall_score(),
                    source=EvidenceSource.MOCK_INTERVIEW,
                    source_record_id=record.id,
                    polarity=EvidencePolarity.NEUTRAL,
                    notes="; ".join(evaluation.missing_signals),
                )
            )
            self._invalidate_final_reports(
                runtime.state, "Mock answer evidence changed; regenerate evaluation"
            )
            if is_follow_up:
                # The follow-up was answered: the question is complete. Clear the
                # pending flag so the UI shows the actions and a later /next
                # advances instead of re-offering the same follow-up.
                mock_session.pending_follow_up = ""
                mock_session.pending_parent_question_id = None
            runtime.state.next_action = "回答已评价：请选择 重新来 / 下一题 / 结束面试"
            await self._persist(session_id, runtime.state)
        return runtime.state

    async def advance_mock_interview(self, session_id: str) -> InterviewState:
        """Advance to the next question, or offer a follow-up when the latest
        main answer left missing signals. Advancing may skip the current
        question (the interviewer stays in control); refills when low."""
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            questions = runtime.state.mock_interview.questions
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            if mock_session.current_question_index >= len(questions):
                raise MockInterviewStateError("Mock interview has no remaining questions")
            question = questions[mock_session.current_question_index]
            if mock_session.pending_follow_up:
                # A follow-up is pending; advancing skips it and moves on.
                mock_session.pending_follow_up = ""
                mock_session.pending_parent_question_id = None
                mock_session.current_question_index += 1
            else:
                last_response = next(
                    (
                        item
                        for item in reversed(mock_session.responses)
                        if item.question_id == question.id
                    ),
                    None,
                )
                last_follow_up = last_response is not None and last_response.is_follow_up
                last_main = (
                    last_response
                    if last_response is not None and not last_response.is_follow_up
                    else None
                )
                if (
                    not last_follow_up
                    and last_main is not None
                    and last_main.evaluation.missing_signals
                    and question.follow_ups
                ):
                    # First advance after a main answer offers the follow-up
                    # (the user can skip it by advancing again).
                    mock_session.pending_follow_up = question.follow_ups[0]
                    mock_session.pending_parent_question_id = question.id
                    runtime.state.next_action = "Answer the evidence-seeking follow-up question"
                    await self._persist(session_id, runtime.state)
                    return runtime.state
                mock_session.current_question_index += 1
            if mock_session.current_question_index >= len(questions):
                # The model refill stays asynchronous, but an active session must
                # never expose an empty current question at the pool boundary.
                agent = runtime.get_agent("mock_interview_agent")
                if agent is not None and hasattr(agent, "deterministic_refill_question"):
                    questions.append(
                        agent.deterministic_refill_question(  # type: ignore[union-attr]
                            runtime.state, ordinal=len(questions) + 1
                        )
                    )
            runtime.state.next_action = "Answer the next mock interview question"
            await self._maybe_refill_mock_questions(session_id, runtime, mock_session, questions)
            await self._persist(session_id, runtime.state)
        return runtime.state

    async def previous_mock_question(self, session_id: str) -> InterviewState:
        """Go back to the previous question (no-op at the first question)."""
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            if mock_session.current_question_index <= 0:
                raise MockInterviewStateError("已回到第一题")
            # Clear any follow-up state so the previous question renders cleanly.
            mock_session.pending_follow_up = ""
            mock_session.pending_parent_question_id = None
            mock_session.current_question_index -= 1
            runtime.state.next_action = "Answer the current mock interview question"
            await self._persist(session_id, runtime.state)
        return runtime.state

    async def finish_mock_interview(self, session_id: str) -> InterviewState:
        """End the mock interview manually and run the evaluation when answers exist."""
        runtime = await self._get_runtime(session_id)
        should_evaluate = False
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            if mock_session.status == MockSessionStatus.EVALUATING:
                raise MockInterviewStateError("Mock interview evaluation is already running")
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            should_evaluate = bool(mock_session.responses)
            if should_evaluate:
                mock_session.status = MockSessionStatus.EVALUATING
                runtime.state.next_action = "Generate evidence-based evaluation"
            else:
                mock_session.status = MockSessionStatus.COMPLETED
                mock_session.completed_at = datetime.now(timezone.utc)
                runtime.state.next_action = "Mock interview completed without answers"
            await self._persist(session_id, runtime.state)
        if should_evaluate:
            try:
                state = await self.run_evaluation(session_id)
            except Exception:
                # Keep all answers and make the normal finish action retryable.
                async with self._lock_for(session_id):
                    runtime.state.mock_session.status = MockSessionStatus.ACTIVE
                    runtime.state.mock_session.completed_at = None
                    runtime.state.next_action = (
                        "Final evaluation failed; retry ending the interview"
                    )
                    await self._persist(session_id, runtime.state)
                raise
            async with self._lock_for(session_id):
                state.mock_session.status = MockSessionStatus.COMPLETED
                state.mock_session.completed_at = datetime.now(timezone.utc)
                await self._persist(session_id, state)
            return state
        return runtime.state

    async def _maybe_refill_mock_questions(
        self, session_id: str, runtime, mock_session, questions
    ) -> None:
        pending = len(questions) - mock_session.current_question_index
        if (
            pending < MOCK_CACHE_MIN
            and not mock_session.refill_in_flight
            and mock_session.status == MockSessionStatus.ACTIVE
        ):
            mock_session.refill_in_flight = True
            self._record_debug("mock_refill_scheduled", session_id)
            owner = current_owner()
            self._background.schedule(self._refill_mock_questions_task(session_id, owner=owner))

    async def _refill_mock_questions_task(self, session_id: str, *, owner: str) -> None:
        """Background: generate more mock questions when the cache runs low."""
        runtime = self._runtimes.get((owner, session_id))
        if runtime is None:
            return
        agent = runtime.get_agent("mock_interview_agent")
        recent = [
            f"{item.competency}：{item.answer[:120]}"
            for item in runtime.state.mock_session.responses
        ]
        new_questions: list[Any] = []
        try:
            if agent is not None and hasattr(agent, "generate_mock_questions"):
                new_questions = await agent.generate_mock_questions(  # type: ignore[union-attr]
                    runtime.state, recent_answers=recent
                )
        except Exception as exc:  # noqa: BLE001 - background boundary
            logger.error("Mock question refill failed for %s (%s)", session_id, type(exc).__name__)
            new_questions = []
        async with self._lock_for(session_id):
            current = self._runtimes.get((owner, session_id))
            if current is None:
                return
            mock_session = current.state.mock_session
            mock_session.refill_in_flight = False
            if mock_session.status == MockSessionStatus.ACTIVE and new_questions:
                existing_ids = {q.id for q in current.state.mock_interview.questions}
                existing_texts = {
                    q.question.strip() for q in current.state.mock_interview.questions
                }
                added = 0
                for q in new_questions:
                    if q.id not in existing_ids and q.question.strip() not in existing_texts:
                        current.state.mock_interview.questions.append(q)
                        existing_texts.add(q.question.strip())
                        added += 1
                self._record_debug("mock_refill_completed", session_id, detail=f"added={added}")
            await self._persist(session_id, current.state)

    async def run_evaluation(self, session_id: str) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        state = await self._execute_workflow(
            session_id,
            runtime,
            "evaluation",
            [
                ("evaluation_agent", "Aggregate evidence and evaluate competencies"),
                ("feedback_agent", "Generate candidate and interviewer feedback"),
            ],
        )
        if state.autopilot.enabled:
            state.autopilot.status = AutopilotStatus.COMPLETED
            state.autopilot.phase = "completed"
            state.autopilot.pause_reason = ""
            state.autopilot.completed_actions.extend(
                action
                for action in ("answer_evaluation", "final_evaluation", "feedback_generation")
                if action not in state.autopilot.completed_actions
            )
            state.autopilot.updated_at = datetime.now(timezone.utc)
            await self._persist(session_id, state)
            self._record_debug("autopilot_completed", session_id)
        return state

    async def review_answer_evaluation(
        self,
        session_id: str,
        record_id: UUID,
        *,
        content: float,
        technical_depth: float,
        structure: float,
        impact: float,
        evidence_polarity: EvidencePolarity = EvidencePolarity.NEUTRAL,
        note: str = "",
    ) -> InterviewState:
        """Persist a human-reviewed score for either a mock or live answer."""
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            records: list[MockAnswerRecord | LiveInterviewRecord] = [
                *runtime.state.mock_session.responses,
                *runtime.state.live_interview_records,
            ]
            record = next((item for item in records if item.id == record_id), None)
            if record is None:
                raise EvaluationStateError("Answer record was not found")
            linked_evidence = next(
                (item for item in runtime.state.evidence if item.source_record_id == record_id),
                None,
            )
            if linked_evidence is None:
                raise EvaluationStateError("Linked evidence was not found")
            evaluation = record.evaluation.model_copy(deep=True)
            evaluation.content = content
            evaluation.technical_depth = technical_depth
            evaluation.structure = structure
            evaluation.impact = impact
            evaluation.scoring_source = AnswerScoringSource.HUMAN
            evaluation.review_status = AnswerReviewStatus.REVIEWED
            apply_specific_feedback(
                evaluation,
                record.answer,
                question=record.question,
                competency=record.competency,
            )
            clean_note = note.strip()
            if clean_note:
                review_feedback = f"人工复核：{clean_note}"
                evaluation.feedback = list(dict.fromkeys([*evaluation.feedback, review_feedback]))
            record.evaluation = evaluation
            if isinstance(record, LiveInterviewRecord):
                record.scoring_status = "scored"
                record.scoring_error = ""

            linked_evidence.confidence = evaluation.overall_score()
            linked_evidence.polarity = evidence_polarity

            # Any existing final report was calculated from the superseded score.
            self._invalidate_final_reports(
                runtime.state, "Regenerate evaluation after human score review"
            )
            runtime.state.next_action = "Regenerate evaluation after human score review"
            runtime.state.missing_signals = list(
                dict.fromkeys(gap for item in records for gap in item.evaluation.missing_signals)
            )
            self._refresh_live_coverage_guidance(runtime.state)
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "answer_score_human_reviewed",
            session_id,
            detail=(
                f"record={str(record_id)[:8]}; polarity={evidence_polarity.value}; "
                f"score={evaluation.overall_score():.2f}"
            ),
        )
        return runtime.state

    async def run_autopilot(
        self,
        session_id: str,
        *,
        role: str,
        resume_text: str,
        job_description: str,
        company_name: str,
        company_context: str = "",
        interviewer_name: str = "",
        interviewer_position: str = "",
        authorized_public_research: bool = False,
    ) -> InterviewState:
        """Advance every safe deterministic/agent step and pause at real-world input."""
        runtime = await self._get_runtime(session_id)
        now = datetime.now(timezone.utc)
        async with self._lock_for(session_id):
            self._assert_candidate_reset_allowed(runtime.state)
            previous_candidate = runtime.state.candidate
            same_resume = previous_candidate.raw_resume_text.strip() == resume_text.strip()
            runtime.state.candidate = (
                previous_candidate.model_copy(deep=True)
                if same_resume
                else CandidateProfile(
                    # The session label is user-controlled identity metadata, not
                    # a derived resume artifact. Preserve it while clearing every
                    # analyzed skill/claim from the prior document.
                    name=previous_candidate.name,
                    raw_resume_text=resume_text,
                )
            )
            if not same_resume:
                # Confirmations belong to a specific resume. Reusing them after
                # the document changes can leak one candidate's facts into the
                # next candidate's prompts and public-employer research.
                runtime.state.resume_review = ResumeReview()
            runtime.state.candidate.raw_resume_text = resume_text
            runtime.state.job = JobDescription()
            runtime.state.job_review = JobDescriptionReview()
            runtime.state.company = CompanyInfo(name=company_name)
            runtime.state.interviewer = InterviewerProfile(
                name=interviewer_name,
                position=interviewer_position,
                company=company_name,
            )
            runtime.state.entity_resolutions = []
            runtime.state.fact_cards = []
            runtime.state.past_employer_sources = []
            runtime.state.past_employer_research_status = "not_requested"
            runtime.state.past_employer_block = ""
            runtime.state.strategy = InterviewStrategy()
            runtime.state.blueprint = InterviewBlueprint()
            runtime.state.mock_interview = MockInterviewPlan()
            runtime.state.mock_session = MockInterviewSession()
            runtime.state.evaluation = EvaluationReport()
            runtime.state.feedback = FeedbackReport()
            runtime.state.live_interview_records = []
            runtime.state.live_interview = LiveInterviewSession()
            runtime.state.evidence.clear()
            runtime.state.evaluated_competencies.clear()
            runtime.state.missing_signals.clear()
            runtime.state.conversation_history.clear()
            runtime.state.current_stage = InterviewStage.NOT_STARTED
            runtime.state.autopilot = AutopilotState(
                enabled=True,
                status=AutopilotStatus.RUNNING,
                phase="intelligence",
                authorized_public_research=authorized_public_research,
                started_at=now,
                updated_at=now,
            )
            await self._persist(session_id, runtime.state)
        self._record_debug("autopilot_started", session_id, detail=role)
        try:
            if role == "candidate":
                state = await self.run_candidate_prep(
                    session_id,
                    resume_text=resume_text,
                    job_description=job_description,
                    company_name=company_name,
                    company_context=company_context,
                    interviewer_name=interviewer_name,
                    interviewer_position=interviewer_position,
                    authorized_public_research=authorized_public_research,
                    prepare_resume_transition=False,
                )
                state = await self.start_mock_interview(session_id)
                state.autopilot.completed_actions = [
                    "resume_analysis",
                    "job_analysis",
                    "company_research",
                    "strategy_generation",
                    "mock_plan_generation",
                ]
                state.autopilot.phase = "interview"
                state.autopilot.status = AutopilotStatus.WAITING_FOR_INPUT
                state.autopilot.pause_reason = "等待候选人回答当前问题"
                state.next_action = "Answer the current AI-led interview question"
            else:
                state = await self.run_enterprise_design(
                    session_id,
                    resume_text=resume_text,
                    job_description=job_description,
                    company_name=company_name,
                    company_context=company_context,
                    authorized_public_research=authorized_public_research,
                    prepare_resume_transition=False,
                )
                state.autopilot.completed_actions = [
                    "resume_analysis",
                    "job_analysis",
                    "company_research",
                    "interview_blueprint_generation",
                ]
                state.autopilot.phase = "interview_execution"
                state.autopilot.status = AutopilotStatus.WAITING_FOR_INPUT
                state.autopilot.pause_reason = "等待真实面试回答或面试记录，不能由 AI 代造证据"
                state.next_action = "Conduct the structured interview and collect evidence"
            state.autopilot.updated_at = datetime.now(timezone.utc)
            await self._persist(session_id, state)
            self._record_debug("autopilot_paused", session_id, detail=state.autopilot.pause_reason)
            return state
        except Exception:
            runtime.state.autopilot.status = AutopilotStatus.FAILED
            runtime.state.autopilot.phase = "error"
            runtime.state.autopilot.updated_at = datetime.now(timezone.utc)
            await self._persist(session_id, runtime.state)
            raise

    @staticmethod
    def current_mock_question(state: InterviewState):
        session = state.mock_session
        questions = state.mock_interview.questions
        if session.status != MockSessionStatus.ACTIVE or session.current_question_index >= len(
            questions
        ):
            return None
        question = questions[session.current_question_index]
        if session.pending_follow_up:
            return question.model_copy(
                update={
                    "question": session.pending_follow_up,
                    "rationale": "根据上一回答中缺失的证据进行追问",
                }
            )
        return question

    async def import_interview_transcript(
        self,
        session_id: str,
        entries: list[dict[str, str]],
        *,
        auto_evaluate: bool = True,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            for entry in entries[:50]:
                question = entry.get("question", "").strip()
                answer = entry.get("answer", "").strip()
                competency = entry.get("competency", "").strip() or "综合能力"
                if not question or not answer:
                    raise EvaluationStateError("Transcript entries require question and answer")
                evidence_count_before = len(runtime.state.evidence)
                message = await runtime.run(
                    "coach_agent",
                    json.dumps(
                        {
                            "question": question,
                            "answer": answer,
                            "competency": competency,
                            "evidence_source": "live_interview",
                            "answer_modality": "typed",
                        },
                        ensure_ascii=False,
                    ),
                )
                try:
                    evaluation = AnswerEvaluation.model_validate_json(message.content)
                except (ValueError, TypeError) as exc:
                    raise EvaluationStateError("Transcript answer evaluation failed") from exc
                record = LiveInterviewRecord(
                    question=question,
                    answer=answer,
                    competency=competency,
                    evaluation=evaluation,
                    source="live_interview",
                    scoring_status="scored",
                )
                runtime.state.live_interview_records.append(record)
                for evidence in runtime.state.evidence[evidence_count_before:]:
                    if evidence.source.value == "live_interview":
                        evidence.source_record_id = record.id
            runtime.state.next_action = "Generate evidence-based hiring evaluation"
            self._invalidate_final_reports(
                runtime.state, "Transcript evidence changed; regenerate evaluation"
            )
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "interview_transcript_imported",
                session_id,
                detail=f"{min(len(entries), 50)} entries",
            )
        if auto_evaluate:
            return await self.run_evaluation(session_id)
        return runtime.state

    async def _execute_workflow(
        self,
        session_id: str,
        runtime: AgentRuntime,
        name: str,
        steps: list[tuple[str, str]],
        company_name: str | None = None,
        interviewer: InterviewerProfile | None = None,
        parallel_prefix: int = 0,
        authorized_public_research: bool = False,
    ) -> InterviewState:
        workflow_started = perf_counter()
        async with self._lock_for(session_id):
            if name == "evaluation" and not runtime.state.evidence:
                raise EvaluationStateError(
                    "No interview evidence is available; complete a mock or live interview first"
                )
            if name == "evaluation":
                if runtime.state.mock_session.status == MockSessionStatus.ACTIVE:
                    raise EvaluationStateError(
                        "Finish the active mock interview before generating the final report"
                    )
                if runtime.state.live_interview.status in {
                    LiveInterviewStatus.ACTIVE,
                    LiveInterviewStatus.PAUSED,
                }:
                    raise EvaluationStateError(
                        "Complete the live interview before generating the final report"
                    )
                if any(
                    record.scoring_status != "scored"
                    for record in runtime.state.live_interview_records
                ):
                    raise EvaluationStateError(
                        "Wait for live answer scoring or complete human review before evaluation"
                    )
            if company_name is not None:
                runtime.state.company.name = company_name
            if interviewer is not None:
                runtime.state.interviewer = interviewer
            # Record explicit public-research authorization so the past-employer
            # search respects it even in the non-autopilot workflow paths.
            runtime.state.autopilot.authorized_public_research = authorized_public_research
            runtime.state.workflow = WorkflowProgress(
                name=name,
                status=WorkflowStatus.RUNNING,
                total_steps=len(steps),
            )
            self._record_debug("workflow_started", session_id, detail=name)
            await self._persist(session_id, runtime.state)
            try:
                start_index = 0
                if parallel_prefix:
                    initial = steps[:parallel_prefix]
                    runtime.state.workflow.current_step = " + ".join(
                        agent_name for agent_name, _ in initial
                    )
                    await self._persist(session_id, runtime.state)
                    await asyncio.gather(
                        *(
                            runtime.run(agent_name, instruction)
                            for agent_name, instruction in initial
                        )
                    )
                    start_index = len(initial)
                    runtime.state.workflow.completed_steps = start_index
                    self._sync_intelligence(runtime.state)
                    self._refresh_live_question_usage(runtime.state)
                    await self._persist(session_id, runtime.state)
                    # After candidate/job/company analysis, research the
                    # candidate's recent/important past employers so strategy and
                    # design agents can reference what those companies do.
                    if any(agent_name == "candidate_agent" for agent_name, _ in initial):
                        block = await self._research_employers_inline(runtime.state)
                        if block:
                            runtime.state.past_employer_block = block
                            await self._persist(session_id, runtime.state)
                for index, (agent_name, instruction) in enumerate(
                    steps[start_index:], start=start_index + 1
                ):
                    runtime.state.workflow.current_step = agent_name
                    await self._persist(session_id, runtime.state)
                    await runtime.run(agent_name, instruction)
                    runtime.state.workflow.completed_steps = index
                    self._sync_intelligence(runtime.state)
                    self._refresh_live_question_usage(runtime.state)
                    await self._persist(session_id, runtime.state)
                self._validate_workflow_result(name, runtime.state)
            except Exception as exc:
                runtime.state.workflow.status = WorkflowStatus.FAILED
                runtime.state.workflow.error = str(exc)
                await self._persist(session_id, runtime.state)
                self._record_debug(
                    "workflow_failed",
                    session_id,
                    level=DebugLevel.ERROR,
                    detail=f"{name}: {exc}",
                )
                raise WorkflowExecutionError(str(exc)) from exc
            runtime.state.workflow.status = WorkflowStatus.COMPLETED
            runtime.state.workflow.current_step = ""
            runtime.state.next_action = (
                "Start mock interview"
                if name == "candidate_prep"
                else "Execute interview blueprint"
            )
            if name == "evaluation":
                runtime.state.current_stage = InterviewStage.COMPLETED
                runtime.state.next_action = "Review the final evaluation and feedback"
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "workflow_completed",
                session_id,
                detail=name,
                duration_ms=(perf_counter() - workflow_started) * 1000,
            )
            return runtime.state

    async def resolve_entity_candidate(
        self,
        session_id: str,
        resolution_id: UUID,
        *,
        accept: bool,
        proposed_name: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            try:
                resolution: EntityResolution = resolve_entity(
                    runtime.state,
                    resolution_id,
                    accept=accept,
                    proposed_name=proposed_name,
                )
            except LookupError as exc:
                raise ResumeReviewStateError(str(exc)) from exc
            self._sync_intelligence(runtime.state)
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "entity_resolution_updated",
                session_id,
                detail=f"{resolution.input_name} -> {resolution.proposed_name}: {resolution.status.value}",
            )
            return runtime.state

    async def decide_fact_card(
        self, session_id: str, card_id: UUID, *, action: str, note: str = ""
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            try:
                card = decide_fact_card(runtime.state, card_id, action=action, note=note)
            except (LookupError, ValueError) as exc:
                raise ResumeReviewStateError(str(exc)) from exc
            runtime.state.next_action = (
                "Use confirmed fact cards or continue public research review"
            )
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "fact_card_decided",
                session_id,
                detail=f"{card.category}: {card.status.value}",
            )
            return runtime.state

    @staticmethod
    def _sync_intelligence(state: InterviewState) -> None:
        sync_entity_resolutions(state)
        build_fact_cards(state)

    @staticmethod
    def _resolve_live_question_segment(
        segments: list[TranscriptSegment],
        answer_segment: TranscriptSegment,
        question_segment_id: UUID | None,
    ) -> TranscriptSegment | None:
        if question_segment_id is not None:
            explicit = next((item for item in segments if item.id == question_segment_id), None)
            if explicit is None:
                raise LiveInterviewStateError("Question segment was not found")
            if explicit.speaker != TranscriptSpeaker.INTERVIEWER:
                raise LiveInterviewStateError("Question segment must belong to the interviewer")
            return explicit
        prior_questions = [
            item
            for item in segments
            if item.sequence < answer_segment.sequence
            and item.speaker == TranscriptSpeaker.INTERVIEWER
            and item.stable
            and item.confirmed
        ]
        return prior_questions[-1] if prior_questions else None

    def _refresh_live_answer_boundaries(self, state: InterviewState) -> None:
        live = state.live_interview
        recorded_segment_ids = {
            segment_id
            for record in state.live_interview_records
            for segment_id in record.transcript_segment_ids
        }
        suggestions: list[LiveAnswerBoundarySuggestion] = []
        current_question: TranscriptSegment | None = None
        candidate_run: list[TranscriptSegment] = []

        def flush_run() -> None:
            if len(candidate_run) < MIN_ANSWER_BOUNDARY_SEGMENTS:
                return
            question_text = current_question.text if current_question else ""
            confidence, factors = self._score_live_boundary(candidate_run, current_question)
            suggested_competency = (
                self._infer_live_competency(state, question_text)
                or (state.job.competencies[0] if state.job.competencies else "")
                or "综合能力"
            )
            suggestions.append(
                LiveAnswerBoundarySuggestion(
                    question_segment_id=current_question.id if current_question else None,
                    answer_segment_ids=[item.id for item in candidate_run],
                    suggested_competency=suggested_competency,
                    confidence=confidence,
                    confidence_factors=factors,
                    reason=(
                        "连续候选人片段之间没有新的面试官问题，"
                        "建议作为同一回答合并复核后再归档证据。"
                    ),
                )
            )

        for segment in live.segments:
            if segment.speaker == TranscriptSpeaker.INTERVIEWER:
                flush_run()
                current_question = segment if segment.stable and segment.confirmed else None
                candidate_run = []
                continue
            if (
                segment.speaker == TranscriptSpeaker.CANDIDATE
                and segment.stable
                and segment.confirmed
                and segment.id not in recorded_segment_ids
            ):
                candidate_run.append(segment)
                continue
            flush_run()
            candidate_run = []
        flush_run()
        live.answer_boundary_suggestions = suggestions[-ANSWER_BOUNDARY_SUGGESTIONS_MAX:]

    def _score_live_boundary(
        self,
        candidate_run: list[TranscriptSegment],
        current_question: TranscriptSegment | None,
    ) -> tuple[float, list[str]]:
        score = BOUNDARY_BASE_SCORE
        factors: list[str] = []
        if current_question is not None:
            score += 0.12
            factors.append("已关联最近面试官问题")
        else:
            score -= 0.08
            factors.append("未找到明确面试官问题，需人工确认上下文")
        run_bonus = min(0.18, 0.06 * (len(candidate_run) - 1))
        score += run_bonus
        factors.append(f"{len(candidate_run)} 段连续候选人发言")
        total_chars = sum(len(item.text) for item in candidate_run)
        if total_chars >= 160:
            score += 0.1
            factors.append("回答较长，适合作为完整回答合并")
        elif total_chars >= 70:
            score += 0.06
            factors.append("回答长度达到合并阈值")
        else:
            score -= 0.04
            factors.append("回答较短，可能只是补充短句")
        last_text = candidate_run[-1].text
        if re.search(
            r"(以上|大概就是|基本就是|总结|最后|最终|就这些|谢谢|that's all)",
            last_text,
            re.IGNORECASE,
        ):
            score += 0.08
            factors.append("末段出现回答结束信号")
        if any(item.source == "asr" for item in candidate_run):
            factors.append("包含 ASR 分段，建议复核文本后再合并")
        return max(BOUNDARY_MIN_SCORE, min(BOUNDARY_MAX_SCORE, round(score, 2))), factors[:8]

    def _refresh_live_rolling_summary(self, state: InterviewState) -> None:
        live = state.live_interview
        stable_segments = [
            item
            for item in live.segments
            if item.stable
            and item.confirmed
            and item.speaker != TranscriptSpeaker.UNKNOWN
        ]
        if len(stable_segments) <= LIVE_RECENT_SEGMENT_WINDOW:
            live.rolling_summary = ""
            live.summarized_until_sequence = 0
            return
        cutoff = stable_segments[-LIVE_RECENT_SEGMENT_WINDOW].sequence - 1
        summarized = [item for item in stable_segments if item.sequence <= cutoff]
        speaker_labels = {
            TranscriptSpeaker.INTERVIEWER: "面试官",
            TranscriptSpeaker.CANDIDATE: "候选人",
            TranscriptSpeaker.UNKNOWN: "待确认",
        }
        lines = [
            f"#{item.sequence} {speaker_labels[item.speaker]}: {item.text[:180]}"
            for item in summarized
        ]
        summary = "\n".join(lines)
        if len(summary) > LIVE_SUMMARY_CHAR_LIMIT:
            summary = "…\n" + summary[-LIVE_SUMMARY_CHAR_LIMIT:]
        live.rolling_summary = summary
        live.summarized_until_sequence = cutoff

    def _find_duplicate_live_segment(
        self,
        segments: list[TranscriptSegment],
        text: str,
        speaker: TranscriptSpeaker,
    ) -> TranscriptSegment | None:
        normalized = self._normalize_live_segment_text(text)
        if len(normalized) < 8:
            return None
        for segment in reversed(segments[-LIVE_RECENT_SEGMENT_WINDOW:]):
            if segment.speaker != speaker or not segment.stable or not segment.confirmed:
                continue
            prior = self._normalize_live_segment_text(segment.text)
            if normalized == prior:
                return segment
        return None

    @staticmethod
    def _normalize_live_segment_text(value: str) -> str:
        return re.sub(r"[\W_]+", "", value.lower(), flags=re.UNICODE)

    def _refresh_live_coverage_guidance(self, state: InterviewState) -> None:
        competencies = self._live_target_competencies(state)
        if not competencies:
            state.live_interview.coverage_guidance = []
            return
        by_competency: dict[str, list[float]] = {name: [] for name in competencies}
        for evidence in state.evidence:
            competency = evidence.competency.strip()
            if competency not in by_competency:
                by_competency[competency] = []
            by_competency[competency].append(evidence.confidence)
        guidance: list[LiveCoverageGuidance] = []
        for competency in competencies:
            confidences = by_competency.get(competency, [])
            count = len(confidences)
            strongest = max(confidences, default=0.0)
            if count == 0:
                priority = "high"
                reason = "尚无已确认 live 证据，最终评价无法覆盖该能力。"
                question_type = "main"
                sample = f"请讲一个最能体现你“{competency}”能力的真实项目。"
            elif count < CROSS_VALIDATION_EVIDENCE_COUNT:
                priority = "medium"
                reason = "已有 1 条证据，但缺少交叉验证，建议再问一个独立场景。"
                question_type = "follow_up"
                sample = f"刚才关于“{competency}”的例子还有哪些量化结果或风险权衡？"
            elif strongest <= WEAK_SIGNAL_THRESHOLD:
                priority = "medium"
                reason = "已有多条证据，但评分信号偏弱，需要更具体的行为和结果。"
                question_type = "deep_dive"
                sample = f"请把“{competency}”相关经历拆成目标、行动、指标和复盘。"
            else:
                priority = "low"
                reason = "已有可用证据；除非该能力是关键门槛，否则可转向其他缺口。"
                question_type = "optional"
                sample = f"如果时间允许，可补一个“{competency}”失败或复盘案例。"
            guidance.append(
                LiveCoverageGuidance(
                    competency=competency,
                    evidence_count=count,
                    strongest_confidence=strongest,
                    priority=priority,
                    reason=reason,
                    suggested_question_type=question_type,
                    sample_question=sample,
                )
            )
        priority_order = {"high": 0, "medium": 1, "low": 2}
        guidance.sort(
            key=lambda item: (
                priority_order.get(item.priority, 3),
                item.evidence_count,
                -item.strongest_confidence,
                item.competency,
            )
        )
        state.live_interview.coverage_guidance = guidance[:COVERAGE_GUIDANCE_MAX_ITEMS]

    def _refresh_live_question_usage(self, state: InterviewState) -> None:
        questions = self._live_blueprint_questions(state)
        if not questions:
            state.live_interview.question_usage = []
            return
        by_question_id: dict[str, list[QuestionSuggestion]] = {item[0]: [] for item in questions}
        for suggestion in state.live_interview.suggestions:
            source_id = suggestion.source_question_id.strip()
            if source_id in by_question_id:
                by_question_id[source_id].append(suggestion)
        usage: list[LiveQuestionUsage] = []
        used_ids = set(state.live_interview.used_question_ids)
        for question_id, round_name, question in questions:
            related = by_question_id.get(question_id, [])
            latest = related[-1] if related else None
            status = "used" if question_id in used_ids else "suggested" if related else "pending"
            usage.append(
                LiveQuestionUsage(
                    question_id=question_id,
                    round_name=round_name,
                    question=question.question,
                    competency=question.competency,
                    status=status,
                    suggested_count=len(related),
                    last_suggestion_status=latest.status.value if latest else "",
                    last_decided_at=latest.created_at if latest and status == "used" else None,
                )
            )
        status_order = {"pending": 0, "suggested": 1, "used": 2}
        usage.sort(
            key=lambda item: (status_order.get(item.status, 3), item.round_name, item.question)
        )
        state.live_interview.question_usage = usage[:QUESTION_USAGE_MAX_ITEMS]

    def _refresh_live_action_card(self, state: InterviewState) -> None:
        live = state.live_interview
        live_evidence = [
            item for item in state.live_interview_records if item.source == "live_interview"
        ]
        recorded_segment_ids = {
            segment_id for record in live_evidence for segment_id in record.transcript_segment_ids
        }
        pending_candidate_segments = [
            item
            for item in live.segments
            if item.speaker == TranscriptSpeaker.CANDIDATE
            and item.stable
            and item.confirmed
            and item.id not in recorded_segment_ids
        ]
        unknown_segments = [
            item
            for item in live.segments
            if item.speaker == TranscriptSpeaker.UNKNOWN and item.stable and item.confirmed
        ]
        pending_suggestion = next(
            (
                item
                for item in reversed(live.suggestions)
                if item.status == QuestionSuggestionStatus.PENDING
            ),
            None,
        )
        boundary = max(
            live.answer_boundary_suggestions,
            key=lambda item: item.confidence,
            default=None,
        )
        total_evidence = len(state.evidence)
        covered_competencies = {
            item.competency for item in state.evidence if item.competency.strip()
        }
        evidence_status = f"{total_evidence} 条证据 · {len(covered_competencies)} 个能力覆盖"
        top_gap = live.coverage_guidance[0] if live.coverage_guidance else None
        source_refs = [
            f"segments={len(live.segments)}",
            f"pending_candidate={len(pending_candidate_segments)}",
            f"live_evidence={len(live_evidence)}",
        ]

        def card(
            action_type: str,
            priority: str,
            title: str,
            detail: str,
            primary_cta: str,
            secondary_cta: str = "",
            refs: list[str] | None = None,
        ) -> LiveActionCard:
            return LiveActionCard(
                action_type=action_type,
                priority=priority,
                title=title,
                detail=detail,
                primary_cta=primary_cta,
                secondary_cta=secondary_cta,
                evidence_status=evidence_status,
                source_refs=[*source_refs, *(refs or [])][:ACTION_CARD_SOURCE_REFS_MAX],
            )

        if live.status == LiveInterviewStatus.IDLE:
            live.action_card = card(
                "start",
                "medium",
                "开始实时面试",
                "确认候选人已知情并同意转写后，启动监听并记录第一轮问题。",
                "开始实时会话",
            )
            return
        if live.status == LiveInterviewStatus.PAUSED:
            live.action_card = card(
                "resume",
                "medium",
                "实时面试已暂停",
                "恢复监听前，可以先审阅已产生的转写片段和证据归档状态。",
                "恢复监听",
                "审阅证据",
            )
            return
        if live.status == LiveInterviewStatus.COMPLETED:
            ready = (
                total_evidence >= MIN_EVIDENCE_COUNT
                and len(covered_competencies) >= MIN_COMPETENCY_COVERAGE
            )
            live.action_card = card(
                "evaluate" if ready else "review_evidence",
                "high" if not ready else "medium",
                "生成最终评价" if ready else "先补齐证据再评价",
                (
                    "证据数量和能力覆盖已达到招聘评价门槛，可以聚合证据生成报告。"
                    if ready
                    else "实时会话已结束，但证据数量或能力覆盖不足；请先确认候选人回答或导入记录。"
                ),
                "聚合证据并评估" if ready else "审阅待确认回答",
                refs=["completed=true"],
            )
            return
        if unknown_segments:
            live.action_card = card(
                "review_speaker",
                "high",
                "先确认待识别说话人",
                "连续监听产生了待确认片段；确认说话人与文本后，后续 Agent 才能安全使用这些事实。",
                "审阅转写片段",
                refs=[f"unknown={len(unknown_segments)}"],
            )
            return
        if boundary is not None and boundary.confidence >= BOUNDARY_MERGE_THRESHOLD:
            live.action_card = card(
                "merge_boundary",
                "high",
                "合并连续候选人回答",
                (
                    f"检测到 {len(boundary.answer_segment_ids)} 段连续候选人发言，"
                    f"边界置信度 {round(boundary.confidence * 100)}%；建议先合并确认成一条证据。"
                ),
                "按边界建议合并",
                "逐条确认",
                refs=[f"boundary={round(boundary.confidence, 2)}"],
            )
            return
        if pending_candidate_segments:
            live.action_card = card(
                "confirm_evidence",
                "high",
                "归档候选人回答为证据",
                (
                    f"还有 {len(pending_candidate_segments)} 段候选人回答未进入证据链；"
                    "先确认能力维度，再让下一问题建议基于已确认事实。"
                ),
                "确认为证据",
                "生成下一问题",
            )
            return
        if pending_suggestion is not None:
            live.action_card = card(
                "decide_question",
                "medium",
                "处理下一问题建议",
                (
                    f"AI 已准备一个聚焦“{pending_suggestion.competency}”的问题；"
                    "采用、编辑或跳过后，蓝图问题使用图会同步更新。"
                ),
                "采用或编辑问题",
                "跳过建议",
                refs=[f"suggestion={str(pending_suggestion.id)[:8]}"],
            )
            return
        if top_gap is not None and (
            total_evidence < MIN_EVIDENCE_COUNT
            or len(covered_competencies) < MIN_COMPETENCY_COVERAGE
            or top_gap.priority in {"high", "medium"}
        ):
            live.action_card = card(
                "plan_gap_question",
                "medium" if total_evidence >= CROSS_VALIDATION_EVIDENCE_COUNT else "high",
                f"补齐“{top_gap.competency}”证据",
                f"{top_gap.reason} 建议下一问：{top_gap.sample_question}",
                "根据回答生成",
                "继续提问",
                refs=[f"gap={top_gap.competency}", f"priority={top_gap.priority}"],
            )
            return
        if (
            total_evidence < MIN_EVIDENCE_COUNT
            or len(covered_competencies) < MIN_COMPETENCY_COVERAGE
        ):
            live.action_card = card(
                "plan_gap_question",
                "high",
                "继续补齐招聘评价证据",
                "当前证据数量或能力覆盖仍不足；如果 JD 能力维度尚未完善，请先补充岗位职责、任职要求和团队背景。",
                "根据回答生成",
                "补充 JD 信息",
                refs=["gap=generic"],
            )
            return
        live.action_card = card(
            "ready_to_evaluate",
            "low",
            "证据链已基本可用",
            "当前证据数量和能力覆盖达到基础招聘评价门槛；可以继续深挖，也可以结束后生成评价。",
            "结束并生成评价",
            "继续追问",
        )

    @staticmethod
    def _live_blueprint_questions(
        state: InterviewState,
    ) -> list[tuple[str, str, InterviewQuestion]]:
        return [
            (str(question.id), interview_round.name, question)
            for interview_round in state.blueprint.rounds
            for question in interview_round.questions
            if question.question.strip()
        ]

    @staticmethod
    def _live_target_competencies(state: InterviewState) -> list[str]:
        values = [
            *state.job.competencies,
            *[
                question.competency
                for interview_round in state.blueprint.rounds
                for question in interview_round.questions
            ],
            *[question.competency for question in state.mock_interview.questions],
        ]
        return list(dict.fromkeys(item.strip() for item in values if item.strip()))

    @staticmethod
    def _infer_live_competency(state: InterviewState, question: str) -> str:
        for suggestion in reversed(state.live_interview.suggestions):
            final_question = suggestion.final_question or suggestion.suggested_question
            if final_question and (
                final_question == question
                or final_question[:80] in question
                or question[:80] in final_question
            ):
                return suggestion.competency
        for round_item in state.blueprint.rounds:
            for item in round_item.questions:
                if item.question and (
                    item.question == question
                    or item.question[:80] in question
                    or question[:80] in item.question
                ):
                    return item.competency
        if state.job.competencies:
            return state.job.competencies[0]
        if state.mock_interview.questions:
            return state.mock_interview.questions[0].competency
        return ""

    @staticmethod
    def _validate_workflow_result(name: str, state: InterviewState) -> None:
        if name == "candidate_prep":
            if not state.strategy.summary and not state.strategy.key_risks:
                raise ValueError("Strategy agent did not return a valid structured result")
            if not state.mock_interview.questions:
                raise ValueError("Mock interview agent did not return any questions")
        elif name == "enterprise_design" and not state.blueprint.rounds:
            raise ValueError("Interview design agent did not return any rounds")
        elif name == "evaluation":
            if not state.evaluation.competencies:
                raise ValueError("Evaluation agent did not return competency results")
            if not state.feedback.overall:
                raise ValueError("Feedback agent did not return a valid report")

    async def _run(self, session_id: str, agent_name: str, instruction: str) -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            message = await runtime.run(agent_name, instruction)
            await self._persist(session_id, runtime.state)
            return message

    async def _get_runtime(self, session_id: str) -> AgentRuntime:
        owner = current_owner()
        cached = self._runtimes.get((owner, session_id))
        if cached is not None:
            return cached
        state = await self.storage.get_session_state(session_id, owner_id=owner)
        if state is None:
            raise SessionNotFoundError(session_id)
        runtime = create_runtime(
            self.llm_client, self.search_provider, self.debug_events, session_id
        )
        runtime.state = InterviewState.model_validate(state)
        # Upgrade persisted pre-quality-contract questions in place. This keeps
        # existing user sessions usable after deployment instead of requiring a
        # new candidate workflow solely to obtain bounded questions/examples.
        mock_agent = runtime.get_agent("mock_interview_agent")
        if mock_agent is not None and hasattr(mock_agent, "enrich_question"):
            for question in runtime.state.mock_interview.questions:
                # User-supplied wording is itself part of the practice contract.
                # It already receives requirements/framework/example at insert
                # time and must not be rewritten by legacy-question migration.
                if question.source == "custom" and question.understanding is None:
                    question.understanding = deterministic_question_understanding(
                        question.question,
                        competency=(
                            "" if question.competency == "自定义问题" else question.competency
                        ),
                    )
                elif question.source != "custom":
                    mock_agent.enrich_question(question)  # type: ignore[union-attr]
        # In-process refill tasks do not survive a service restart.
        runtime.state.mock_session.refill_in_flight = False
        if runtime.state.mock_session.status == MockSessionStatus.EVALUATING:
            # The evaluation coroutine was process-local. Restore a retryable
            # state rather than leaving the session permanently in-flight.
            runtime.state.mock_session.status = MockSessionStatus.ACTIVE
            runtime.state.mock_session.completed_at = None
            runtime.state.next_action = "Evaluation was interrupted; retry ending the interview"
        if not runtime.state.job_review.requirements and (
            runtime.state.job.raw_description or runtime.state.job.title
        ):
            inferred = [
                *runtime.state.job.required_skills,
                *runtime.state.job.preferred_skills,
                *runtime.state.job.competencies,
            ]
            runtime.state.job_review = review_job_description(
                runtime.state.job.raw_description or runtime.state.job.title,
                list(dict.fromkeys(inferred)),
            )
        self._sync_intelligence(runtime.state)
        self._refresh_live_coverage_guidance(runtime.state)
        self._refresh_live_question_usage(runtime.state)
        if runtime.state.evaluation.finalized_at:
            runtime.state.enforce_evaluation_evidence_floor()
        self._runtimes[(owner, session_id)] = runtime
        self._locks.setdefault((owner, session_id), asyncio.Lock())
        await self._persist(session_id, runtime.state)
        return runtime

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        owner = current_owner()
        return self._locks.setdefault((owner, session_id), asyncio.Lock())

    async def _persist(self, session_id: str, state: InterviewState) -> None:
        self._refresh_live_action_card(state)
        await self.storage.save_session(
            session_id, state.model_dump(mode="json"), owner_id=current_owner()
        )

    def get_cached_runtime(
        self, session_id: str, *, owner_id: str | None = None
    ) -> AgentRuntime | None:
        """Return a cached runtime for a session.

        The debug console (localhost-only) inspects any session, so by default
        this scans all owners. Owner-scoped callers can pass ``owner_id``.
        """
        if owner_id is not None:
            return self._runtimes.get((owner_id, session_id))
        for (owner, sid), runtime in self._runtimes.items():
            if sid == session_id:
                return runtime
        return None

    def _record_debug(
        self,
        action: str,
        session_id: str,
        *,
        level: DebugLevel = DebugLevel.INFO,
        detail: str = "",
        duration_ms: float | None = None,
    ) -> None:
        if self.debug_events is not None:
            self.debug_events.record(
                DebugEvent(
                    level=level,
                    category="service",
                    action=action,
                    session_id=session_id,
                    duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
                    detail=detail[:1000],
                )
            )
