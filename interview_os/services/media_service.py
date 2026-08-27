"""Audio transcription, recording, speech feedback, and synthesis operations."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID

from interview_os.core.debug import DebugLevel
from interview_os.core.evidence import EvidenceSource
from interview_os.core.request_context import current_owner
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import (
    LIVE_RECENT_SEGMENT_WINDOW,
    OMNI_CONTEXT_CHAR_LIMIT,
    InterviewState,
    LiveInterviewStatus,
    QuestionSuggestion,
    QuestionSuggestionType,
    SpeechDeliveryFeedback,
    TranscriptSpeaker,
)
from interview_os.tools.asr import ASRError
from interview_os.tools.tts import TTSError

logger = logging.getLogger(__name__)

from interview_os.services.service_contracts import (
    SPEECH_FEEDBACK_PROHIBITED_TERMS,
    LiveInterviewStateError,
    MockInterviewStateError,
    WorkflowExecutionError,
)
from interview_os.services.service_mixin import InterviewServiceMixin


class MediaServiceMixin(InterviewServiceMixin):
    """Audio transcription, recording, speech feedback, and synthesis operations."""

    def set_live_audio_mode(self, mode: str) -> None:
        if mode not in {"asr_text", "audio_direct"}:
            raise ValueError(f"Unsupported live audio mode: {mode}")
        if mode == "audio_direct" and self.omni_client is None:
            raise ValueError("Live audio direct requires an omni client")
        self.live_audio_mode = mode

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
            if item.stable and item.confirmed and item.speaker != TranscriptSpeaker.UNKNOWN
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
