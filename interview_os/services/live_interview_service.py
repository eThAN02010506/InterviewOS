"""Live interview transcript, question planning, and evidence operations."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from uuid import UUID

from interview_os.core.state import (
    ANSWER_BOUNDARY_SUGGESTIONS_MAX,
    BOUNDARY_BASE_SCORE,
    BOUNDARY_MAX_SCORE,
    BOUNDARY_MIN_SCORE,
    LIVE_RECENT_SEGMENT_WINDOW,
    LIVE_SUMMARY_CHAR_LIMIT,
    MIN_ANSWER_BOUNDARY_SEGMENTS,
    InterviewState,
    LiveAnswerBoundarySuggestion,
    LiveInterviewSession,
    LiveInterviewStatus,
    TranscriptSegment,
    TranscriptSpeaker,
)

logger = logging.getLogger(__name__)

from interview_os.services.service_contracts import (
    LiveInterviewStateError,
)
from interview_os.services.service_mixin import InterviewServiceMixin


class LiveInterviewServiceMixin(InterviewServiceMixin):
    """Live interview transcript, question planning, and evidence operations."""

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

    @staticmethod
    def assert_live_active(state: InterviewState) -> None:
        if state.live_interview.status != LiveInterviewStatus.ACTIVE:
            raise LiveInterviewStateError("Live interview must be active")

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
            if item.stable and item.confirmed and item.speaker != TranscriptSpeaker.UNKNOWN
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
