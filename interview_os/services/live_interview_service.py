"""Live interview transcript, question planning, and evidence operations."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from uuid import UUID

from interview_os.core.evidence import Evidence, EvidencePolarity, EvidenceSource
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
    EVIDENCE_SIGNAL_CHAR_LIMIT,
    LIVE_RECENT_SEGMENT_WINDOW,
    LIVE_SUMMARY_CHAR_LIMIT,
    MIN_ANSWER_BOUNDARY_SEGMENTS,
    MIN_COMPETENCY_COVERAGE,
    MIN_EVIDENCE_COUNT,
    QUESTION_USAGE_MAX_ITEMS,
    WEAK_SIGNAL_THRESHOLD,
    AnswerEvaluation,
    AnswerReviewStatus,
    AnswerScoringSource,
    InterviewQuestion,
    InterviewState,
    LiveActionCard,
    LiveAnswerBoundarySuggestion,
    LiveCoverageGuidance,
    LiveInterviewRecord,
    LiveInterviewSession,
    LiveInterviewStatus,
    LiveQuestionUsage,
    QuestionSuggestion,
    QuestionSuggestionStatus,
    TranscriptSegment,
    TranscriptSpeaker,
)

logger = logging.getLogger(__name__)

from interview_os.services.service_contracts import (
    EvaluationStateError,
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
                signal=answer_text[:EVIDENCE_SIGNAL_CHAR_LIMIT],
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
            placeholder.signal = record.answer[:EVIDENCE_SIGNAL_CHAR_LIMIT]
            placeholder.confidence = evaluation.overall_score()
            # The signal has just been replaced by model output. Never carry a
            # human classification from the previous signal across that change.
            placeholder.polarity = EvidencePolarity.NEUTRAL
            placeholder.notes = "; ".join(evaluation.missing_signals)
            return
        state.evidence.append(
            Evidence(
                competency=record.competency,
                signal=record.answer[:EVIDENCE_SIGNAL_CHAR_LIMIT],
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
