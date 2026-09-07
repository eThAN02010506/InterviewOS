"""Live answer confirmation, evidence scoring, and review operations."""

from __future__ import annotations

import json
import logging
from uuid import UUID

from interview_os.core.evidence import Evidence, EvidencePolarity, EvidenceSource
from interview_os.core.request_context import current_owner
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import (
    COVERAGE_GUIDANCE_MAX_ITEMS,
    CROSS_VALIDATION_EVIDENCE_COUNT,
    EVIDENCE_SIGNAL_CHAR_LIMIT,
    WEAK_SIGNAL_THRESHOLD,
    AnswerEvaluation,
    AnswerReviewStatus,
    AnswerScoringSource,
    InterviewState,
    LiveCoverageGuidance,
    LiveInterviewRecord,
    LiveInterviewStatus,
    TranscriptSpeaker,
)

logger = logging.getLogger(__name__)

from interview_os.services.service_contracts import (
    EvaluationStateError,
    LiveInterviewStateError,
)
from interview_os.services.service_mixin import InterviewServiceMixin


class LiveEvidenceServiceMixin(InterviewServiceMixin):
    """Live answer confirmation, evidence scoring, and review operations."""

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
            logger.error("Live scoring failed for %s (%s)", record_id, type(exc).__name__)
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
                current.scoring_error = f"Scoring failed ({type(exc).__name__})"
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
