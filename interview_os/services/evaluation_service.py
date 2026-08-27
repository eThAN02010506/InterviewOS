"""Evidence aggregation and human score review operations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from interview_os.core.answer_feedback import apply_specific_feedback
from interview_os.core.evidence import EvidencePolarity
from interview_os.core.state import (
    AnswerReviewStatus,
    AnswerScoringSource,
    AutopilotStatus,
    InterviewState,
    LiveInterviewRecord,
    MockAnswerRecord,
)
from interview_os.services.service_contracts import EvaluationStateError
from interview_os.services.service_mixin import InterviewServiceMixin


class EvaluationServiceMixin(InterviewServiceMixin):
    """Generate final reports and apply explicit human score overrides."""

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
                for action in (
                    "answer_evaluation",
                    "final_evaluation",
                    "feedback_generation",
                )
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
