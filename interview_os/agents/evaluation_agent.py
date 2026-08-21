"""Evaluation Agent - aggregates evidence for evidence-based decisions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from uuid import UUID

from interview_os.core.agent import Agent
from interview_os.core.evidence import Evidence, EvidencePolarity
from interview_os.core.message import Message
from interview_os.core.state import (
    AnswerReviewStatus,
    AnswerScoringSource,
    CompetencyEvaluation,
    EvaluationReport,
    HiringRecommendation,
    InterviewState,
)
from interview_os.models.prompt_templates import EVALUATION_NARRATIVE_PROMPT
from interview_os.models.structured import EvaluationNarrativeDraft


class EvaluationAgent(Agent):
    """Aggregates all evidence to produce competency scores and hiring recommendation."""

    def __init__(self, **kwargs):
        super().__init__(
            name="evaluation_agent",
            role="Evidence Evaluator",
            goal="Aggregate evidence into competency scores and hiring recommendation",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        # Recruitment decisions are a trust boundary. Scores and factual fields are
        # rebuilt from persisted Evidence; the optional model pass can only explain
        # that locked result through validated same-competency references.
        report = self._fallback_report(state)
        await self._add_model_narrative(state, report)
        provisional_scoring = self._has_provisional_scores(state)
        self._calibrate_recommendation(report, provisional_scoring=provisional_scoring)
        state.evaluation = report
        state.enforce_evaluation_evidence_floor()
        report = state.evaluation
        report.finalized_at = datetime.now(timezone.utc)
        state.evaluated_competencies = {item.competency: item.score for item in report.competencies}
        state.missing_signals = list(
            dict.fromkeys(gap for item in report.competencies for gap in item.gaps)
        )
        return self.make_response(report.model_dump_json())

    async def _add_model_narrative(
        self, state: InterviewState, report: EvaluationReport
    ) -> None:
        """Add explanation and next probes without delegating factual aggregation."""
        if self.llm_client is None or not report.competencies:
            return
        frame = []
        evidence_ids: dict[str, dict[int, UUID]] = {}
        competency_names: dict[str, str] = {}
        independent_case_counts: dict[str, int] = {}
        evidence_by_competency: dict[str, list[Evidence]] = {}
        mock_records = {record.id: record for record in state.mock_session.responses}
        live_record_ids = {record.id for record in state.live_interview_records}
        for item in state.evidence:
            evidence_by_competency.setdefault(item.competency or "综合能力", []).append(item)
        for competency_index, result in enumerate(report.competencies, start=1):
            competency_id = f"C{competency_index}"
            competency_names[competency_id] = result.competency
            items = evidence_by_competency.get(result.competency, [])
            numbered = {}
            evidence_rows = []
            independent_case_keys: set[str] = set()
            for number, item in enumerate(items, start=1):
                numbered[number] = item.id
                mock_record = (
                    mock_records.get(item.source_record_id)
                    if item.source_record_id is not None
                    else None
                )
                if mock_record is not None:
                    case_key = f"mock:{mock_record.question_id}"
                    answer_kind = (
                        "follow_up_same_case" if mock_record.is_follow_up else "main_answer"
                    )
                elif item.source_record_id in live_record_ids:
                    case_key = f"live:{item.source_record_id}"
                    answer_kind = "live_answer"
                else:
                    case_key = f"evidence:{item.id}"
                    answer_kind = "other_evidence"
                independent_case_keys.add(case_key)
                evidence_rows.append(
                    {
                        "number": number,
                        "candidate_evidence": item.signal[:500],
                        "confidence": round(item.confidence, 4),
                        "polarity": item.polarity.value,
                        "answer_kind": answer_kind,
                        "case_key": case_key,
                    }
                )
            evidence_ids[competency_id] = numbered
            independent_case_counts[competency_id] = len(independent_case_keys)
            frame.append(
                {
                    "competency_id": competency_id,
                    "competency_label": result.competency,
                    "fixed_score": round(result.score, 4),
                    "fixed_gaps": result.gaps,
                    "evidence": evidence_rows,
                }
            )
        prompt = EVALUATION_NARRATIVE_PROMPT.format(
            evaluation_frame=json.dumps(frame, ensure_ascii=False)
        )
        try:
            draft = await self.think_structured(
                prompt,
                EvaluationNarrativeDraft,
                max_tokens=1800,
            )
            if not self._apply_model_narrative(
                report,
                draft,
                evidence_ids,
                competency_names,
                independent_case_counts,
            ):
                raise ValueError("narrative competency or evidence references did not match")
        except Exception as exc:  # noqa: BLE001 - optional narrative must not block report
            self.record_degradation(
                f"Final narrative validation failed; deterministic report retained ({type(exc).__name__})"
            )

    @staticmethod
    def _normalize_assessment_for_locked_gaps(assessment: str, gaps: list[str]) -> str:
        """Prevent a model narrative from contradicting the locked evidence frame."""
        independent_gap = "需要更多独立回答交叉验证"
        replacement = (
            "现有回答已形成证据，仍需要第二个独立案例交叉验证"
            if independent_gap in gaps
            else "现有回答已形成证据，但该结论仍需补充验证"
        )
        # A validated narrative must cite at least one evidence number, so any
        # blanket statement that the answer supplied no evidence is false even
        # when additional locked gaps are present.
        for contradiction in (
            "当前回答尚未提供证据",
            "当前没有提供证据",
            "缺少任何证据",
            "没有证据",
        ):
            assessment = assessment.replace(contradiction, replacement)
        return assessment

    @staticmethod
    def _apply_model_narrative(
        report: EvaluationReport,
        draft: EvaluationNarrativeDraft,
        evidence_ids: dict[str, dict[int, UUID]],
        competency_names: dict[str, str],
        independent_case_counts: dict[str, int],
    ) -> bool:
        """Atomically apply a complete narrative whose references all resolve."""
        reviews = {item.competency_id: item for item in draft.competency_reviews}
        expected = set(competency_names)
        if (
            not draft.summary.strip()
            or len(reviews) != len(draft.competency_reviews)
            or set(reviews) != expected
        ):
            return False
        resolved: dict[str, tuple[str, str, list[UUID]]] = {}
        asserted_multiple_cases = re.compile(
            r"(?:两次|两个|第二个|2个).{0,3}(?:案例|经历).{0,8}(?:显示|证明|体现|提升|说明)|"
            r"(?:通过|已有|展示了).{0,5}(?:两次|两个|2个)(?:案例|经历)"
        )
        if sum(independent_case_counts.values()) < 2 and asserted_multiple_cases.search(
            draft.summary
        ):
            return False
        for competency_id in expected:
            review = reviews[competency_id]
            allowed = evidence_ids.get(competency_id, {})
            numbers = list(dict.fromkeys(review.evidence_numbers))
            if (
                not review.assessment.strip()
                or not review.next_probe.strip()
                or not numbers
                or any(number not in allowed for number in numbers)
                or (
                    independent_case_counts.get(competency_id, 0) < 2
                    and asserted_multiple_cases.search(review.assessment)
                )
            ):
                return False
            resolved[competency_names[competency_id]] = (
                review.assessment.strip()[:600],
                review.next_probe.strip()[:300],
                [allowed[number] for number in numbers],
            )
        for result in report.competencies:
            assessment, next_probe, refs = resolved[result.competency]
            result.assessment = EvaluationAgent._normalize_assessment_for_locked_gaps(
                assessment, result.gaps
            )
            result.next_probe = next_probe
            result.narrative_evidence_ids = refs
        report.summary = re.sub(r"\bC\d+\s*", "", draft.summary.strip())[:1000]
        report.narrative_source = "model"
        return True

    @staticmethod
    def _has_provisional_scores(state: InterviewState) -> bool:
        evaluations = [
            response.evaluation for response in state.mock_session.responses
        ] + [record.evaluation for record in state.live_interview_records]
        has_unreviewed_rule_score = any(
            evaluation.scoring_source == AnswerScoringSource.DETERMINISTIC_RULE
            and evaluation.review_status != AnswerReviewStatus.REVIEWED
            for evaluation in evaluations
        )
        has_unfinished_live_score = any(
            record.scoring_status != "scored" for record in state.live_interview_records
        )
        return has_unreviewed_rule_score or has_unfinished_live_score

    @staticmethod
    def _calibrate_recommendation(
        report: EvaluationReport, *, provisional_scoring: bool = False
    ) -> None:
        """Bind hiring labels to the persisted scores and confidence."""
        if not report.competencies:
            report.overall_score = 0.0
            report.recommendation = HiringRecommendation.INSUFFICIENT_EVIDENCE
            return
        overall = sum(item.score for item in report.competencies) / len(report.competencies)
        confidence = sum(item.confidence for item in report.competencies) / len(
            report.competencies
        )
        original = report.recommendation
        report.overall_score = round(overall, 4)
        if provisional_scoring or confidence < 0.5:
            calibrated = HiringRecommendation.INSUFFICIENT_EVIDENCE
        elif overall >= 0.85 and confidence >= 0.75:
            calibrated = HiringRecommendation.STRONG_HIRE
        elif overall >= 0.75 and confidence >= 0.65:
            calibrated = HiringRecommendation.HIRE
        elif overall >= 0.6:
            calibrated = HiringRecommendation.LEAN_HIRE
        elif overall >= 0.45:
            calibrated = HiringRecommendation.LEAN_NO_HIRE
        else:
            calibrated = HiringRecommendation.NO_HIRE
        report.recommendation = calibrated
        if provisional_scoring:
            report.risks.append(
                "部分回答仅完成确定性规则评分，尚未人工复核，不能形成录用或不录用建议。"
            )
        if calibrated != original:
            report.risks.append(
                "初始建议与证据分数或置信度不一致，已按确定性阈值校准。"
            )

    @staticmethod
    def _fallback_report(state: InterviewState) -> EvaluationReport:
        grouped: dict[str, list] = {}
        mock_records = {record.id: record for record in state.mock_session.responses}
        live_record_ids = {record.id for record in state.live_interview_records}
        for evidence in state.evidence:
            grouped.setdefault(evidence.competency or "综合能力", []).append(evidence)
        competencies = []
        for competency, evidence_items in grouped.items():
            # Positive/neutral evidence retains its demonstrated competency
            # score. Explicitly human-classified negative evidence can never
            # improve a hiring recommendation: confidence in an adverse signal
            # moves the contribution down, capped below the lean-hire boundary.
            directional_scores = [
                min(0.4, 1.0 - item.confidence)
                if item.polarity == EvidencePolarity.NEGATIVE
                else item.confidence
                for item in evidence_items
            ]
            score = sum(directional_scores) / len(directional_scores)
            independent_case_keys: set[str] = set()
            for item in evidence_items:
                mock_record = (
                    mock_records.get(item.source_record_id)
                    if item.source_record_id is not None
                    else None
                )
                if mock_record is not None:
                    independent_case_keys.add(f"mock:{mock_record.question_id}")
                elif item.source_record_id in live_record_ids:
                    independent_case_keys.add(f"live:{item.source_record_id}")
                else:
                    independent_case_keys.add(f"evidence:{item.id}")
            independent_case_count = len(independent_case_keys)
            recorded_gaps = list(
                dict.fromkeys(
                    gap.strip()
                    for item in evidence_items
                    for gap in re.split(r"[;；]", item.notes)
                    if gap.strip()
                )
            )
            if independent_case_count < 2:
                recorded_gaps.append("需要更多独立回答交叉验证")
            competencies.append(
                CompetencyEvaluation(
                    competency=competency,
                    score=score,
                    confidence=min(0.85, 0.45 + 0.1 * independent_case_count),
                    supporting_evidence=[item.signal for item in evidence_items if item.signal],
                    gaps=list(dict.fromkeys(recorded_gaps)),
                )
            )
        overall = (
            sum(item.score for item in competencies) / len(competencies) if competencies else 0.0
        )
        sufficient = len(state.evidence) >= 3 and len(competencies) >= 2
        recommendation = HiringRecommendation.INSUFFICIENT_EVIDENCE
        if sufficient:
            recommendation = (
                HiringRecommendation.HIRE
                if overall >= 0.75
                else HiringRecommendation.LEAN_HIRE
                if overall >= 0.6
                else HiringRecommendation.LEAN_NO_HIRE
            )
        return EvaluationReport(
            competencies=competencies,
            overall_score=overall,
            recommendation=recommendation,
            summary=(
                "已按已记录证据完成确定性聚合；当前证据量不足以形成稳健招聘结论。"
                if not sufficient
                else "已按已记录证据完成确定性聚合。"
            ),
            risks=[] if sufficient else ["当前证据数量或胜任力覆盖尚未达到招聘决策门槛"],
        )
