"""Evaluation Agent - aggregates evidence for evidence-based decisions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import (
    AnswerReviewStatus,
    AnswerScoringSource,
    CompetencyEvaluation,
    EvaluationReport,
    HiringRecommendation,
    InterviewState,
)
from interview_os.models.prompt_templates import EVALUATION_PROMPT

logger = logging.getLogger(__name__)


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
        evidence_text = "\n".join(
            f"- [{e.source.value}] {e.competency}: {e.signal} (confidence: {e.confidence})"
            for e in state.evidence
        )
        prompt = EVALUATION_PROMPT.format(
            evidence_list=evidence_text or "No evidence collected yet."
        )
        try:
            report = await self.think_structured(prompt, EvaluationReport, context=state.summary())
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse evaluation report: %s", exc)
            report = self._fallback_report(state)
            self.record_degradation(
                "Structured evaluation failed; aggregated recorded evidence without adding facts"
            )
        self._ground_competencies(report, state)
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

    @classmethod
    def _ground_competencies(cls, report: EvaluationReport, state: InterviewState) -> None:
        """Replace model-controlled numeric fields with persisted evidence aggregates."""
        grounded = cls._fallback_report(state)
        if not report.competencies:
            report.summary = report.summary or grounded.summary
            report.risks = list(dict.fromkeys([*report.risks, *grounded.risks]))
        reported_by_name = {
            item.competency.strip().casefold(): item
            for item in report.competencies
            if item.competency.strip()
        }
        for item in grounded.competencies:
            model_item = reported_by_name.get(item.competency.strip().casefold())
            if model_item is not None:
                item.gaps = list(dict.fromkeys([*model_item.gaps, *item.gaps]))
        report.competencies = grounded.competencies

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
                "模型招聘建议与证据分数或置信度不一致，已按确定性阈值校准。"
            )

    @staticmethod
    def _fallback_report(state: InterviewState) -> EvaluationReport:
        grouped: dict[str, list] = {}
        for evidence in state.evidence:
            grouped.setdefault(evidence.competency or "综合能力", []).append(evidence)
        competencies = []
        for competency, evidence_items in grouped.items():
            score = sum(item.confidence for item in evidence_items) / len(evidence_items)
            competencies.append(
                CompetencyEvaluation(
                    competency=competency,
                    score=score,
                    confidence=min(0.85, 0.45 + 0.1 * len(evidence_items)),
                    supporting_evidence=[item.signal for item in evidence_items if item.signal],
                    gaps=[] if len(evidence_items) >= 2 else ["需要更多独立回答交叉验证"],
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
            risks=[] if sufficient else ["结构化模型输出失败；当前报告使用证据聚合兜底"],
        )
