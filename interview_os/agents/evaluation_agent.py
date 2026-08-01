"""Evaluation Agent - aggregates evidence for evidence-based decisions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import (
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
        evidence_competencies = {
            evidence.competency.strip().casefold() for evidence in state.evidence if evidence.competency
        }
        reported_competencies = {
            item.competency.strip().casefold() for item in report.competencies if item.competency
        }
        ungrounded_competencies = reported_competencies - evidence_competencies
        if state.evidence and (not reported_competencies or ungrounded_competencies):
            logger.warning(
                "Evaluation report omitted or invented competency results; using evidence aggregate"
            )
            report = self._fallback_report(state)
            self.record_degradation(
                "Structured evaluation competencies were not grounded; aggregated recorded evidence"
            )
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
