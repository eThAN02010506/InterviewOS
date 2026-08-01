"""Feedback Agent - generates structured feedback report."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import FeedbackReport, InterviewState

logger = logging.getLogger(__name__)


class FeedbackAgent(Agent):
    """Generates structured feedback for both enterprise and candidate sides."""

    def __init__(self, **kwargs):
        super().__init__(
            name="feedback_agent",
            role="Feedback Report Generator",
            goal="Generate actionable, evidence-based feedback reports",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        context = (
            f"Evaluated competencies: {state.evaluated_competencies}\n"
            f"Evidence count: {len(state.evidence)}\n"
            f"Missing signals: {state.missing_signals}\n"
            f"Evaluation: {state.evaluation.model_dump_json()}"
        )
        prompt = (
            "Generate evidence-based feedback as JSON with overall, strengths, improvements, "
            "action_plan, interviewer_notes, and recommendation_reasoning. All fields except "
            "overall and recommendation_reasoning are lists of strings. Candidate-facing "
            "improvements must be actionable; interviewer notes must distinguish missing "
            "signals from negative evidence. Use Chinese for every narrative field."
        )
        try:
            state.feedback = await self.think_structured(prompt, FeedbackReport, context=context)
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse feedback report: %s", exc)
            evidence_count = len(state.evidence)
            state.feedback = FeedbackReport(
                overall=state.evaluation.summary or "已根据现有证据完成评价。",
                strengths=[
                    item.signal
                    for item in state.evidence
                    if item.confidence >= 0.7 and item.signal
                ][:5],
                improvements=list(dict.fromkeys(state.missing_signals))[:5],
                action_plan=["补充至少两道不同胜任力问题，以形成交叉验证"],
                interviewer_notes=[f"当前共有 {evidence_count} 条可追溯证据"],
                recommendation_reasoning=(
                    "结构化反馈生成失败，本段仅转述确定性证据聚合结果；未添加新事实。"
                ),
            )
            self.record_degradation(
                "Structured feedback failed; summarized the finalized evidence report"
            )
        state.enforce_evaluation_evidence_floor()
        return self.make_response(state.feedback.model_dump_json())
