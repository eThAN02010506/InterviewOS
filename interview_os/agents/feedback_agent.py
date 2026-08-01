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
            return self.make_response("{}")
        return self.make_response(state.feedback.model_dump_json())
