"""Feedback Agent - generates structured feedback report."""
from __future__ import annotations

import logging

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewState

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
            f"Missing signals: {state.missing_signals}"
        )
        prompt = (
            "Generate a structured feedback report including: "
            "1. Overall assessment, "
            "2. Strengths demonstrated, "
            "3. Areas for improvement, "
            "4. Specific evidence supporting each point, "
            "5. Recommendation (hire/no-hire/strong-hire) with reasoning."
        )
        raw = await self.think(prompt, context=context)
        return self.make_response(raw)
