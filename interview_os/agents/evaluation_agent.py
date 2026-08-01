"""Evaluation Agent - aggregates evidence for evidence-based decisions."""
from __future__ import annotations

import logging

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewStage, InterviewState
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
        prompt = EVALUATION_PROMPT.format(evidence_list=evidence_text or "No evidence collected yet.")
        raw = await self.think(prompt, context=state.summary())
        state.current_stage = InterviewStage.COMPLETED
        return self.make_response(raw)
