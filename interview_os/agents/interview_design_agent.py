"""Interview Design Agent - creates interview blueprint for enterprise side."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewBlueprint, InterviewState

logger = logging.getLogger(__name__)


class InterviewDesignAgent(Agent):
    """Designs interview blueprint: rounds, goals, questions, evaluation criteria."""

    def __init__(self, **kwargs):
        super().__init__(
            name="interview_design_agent",
            role="Interview Architect",
            goal="Design a structured interview blueprint based on candidate, job, and company analysis",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        context = (
            f"Position: {state.job.title}\n"
            f"Job review with explicit/inferred labels: {state.job_review.model_dump_json()}\n"
            f"Confirmed candidate facts: {state.candidate_evidence_context()}\n"
            f"Company DNA: {state.company.dna}\n"
            f"Company preferences: {state.company.preferences}"
        )
        prompt = (
            "Design an interview blueprint and output JSON with position and rounds. "
            "Each round has name, goal, evaluation_criteria (list), and 3-5 questions. "
            "Each question has question, competency, rationale, strong_signals (list), "
            "and follow_ups (list). Map every question to a job competency. Use Chinese for "
            "round names, goals, questions, criteria, rationale, signals, and follow-ups."
        )
        try:
            state.blueprint = await self.think_structured(
                prompt, InterviewBlueprint, context=context
            )
            state.blueprint.position = state.blueprint.position or state.job.title
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse interview blueprint: %s", exc)
        state.next_action = "Execute interview blueprint"
        return self.make_response(state.blueprint.model_dump_json(indent=2))
