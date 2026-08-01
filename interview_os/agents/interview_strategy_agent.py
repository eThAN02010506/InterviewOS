"""Interview Strategy Agent - three-way fusion: Candidate + Job + Interviewer."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewState, InterviewStrategy
from interview_os.models.prompt_templates import STRATEGY_FUSION_PROMPT
from interview_os.models.structured import parse_model_output

logger = logging.getLogger(__name__)


class InterviewStrategyAgent(Agent):
    """Fuses candidate, job, and interviewer analysis into personalized strategy.

    This is the three-way fusion model: the core value proposition.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="interview_strategy_agent",
            role="Interview Strategy Advisor",
            goal="Fuse candidate, job, and interviewer insights into a personalized interview strategy",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        prompt = STRATEGY_FUSION_PROMPT.format(
            candidate_profile=str(state.candidate.strengths + state.candidate.weaknesses),
            job_requirements=str(state.job.competencies),
            interviewer_profile=str(state.interviewer.likely_preferences),
        )
        raw = await self.think(prompt, context=state.summary())
        try:
            state.strategy = parse_model_output(raw, InterviewStrategy)
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse interview strategy: %s", exc)
        return self.make_response(state.strategy.model_dump_json(indent=2))
