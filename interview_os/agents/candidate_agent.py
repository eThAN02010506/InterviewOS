"""Candidate Intelligence Agent - analyzes resume to build candidate profile."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import CandidateProfile, InterviewState
from interview_os.models.prompt_templates import CANDIDATE_ANALYSIS_PROMPT
from interview_os.models.structured import parse_model_output

logger = logging.getLogger(__name__)


class CandidateAgent(Agent):
    """Analyzes candidate resume to generate a structured CandidateProfile."""

    def __init__(self, **kwargs):
        super().__init__(
            name="candidate_agent",
            role="Candidate Intelligence Analyst",
            goal="Understand candidate strengths, weaknesses, and unique advantages",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        resume_text = state.candidate.raw_resume_text or instruction
        state.candidate.raw_resume_text = resume_text
        prompt = CANDIDATE_ANALYSIS_PROMPT.format(resume_text=resume_text)
        raw = await self.think(prompt, context=state.summary())

        try:
            parsed = parse_model_output(raw, CandidateProfile)
            parsed.raw_resume_text = resume_text
            state.candidate = parsed
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse candidate profile: %s", exc)

        content = (
            f"Candidate: {state.candidate.name}\n"
            f"Strengths: {state.candidate.strengths}\n"
            f"Weaknesses: {state.candidate.weaknesses}\n"
            f"Unique Advantages: {state.candidate.unique_advantages}"
        )
        return self.make_response(content)
