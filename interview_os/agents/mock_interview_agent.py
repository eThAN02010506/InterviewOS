"""Mock Interview Agent - generates personalized questions."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewState, MockInterviewPlan
from interview_os.models.prompt_templates import MOCK_QUESTION_PROMPT

logger = logging.getLogger(__name__)


class MockInterviewAgent(Agent):
    """Generates personalized interview questions.

    Question = Candidate Background + Job Requirement + Interviewer Preference
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="mock_interview_agent",
            role="Mock Interview Simulator",
            goal="Generate personalized interview questions based on candidate, job, and interviewer",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        prompt = MOCK_QUESTION_PROMPT.format(
            candidate_background=state.candidate.model_dump_json(),
            job_requirement=str(state.job.competencies),
            interviewer_preference=str(state.interviewer.likely_preferences),
        )
        try:
            state.mock_interview = await self.think_structured(
                prompt, MockInterviewPlan, context=state.summary()
            )
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse mock interview plan: %s", exc)
        return self.make_response(state.mock_interview.model_dump_json(indent=2))
