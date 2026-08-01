"""Job Intelligence Agent - analyzes JD to extract competencies."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewState, JobDescription
from interview_os.models.prompt_templates import JOB_ANALYSIS_PROMPT
from interview_os.models.structured import parse_model_output

logger = logging.getLogger(__name__)


class JobAgent(Agent):
    """Analyzes job description to generate structured JobDescription."""

    def __init__(self, **kwargs):
        super().__init__(
            name="job_agent",
            role="Job Intelligence Analyst",
            goal="Extract required competencies and evaluation dimensions from JD",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        jd_text = instruction or state.job.title
        prompt = JOB_ANALYSIS_PROMPT.format(jd_text=jd_text)
        raw = await self.think(prompt, context=state.summary())

        try:
            state.job = parse_model_output(raw, JobDescription)
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse job description: %s", exc)

        content = (
            f"Job: {state.job.title}\n"
            f"Required Skills: {state.job.required_skills}\n"
            f"Competencies: {state.job.competencies}"
        )
        return self.make_response(content)
