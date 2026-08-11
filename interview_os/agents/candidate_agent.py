"""Candidate Intelligence Agent - analyzes resume to build candidate profile."""
from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import CandidateProfile, InterviewState
from interview_os.models.prompt_templates import CANDIDATE_ANALYSIS_PROMPT

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
        existing_name = state.candidate.name
        resume_text = instruction or state.candidate.raw_resume_text
        resume_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", resume_text)
        state.candidate.raw_resume_text = resume_text
        prompt = CANDIDATE_ANALYSIS_PROMPT.format(resume_text=resume_text)

        try:
            parsed = await self.think_structured(
                prompt, CandidateProfile, context=state.summary()
            )
            parsed.raw_resume_text = resume_text
            self._ground_profile(parsed, resume_text)
            if not parsed.name:
                parsed.name = existing_name
            state.candidate = parsed
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse candidate profile: %s", exc)
            if not state.candidate.name:
                state.candidate.name = self._explicit_name(resume_text)

        content = (
            f"Candidate: {state.candidate.name}\n"
            f"Strengths: {state.candidate.strengths}\n"
            f"Weaknesses: {state.candidate.weaknesses}\n"
            f"Unique Advantages: {state.candidate.unique_advantages}"
        )
        return self.make_response(content)

    @classmethod
    def _ground_profile(cls, profile: CandidateProfile, resume_text: str) -> None:
        """Keep identity and resume claims anchored to explicit document evidence."""
        explicit_name = cls._explicit_name(resume_text)
        if explicit_name:
            profile.name = explicit_name

        # A resume is not evidence of personal weaknesses. Gap analysis belongs in
        # the strategy agent where it can be compared with explicit JD requirements.
        profile.weaknesses = []

        compact_resume = cls._compact(resume_text)
        profile.achievements = [
            claim for claim in profile.achievements if cls._compact(claim) in compact_resume
        ]

    @staticmethod
    def _explicit_name(resume_text: str) -> str:
        name_match = re.search(
            r"\bName\s*[:：]\s*([A-Za-z][A-Za-z .'-]{0,40}?)(?=\s+(?:Led|Managed|Built|Partnered|Designed|WORK|SKILLS)\b)",
            resume_text,
            flags=re.IGNORECASE,
        )
        return name_match.group(1).strip() if name_match else ""

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]", "", value).lower()
