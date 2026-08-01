"""Interviewer Intelligence Agent - THE core innovation.

Analyzes interviewer background, career pattern, communication style,
and likely preferences to help candidates prepare strategically.
"""
from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewerProfile, InterviewState
from interview_os.models.prompt_templates import INTERVIEWER_ANALYSIS_PROMPT
from interview_os.models.structured import parse_model_output
from interview_os.tools.web_search import format_search_results

logger = logging.getLogger(__name__)


class InterviewerAgent(Agent):
    """Analyzes interviewer to generate InterviewerProfile.

    This is the key differentiator: understanding WHO will interview you,
    not just WHAT to prepare.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="interviewer_agent",
            role="Interviewer Intelligence Analyst",
            goal="Understand interviewer thinking style, preferences, and evaluation criteria",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        info = state.interviewer
        existing_sources = list(info.public_expressions)
        identity = " ".join(part for part in (info.name or instruction, info.position, info.company) if part)
        search = await self.tools.call(
            "web_search",
            query=f'"{info.name or instruction}" {info.company} {info.position} talk interview blog',
            num_results=6,
        )
        if search.success:
            existing_sources = search.data["results"]
        research = format_search_results(existing_sources)
        prompt = INTERVIEWER_ANALYSIS_PROMPT.format(
            name=info.name or instruction,
            position=info.position,
            company=info.company,
            public_info=research or instruction or f"No public sources found for {identity}",
        )
        raw = await self.think(prompt, context=state.summary())

        try:
            parsed = parse_model_output(raw, InterviewerProfile)
            parsed.name = parsed.name or info.name
            parsed.position = parsed.position or info.position
            parsed.company = parsed.company or info.company
            parsed.public_expressions = parsed.public_expressions or existing_sources
            state.interviewer = parsed
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse interviewer profile: %s", exc)

        content = (
            f"Interviewer: {state.interviewer.name}\n"
            f"Career Pattern: {state.interviewer.career_pattern}\n"
            f"Communication Style: {state.interviewer.communication_style}\n"
            f"Likely Preferences: {state.interviewer.likely_preferences}"
        )
        return self.make_response(content)
