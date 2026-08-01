"""Company Intelligence Agent - analyzes company DNA."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import CompanyInfo, InterviewState
from interview_os.models.prompt_templates import COMPANY_ANALYSIS_PROMPT
from interview_os.models.structured import parse_model_output
from interview_os.tools.web_search import format_search_results

logger = logging.getLogger(__name__)


class CompanyAgent(Agent):
    """Analyzes company to generate CompanyInfo (Company DNA)."""

    def __init__(self, **kwargs):
        super().__init__(
            name="company_agent",
            role="Company Intelligence Analyst",
            goal="Understand company technology, culture, and hiring preferences",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        company_name = state.company.name or instruction or "Unknown"
        existing_sources = list(state.company.public_sources)
        research_allowed = (
            not state.autopilot.enabled or state.autopilot.authorized_public_research
        )
        if research_allowed:
            search = await self.tools.call(
                "web_search",
                query=(
                    f'"{company_name}" official product engineering technology '
                    "company culture hiring news"
                ),
                num_results=6,
            )
            if search.success:
                existing_sources = search.data["results"]
        research = format_search_results(existing_sources) if existing_sources else instruction
        prompt = COMPANY_ANALYSIS_PROMPT.format(
            company_name=company_name,
            company_info=research or state.company.dna,
        )
        raw = await self.think(prompt, context=state.summary())

        try:
            parsed = parse_model_output(raw, CompanyInfo)
            parsed.name = parsed.name or company_name
            parsed.public_sources = parsed.public_sources or existing_sources
            state.company = parsed
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse company info: %s", exc)

        content = (
            f"Company: {state.company.name}\n"
            f"DNA: {state.company.dna}\n"
            f"Preferences: {state.company.preferences}"
        )
        return self.make_response(content)
