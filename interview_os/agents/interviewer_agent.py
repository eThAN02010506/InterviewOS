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
from interview_os.tools.web_search import (
    filter_entity_results,
    format_search_results,
    merge_search_results,
)

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
        research_allowed = (
            not state.autopilot.enabled or state.autopilot.authorized_public_research
        )
        research_status = "not_requested"
        if research_allowed:
            search = await self.tools.call(
                "web_search",
                query=f'"{info.name or instruction}" {info.company} {info.position} talk interview blog',
                num_results=6,
            )
            first_results = search.data["results"] if search.success else []
            accepted_results = filter_entity_results(
                first_results,
                entity=info.name or instruction,
                required_context=info.company,
                allow_context_alias=True,
            )
            second_results: list[dict] = []
            if not accepted_results:
                localized = await self.tools.call(
                    "web_search",
                    query=(
                        f'"{info.name or instruction}" "{info.company}" '
                        f"{info.position} 采访 演讲 个人资料"
                    ),
                    num_results=10,
                    search_depth="advanced",
                )
                second_results = localized.data["results"] if localized.success else []
                research_status = "completed" if localized.success else "failed"
            else:
                research_status = "completed"
            existing_sources = filter_entity_results(
                merge_search_results(first_results, second_results),
                entity=info.name or instruction,
                required_context=info.company,
                allow_context_alias=True,
            )
            resolved_company = next(
                (
                    str(item.get("matched_identity", ""))
                    for item in existing_sources
                    if item.get("identity_match") == "corroborated_alias"
                ),
                "",
            )
            if resolved_company:
                canonical = await self.tools.call(
                    "web_search",
                    query=(
                        f'"{resolved_company}" "{info.name or instruction}" '
                        "官网 创始人"
                    ),
                    num_results=6,
                )
                canonical_results = filter_entity_results(
                    canonical.data["results"] if canonical.success else [],
                    entity=info.name or instruction,
                    required_context=resolved_company,
                )
                for item in canonical_results:
                    item.update(
                        identity_match="corroborated_alias",
                        input_identity=info.company,
                        matched_identity=resolved_company,
                    )
                existing_sources = merge_search_results(
                    existing_sources, canonical_results, limit=8
                )
            if research_status == "completed" and not existing_sources:
                research_status = "no_reliable_sources"
        research = format_search_results(existing_sources)
        prompt = INTERVIEWER_ANALYSIS_PROMPT.format(
            name=info.name or instruction,
            position=info.position,
            company=info.company,
            public_info=research or instruction or f"No public sources found for {identity}",
        )
        try:
            parsed = await self.think_structured(
                prompt, InterviewerProfile, context=state.summary()
            )
            # Search evidence can enrich a profile but cannot replace the selected identity.
            parsed.name = info.name
            parsed.position = info.position
            parsed.company = info.company
            # Provenance belongs to the search tool, never to model-generated JSON.
            parsed.public_expressions = existing_sources
            parsed.public_research_status = research_status
            state.interviewer = parsed
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse interviewer profile (%s)", type(exc).__name__)
            info.public_expressions = existing_sources
            info.public_research_status = research_status
            state.interviewer = info

        content = (
            f"Interviewer: {state.interviewer.name}\n"
            f"Career Pattern: {state.interviewer.career_pattern}\n"
            f"Communication Style: {state.interviewer.communication_style}\n"
            f"Likely Preferences: {state.interviewer.likely_preferences}"
        )
        return self.make_response(content)
