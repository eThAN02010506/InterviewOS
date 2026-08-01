"""Company Intelligence Agent - analyzes company DNA."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import CompanyInfo, InterviewState
from interview_os.models.prompt_templates import COMPANY_ANALYSIS_PROMPT
from interview_os.tools.web_search import (
    filter_entity_results,
    format_search_results,
    merge_search_results,
)

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
        research_status = "not_requested"
        if research_allowed:
            search = await self.tools.call(
                "web_search",
                query=(
                    f'"{company_name}" official product engineering technology '
                    "company culture hiring news"
                ),
                num_results=6,
            )
            first_results = search.data["results"] if search.success else []
            accepted_results = filter_entity_results(
                first_results,
                entity=company_name,
                corroborating_entity=state.interviewer.name,
            )
            second_results: list[dict] = []
            if not accepted_results:
                localized = await self.tools.call(
                    "web_search",
                    query=(
                        f'"{company_name}" {state.interviewer.name} '
                        "公司 官网 创始人 产品 招聘"
                    ),
                    num_results=6,
                )
                second_results = localized.data["results"] if localized.success else []
                research_status = "completed" if localized.success else "failed"
            else:
                research_status = "completed"
            existing_sources = filter_entity_results(
                merge_search_results(first_results, second_results),
                entity=company_name,
                corroborating_entity=state.interviewer.name,
            )
            resolved_alias = next(
                (
                    str(item.get("matched_identity", ""))
                    for item in existing_sources
                    if item.get("identity_match") == "corroborated_alias"
                ),
                "",
            )
            if resolved_alias:
                canonical = await self.tools.call(
                    "web_search",
                    query=f'"{resolved_alias}" "{state.interviewer.name}" 官网 公司 创始人 产品',
                    num_results=10,
                    search_depth="advanced",
                )
                canonical_results = filter_entity_results(
                    canonical.data["results"] if canonical.success else [],
                    entity=resolved_alias,
                )
                for item in canonical_results:
                    item.update(
                        identity_match="corroborated_alias",
                        input_identity=company_name,
                        matched_identity=resolved_alias,
                    )
                existing_sources = merge_search_results(
                    existing_sources, canonical_results, limit=8
                )
            if research_status == "completed" and not existing_sources:
                research_status = "no_reliable_sources"
        research = format_search_results(existing_sources) if existing_sources else instruction
        prompt = COMPANY_ANALYSIS_PROMPT.format(
            company_name=company_name,
            company_info=research or state.company.dna,
        )
        try:
            parsed = await self.think_structured(
                prompt, CompanyInfo, context=state.summary()
            )
            # Entity identity is authoritative user input; the model may only enrich it.
            parsed.name = company_name
            # Provenance belongs to the search tool, never to model-generated JSON.
            parsed.public_sources = existing_sources
            parsed.public_research_status = research_status
            state.company = parsed
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse company info: %s", exc)
            state.company.public_sources = existing_sources
            state.company.public_research_status = research_status

        content = (
            f"Company: {state.company.name}\n"
            f"DNA: {state.company.dna}\n"
            f"Preferences: {state.company.preferences}"
        )
        return self.make_response(content)
