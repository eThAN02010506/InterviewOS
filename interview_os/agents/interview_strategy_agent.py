"""Interview Strategy Agent - three-way fusion: Candidate + Job + Interviewer."""
from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewState, InterviewStrategy
from interview_os.models.prompt_templates import STRATEGY_FUSION_PROMPT

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
            candidate_profile=(
                state.candidate.model_dump_json(exclude={"raw_resume_text"})
                + f"\nResume evidence excerpt: {state.candidate.raw_resume_text[:12000]}"
            ),
            job_requirements=state.job.model_dump_json(),
            interviewer_profile=state.interviewer.model_dump_json(),
        )
        try:
            state.strategy = await self.think_structured(
                prompt, InterviewStrategy, context=state.summary()
            )
            if self._is_title_only(state.job.raw_description):
                # A title supports preparation topics, but not a claim that the candidate
                # fails a requirement that was never explicitly provided.
                state.strategy.key_risks = []
            self._remove_unsupported_quantitative_claims(state)
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse interview strategy: %s", exc)
        return self.make_response(state.strategy.model_dump_json(indent=2))

    @staticmethod
    def _is_title_only(raw_description: str) -> bool:
        text = raw_description.strip()
        return bool(text) and len(text) <= 120 and "\n" not in text and not any(
            marker in text
            for marker in ("：", ":", "职责", "要求", "responsibilities", "requirements")
        )

    @staticmethod
    def _remove_unsupported_quantitative_claims(state: InterviewState) -> None:
        """Drop strategy statements containing metrics absent from supplied evidence."""
        evidence = "\n".join(
            (
                state.candidate.raw_resume_text,
                state.job.raw_description,
                " ".join(
                    str(item.get("snippet", "")) for item in state.company.public_sources
                ),
                " ".join(
                    str(item.get("snippet", ""))
                    for item in state.interviewer.public_expressions
                ),
            )
        )

        def grounded(statement: str) -> bool:
            material_numbers = re.findall(r"(?<!\d)(?:\d{2,}|\d+(?:\.\d+)?%)(?!\d)", statement)
            return all(number in evidence for number in material_numbers)

        for field in (
            "key_risks",
            "answer_framework",
            "topics_to_emphasize",
            "topics_to_avoid",
            "likely_questions",
        ):
            values = getattr(state.strategy, field)
            setattr(state.strategy, field, [value for value in values if grounded(value)])
        if not grounded(state.strategy.summary):
            state.strategy.summary = "请围绕简历中的真实经历和可核验结果准备回答。"
