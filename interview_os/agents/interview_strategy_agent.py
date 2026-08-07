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
        employer_block = (
            f"\n{state.past_employer_block}" if state.past_employer_block else ""
        )
        prompt = STRATEGY_FUSION_PROMPT.format(
            candidate_profile=(
                f"Candidate name: {state.candidate.name}\n"
                "Candidate evidence (confirmed claims + full resume):\n"
                f"{state.candidate_evidence_context(structure_required=True)[:12000]}"
                f"{employer_block}"
            ),
            job_requirements=(
                state.job.model_dump_json()
                + f"\nRequirement provenance: {state.job_review.model_dump_json()}"
            ),
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
        if not state.strategy.summary and not state.strategy.key_risks:
            # The three-way fusion output was empty or ungrounded; give the
            # candidate a deterministic fallback so preparation never hard-fails.
            self.record_degradation(
                "Strategy agent returned no usable strategy; built a generic preparation plan"
            )
            state.strategy = self._generic_strategy(state)
        return self.make_response(state.strategy.model_dump_json(indent=2))

    @staticmethod
    def _generic_strategy(state: InterviewState) -> InterviewStrategy:
        from interview_os.core.state import InterviewStrategy

        competencies = [item.strip() for item in state.job.competencies if item.strip()]
        topics = competencies or (["核心技术能力"] if state.job.title else [])
        return InterviewStrategy(
            summary=(
                f"围绕岗位“{state.job.title or '目标岗位'}”准备：结合简历中的真实经历，"
                "用具体行动和可量化结果支撑每一个能力维度。"
            ),
            key_risks=(
                ["简历信息未经确认，回答中避免引入简历之外无法核验的量化结果"]
                if not state.candidate.name
                else ["注意把经历落到岗位要求的能力维度上"]
            ),
            answer_framework=["先说结论", "再讲行动与权衡", "最后给可量化结果"],
            topics_to_emphasize=topics,
            topics_to_avoid=[],
            likely_questions=[
                f"请讲一个最能体现你“{topic}”能力的真实项目。" for topic in topics
            ],
        )

    @staticmethod
    def _is_title_only(raw_description: str) -> bool:
        text = raw_description.strip()
        return (
            bool(text)
            and len(text) <= 120
            and "\n" not in text
            and not any(
                marker in text
                for marker in ("：", ":", "职责", "要求", "responsibilities", "requirements")
            )
        )

    @staticmethod
    def _remove_unsupported_quantitative_claims(state: InterviewState) -> None:
        """Drop strategy statements containing metrics absent from supplied evidence."""
        evidence = "\n".join(
            (
                state.candidate_evidence_context(structure_required=True),
                state.job.raw_description,
                " ".join(str(item.get("snippet", "")) for item in state.company.public_sources),
                " ".join(
                    str(item.get("snippet", "")) for item in state.interviewer.public_expressions
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
