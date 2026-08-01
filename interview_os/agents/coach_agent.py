"""Answer Coach Agent - analyzes and improves interview answers."""
from __future__ import annotations

import logging

from pydantic import BaseModel, ValidationError

from interview_os.core.agent import Agent
from interview_os.core.evidence import Evidence, EvidenceSource
from interview_os.core.message import Message
from interview_os.core.state import AnswerEvaluation, InterviewState
from interview_os.models.prompt_templates import ANSWER_COACH_PROMPT

logger = logging.getLogger(__name__)


class CoachInput(BaseModel):
    question: str
    answer: str
    competency: str = "Answer Quality"
    evidence_source: EvidenceSource = EvidenceSource.MOCK_INTERVIEW


class CoachAgent(Agent):
    """Coaches candidates on answer quality across 4 dimensions.

    Dimensions: Content, Technical Depth, Structure, Impact.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="coach_agent",
            role="Answer Coach",
            goal="Help candidates improve answer quality across content, depth, structure, and impact",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        try:
            coach_input = CoachInput.model_validate_json(instruction)
        except ValidationError:
            parts = instruction.split("---ANSWER---", 1)
            coach_input = CoachInput(
                question=parts[0].strip() if len(parts) > 1 else "",
                answer=parts[1].strip() if len(parts) > 1 else instruction,
            )

        prompt = ANSWER_COACH_PROMPT.format(
            question=coach_input.question,
            answer=coach_input.answer,
            competency=coach_input.competency,
        )
        try:
            evaluation = await self.think_structured(
                prompt, AnswerEvaluation, context=state.summary()
            )
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse answer evaluation: %s", exc)
            evaluation = AnswerEvaluation(
                content=0.4,
                technical_depth=0.4,
                structure=0.4,
                impact=0.4,
                feedback=["自动评分输出无效，本回答需要人工复核。"],
                improved_answer=coach_input.answer,
                observed_signals=[],
                missing_signals=["AI structured scoring failed; human review required"],
            )

        ev = Evidence(
            competency=coach_input.competency,
            signal="; ".join(evaluation.observed_signals) or coach_input.answer[:200],
            confidence=evaluation.overall_score(),
            source=coach_input.evidence_source,
            notes="; ".join(evaluation.missing_signals),
        )
        state.add_evidence(ev)
        return self.make_response(evaluation.model_dump_json())
