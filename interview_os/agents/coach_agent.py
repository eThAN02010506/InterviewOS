"""Answer Coach Agent - analyzes and improves interview answers."""
from __future__ import annotations

import logging

from pydantic import BaseModel, ValidationError

from interview_os.core.agent import Agent
from interview_os.core.evidence import Evidence, EvidenceSource
from interview_os.core.message import Message
from interview_os.core.state import AnswerEvaluation, InterviewState
from interview_os.models.prompt_templates import ANSWER_COACH_PROMPT
from interview_os.models.structured import parse_model_output

logger = logging.getLogger(__name__)


class CoachInput(BaseModel):
    question: str
    answer: str
    competency: str = "Answer Quality"


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
        raw = await self.think(prompt, context=state.summary())

        try:
            evaluation = parse_model_output(raw, AnswerEvaluation)
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse answer evaluation: %s", exc)
            return self.make_response("{}")

        ev = Evidence(
            competency=coach_input.competency,
            signal="; ".join(evaluation.observed_signals) or coach_input.answer[:200],
            confidence=evaluation.overall_score(),
            source=EvidenceSource.MOCK_INTERVIEW,
            notes="; ".join(evaluation.missing_signals),
        )
        state.add_evidence(ev)
        return self.make_response(evaluation.model_dump_json())
