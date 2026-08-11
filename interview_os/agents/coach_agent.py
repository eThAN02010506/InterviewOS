"""Answer Coach Agent - analyzes and improves interview answers."""
from __future__ import annotations

import logging
import re
from uuid import UUID

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
    record_id: str | None = None


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
        self._ground_improved_answer(evaluation, coach_input.answer)

        ev = Evidence(
            competency=coach_input.competency,
            signal="; ".join(evaluation.observed_signals) or coach_input.answer[:200],
            confidence=evaluation.overall_score(),
            source=coach_input.evidence_source,
            notes="; ".join(evaluation.missing_signals),
        )
        if coach_input.record_id is not None:
            # Live confirmation already created a placeholder evidence row for
            # this record; backfill it instead of appending a duplicate so the
            # evidence chain stays consistent between confirm and scoring.
            placeholder = next(
                (
                    item
                    for item in state.evidence
                    if item.source == EvidenceSource.LIVE_INTERVIEW
                    and item.source_record_id is not None
                    and str(item.source_record_id) == coach_input.record_id
                ),
                None,
            )
            if placeholder is not None:
                placeholder.competency = ev.competency
                placeholder.signal = ev.signal
                placeholder.confidence = ev.confidence
                placeholder.notes = ev.notes
                return self.make_response(evaluation.model_dump_json())
            # Mock answers carry their record id so a retry can replace this
            # evidence row instead of duplicating it.
            try:
                ev.source_record_id = UUID(coach_input.record_id)
            except (ValueError, TypeError):
                pass
        state.add_evidence(ev)
        return self.make_response(evaluation.model_dump_json())

    @staticmethod
    def _ground_improved_answer(evaluation: AnswerEvaluation, answer: str) -> None:
        """Reject model rewrites that introduce unsupported numeric facts.

        A coaching rewrite may improve structure and wording, but it must never
        fabricate dates, percentages, headcount, money, or performance metrics.
        Numeric claims are deterministic to audit and cover the highest-risk
        hallucinations observed with local models.
        """

        def numeric_facts(value: str) -> set[str]:
            return {
                token.replace("，", ",")
                for token in re.findall(
                    r"(?<![A-Za-z0-9_])\d+(?:[.,，]\d+)?\s*(?:%|％)?(?![A-Za-z0-9_])",
                    value,
                )
            }

        unsupported = numeric_facts(evaluation.improved_answer) - numeric_facts(answer)
        if not unsupported:
            return
        evaluation.improved_answer = (
            "基于已提供事实的版本（未新增未经核验的数据）：\n\n" + answer.strip()
        )
        warning = "模型优化稿引入了原回答未提供的数字，已移除；请从 ATS 或原始材料核对后补充。"
        if warning not in evaluation.feedback:
            evaluation.feedback.append(warning)
        missing = "需要核验并补充真实量化结果"
        if missing not in evaluation.missing_signals:
            evaluation.missing_signals.append(missing)
