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
        self._build_grounded_improvement(evaluation, coach_input.answer)

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
    def _build_grounded_improvement(evaluation: AnswerEvaluation, answer: str) -> None:
        """Build a useful coaching scaffold without model-generated claims.

        Local models can fabricate employers, meetings, tools, or outcomes even
        without adding numbers. The model therefore scores and identifies gaps,
        while this deterministic layer preserves the original answer verbatim and
        provides only explicit placeholders for facts the candidate should add.
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
        evaluation.improved_answer = (
            "已提供事实（原文保留）：\n"
            + answer.strip()
            + "\n\nSTAR 补充框架（请只填写真实、可核验的信息）：\n"
            "- 情境：[补充业务背景与目标]\n"
            "- 任务：[补充你的具体职责与约束]\n"
            "- 行动：[按步骤重组上面的真实行动]\n"
            "- 结果：[补充已核验的结果；没有数据时明确说明]"
        )
        if unsupported:
            warning = "模型草稿引入了原回答未提供的数字，已丢弃；请从 ATS 或原始材料核对。"
            if warning not in evaluation.feedback:
                evaluation.feedback.append(warning)
            missing = "需要核验并补充真实量化结果"
            if missing not in evaluation.missing_signals:
                evaluation.missing_signals.append(missing)
