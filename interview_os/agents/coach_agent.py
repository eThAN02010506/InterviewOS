"""Answer Coach Agent - analyzes and improves interview answers."""
from __future__ import annotations

import logging
import re
from uuid import UUID

from pydantic import BaseModel, ValidationError

from interview_os.core.agent import Agent
from interview_os.core.answer_feedback import apply_specific_feedback
from interview_os.core.evidence import Evidence, EvidencePolarity, EvidenceSource
from interview_os.core.message import Message
from interview_os.core.state import (
    AnswerEvaluation,
    AnswerEvaluationDraft,
    AnswerReviewStatus,
    AnswerScoringSource,
    InterviewState,
)
from interview_os.models.prompt_templates import ANSWER_COACH_PROMPT

logger = logging.getLogger(__name__)


class CoachInput(BaseModel):
    question: str
    answer: str
    competency: str = "Answer Quality"
    evidence_source: EvidenceSource = EvidenceSource.MOCK_INTERVIEW
    record_id: str | None = None
    persist_evidence: bool = True


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
            draft = await self.think_structured(
                prompt, AnswerEvaluationDraft, context=state.summary()
            )
            evaluation = AnswerEvaluation(
                **draft.model_dump(),
                scoring_source=AnswerScoringSource.MODEL,
                review_status=AnswerReviewStatus.NOT_REQUIRED,
            )
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse answer evaluation: %s", exc)
            self.record_degradation("Invalid structured answer score; deterministic rubric used")
            evaluation = self._deterministic_evaluation(coach_input.answer)
        self._build_grounded_improvement(evaluation, coach_input.answer)
        apply_specific_feedback(
            evaluation,
            coach_input.answer,
            question=coach_input.question,
            competency=coach_input.competency,
        )

        # Background live scoring must remain a pure calculation until the
        # service reacquires its session lock. Otherwise a slow model response
        # could mutate evidence after a human review has already won the race.
        if not coach_input.persist_evidence:
            return self.make_response(evaluation.model_dump_json())

        ev = Evidence(
            competency=coach_input.competency,
            # Model-generated observed_signals remain coaching hints. Persist the
            # candidate's own words as the auditable evidence signal instead of
            # allowing an untrusted model to introduce a new achievement claim.
            signal=coach_input.answer[:200],
            confidence=evaluation.overall_score(),
            source=coach_input.evidence_source,
            # Only a human review may classify an answer as positive/negative.
            polarity=EvidencePolarity.NEUTRAL,
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

    @staticmethod
    def _deterministic_evaluation(answer: str) -> AnswerEvaluation:
        """Score answer features transparently when model JSON is unusable."""
        clean = re.sub(r"\s+", " ", answer).strip()
        action_markers = re.findall(
            r"我(?:先|再|通过|建立|制定|设计|推动|负责|主导)|"
            r"(?:分析|监控|复盘|调整|协作|解决|实施|优化)",
            clean,
            re.IGNORECASE,
        )
        structure_markers = re.findall(
            r"(?:首先|其次|然后|最后|最终|背景|目标|挑战|行动|结果|因为|因此|但是)",
            clean,
            re.IGNORECASE,
        )
        impact_markers = re.findall(
            r"(?:最终|结果|支持|获得|认可|交付|改善|提升|降低|缩短|增长|影响)",
            clean,
            re.IGNORECASE,
        )
        data_markers = re.findall(
            r"(?:数据|指标|转化率|接受率|周期|漏斗|分析|ATS)",
            clean,
            re.IGNORECASE,
        )
        verified_metric = bool(
            re.search(r"\d+(?:[.,]\d+)?\s*(?:%|％|人|天|周|月|年|倍)", clean)
            and not re.search(r"(?:没有|缺少|未提供|待核对|不会编造).{0,12}(?:数字|数据|百分比|指标)", clean)
        )

        content = min(0.8, 0.4 + len(clean) / 1200)
        depth = min(0.75, 0.4 + min(len(action_markers), 4) * 0.08)
        structure = min(0.75, 0.4 + min(len(structure_markers), 4) * 0.08)
        impact = min(0.75, 0.4 + min(len(impact_markers), 3) * 0.08)

        observed = []
        if action_markers:
            observed.append("回答描述了候选人的具体行动")
        if data_markers:
            observed.append("回答提到了数据或招聘指标的使用")
        if impact_markers:
            observed.append("回答说明了结果或业务影响")

        missing = []
        if not verified_metric:
            missing.append("缺少已核验的量化结果")
        if len(structure_markers) < 2:
            missing.append("可以更明确地区分背景、行动与结果")
        if len(action_markers) < 2:
            missing.append("需要补充关键决策或执行步骤")

        feedback = ["本次模型结构化评分无效，以下为可解释规则评分，建议人工复核。"]
        if not verified_metric:
            feedback.append("请从 ATS 或原始材料核对真实指标后再补充，不要估算数字。")
        if len(structure_markers) < 2:
            feedback.append("建议按 STAR 顺序明确呈现背景、任务、行动和结果。")

        return AnswerEvaluation(
            content=round(content, 2),
            technical_depth=round(depth, 2),
            structure=round(structure, 2),
            impact=round(impact, 2),
            feedback=feedback[:3],
            observed_signals=observed[:3],
            missing_signals=missing[:3],
            scoring_source=AnswerScoringSource.DETERMINISTIC_RULE,
            review_status=AnswerReviewStatus.PENDING,
        )
