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
from interview_os.core.spoken_answer import (
    analyze_spoken_answer,
    calibrate_evaluation,
    grounded_missing_signals,
)
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
    answer_modality: str = "typed"


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
        analysis = analyze_spoken_answer(
            coach_input.question,
            coach_input.answer,
            answer_modality=coach_input.answer_modality,
        )
        calibrate_evaluation(evaluation, analysis)
        # Persist only gaps supported by the same coverage contract used to
        # calibrate scores. Model-only gaps remain too easy to contradict with
        # evidence present in the answer.
        evaluation.missing_signals = grounded_missing_signals(analysis)
        self._build_grounded_improvement(
            evaluation, coach_input.answer, question=coach_input.question
        )
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
    def _build_grounded_improvement(
        evaluation: AnswerEvaluation, answer: str, *, question: str = ""
    ) -> None:
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
        analysis = evaluation.spoken_analysis
        coverage = {item.requirement: item for item in analysis.question_coverage}

        def evidence_for(*names: str) -> str:
            for name in names:
                item = coverage.get(name)
                if item and item.evidence:
                    return item.evidence.strip()
            return ""

        context = evidence_for("明确具体公司/业务场景", "提供一个真实案例")
        personal = evidence_for("明确个人职责与关键决策")
        outcome = evidence_for("给出结果与验证方式")
        outcome_item = coverage.get("给出结果与验证方式")
        if outcome and outcome_item and outcome_item.status == "partial":
            outcome += "；[补充真实结果、验证方式、指标口径和时间范围]"
        actions = []
        for step in analysis.semantic_steps:
            if step.evidence and step.evidence not in actions:
                actions.append(step.evidence.strip())
        action_text = "；随后，".join(actions[:4])
        requirement_names = "、".join(item.requirement for item in analysis.question_coverage)
        contract_complete = bool(analysis.question_coverage) and all(
            item.status == "covered" for item in analysis.question_coverage
        )
        metric_follow_up = "说明指标、口径与决策关系" in coverage and set(coverage) <= {
            "直接回应题目核心",
            "说明指标、口径与决策关系",
        }
        if contract_complete:
            organization = (
                "建议保持“指标 → 统计口径 → 阈值 → 触发动作”的顺序，不必补讲一套新的 STAR 案例。"
                if metric_follow_up
                else "建议按“背景与目标 → 个人决定 → 执行动作 → 实际结果 → 复盘”分句表达。"
            )
            evaluation.improved_answer = (
                "你的原回答（作为唯一事实来源）：\n"
                + answer.strip()
                + "\n\n基于你本次回答的可直接使用版本（未添加新事实）：\n"
                + analysis.cleaned_transcript
                + "\n\n表达优化："
                + organization
            )
        else:
            evaluation.improved_answer = (
                "你的原回答（作为唯一事实来源）：\n"
                + answer.strip()
                + "\n\n基于你本次回答的重组示范（未添加新事实）：\n"
                f"针对“{question[:100] or '本题'}”，我的核心做法是"
                f"{personal or '[补充你本人承担的职责和关键决定]'}。\n"
                f"当时的具体背景是：{context or '[补充公司/项目、业务阶段、目标和约束]'}。\n"
                f"我采取的关键行动是：{action_text or '[按先后顺序补充二至四个本人动作及判断依据]'}。\n"
                f"最终结果是：{outcome or '[补充真实结果、验证方式、指标口径和时间范围]'}。\n"
                "如果重新处理，我会：[补充一项真实复盘或下一次会改变的做法]。\n"
                f"本题需要完整回应：{requirement_names or '问题中的核心要求'}。"
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

        feedback = [
            "评分模型经自动纠错后仍未返回有效 JSON；本题暂用可解释规则评分，建议人工复核。"
        ]
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
