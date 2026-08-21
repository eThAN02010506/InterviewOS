"""Feedback Agent - generates structured feedback report."""

from __future__ import annotations

from interview_os.core.agent import Agent
from interview_os.core.evidence import EvidencePolarity
from interview_os.core.message import Message
from interview_os.core.state import FeedbackReport, InterviewState


class FeedbackAgent(Agent):
    """Generates structured feedback for both enterprise and candidate sides."""

    def __init__(self, **kwargs):
        super().__init__(
            name="feedback_agent",
            role="Feedback Report Generator",
            goal="Generate actionable, evidence-based feedback reports",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        state.feedback = FeedbackReport()
        state.enforce_evaluation_evidence_floor()
        self._enforce_candidate_voice(state)
        return self.make_response(state.feedback.model_dump_json())

    @staticmethod
    def _enforce_candidate_voice(state: InterviewState) -> None:
        """Build both report views only from persisted evidence and evaluation gaps."""
        report = state.feedback
        competencies = state.evaluation.competencies
        average = (
            sum(item.score for item in competencies) / len(competencies)
            if competencies
            else state.evaluation.overall_score
        )
        evidence_prefix = (
            "当前证据不足；"
            if state.evaluation.recommendation.value == "insufficient_evidence"
            else ""
        )
        report.overall = (
            f"{evidence_prefix}本次练习覆盖 {len(competencies)} 个能力维度，"
            f"当前证据平均得分约 {average:.2f}。"
            "该结果仅用于面试准备，请优先补强下列证据缺口。"
        )
        report.strengths = list(
            dict.fromkeys(
                item.signal
                for item in state.evidence
                if item.signal
                and item.confidence >= 0.7
                and item.polarity != EvidencePolarity.NEGATIVE
            )
        )[:5]
        gaps = list(
            dict.fromkeys(
                [*state.missing_signals]
                + [gap for item in competencies for gap in item.gaps]
            )
        )[:5]
        candidate_gaps = [
            (
                "再准备一个不同业务场景的真实案例，并讲清数据来源、个人决策和已核验结果"
                if gap == "需要更多独立回答交叉验证"
                else gap
            )
            for gap in gaps
        ]
        report.improvements = candidate_gaps
        report.action_plan = (
            [f"准备并练习：{gap}" for gap in candidate_gaps]
            if candidate_gaps
            else ["准备并练习：可交叉验证的具体案例与真实结果"]
        )
        negative_notes = [
            f"负面证据：{item.signal}（Evidence {item.id}）"
            for item in state.evidence
            if item.signal and item.polarity == EvidencePolarity.NEGATIVE
        ]
        pending_notes = [f"仍待核验：{gap}" for gap in gaps]
        evidence_competencies = {
            item.competency for item in state.evidence if item.competency.strip()
        }
        threshold_met = len(state.evidence) >= 3 and len(evidence_competencies) >= 2
        evidence_summary = (
            f"当前共有 {len(state.evidence)} 条可追溯证据"
            if threshold_met
            else (
                f"当前仅有 {len(state.evidence)} 条证据，覆盖 "
                f"{len(evidence_competencies)} 个胜任力；未达到招聘决策门槛。"
            )
        )
        report.interviewer_notes = [
            evidence_summary,
            *negative_notes[:5],
            *pending_notes[:5],
        ]
        if not threshold_met:
            report.interviewer_notes.insert(1, "证据不足不是负面证据，需要继续采集独立回答")
        recommendation = state.evaluation.recommendation
        if recommendation.value == "insufficient_evidence":
            report.recommendation_reasoning = (
                "当前证据包含尚未人工复核的规则评分，或未达到招聘决策门槛，"
                "因此不能给出录用或不录用建议；缺失信号不是负面证据。"
            )
        else:
            labels = {
                "strong_hire": "强烈建议录用",
                "hire": "建议录用",
                "lean_hire": "倾向录用",
                "lean_no_hire": "倾向不录用",
                "no_hire": "不建议录用",
            }
            report.recommendation_reasoning = (
                f"当前校准建议为“{labels.get(recommendation.value, recommendation.value)}”，"
                f"总分 {state.evaluation.overall_score:.2f}；该结论依据持久化胜任力分数、"
                "证据置信度和固定阈值生成。仍缺信号需单独核验，不能当作负面证据。"
            )
