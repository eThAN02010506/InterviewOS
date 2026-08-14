"""Evidence-aware next-question planner for live interviews."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.evidence import EvidenceSource
from interview_os.core.message import Message, MessageType
from interview_os.core.state import (
    LIVE_RECENT_SEGMENT_WINDOW,
    PLANNER_MAX_TOKENS,
    PLANNER_QUESTION_MAP_MAX,
    PLANNER_SUMMARY_CHAR_LIMIT,
    InterviewState,
    QuestionSuggestion,
    QuestionSuggestionType,
    TranscriptSpeaker,
)

logger = logging.getLogger(__name__)


class LiveInterviewAgent(Agent):
    """Prepare one bounded, evidence-linked question for the interviewer."""

    def __init__(self, **kwargs):
        super().__init__(
            name="live_interview_agent",
            role="Real-time Interview Copilot",
            goal="Prepare the next evidence-seeking interview question without taking control",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        recent_segments = [
            item
            for item in state.live_interview.segments
            if item.sequence > state.live_interview.summarized_until_sequence
            and item.speaker != TranscriptSpeaker.UNKNOWN
        ][-LIVE_RECENT_SEGMENT_WINDOW:]
        transcript = "\n".join(
            f"[{segment.speaker.value}] {segment.text}"
            for segment in recent_segments
            if segment.stable and segment.confirmed
        )
        used_question_ids = set(state.live_interview.used_question_ids)
        last_source_id = ""
        for suggestion in reversed(state.live_interview.suggestions):
            if suggestion.source_question_id.strip():
                last_source_id = suggestion.source_question_id.strip()
                break
        blueprint_questions = [
            question
            for interview_round in state.blueprint.rounds
            for question in interview_round.questions
        ]
        mapped_questions = [
            question
            for question in blueprint_questions
            if str(question.id) not in used_question_ids
        ][:PLANNER_QUESTION_MAP_MAX]
        if last_source_id and not any(str(q.id) == last_source_id for q in mapped_questions):
            linked = next(
                (q for q in blueprint_questions if str(q.id) == last_source_id), None
            )
            if linked is not None:
                mapped_questions.append(linked)
        question_map = "\n".join(
            f"- id={question.id}; competency={question.competency}; question={question.question}; "
            f"signals={question.strong_signals}"
            for question in mapped_questions
        ) or "No prepared blueprint questions."
        question_usage = "\n".join(
            f"- status={item.status}; id={item.question_id}; round={item.round_name}; "
            f"competency={item.competency}; suggested={item.suggested_count}; "
            f"last={item.last_suggestion_status}; question={item.question}"
            for item in state.live_interview.question_usage[:12]
        ) or "No blueprint question usage yet."
        evidence_map = "\n".join(
            f"- {item.competency}: {item.signal}"
            for item in state.evidence[-20:]
            if item.source == EvidenceSource.LIVE_INTERVIEW
        ) or "No confirmed live evidence yet."
        coverage_guidance = "\n".join(
            f"- priority={item.priority}; competency={item.competency}; "
            f"evidence={item.evidence_count}; confidence={item.strongest_confidence:.2f}; "
            f"reason={item.reason}; suggested={item.sample_question}"
            for item in state.live_interview.coverage_guidance[:5]
        ) or "No coverage guidance available."
        rolling_summary = state.live_interview.rolling_summary
        if len(rolling_summary) > PLANNER_SUMMARY_CHAR_LIMIT:
            rolling_summary = "…\n" + rolling_summary[-PLANNER_SUMMARY_CHAR_LIMIT:]
        prompt = (
            "Return exactly one JSON object matching the QuestionSuggestion schema. "
            "Prepare a question for the interviewer; do not answer it and do not address the "
            "candidate directly outside suggested_question. Prefer one evidence-seeking follow-up "
            "when the latest candidate answer has a material gap. Otherwise choose an unused main "
            "question from the blueprint. If the answer is incomplete, use question_type=clarify. "
            "Never infer negative evidence from silence. Keep alternatives to at most two. "
            f"Additional instruction: {instruction or 'Prepare the next question.'}"
        )
        context = (
            f"Job competencies: {state.job.competencies}\n"
            f"Missing signals: {state.missing_signals}\n"
            f"Used question IDs: {state.live_interview.used_question_ids}\n"
            f"Prepared question map:\n{question_map}\n"
            f"Blueprint question usage:\n{question_usage}\n"
            f"Recorded evidence:\n{evidence_map}\n"
            f"Coverage guidance:\n{coverage_guidance}\n"
            f"Rolling transcript summary:\n"
            f"{rolling_summary or 'No older transcript summary yet.'}\n"
            f"Recent confirmed transcript:\n{transcript}"
        )
        try:
            suggestion = await self.think_structured(
                prompt, QuestionSuggestion, context=context, max_tokens=PLANNER_MAX_TOKENS
            )
            matched_competency = self._match_competency(suggestion.competency, state)
            if matched_competency is None:
                raise ValueError(
                    f"Suggestion competency '{suggestion.competency}' is not grounded"
                )
            suggestion.competency = matched_competency
            if not suggestion.rationale.strip() or not suggestion.evidence_gap.strip():
                raise ValueError("Suggestion omitted its rationale or evidence gap")
            if not [item for item in suggestion.expected_signals if item.strip()]:
                raise ValueError("Suggestion omitted expected evidence signals")
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Live question planning degraded: %s", exc)
            suggestion = self._fallback_suggestion(state)
            self.record_degradation(
                "Live question planner used a deterministic unused blueprint question"
            )
        state.live_interview.suggestions.append(suggestion)
        return Message(
            type=MessageType.SUGGESTION,
            sender=self.name,
            content=suggestion.model_dump_json(),
        )

    @staticmethod
    def _allowed_competencies(state: InterviewState) -> set[str]:
        values = {item.strip() for item in state.job.competencies if item.strip()}
        values.update(
            question.competency.strip()
            for interview_round in state.blueprint.rounds
            for question in interview_round.questions
            if question.competency.strip()
        )
        if not values:
            values.add("综合能力")
        return values

    @classmethod
    def _match_competency(cls, value: str, state: InterviewState) -> str | None:
        requested = "".join(value.casefold().split())
        for competency in cls._allowed_competencies(state):
            candidate = "".join(competency.casefold().split())
            if requested == candidate:
                return competency
            if len(requested) >= 2 and len(candidate) >= 2 and (
                requested in candidate or candidate in requested
            ):
                return competency
        return None

    @classmethod
    def _fallback_suggestion(cls, state: InterviewState) -> QuestionSuggestion:
        used = set(state.live_interview.used_question_ids)
        questions = [
            question
            for interview_round in state.blueprint.rounds
            for question in interview_round.questions
        ]
        selected_question = next((item for item in questions if str(item.id) not in used), None)
        if selected_question is not None:
            return QuestionSuggestion(
                suggested_question=selected_question.question,
                question_type=QuestionSuggestionType.NEXT_MAIN,
                competency=selected_question.competency or "综合能力",
                rationale=selected_question.rationale or "按结构化面试计划推进未覆盖能力",
                evidence_gap="该计划问题尚未形成确认后的回答证据",
                expected_signals=selected_question.strong_signals
                or ["候选人本人采取的行动", "关键技术或业务权衡", "可验证结果"],
                source_question_id=str(selected_question.id),
                confidence=0.6,
                alternatives=selected_question.follow_ups[:2],
            )
        latest_candidate = next(
            (
                segment
                for segment in reversed(state.live_interview.segments)
                if segment.speaker == TranscriptSpeaker.CANDIDATE
            ),
            None,
        )
        competency = state.job.competencies[0] if state.job.competencies else "综合能力"
        answer_text = latest_candidate.text if latest_candidate else ""
        has_measured_result = any(
            marker in answer_text
            for marker in ("%", "百分", "p95", "P95", "降低", "提升", "下降", "增长")
        )
        if latest_candidate and has_measured_result:
            question_text = (
                f"围绕{competency}，你当时还比较了哪些替代方案？"
                "请说明最终选择依据、主要风险和触发回滚的条件。"
            )
            rationale = "回答已有行动和量化结果，下一步验证方案权衡与风险控制深度"
            evidence_gap = "替代方案、选择依据、主要风险与回滚条件"
            expected_signals = ["被放弃方案", "选择标准", "风险监控", "回滚阈值"]
        else:
            question_text = (
                f"围绕{competency}，请补充你当时最关键的技术权衡、验证方法和可量化结果。"
                if latest_candidate
                else f"请介绍一个最能体现你{competency}能力的项目。"
            )
            rationale = "当前缺少可以支撑评价的完整候选人回答"
            evidence_gap = "具体行动、权衡与结果"
            expected_signals = ["候选人本人采取的行动", "技术或业务权衡", "可验证结果"]
        return QuestionSuggestion(
            suggested_question=question_text,
            question_type=(
                QuestionSuggestionType.CLARIFY
                if latest_candidate
                else QuestionSuggestionType.NEXT_MAIN
            ),
            competency=competency,
            rationale=rationale,
            evidence_gap=evidence_gap,
            expected_signals=expected_signals,
            confidence=0.45,
        )
