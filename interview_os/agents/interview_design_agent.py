"""Interview Design Agent - creates interview blueprint for enterprise side."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewBlueprint, InterviewState

logger = logging.getLogger(__name__)


class InterviewDesignAgent(Agent):
    """Designs interview blueprint: rounds, goals, questions, evaluation criteria."""

    def __init__(self, **kwargs):
        super().__init__(
            name="interview_design_agent",
            role="Interview Architect",
            goal="Design a structured interview blueprint based on candidate, job, and company analysis",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        confirmed_facts = state.candidate_evidence_context()
        candidate_section = (
            f"Confirmed candidate facts: {confirmed_facts}\n"
            if confirmed_facts.strip()
            else "Confirmed candidate facts: none provided. Design generic rounds from the "
            "job competencies; do not leave rounds empty.\n"
        )
        context = (
            f"Position: {state.job.title}\n"
            f"Job review with explicit/inferred labels: {state.job_review.model_dump_json()}\n"
            f"{candidate_section}"
            f"Company DNA: {state.company.dna}\n"
            f"Company preferences: {state.company.preferences}"
        )
        prompt = (
            "Design an interview blueprint and output JSON with position and rounds. "
            "Each round has name, goal, evaluation_criteria (list), and 3-5 questions. "
            "Each question has question, competency, rationale, strong_signals (list), "
            "and follow_ups (list). Map every question to a job competency. Use Chinese for "
            "round names, goals, questions, criteria, rationale, signals, and follow-ups. "
            "Produce at least one round with questions. If no candidate resume is available, "
            "design rounds from the posted requirements and competencies alone."
        )
        try:
            state.blueprint = await self.think_structured(
                prompt, InterviewBlueprint, context=context
            )
            state.blueprint.position = state.blueprint.position or state.job.title
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse interview blueprint: %s", exc)
        if not state.blueprint.rounds:
            # Local models can return a valid blueprint with empty rounds when no
            # candidate resume is available; fall back to a deterministic generic
            # blueprint so enterprise design never fails on that path.
            self.record_degradation(
                "Interview design returned no rounds; built a generic JD-based blueprint"
            )
            state.blueprint = self._generic_blueprint(state)
        state.next_action = "Execute interview blueprint"
        return self.make_response(state.blueprint.model_dump_json(indent=2))

    @staticmethod
    def _generic_blueprint(state: InterviewState) -> InterviewBlueprint:
        from interview_os.core.state import InterviewQuestion, InterviewRound

        competencies = [
            item.strip()
            for item in state.job.competencies
            if item.strip()
        ] or (["综合能力"] if state.job.title else [])
        rounds = []
        for competency in competencies:
            questions = [
                InterviewQuestion(
                    question=f"请讲一个最能体现你“{competency}”能力的真实项目，说明你的具体行动与可量化结果。",
                    competency=competency,
                    rationale=f"围绕 JD 要求的“{competency}”采集结构化行为证据",
                    strong_signals=["候选人本人采取的行动", "关键技术或业务权衡", "可量化结果"],
                    follow_ups=[
                        "当时对比过哪些替代方案？为什么最终选择这个？",
                        "过程中遇到的主要风险是什么，如何控制和回滚？",
                    ],
                )
            ]
            rounds.append(
                InterviewRound(
                    name=f"{competency}验证",
                    goal=f"验证候选人“{competency}”的证据强度与深度",
                    questions=questions,
                    evaluation_criteria=[
                        f"{competency}相关经历的真实性与细节",
                        "行为是否具体、结果是否可量化",
                        "能否讲清权衡、风险与复盘",
                    ],
                )
            )
        return InterviewBlueprint(position=state.job.title, rounds=rounds)
