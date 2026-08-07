"""Mock Interview Agent - generates personalized questions."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewQuestion, InterviewState, MockInterviewPlan
from interview_os.models.prompt_templates import MOCK_QUESTION_PROMPT

logger = logging.getLogger(__name__)


class MockInterviewAgent(Agent):
    """Generates personalized interview questions.

    Question = Candidate Background + Job Requirement + Interviewer Preference
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="mock_interview_agent",
            role="Mock Interview Simulator",
            goal="Generate personalized interview questions based on candidate, job, and interviewer",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        employer_block = (
            f"\n{state.past_employer_block}" if state.past_employer_block else ""
        )
        prompt = MOCK_QUESTION_PROMPT.format(
            candidate_background=(
                state.candidate_evidence_context(structure_required=True) + employer_block
            ),
            job_requirement=state.job_review.model_dump_json(),
            interviewer_preference=str(state.interviewer.likely_preferences),
        )
        try:
            state.mock_interview = await self.think_structured(
                prompt, MockInterviewPlan, context=state.summary()
            )
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse mock interview plan: %s", exc)
        if not state.mock_interview.questions:
            competencies = state.job.competencies or ["岗位核心能力"]
            state.mock_interview = MockInterviewPlan(
                questions=[
                    InterviewQuestion(
                        question=f"请结合一段真实经历，说明你如何运用{competency}解决问题。",
                        competency=competency,
                        rationale="结构化输出失败后的可审计降级问题，用于采集当前岗位所需证据。",
                        strong_signals=["具体情境", "个人行动", "量化结果", "复盘与取舍"],
                        follow_ups=["你个人具体负责什么？", "结果如何衡量？"],
                    )
                    for competency in competencies[:5]
                ]
            )
        return self.make_response(state.mock_interview.model_dump_json(indent=2))
