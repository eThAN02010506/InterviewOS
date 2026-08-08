"""Mock Interview Agent - generates personalized questions."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import (
    MOCK_POOL_TARGET,
    MOCK_REFILL_BATCH,
    InterviewQuestion,
    InterviewState,
    MockInterviewPlan,
)
from interview_os.models.prompt_templates import (
    MOCK_FRAMEWORK_PROMPT,
    MOCK_QUESTION_PROMPT,
    MOCK_REFILL_PROMPT,
)
from interview_os.models.structured import FrameworkMap

logger = logging.getLogger(__name__)

_COMPETENCY_TEMPLATES = (
    "请结合一段真实经历，说明你如何运用{competency}解决问题。",
    "讲一个你在{competency}上最关键的决策，以及你如何衡量它是否成功。",
    "分享一次你通过{competency}推动结果、并面对取舍的经历。",
    "请用一个具体项目说明你在{competency}方面的做法与复盘。",
)


class MockInterviewAgent(Agent):
    """Generates personalized interview questions.

    Question = Candidate Background + Job Requirement + Interviewer Preference.
    The question pool is built from the strategy's likely_questions plus job
    competencies, then kept topped-up in the background during the interview.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="mock_interview_agent",
            role="Mock Interview Simulator",
            goal="Generate personalized interview questions based on candidate, job, and interviewer",
            **kwargs,
        )

    @staticmethod
    def deterministic_framework(state: InterviewState, competency: str) -> str:
        """Build a per-question reference-answer hint from the resume materials.

        Picks 2-3 distinct, competency-relevant angles from the strategy's
        answer framework, the candidate's unique advantages, and strengths, so
        different questions get different concrete pointers instead of one
        repeated template. Rendered as a compact bulleted list.
        """
        competency = competency or "岗位核心能力"

        # Pool of candidate-specific material lines.
        materials: list[str] = [
            line.strip()
            for line in state.strategy.answer_framework
            if line.strip()
        ]
        materials += [f"突出你的优势：{a}" for a in state.candidate.unique_advantages]
        materials += [f"体现你的擅长：{s}" for s in state.candidate.strengths]

        # Match materials whose text plausibly relates to this competency.
        def matches(text: str) -> bool:
            hay = text.lower()
            key = competency.lower()
            return key in hay or any(
                word in hay
                for word in ("团队", "项目", "技术", "数据", "指标", "性能", "架构")
                if word in competency
            )

        matched = [m for m in materials if matches(m)]
        if not matched:
            matched = materials[:3]
        # Prefer the most specific: skip the generic opener if something richer.
        chosen = matched[:3]
        items = ["先点明该能力对应的真实经历与你的角色", *chosen]
        return "建议这样组织：\n" + "\n".join(f"· {item}" for item in items)

    def _expand_pool(self, state: InterviewState) -> None:
        """Build the initial question pool from likely_questions + competencies."""
        competencies = state.job.competencies or []
        current = state.mock_interview.questions
        seen: set[str] = set()
        for q in current:
            seen.add(q.question.strip())
        # 1. Strategy likely_questions.
        for question_text in state.strategy.likely_questions:
            q_text = question_text.strip()
            if not q_text or q_text in seen:
                continue
            competency = (
                next(
                    (c for c in competencies if c in q_text),
                    competencies[0] if competencies else "综合能力",
                )
            )
            current.append(
                InterviewQuestion(
                    question=q_text,
                    competency=competency,
                    rationale="来自准备策略中的可能问题",
                    strong_signals=["具体情境", "个人行动", "量化结果"],
                    follow_ups=["你个人具体负责什么？", "结果如何衡量？"],
                    answer_framework="",
                    source="likely",
                )
            )
            seen.add(q_text)
            if len(current) >= MOCK_POOL_TARGET:
                break
        # 2. Pad with competency templates to the pool target. A double loop
        # over templates then competencies yields unique (template, competency)
        # combinations instead of the synchronized-index alias bug.
        if not competencies:
            competencies = ["岗位核心能力"]
        for template in _COMPETENCY_TEMPLATES:
            if len(current) >= MOCK_POOL_TARGET:
                break
            for competency in competencies:
                if len(current) >= MOCK_POOL_TARGET:
                    break
                q_text = template.format(competency=competency)
                if q_text in seen:
                    continue
                current.append(
                    InterviewQuestion(
                        question=q_text,
                        competency=competency,
                        rationale="基于岗位能力生成的通用练习问题",
                        strong_signals=["具体情境", "个人行动", "量化结果", "复盘与取舍"],
                        follow_ups=["你个人具体负责什么？", "结果如何衡量？"],
                        answer_framework="",
                        source="competency",
                    )
            )
            seen.add(q_text)
        # Answer frameworks are filled by _generate_frameworks (per-question LLM
        # pass, deterministic backfill), so leave them empty here.

    async def _generate_frameworks(self, state: InterviewState) -> None:
        """Fill answer_framework for every pool question lacking one.

        Uses one batched LLM call keyed by question index so each question gets
        its own tailored reference-answer hint. Deterministic_framework backfills
        anything the LLM misses so no question ships empty.
        """
        pending = [q for q in state.mock_interview.questions if not q.answer_framework]
        if not pending:
            return
        if self.llm_client is not None:
            employer_block = (
                f"\n{state.past_employer_block}" if state.past_employer_block else ""
            )
            numbered = "\n".join(
                f"{i}. [{q.competency}] {q.question}" for i, q in enumerate(pending)
            )
            prompt = MOCK_FRAMEWORK_PROMPT.format(
                candidate_background=(
                    state.candidate_evidence_context(structure_required=True) + employer_block
                ),
                job_requirement=state.job_review.model_dump_json(),
                questions=numbered,
            )
            raw = await self.think(prompt, context=state.summary())
            parsed = self._parse_structured(raw, FrameworkMap)
            if parsed is not None:
                for index, framework in parsed.index().items():
                    if 0 <= index < len(pending) and framework:
                        pending[index].answer_framework = framework.strip()
        for q in state.mock_interview.questions:
            if not q.answer_framework:
                q.answer_framework = self.deterministic_framework(state, q.competency)

    @classmethod
    def deterministic_refill_question(
        cls, state: InterviewState, *, ordinal: int
    ) -> InterviewQuestion:
        """Create one unique local question when the async refill has not landed."""
        competencies = state.job.competencies or ["岗位核心能力"]
        competency = competencies[(ordinal - 1) % len(competencies)]
        return InterviewQuestion(
            question=(
                f"继续深入「{competency}」：请补充第 {ordinal} 个不同的真实案例，"
                "说明你的个人行动、关键取舍和可验证结果。"
            ),
            competency=competency,
            rationale="后台补题尚未完成时的本地连续面试兜底",
            strong_signals=["具体情境", "个人行动", "关键取舍", "可验证结果"],
            follow_ups=["这个案例与前面的案例有什么不同？"],
            answer_framework=cls.deterministic_framework(state, competency),
            source="refill",
        )

    async def generate_mock_questions(
        self,
        state: InterviewState,
        *,
        recent_answers: list[str] | tuple[str, ...] = (),
        count: int = MOCK_REFILL_BATCH,
    ) -> list[InterviewQuestion]:
        """Generate a batch of follow-up questions without mutating state."""
        employer_block = (
            f"\n{state.past_employer_block}" if state.past_employer_block else ""
        )
        recent_block = "\n".join(
            f"- {item[:200]}" for item in recent_answers[-6:]
        ) or "（暂无已答内容）"
        prompt = MOCK_REFILL_PROMPT.format(
            candidate_background=(
                state.candidate_evidence_context(structure_required=True) + employer_block
            ),
            job_requirement=state.job_review.model_dump_json(),
            interviewer_preference=str(state.interviewer.likely_preferences),
            recent_answers=recent_block,
            count=count,
        )
        try:
            plan = await self.think_structured(
                prompt, MockInterviewPlan, context=state.summary()
            )
            questions = plan.questions[:count]
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse mock refill: %s", exc)
            questions = []
        if not questions:
            # Deterministic fallback from competencies not yet covered.
            answered = set()
            for record in state.mock_session.responses:
                answered.add(record.competency)
            competencies = [
                c for c in (state.job.competencies or ["岗位核心能力"]) if c not in answered
            ] or (state.job.competencies or ["岗位核心能力"])
            questions = [
                InterviewQuestion(
                    question=_COMPETENCY_TEMPLATES[i % len(_COMPETENCY_TEMPLATES)].format(
                        competency=competency
                    ),
                    competency=competency,
                    rationale="后台补题失败后的确定性兜底问题",
                    strong_signals=["具体情境", "个人行动", "量化结果"],
                    follow_ups=["你个人具体负责什么？", "结果如何衡量？"],
                    answer_framework=self.deterministic_framework(state, competency),
                    source="refill",
                )
                for i, competency in enumerate(competencies[:count])
            ]
        else:
            for q in questions:
                q.source = "refill"
                if not q.answer_framework:
                    q.answer_framework = self.deterministic_framework(state, q.competency)
        return questions

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
                        question=_COMPETENCY_TEMPLATES[i % len(_COMPETENCY_TEMPLATES)].format(
                            competency=competency
                        ),
                        competency=competency,
                        rationale="结构化输出失败后的可审计降级问题，用于采集当前岗位所需证据。",
                        strong_signals=["具体情境", "个人行动", "量化结果", "复盘与取舍"],
                        follow_ups=["你个人具体负责什么？", "结果如何衡量？"],
                        answer_framework=self.deterministic_framework(state, competency),
                        source="competency",
                    )
                    for i, competency in enumerate(competencies[:5])
                ]
            )
        self._expand_pool(state)
        await self._generate_frameworks(state)
        return self.make_response(state.mock_interview.model_dump_json(indent=2))
