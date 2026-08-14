"""Mock Interview Agent - generates personalized questions."""

from __future__ import annotations

import logging
import re

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
from interview_os.tools.web_search import format_employer_business_context

logger = logging.getLogger(__name__)

_COMPETENCY_TEMPLATES = (
    "请选一个最近三年你亲自负责、最能体现{competency}的真实案例：当时要解决什么业务问题，你做了什么关键决定，结果如何验证？",
    "请讲一次你在{competency}上遇到明显约束或意见分歧的经历：你比较了哪些方案，为什么这样选择，最终结果和复盘是什么？",
    "请用一个已经结束且结果可核验的项目说明你如何通过{competency}推动落地；请区分团队工作与你个人负责的部分。",
    "请讲一次{competency}没有按原计划推进的真实经历：你如何识别问题、调整方案，并用什么证据判断调整有效？",
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
    def _is_behavioral_question(question: str) -> bool:
        folded = question.casefold()
        return any(
            word in folded
            for word in (
                "案例",
                "一次",
                "已经发生",
                "已经结束",
                "请描述你在",
                "请说明你在",
                "tell me about a time",
                "describe a time",
            )
        ) or bool(
            re.search(
                r"(?:请讲|谈谈|描述|说明).{0,30}经历|"
                r"(?:你)?在.{2,80}(?:时|中)[，,]?(?:你)?(?:是)?如何",
                question,
            )
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
        if not state.resume_review.claims:
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

    @staticmethod
    def question_requirements(question: str) -> list[str]:
        """Return explicit requirements the answer and feedback must share."""
        folded = question.casefold()
        behavioral = MockInterviewAgent._is_behavioral_question(question)
        motivation = any(
            word in folded for word in ("为什么", "动机", "why this", "why do you")
        ) and not behavioral
        if motivation:
            return ["说明动机与匹配关系"]
        if any(
            word in folded for word in ("如果", "假如", "会怎么", "what would", "how would")
        ):
            return ["说明方法或制定过程", "给出结果与验证方式"]
        if not behavioral and any(
            word in folded
            for word in (
                "如何",
                "怎么",
                "流程",
                "方法",
                "设计",
                "步骤",
                "how do",
                "approach",
                "process",
            )
        ):
            return ["说明方法或制定过程", "给出结果与验证方式"]
        requirements = [
            "提供一个真实案例",
            "明确具体公司/业务场景",
            "明确个人职责与关键决策",
            "说明执行动作",
        ]
        if any(
            word in folded
            for word in (
                "如何",
                "怎么",
                "方法",
                "取舍",
                "权衡",
                "方案",
                "选择",
                "分歧",
                "约束",
                "关键决定",
                "关键决策",
            )
        ):
            requirements.append("说明方法或制定过程")
        requirements.append("给出结果与验证方式")
        return requirements

    @staticmethod
    def teaching_example(competency: str) -> str:
        """Return a realistic but explicitly fictional behavior example."""
        folded = (competency or "").casefold()
        if any(word in folded for word in ("招聘", "人才", "寻访", "组织")):
            return (
                "在一家进入新市场的企业中，业务要求八周内组建首批核心团队，但岗位画像频繁变化。"
                "示范候选人先与业务负责人把目标拆成必须具备、可培养和文化风险三类标准，随后用人才地图"
                "比较三个来源渠道，并每周按有效候选人率、面试通过率和接受率复盘。发现技术负责人对经验"
                "年限要求过高后，他用前两周漏斗数据推动团队改成能力证据面试。最终关键岗位按期完成，"
                "无效面试明显下降。复盘来看，最重要的不是扩大搜索量，而是尽早固定决策标准和调整机制。"
            )
        if any(word in folded for word in ("架构", "技术", "系统", "性能", "工程")):
            return (
                "在某交易系统流量增长后，接口高峰期延迟从两百毫秒升到一秒以上。示范候选人负责定位和"
                "改造方案，他先用链路追踪确认瓶颈在同步写入，再比较扩容、异步队列和缓存三种方案。考虑"
                "一致性与回滚成本后，他选择先拆出可重试队列并保留双写校验，分两批灰度上线。两周后高峰"
                "延迟稳定在三百毫秒以内，错误率没有上升。复盘时他补充了容量预警，避免团队再次被动救火。"
            )
        if any(word in folded for word in ("领导", "团队", "协作", "沟通", "管理")):
            return (
                "在一个跨部门项目中，产品、销售和交付团队对上线范围意见不一致。示范候选人承担推进责任，"
                "先把争议拆成客户价值、交付风险和不可逆成本三项，再分别访谈负责人并形成两套范围方案。"
                "他建议先上线覆盖主要客户的最小范围，同时设立两周验证指标和回退条件。项目按期发布，首批"
                "用户完成核心流程，未解决需求进入下一迭代。复盘中他认识到，推进不是替大家决定，而是让"
                "约束、责任人和决策门槛变得透明。"
            )
        return (
            f"在某项需要体现“{competency or '岗位核心能力'}”的跨团队项目中，示范候选人先把模糊目标"
            "转成三项可验证结果，并明确自己负责方案设计和推进落地。他比较了快速临时方案与长期改造方案，"
            "结合时间、风险和可逆性选择分阶段实施；第一阶段用小范围试点验证假设，第二阶段根据数据调整"
            "流程并扩大范围。项目最终达到约定目标，且没有突破风险边界。复盘时他指出，如果重来，会更早"
            "统一指标口径并邀请执行团队参与方案设计。"
        )

    @classmethod
    def enrich_question(cls, question: InterviewQuestion) -> None:
        text = question.question.strip()
        folded = text.casefold()
        behavioral = cls._is_behavioral_question(text)
        motivation = any(
            word in folded for word in ("为什么", "动机", "why this", "why do you")
        ) and not behavioral
        bounded = any(
            marker in folded
            for marker in ("结果", "验证", "复盘", "影响", "result", "impact", "outcome")
        )
        if text and (len(text) < 28 or not bounded):
            stem = text.rstrip("。？? ")
            if motivation:
                suffix = "。请分别连接这个岗位最吸引你的具体要素、你已有的相关经历，以及下一阶段希望解决的问题。"
            elif any(
                word in folded
                for word in ("如果", "假如", "会怎么", "what would", "how would")
            ):
                suffix = "。请说明你的前提假设、处理步骤、主要风险，以及用什么结果判断方案有效。"
            elif not behavioral and any(
                word in folded
                for word in ("如何", "怎么", "流程", "方法", "设计", "步骤", "how do", "approach", "process")
            ):
                suffix = "。请说明适用场景、按顺序执行的步骤、关键判断依据，以及用什么结果验证方法有效。"
            else:
                suffix = "。请选一个具体且已经发生的案例，说明当时的目标与约束、你本人做出的关键决定和行动，以及结果如何验证。"
            question.question = stem + suffix
        # One deterministic vocabulary is shared with spoken-answer coverage;
        # accepting arbitrary model labels here would make pre-answer guidance
        # disagree with post-answer feedback.
        question.question_requirements = cls.question_requirements(question.question)
        if not question.example_answer:
            question.example_answer = cls.teaching_example(question.competency)

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
            employer_context = format_employer_business_context(state.past_employer_sources)
            employer_block = f"\n{employer_context}" if employer_context else ""
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
                for index, item in parsed.index().items():
                    if 0 <= index < len(pending) and item.answer_framework:
                        pending[index].answer_framework = item.answer_framework.strip()
        for q in state.mock_interview.questions:
            if not q.answer_framework:
                q.answer_framework = self.deterministic_framework(state, q.competency)
            self.enrich_question(q)

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
            question_requirements=cls.question_requirements("真实案例、关键取舍和可验证结果"),
            example_answer=cls.teaching_example(competency),
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
        employer_context = format_employer_business_context(state.past_employer_sources)
        employer_block = f"\n{employer_context}" if employer_context else ""
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
        for q in questions:
            self.enrich_question(q)
        return questions

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        employer_context = format_employer_business_context(state.past_employer_sources)
        employer_block = f"\n{employer_context}" if employer_context else ""
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
        for question in state.mock_interview.questions:
            self.enrich_question(question)
        await self._generate_frameworks(state)
        return self.make_response(state.mock_interview.model_dump_json(indent=2))
