"""Mock Interview Agent - generates personalized questions."""

from __future__ import annotations

import asyncio
import logging
import re

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.question_understanding import (
    deterministic_question_understanding,
    merge_model_understanding,
    reconcile_question_understanding,
)
from interview_os.core.spoken_answer import analyze_spoken_answer
from interview_os.core.spoken_answer import (
    question_requirements as derive_question_requirements,
)
from interview_os.core.state import (
    MOCK_POOL_TARGET,
    MOCK_REFILL_BATCH,
    InterviewQuestion,
    InterviewState,
    MockInterviewPlan,
    QuestionUnderstanding,
)
from interview_os.models.prompt_templates import (
    CUSTOM_QUESTION_ANALYSIS_PROMPT,
    MOCK_QUESTION_PROMPT,
    MOCK_REFILL_PROMPT,
)
from interview_os.models.structured import CustomQuestionAnalysisDraft
from interview_os.tools.web_search import format_employer_business_context

logger = logging.getLogger(__name__)

_COMPETENCY_TEMPLATES = (
    "请选一个最近三年你亲自负责、最能体现{competency}的真实案例：当时要解决什么业务问题，你做了什么关键决定，结果如何验证？",
    "请讲一次你在{competency}上遇到明显约束或意见分歧的经历：你比较了哪些方案，为什么这样选择，最终结果和复盘是什么？",
    "请用一个已经结束且结果可核验的项目说明你如何通过{competency}推动落地；请区分团队工作与你个人负责的部分。",
    "请讲一次{competency}没有按原计划推进的真实经历：你如何识别问题、调整方案，并用什么证据判断调整有效？",
)

_FACTUAL_NUMBER_PATTERN = re.compile(
    r"(?:\b(?:19|20)\d{2}\b)|"
    r"(?:\d+(?:\.\d+)?\s*(?:万\s*)?(?:qps|rps|tps|%|倍|人|年|个月|ms|毫秒|秒))",
    re.IGNORECASE,
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
        """Build a source-bounded reference-answer hint from candidate facts."""
        competency = competency or "岗位核心能力"
        folded_competency = competency.casefold()
        if any(word in folded_competency for word in ("动机", "求职", "意愿", "motivation")):
            return (
                "建议这样组织：\n"
                "· 先说明你主动选择什么，而不是抱怨正在离开什么\n"
                "· 用一段真实经历证明你在哪类问题、责任范围和工作节奏中最投入\n"
                "· 对照目标公司阶段与岗位任务，说明双方匹配点和你能立即贡献什么\n"
                "· 最后说明你已考虑的风险，以及入职后前三个月会如何验证选择"
            )

        facts = state.confirmed_resume_facts()
        source_label = "已确认简历事实"
        if not facts and not state.resume_review.claims:
            facts = [
                re.sub(
                    r"^\s*(?:(?:[•·*\-—]+)|(?:[（(]?\d{1,2}[、.．)）]\s*))\s*",
                    "",
                    line,
                ).strip()
                for line in state.candidate.raw_resume_text.splitlines()
                if len(line.strip()) >= 10
            ]
            source_label = "候选人简历自述（使用前确认）"
        vocabulary = (
            "产品", "战略", "路线图", "客户", "留存", "续费", "商业化", "增长",
            "团队", "管理", "辅导", "协作", "跨部门", "数据", "指标", "技术", "架构",
            "系统", "性能", "稳定性", "风险", "招聘", "人才",
        )
        requested = [term for term in vocabulary if term in competency]
        ranked = sorted(
            dict.fromkeys(facts),
            key=lambda fact: (
                -sum(term in fact for term in requested),
                -int(bool(re.search(r"\d", fact))),
                -int(any(term in fact for term in ("负责", "主导", "决定", "推动", "上线"))),
            ),
        )
        chosen = ranked[:2] if ranked else []
        if not chosen:
            return (
                "建议这样组织：\n"
                "· 当前没有可直接引用的已确认候选人事实，先选择并确认一段真实经历\n"
                "· 按背景与目标 → 本人决策 → 行动与取舍 → 已核验结果 → 复盘组织\n"
                "· 没有的经历、数字、团队规模或验证方式不要补写"
            )
        return "建议这样组织：\n" + "\n".join(
            [
                *[f"· 可选材料（{source_label}）：{fact}" for fact in chosen],
                "· 按背景与目标 → 本人决策 → 行动与取舍 → 已核验结果 → 复盘组织",
                "· 只补充你能确认的事实；系统不会替你生成经历、动作或结果",
            ]
        )

    @staticmethod
    def question_requirements(question: str) -> list[str]:
        """Return explicit requirements the answer and feedback must share."""
        return derive_question_requirements(question)

    @staticmethod
    def _factual_numbers(text: str) -> set[str]:
        return {
            re.sub(r"\s+", "", match.group(0)).casefold()
            for match in _FACTUAL_NUMBER_PATTERN.finditer(text)
        }

    @classmethod
    def _ground_question_premises(
        cls, state: InterviewState, questions: list[InterviewQuestion]
    ) -> None:
        """Remove numeric premises that the explicit JD did not establish.

        Candidate metrics are excellent answer evidence, but are unsafe as
        question premises: a model can combine an actual production peak with
        a different load-test statement. Replacing such a question with a
        competency template asks for the evidence without asserting it.
        """
        allowed = cls._factual_numbers(state.job.raw_description)
        for index, question in enumerate(questions):
            unsupported = cls._factual_numbers(question.question) - allowed
            if not unsupported:
                continue
            competency = question.competency or "岗位核心能力"
            question.question = _COMPETENCY_TEMPLATES[
                index % len(_COMPETENCY_TEMPLATES)
            ].format(competency=competency)
            question.rationale = (
                "模型问题含未由明确 JD 支持的数字前提，已改为不预设事实的能力问题"
            )
            question.strong_signals = ["具体情境", "个人行动", "关键取舍", "可验证结果"]
            question.follow_ups = ["结果如何核验？", "你个人具体负责什么？"]
            question.answer_framework = ""
            question.source = "competency"

    async def analyze_custom_question(
        self,
        state: InterviewState,
        question: str,
        *,
        competency: str = "",
    ) -> QuestionUnderstanding:
        """Interpret a user question with a deterministic, model-enhanced contract.

        Model analysis is optional and bounded. The deterministic answer
        requirements remain authoritative so pre-answer coaching cannot drift
        away from the post-answer scoring contract.
        """
        explicit_requirements = [
            item.text
            for item in state.job_review.requirements
            if item.origin.value == "explicit"
        ]
        if not explicit_requirements and state.job.raw_description.strip():
            explicit_requirements = list(state.job.responsibilities)
        confirmed_claims = [
            (str(claim.id), claim.statement)
            for claim in state.resume_review.claims
            if claim.status.value in {"confirmed", "modified"}
        ]
        fallback = deterministic_question_understanding(
            question,
            competency=competency,
            job_title=state.job.title,
            explicit_job_requirements=explicit_requirements,
            confirmed_claims=confirmed_claims,
        )
        if self.llm_client is None:
            return fallback
        prompt = CUSTOM_QUESTION_ANALYSIS_PROMPT.format(
            question=question,
            competency=competency or "（未指定，请从问题推断）",
            job_title=state.job.title or "（未提供）",
            job_requirement=(
                state.job.model_dump_json() + "\n" + state.job_review.model_dump_json()
            )[:5000],
            candidate_facts=(
                "\n".join(
                    f"{index}. {statement}"
                    for index, (_, statement) in enumerate(confirmed_claims, start=1)
                )
                or "（没有已确认的候选人事实；candidate_story_options 必须为空）"
            )[:5000],
        )
        try:
            draft = await asyncio.wait_for(
                self.think_structured(
                    prompt,
                    CustomQuestionAnalysisDraft,
                    context="只使用编号且已确认的候选人事实；不得推断或补写候选人经历。",
                    max_tokens=1200,
                ),
                timeout=30,
            )
        except (TimeoutError, ValueError, TypeError, ValidationError) as exc:
            logger.warning(
                "Custom question semantic analysis fell back to rules (%s)",
                type(exc).__name__,
            )
            self.record_degradation("自定义问题语义解析失败，已使用确定性题型与题族分析")
            return fallback
        return merge_model_understanding(
            fallback,
            **draft.model_dump(),
            original_question=question,
            explicit_competency=competency,
            confirmed_claims=confirmed_claims,
        )

    @staticmethod
    def teaching_example(competency: str, question: str = "") -> str:
        """Return a realistic but explicitly fictional behavior example."""
        folded = f"{competency} {question}".casefold()
        if any(
            word in folded
            for word in (
                "动机",
                "为什么加入",
                "为什么选择",
                "为什么想",
                "转到创业",
                "求职",
                "why join",
                "why this",
                "motivation",
            )
        ):
            return (
                "我考虑转到早期公司，不是因为否定成熟平台，而是过去两年里我最投入的工作，"
                "都是从问题还不清晰时开始：和客户确认需求、定义首版指标，再带团队快速验证。"
                "在成熟组织中，这类从零到一的责任通常被拆给多个团队；我下一阶段希望对产品结果承担"
                "更完整的责任。这个岗位吸引我的具体原因，是公司正在验证新业务的可复制增长方式，"
                "而我做过用户访谈、指标设计和跨团队落地，能先帮助团队缩短验证周期。我也理解早期"
                "公司的资源和流程不完整，所以不会把“创业”想成单纯更自由；入职前三个月，我会先"
                "用客户反馈速度、关键漏斗变化和团队决策周期验证自己是否真的创造了匹配的价值。"
            )
        if any(word in folded for word in ("容量", "qps", "扩容", "吞吐")):
            return (
                "去年一次大型促销前，业务预计订单峰值会达到平日的六倍。我负责八人平台团队的容量方案，"
                "先与业务统一订单量和转化率口径，再按网关、订单、库存、消息队列和数据库拆解链路，建立"
                "QPS、P99、CPU、连接池和积压的单实例基线。我比较了全部预扩容与全部依赖自动扩容两种"
                "方案；考虑数据库扩容耗时和热点迁移风险，最终决定数据库与消息队列按预测峰值的1.3倍"
                "预扩容，无状态服务使用HPA。上线前回放历史流量并完成1.3倍峰值压测和单节点故障演练。"
                "活动实际峰值为十一万QPS，P99保持在二百四十毫秒，错误率为0.2%，资源水位没有越过"
                "回滚线。复盘后我用预测误差修正下一次容量模型。"
            )
        if any(word in folded for word in ("招聘", "人才", "寻访", "组织")):
            return (
                "在一家进入新市场的企业中，业务要求八周内组建首批核心团队，但岗位画像频繁变化。"
                "我负责招聘项目，先与业务负责人把目标拆成必须具备、可培养和文化风险三类标准，随后用人才地图"
                "比较三个来源渠道，并每周按有效候选人率、面试通过率和接受率复盘。发现技术负责人对经验"
                "年限要求过高后，我用前两周漏斗数据推动团队改成能力证据面试。最终关键岗位按期完成，"
                "无效面试明显下降。复盘来看，最重要的不是扩大搜索量，而是尽早固定决策标准和调整机制。"
            )
        if any(
            word in folded
            for word in ("架构", "技术", "系统", "性能", "工程", "稳定性", "故障", "可观测")
        ):
            return (
                "在某交易系统大促流量增长后，接口高峰期延迟从两百毫秒升到一秒以上。我负责定位和改造，"
                "先用链路追踪确认瓶颈在同步写入，再比较扩容、异步队列和缓存三种方案。考虑一致性、成本"
                "与回滚风险后，我决定先拆出可重试队列并保留双写校验，按5%、20%、50%分批灰度上线。"
                "两周后高峰延迟稳定在三百毫秒以内，错误率没有上升。复盘时我补充了容量预警和回滚演练。"
            )
        if any(word in folded for word in ("领导", "团队", "协作", "沟通", "管理")):
            return (
                "在一个跨部门项目中，产品、销售和交付团队对上线范围意见不一致。我承担推进责任，"
                "先把争议拆成客户价值、交付风险和不可逆成本三项，再分别访谈负责人并形成两套范围方案。"
                "我决定先上线覆盖主要客户的最小范围，同时设立两周验证指标和回退条件。项目按期发布，首批"
                "用户完成核心流程，未解决需求进入下一迭代。复盘中我认识到，推进不是替大家决定，而是让"
                "约束、责任人和决策门槛变得透明。"
            )
        return (
            f"在某项需要体现“{competency or '岗位核心能力'}”的跨团队项目中，我先把模糊目标"
            "转成三项可验证结果，并负责方案设计和推进落地。我比较了快速临时方案与长期改造方案，"
            "结合时间、风险和可逆性决定分阶段实施；第一阶段用小范围试点验证假设，第二阶段根据数据调整"
            "流程并扩大范围。项目最终在八周内达到约定目标，且没有突破风险边界。复盘时我认为，如果重来，会更早"
            "统一指标口径并邀请执行团队参与方案设计。"
        )

    @classmethod
    def enrich_question(
        cls, question: InterviewQuestion, state: InterviewState | None = None
    ) -> None:
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
        explicit_requirements = []
        confirmed_claims: list[tuple[str, str]] = []
        job_title = ""
        if state is not None:
            explicit_requirements = [
                item.text
                for item in state.job_review.requirements
                if item.origin.value == "explicit"
            ]
            if not explicit_requirements and state.job.raw_description.strip():
                explicit_requirements = list(state.job.responsibilities)
            confirmed_claims = [
                (str(claim.id), claim.statement)
                for claim in state.resume_review.claims
                if claim.status.value in {"confirmed", "modified"}
            ]
            job_title = state.job.title
        fallback = deterministic_question_understanding(
            question.question,
            competency=question.competency,
            job_title=job_title,
            explicit_job_requirements=explicit_requirements,
            confirmed_claims=confirmed_claims,
        )
        question.understanding = reconcile_question_understanding(
            question.understanding,
            fallback,
        )
        example = question.example_answer or cls.teaching_example(question.competency, text)
        analysis = analyze_spoken_answer(question.question, example)
        if any(item.status == "missing" for item in analysis.question_coverage):
            example = cls.teaching_example(question.competency, question.question)
        question.example_answer = example

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
        # Source-bounded answer frameworks are filled in one deterministic pass.

    async def _generate_frameworks(self, state: InterviewState) -> None:
        """Fill source-bounded frameworks without a free-form generation pass."""
        for q in state.mock_interview.questions:
            q.answer_framework = self.deterministic_framework(state, q.competency)
            self.enrich_question(q, state)

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
            logger.warning("Failed to parse mock refill (%s)", type(exc).__name__)
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
            self._ground_question_premises(state, questions)
            for q in questions:
                q.source = "refill"
                if not q.answer_framework:
                    q.answer_framework = self.deterministic_framework(state, q.competency)
        for q in questions:
            self.enrich_question(q, state)
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
            logger.warning("Failed to parse mock interview plan (%s)", type(exc).__name__)
        self._ground_question_premises(state, state.mock_interview.questions)
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
            self.enrich_question(question, state)
        await self._generate_frameworks(state)
        return self.make_response(state.mock_interview.model_dump_json(indent=2))
