"""Interview Strategy Agent - three-way fusion: Candidate + Job + Interviewer."""

from __future__ import annotations

import logging
import math
import re

from pydantic import ValidationError

from interview_os.core.agent import Agent
from interview_os.core.message import Message
from interview_os.core.state import InterviewState, InterviewStrategy
from interview_os.models.prompt_templates import STRATEGY_FUSION_PROMPT
from interview_os.tools.web_search import format_employer_business_context

logger = logging.getLogger(__name__)


class InterviewStrategyAgent(Agent):
    """Fuses candidate, job, and interviewer analysis into personalized strategy.

    This is the three-way fusion model: the core value proposition.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="interview_strategy_agent",
            role="Interview Strategy Advisor",
            goal="Fuse candidate, job, and interviewer insights into a personalized interview strategy",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        employer_context = format_employer_business_context(state.past_employer_sources)
        employer_block = f"\n{employer_context}" if employer_context else ""
        prompt = STRATEGY_FUSION_PROMPT.format(
            candidate_profile=(
                f"Candidate name: {state.candidate.name}\n"
                "Candidate evidence (confirmed claims + full resume):\n"
                f"{state.candidate_evidence_context(structure_required=True)[:12000]}"
                f"{employer_block}"
            ),
            job_requirements=(
                state.job.model_dump_json()
                + f"\nRequirement provenance: {state.job_review.model_dump_json()}"
            ),
            interviewer_profile=state.interviewer.model_dump_json(),
        )
        try:
            state.strategy = await self.think_structured(
                prompt, InterviewStrategy, context=state.summary()
            )
            if self._is_title_only(state.job.raw_description):
                # A title supports preparation topics, but not a claim that the candidate
                # fails a requirement that was never explicitly provided.
                state.strategy.key_risks = []
            self._remove_unsupported_quantitative_claims(state)
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("Failed to parse interview strategy (%s)", type(exc).__name__)
        if not state.strategy.summary and not state.strategy.key_risks:
            # The three-way fusion output was empty or ungrounded; give the
            # candidate a deterministic fallback so preparation never hard-fails.
            self.record_degradation(
                "Strategy agent returned no usable strategy; built a generic preparation plan"
            )
            state.strategy = self._generic_strategy(state)
        if state.job.raw_description.strip() and not state.job_review.is_title_only:
            # Apply the source boundary to both model output and deterministic
            # fallback. A model parse failure must not reopen the hallucination
            # path for full-JD preparation cards.
            self._ground_full_jd_strategy(state)
        return self.make_response(state.strategy.model_dump_json(indent=2))

    @staticmethod
    def _generic_strategy(state: InterviewState) -> InterviewStrategy:
        from interview_os.core.state import InterviewStrategy

        competencies = [item.strip() for item in state.job.competencies if item.strip()]
        topics = competencies or (["核心技术能力"] if state.job.title else [])
        return InterviewStrategy(
            summary=(
                f"围绕岗位“{state.job.title or '目标岗位'}”准备：结合简历中的真实经历，"
                "用具体行动和可量化结果支撑每一个能力维度。"
            ),
            key_risks=(
                ["简历信息未经确认，回答中避免引入简历之外无法核验的量化结果"]
                if not state.candidate.name
                else ["注意把经历落到岗位要求的能力维度上"]
            ),
            answer_framework=["先说结论", "再讲行动与权衡", "最后给可量化结果"],
            topics_to_emphasize=topics,
            topics_to_avoid=[],
            likely_questions=[
                f"请讲一个最能体现你“{topic}”能力的真实项目。" for topic in topics
            ],
        )

    @staticmethod
    def _is_title_only(raw_description: str) -> bool:
        text = raw_description.strip()
        return (
            bool(text)
            and len(text) <= 120
            and "\n" not in text
            and not any(
                marker in text
                for marker in ("：", ":", "职责", "要求", "responsibilities", "requirements")
            )
        )

    @staticmethod
    def _remove_unsupported_quantitative_claims(state: InterviewState) -> None:
        """Drop strategy statements containing metrics absent from supplied evidence."""
        evidence = "\n".join(
            (
                state.candidate_evidence_context(structure_required=True),
                state.job.raw_description,
                " ".join(str(item.get("snippet", "")) for item in state.company.public_sources),
                " ".join(
                    str(item.get("snippet", "")) for item in state.interviewer.public_expressions
                ),
            )
        )

        def grounded(statement: str) -> bool:
            material_numbers = re.findall(r"(?<!\d)(?:\d{2,}|\d+(?:\.\d+)?%)(?!\d)", statement)
            return all(number in evidence for number in material_numbers)

        for field in (
            "key_risks",
            "answer_framework",
            "topics_to_emphasize",
            "topics_to_avoid",
            "likely_questions",
        ):
            values = getattr(state.strategy, field)
            setattr(state.strategy, field, [value for value in values if grounded(value)])
        if not grounded(state.strategy.summary):
            state.strategy.summary = "请围绕简历中的真实经历和可核验结果准备回答。"

    @staticmethod
    def _ground_full_jd_strategy(state: InterviewState) -> None:
        """Replace free-form fit claims with source-bounded preparation text.

        A model can turn adjacent facts into unsupported conclusions (for
        example, confusing a three-person team with three years of management).
        The preparation card therefore quotes candidate material and explicit
        JD requirements, while leaving the final fit judgment to verification.
        """

        explicit = [
            item.text.strip()
            for item in state.job_review.requirements
            if item.origin.value == "explicit"
            and item.text.strip()
            and not item.text.rstrip("：:")
            in {"岗位职责", "任职要求", "团队背景"}
            and item.text.strip() != state.job.title.strip()
        ]
        facts = state.confirmed_resume_facts()
        fact_label = "已确认简历事实"
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
            fact_label = "候选人简历自述（待确认）"
        facts = list(dict.fromkeys(item for item in facts if item))[:12]

        selected = InterviewStrategyAgent._select_relevant_facts(explicit, facts, limit=3)
        state.strategy.summary = (
            f"目标岗位“{state.job.title or '未命名岗位'}”的准备必须以明确 JD 和候选人事实为边界。"
            + (
                f"{fact_label}中可优先核对：{' ；'.join(selected)}。"
                if selected
                else "当前没有可直接引用的已确认简历事实。"
            )
            + "这些材料是备题线索，不代表系统已判定候选人完全匹配。"
        )
        state.strategy.answer_framework = [
            *[f"可选材料（{fact_label}）：{item}" for item in selected],
            "按“背景与目标 → 本人决策 → 行动与取舍 → 已核验结果 → 复盘”组织。",
            "数字、团队规模、个人归属和验证方式只使用简历原文或用户确认过的事实。",
        ]
        state.strategy.topics_to_emphasize = explicit[:5]
        state.strategy.topics_to_avoid = [
            "不要把团队成果改写成个人成果",
            "不要补充简历未提供或尚未确认的数字、经历和能力",
        ]
        state.strategy.likely_questions = [
            f"请讲一个能够验证这项明确要求的已发生案例：“{requirement}”。"
            "请说明本人职责、关键取舍和可核验结果。"
            for requirement in explicit[:5]
        ]
        candidate_material = "\n".join(facts).casefold()
        gaps = []
        for requirement in explicit:
            concepts = InterviewStrategyAgent._domain_concepts(requirement)
            supported = sum(
                any(alias.casefold() in candidate_material for alias in aliases)
                for aliases in concepts
            )
            # Requirements often combine several related nouns. Treat common
            # aliases (for example “大促” for “重大活动”) as the same evidence
            # concept, while still requiring support for at least half of a
            # multi-part requirement before removing it from the risk list.
            if concepts and supported < math.ceil(len(concepts) / 2):
                gaps.append(
                    f"明确 JD 要求“{requirement}”，但当前简历材料尚未提供直接案例；"
                    "请准备真实经历，若没有则明确说明相邻经验。"
                )
            if len(gaps) >= 3:
                break
        state.strategy.key_risks = gaps

    @staticmethod
    def _domain_terms(text: str) -> list[str]:
        return [
            alias
            for aliases in InterviewStrategyAgent._domain_concepts(text)
            for alias in aliases
        ]

    @staticmethod
    def _domain_concepts(text: str) -> list[tuple[str, ...]]:
        """Map JD wording to evidence aliases instead of literal substrings."""
        groups = (
            ("B2B", ("b2b", "企业客户", "企业级")),
            ("SaaS", ("saas", "订阅", "续费")),
            ("产品", ("产品", "路线图", "需求验证", "用户研究")),
            ("战略", ("战略", "策略", "规划")),
            ("商业化", ("商业化", "销售", "客户成功", "收入")),
            ("团队管理", ("团队管理", "团队负责人", "带领", "领导", "辅导", "培养")),
            ("跨部门", ("跨部门", "跨团队", "业务", "研发", "运维", "利益相关者", "协作")),
            ("增长", ("增长", "大促", "重大活动", "峰值", "扩容")),
            ("重大活动", ("重大活动", "大促", "活动峰值", "峰值流量")),
            ("架构", ("架构", "平台", "分布式", "系统设计")),
            ("容量规划", ("容量", "qps", "峰值", "扩容", "压测", "资源水位")),
            ("稳定性", ("稳定性", "可用性", "slo", "故障", "错误率", "延迟")),
            ("风险回滚", ("风险", "回滚", "演练", "降级", "故障恢复")),
            ("运行指标", ("运行指标", "指标", "qps", "p99", "错误率", "延迟", "cpu")),
            ("可观测", ("可观测", "监控", "告警", "链路追踪")),
            ("成本", ("成本", "资源效率", "基础设施费用")),
            ("招聘", ("招聘", "人才", "候选人", "寻访")),
        )
        folded = text.casefold()
        return [aliases for _, aliases in groups if any(alias in folded for alias in aliases)]

    @staticmethod
    def _select_relevant_facts(
        requirements: list[str], facts: list[str], *, limit: int
    ) -> list[str]:
        requirement_terms = {
            term.casefold()
            for requirement in requirements
            for term in InterviewStrategyAgent._domain_terms(requirement)
        }
        ranked = sorted(
            facts,
            key=lambda fact: (
                -sum(term in fact.casefold() for term in requirement_terms),
                -int(bool(re.search(r"\d", fact))),
                facts.index(fact),
            ),
        )
        return ranked[:limit]
