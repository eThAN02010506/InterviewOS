"""Deterministic interview-question interpretation and transfer guidance."""

from __future__ import annotations

import re

from interview_os.core.spoken_answer import question_answer_type, question_requirements
from interview_os.core.state import QuestionUnderstanding

_TYPE_LABELS = {
    "behavioral_example": "行为经历题",
    "methodology": "方法论题",
    "situational": "情境假设题",
    "motivation": "动机匹配题",
    "technical": "专业判断题",
    "case_analysis": "案例分析题",
    "general": "综合问题",
}


def _infer_competency(question: str, explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    groups = (
        ("求职动机与岗位匹配", ("为什么加入", "为什么选择", "动机", "离职", "转到", "why")),
        ("跨部门协作", ("跨部门", "跨团队", "协作", "利益相关者")),
        ("领导力与团队管理", ("团队", "管理", "辅导", "冲突", "下属", "领导")),
        ("战略思维", ("战略", "业务方向", "优先级", "市场", "长期")),
        ("数据驱动决策", ("数据", "指标", "衡量", "量化", "口径")),
        ("专业判断与问题解决", ("技术", "架构", "系统", "故障", "方案", "设计")),
        ("复盘与学习能力", ("失败", "复盘", "错误", "挫折", "没做好")),
    )
    folded = question.casefold()
    return next((label for label, terms in groups if any(term in folded for term in terms)), "综合能力")


def _normalize_type(question: str) -> str:
    answer_type = question_answer_type(question)
    folded = question.casefold()
    if answer_type == "methodology" and any(
        term in folded for term in ("架构", "技术", "算法", "系统", "代码", "性能")
    ):
        return "technical"
    if any(term in folded for term in ("case study", "案例分析", "估算", "市场规模")):
        return "case_analysis"
    return answer_type if answer_type in _TYPE_LABELS else "general"


def deterministic_question_understanding(
    question: str, *, competency: str = ""
) -> QuestionUnderstanding:
    """Create useful interpretation even when the local model is unavailable."""
    answer_type = _normalize_type(question)
    inferred_competency = _infer_competency(question, competency)
    requirements = question_requirements(question)
    boundaries = requirements or ["直接回应问题核心", "给出判断依据或真实证据"]

    if answer_type == "motivation":
        goal = "验证选择动机是否具体、稳定，并与目标岗位和个人长期方向真正匹配"
        mistakes = ["只讲离开现公司的负面原因", "只表达热情，没有岗位匹配证据", "把宏大愿景当成个人选择依据"]
        variants = [
            "这个岗位最吸引你的具体工作内容是什么？为什么？",
            "如果同时拿到两个机会，你会用哪些标准决定是否加入我们？",
            "这次职业选择与你未来三年的发展目标有什么关系？",
        ]
        transfer = "复用同一段职业选择证据，但分别突出岗位吸引点、决策标准或长期目标，不能背同一套答案。"
        follow_ups = [
            "目标岗位的哪一项具体任务与你过去的哪段经历最匹配？",
            "什么现实风险可能让你重新考虑这个选择？",
            "入职前三个月你会用什么结果验证这次选择？",
        ]
    elif answer_type == "behavioral_example":
        goal = f"用已经发生的事实验证候选人的{inferred_competency}，并区分个人贡献与团队结果"
        mistakes = ["只讲团队做了什么，缺少个人决定", "背景很长但没有关键行动", "用目标或预测冒充实际结果"]
        variants = [
            f"请讲一次你在{inferred_competency}上做出关键取舍的经历。",
            f"请讲一次你在{inferred_competency}上没有按原计划推进、后来调整的经历。",
            "如果让相关同事评价这次经历，他们会怎样描述你的具体贡献？",
        ]
        transfer = "同一案例可复用事实骨架，但要随问法切换主角：取舍题讲备选方案，失败题讲修正，协作题讲他人互动。"
        follow_ups = [
            "这件事中哪一个关键决定是你本人做出的？",
            "有什么已经发生的结果可以验证你的行动有效？",
            "如果重来一次，你会在哪个节点做不同选择？",
        ]
    elif answer_type in {"methodology", "technical", "case_analysis"}:
        goal = f"验证候选人能否把{inferred_competency}拆成有前提、顺序、判断标准和验证闭环的方法"
        mistakes = ["罗列通用步骤但没有判断标准", "忽略适用前提和约束", "没有说明如何验证方法有效"]
        variants = [
            f"你通常如何判断一个{inferred_competency}方案是否值得推进？",
            f"如果时间或资源减半，你会怎样调整这套{inferred_competency}方法？",
            f"请用一个真实案例说明这套{inferred_competency}方法何时失效、你如何修正。",
        ]
        transfer = "保留方法主干，按新问题替换约束、决策门槛和验证指标；案例只用于证明方法，不要喧宾夺主。"
        follow_ups = [
            "这套方法中最关键的判断门槛是什么？",
            "在什么条件下这套方法会失效？",
            "请用一个已发生的结果证明它有效。",
        ]
    elif answer_type == "situational":
        goal = f"验证候选人在信息不完整时运用{inferred_competency}形成假设、排序行动并控制风险的能力"
        mistakes = ["直接给唯一答案却不说明假设", "忽略关键利益相关方", "只有行动清单，没有风险与验证节点"]
        variants = [
            "如果入职前三十天遇到这个场景，你会先确认哪些信息？",
            f"如果关键利益相关方不同意你的{inferred_competency}方案，你会怎么推进？",
            f"出现什么新证据时，你会改变原来的{inferred_competency}判断？",
        ]
        transfer = "复用判断框架而非虚构经历；每个新场景都应重新声明假设、优先级、风险线和验证节点。"
        follow_ups = [
            "你的结论依赖哪两个关键假设？",
            "第一步失败时，你的止损或升级条件是什么？",
            "哪个新事实会让你改变优先级？",
        ]
    else:
        goal = f"确认候选人对{inferred_competency}有清晰立场，并能用具体依据而非口号支撑"
        mistakes = ["没有先给直接结论", "观点过宽，与题目边界无关", "只有抽象判断，没有事实或例子"]
        variants = [
            f"你判断{inferred_competency}做得好的标准是什么？",
            f"在{inferred_competency}上，你最不同意的常见做法是什么？为什么？",
            f"什么事实会让你改变对{inferred_competency}的看法？",
        ]
        transfer = "复用核心立场和证据，但根据问法分别强调评价标准、反方边界或修正条件。"
        follow_ups = [
            "什么具体事实支持这个判断？",
            "你最不确定的前提是什么？",
            "什么证据会让你改变观点？",
        ]
    return QuestionUnderstanding(
        answer_type=answer_type,
        answer_type_label=_TYPE_LABELS[answer_type],
        assessment_goal=goal,
        competency=inferred_competency,
        answer_boundary=list(dict.fromkeys(boundaries))[:5],
        common_mistakes=mistakes,
        transfer_principle=transfer,
        related_questions=_clean_questions(variants, original=question),
        likely_follow_ups=follow_ups,
        analysis_source="rules",
    )


def _clean_questions(items: list[str], *, original: str) -> list[str]:
    """Keep distinct, usable variants and exclude the original wording."""
    original_key = re.sub(r"\s+", "", original).casefold().rstrip("?？。")
    output: list[str] = []
    seen = {original_key}
    for item in items:
        text = re.sub(
            r"^\s*(?:\d{1,2}[.、)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*",
            "",
            " ".join(item.split()).strip(),
        )
        key = re.sub(r"\s+", "", text).casefold().rstrip("?？。")
        if len(text) < 4 or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output[:3]


def _clean_list_items(items: list[str], *, limit: int) -> list[str]:
    output: list[str] = []
    for item in items:
        text = re.sub(
            r"^\s*(?:\d{1,2}[.、)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*",
            "",
            " ".join(item.split()).strip(),
        )
        if text and text not in output:
            output.append(text[:200])
    return output[:limit]


def merge_model_understanding(
    fallback: QuestionUnderstanding,
    *,
    answer_type: str,
    answer_type_label: str,
    assessment_goal: str,
    competency: str,
    answer_boundary: list[str],
    common_mistakes: list[str],
    transfer_principle: str,
    related_questions: list[str],
    likely_follow_ups: list[str],
    original_question: str,
    explicit_competency: str = "",
) -> QuestionUnderstanding:
    """Validate model semantics while preserving the deterministic contract."""
    # The answer taxonomy is shared with coverage scoring. A model label is
    # advisory only and must never make pre-answer guidance disagree with the
    # post-answer contract.
    allowed_type = fallback.answer_type
    model_variants = _clean_questions(related_questions, original=original_question)
    variants = _clean_questions(
        [*model_variants[:2], *fallback.related_questions], original=original_question
    )
    return QuestionUnderstanding(
        answer_type=allowed_type,
        answer_type_label=_TYPE_LABELS[allowed_type],
        assessment_goal=(assessment_goal.strip() or fallback.assessment_goal)[:300],
        competency=(explicit_competency.strip() or competency.strip() or fallback.competency)[:100],
        # Scoring and pre-answer guidance must share the deterministic contract.
        answer_boundary=fallback.answer_boundary,
        common_mistakes=_clean_list_items(common_mistakes, limit=4)
        or fallback.common_mistakes,
        # Transfer coaching must stay stable across model versions; semantic
        # specificity is carried by the goal, mistakes, variants and probes.
        transfer_principle=fallback.transfer_principle,
        related_questions=variants or fallback.related_questions,
        likely_follow_ups=_clean_list_items(likely_follow_ups, limit=4)
        or fallback.likely_follow_ups,
        analysis_source="model",
    )
