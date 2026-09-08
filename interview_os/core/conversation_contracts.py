"""Shared contracts for career conversations that do not require a STAR story."""

from __future__ import annotations

import re

from interview_os.core.state import QuestionCoverageItem

CONVERSATIONAL_TYPES = {"motivation", "compensation", "candidate_question", "constraint"}


def special_question_type(question: str) -> str | None:
    folded = question.casefold()
    for kind, pattern in (
        ("compensation", r"期望薪资|薪资要求|薪酬|salary|compensation"),
        ("candidate_question", r"反问面试官|问面试官|有什么想问|questions for me"),
        ("constraint", r"搬迁|签证|到岗时间|空档|\bgap\b|relocation|visa"),
    ):
        if re.search(pattern, folded):
            return kind
    return None


def requirements(kind: str, question: str) -> list[str]:
    if kind == "motivation":
        result = ["说明动机与匹配关系"]
        if re.search(r"标准|依据|评估|criteria|evaluate", question, re.IGNORECASE):
            result.append("说明机会判断标准")
        return result
    return {
        "compensation": ["说明薪酬口径与预期范围"],
        "candidate_question": ["向面试官提出有助于双向判断的具体问题"],
        "constraint": ["说明客观条件与待确认安排"],
    }[kind]


def clauses(answer: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。！？!?；;\n]|(?<!\d)\.(?!\d)", answer) if s.strip()]


def affirmative_excerpt(answer: str, pattern: str) -> str:
    """Conservative evidence: exclude denial/uncertainty, preserve later contrast."""
    for sentence in re.split(r"[，,]|但是|而是|但", answer):
        for part in clauses(sentence):
            if re.search(pattern, part, re.IGNORECASE) and not re.search(
                r"没有|尚未|未曾|没做|没考虑|不清楚|不知道|不确定|不考虑|不愿|不会|"
                r"(?:不|未)(?:评估|核验|判断|负责|看|验证)|"
                r"\b(?:not|never|no|haven't|don't|didn't|unsure)\b",
                part,
                re.IGNORECASE,
            ):
                return part[:200]
    return ""


def coverage(kind: str, question: str, answer: str) -> list[QuestionCoverageItem]:
    motive = affirmative_excerpt(
        answer, r"希望|吸引|想.{0,16}(?:承担|负责|加入|从事)|because|want|motiv|drawn"
    )
    fit = affirmative_excerpt(
        answer,
        r"(?:我|过去|之前).{0,30}(?:负责|经验|做过|擅长)|my experience|i (?:led|built|worked)",
    )
    criteria = affirmative_excerpt(
        answer,
        r"(?:看|标准|判断|评估).{0,70}(?:客户|责任|产品采用|团队|决策|留存)|criteria.{0,60}(?:customer|ownership|team)",
    )
    salary = affirmative_excerpt(
        answer, r"[\d一二三四五六七八九十百千万].{0,12}(?:元|万|k\b|usd|dollars)|\$\s*\d"
    )
    basis = affirmative_excerpt(
        answer, r"税前|税后|年薪|月薪|总包|annual|monthly|base|total|gross|net"
    )
    inquiry = next(
        (
            s[:200]
            for s in clauses(answer)
            if re.search(
                r"成功标准|挑战|障碍|优先级|决策|职责|协作|考核|团队|success|challenge|priorit|decision|team",
                s,
                re.IGNORECASE,
            )
            and re.search(r"什么|哪些|如何|怎样|是否|多久|谁|what|how|which|who", s, re.IGNORECASE)
        ),
        "",
    )
    condition = next(
        (
            s[:200]
            for s in clauses(answer)
            if re.search(
                r"到岗|搬迁|签证|空档|照顾|离职|交接|[一二三四五六七八九十\d]+[天周月年]|"
                r"relocat|visa|notice|start|week|month|caregiv",
                s,
                re.IGNORECASE,
            )
        ),
        "",
    )
    values = {
        "说明动机与匹配关系": (
            bool(motive and fit),
            motive or fit,
            "连接主动选择的原因与一段相关经历，说明它为何匹配目标岗位。",
        ),
        "说明机会判断标准": (
            bool(criteria),
            criteria,
            "给出具体标准及优先级，例如客户采用、决策责任或团队工作方式。",
        ),
        "说明薪酬口径与预期范围": (
            bool(salary and basis),
            salary or basis,
            "说明税前/税后、年薪/月薪/总包和预期金额或范围；尚未确定时明确待协商。",
        ),
        "向面试官提出有助于双向判断的具体问题": (
            bool(inquiry),
            inquiry,
            "提出关于成功标准、真实挑战或协作方式的具体问题。",
        ),
        "说明客观条件与待确认安排": (
            bool(condition),
            condition,
            "说明实际条件、可行时间线和仍待确认的安排，不必补项目成果。",
        ),
    }
    result = []
    for name in requirements(kind, question):
        complete, evidence, suggestion = values[name]
        if name == "说明动机与匹配关系" and motive and fit:
            evidence = "；".join(dict.fromkeys([motive, fit]))
        if name == "说明薪酬口径与预期范围" and salary and basis:
            evidence = "；".join(dict.fromkeys([salary, basis]))
        result.append(
            QuestionCoverageItem(
                requirement=name,
                status="covered" if complete else "partial" if evidence else "missing",
                evidence=evidence,
                suggestion=suggestion,
            )
        )
    return result


def coaching(kind: str) -> tuple[str, str, str]:
    """Depth, organization, and practical next step appropriate to the question."""
    return {
        "motivation": (
            "说明机会判断标准的优先级和一票否决条件，并区分已核验信息与待确认假设。",
            "主动选择 → 相关经历 → 岗位匹配 → 风险与验证",
            "明确入职前要核验什么，以及何种信息会改变你的选择。",
        ),
        "compensation": (
            "解释预期依据，区分固定薪酬、奖金和股权，并说明可协商条件。",
            "预期与口径 → 依据 → 协商边界",
            "确认整体包的组成与兑现条件，不需要项目量化结果。",
        ),
        "candidate_question": (
            "优先选择当前面试官能回答、且会影响你是否加入的问题。",
            "最想确认的问题 → 为什么重要 → 针对回答继续追问",
            "想清楚对方哪种回答会改变你的判断，避免重复询问已有信息。",
        ),
        "constraint": (
            "区分已确定条件与待确认事项，不承诺无法做到的安排。",
            "客观条件 → 可行时间线 → 待确认事项",
            "明确下一次确认的时间和可行替代安排。",
        ),
    }[kind]


def grounded_rewrite(kind: str, answer: str, items: list[QuestionCoverageItem]) -> str:
    # Extractive sentences preserve negation, quantities and attribution. Missing
    # facts are requested outside the answer, never filled in as first-person claims.
    sentences = list(dict.fromkeys(clauses(answer)))
    if kind == "motivation":

        def rank(sentence: str) -> int:
            if affirmative_excerpt(sentence, r"风险|核验|入职前|入职后|验证|risk|verify"):
                return 3
            if affirmative_excerpt(sentence, r"评估|标准|主要看|criteria"):
                return 2
            if affirmative_excerpt(sentence, r"过去|负责|经验|previous|experience"):
                return 1
            return 0

        sentences.sort(key=rank)
    punctuation = "？" if kind == "candidate_question" else "。"
    draft = (punctuation + "\n").join(sentences) + (punctuation if sentences else "")
    gaps = [item.suggestion for item in items if item.status != "covered"]
    return (
        "基于原话整理的表达稿（保留事实，未补写经历）：\n"
        + draft
        + "\n\n表达顺序："
        + coaching(kind)[1]
        + "；不必套用 STAR。"
        + ("\n仍需你补充：\n" + "\n".join(gaps) if gaps else "")
    )


def organize_evidence_sentences(answer: str, items: list[QuestionCoverageItem]) -> str:
    """Group full source sentences by the evidence contract, without inventing links."""
    sentences = clauses(answer)
    order = (
        ("提供一个真实案例", "明确具体公司/业务场景"),
        ("明确个人职责与关键决策",),
        ("说明备选方案、权衡标准与最终选择", "说明方法或制定过程", "说明执行动作"),
        ("给出结果与验证方式",),
    )

    def rank(sentence: str) -> int:
        for index, names in enumerate(order):
            if any(
                item.requirement in names
                and item.evidence
                and any(
                    part.strip() in sentence for part in item.evidence.split("；") if part.strip()
                )
                for item in items
            ):
                return index
        return len(order)

    # Stable ordering retains local order within each topic. Preserve each full
    # sentence (including qualifications), not a truncated evidence fragment.
    return "。\n".join(sorted(sentences, key=rank)) + ("。" if sentences else "")
