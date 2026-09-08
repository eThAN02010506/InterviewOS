"""Deterministic, behavior-anchored feedback for scored interview answers."""

from __future__ import annotations

import re

from interview_os.core import conversation_contracts
from interview_os.core.state import AnswerEvaluation, DimensionFeedback

DIMENSION_LABELS = {
    "content": "岗位相关证据",
    "technical_depth": "决策与专业深度",
    "structure": "表达结构",
    "impact": "结果与复盘",
}


def _grounded_coaching(evaluation: AnswerEvaluation, answer: str) -> list[dict[str, str]]:
    """Retain bounded model advice, not new facts, with auditable anchors.

    Exact quotes validate attribution, not the truth of a model interpretation.
    Keep these notes separate from scores, coverage and hiring evidence.
    """
    requirements = {item.requirement for item in evaluation.spoken_analysis.question_coverage}
    accepted: list[dict[str, str]] = []
    seen: set[str] = set()
    fields = ("dimension", "requirement", "quote", "interpretation", "action", "next_question")
    for raw in evaluation.coaching_details[:6]:
        if not isinstance(raw, dict) or any(not isinstance(raw.get(key), str) for key in fields):
            continue
        item = {key: raw[key].strip() for key in fields}
        if any(not value or len(value) > 500 for value in item.values()):
            continue
        if item["dimension"] not in DIMENSION_LABELS or item["requirement"] not in requirements:
            continue
        if len(item["quote"]) < 4 or item["quote"] not in answer or item["dimension"] in seen:
            continue
        seen.add(item["dimension"])
        accepted.append(item)
        if len(accepted) == 3:
            break
    # Do not persist rejected/hallucinated quotes in the API response or state.
    evaluation.coaching_details = accepted
    return accepted


def _level(score: float) -> str:
    if score >= 0.8:
        return "表现突出"
    if score >= 0.65:
        return "达到要求"
    if score >= 0.5:
        return "证据有限"
    return "需要补强"


def build_dimension_feedback(
    evaluation: AnswerEvaluation,
    answer: str,
    *,
    question: str = "",
    competency: str = "",
) -> list[DimensionFeedback]:
    """Explain each score using only observable features of the submitted answer."""
    clean = re.sub(r"\s+", " ", answer).strip()
    action_markers = re.findall(
        r"我(?:先|再|通过|建立|制定|设计|推动|负责|主导|决定|选择)|"
        r"(?:分析|监控|复盘|调整|协作|解决|实施|优化|权衡|取舍)|"
        r"\b(?:decided|designed|implemented|analyzed|led|owned|trade-?off)\b",
        clean,
        re.IGNORECASE,
    )
    impact_markers = re.findall(
        r"(?:最终|结果|支持|获得|交付|改善|提升|降低|缩短|增长|影响|复盘|学到)|"
        r"\b(?:result|improved|reduced|increased|delivered|learned|reflection)\b",
        clean,
        re.IGNORECASE,
    )
    metrics = re.findall(
        r"(?<![A-Za-z0-9_])\d+(?:[.,，]\d+)?\s*(?:%|％|人|天|周|月|年|倍|ms|s)?",
        clean,
        re.IGNORECASE,
    )
    target = f"围绕“{question[:60]}”" if question.strip() else "围绕当前问题"
    competency_text = f"“{competency}”" if competency.strip() else "目标能力"
    analysis = evaluation.spoken_analysis
    if analysis.answer_type in conversation_contracts.CONVERSATIONAL_TYPES:
        depth, structure, impact = conversation_contracts.coaching(analysis.answer_type)
        items = analysis.question_coverage
        gaps = [item for item in items if item.status != "covered"]
        quoted = "；".join(f"{item.requirement}：{item.evidence}" for item in items if item.evidence)
        evidence = quoted or "尚未找到与本题要求对应的明确原话。"
        reasoning = conversation_contracts.affirmative_excerpt(
            answer, r"依据|标准|因为|优先|看|criteria|because|basis"
        )
        validation = conversation_contracts.affirmative_excerpt(
            answer, r"核验|确认|如果|协商|验证|时间|verify|confirm|negotiat|timeline"
        )
        evidence_by_dimension = [
            evidence,
            f"判断依据原话：{reasoning}" if reasoning else "尚未提取到明确判断依据；不能用行动关键词数量代替决策质量。",
            f"回答包含 {len(conversation_contracts.clauses(answer))} 个完整语句；按本题沟通顺序审阅，不要求项目 STAR。",
            f"条件或后续安排原话：{validation}" if validation else "尚未提取到后续确认安排；本题不以项目成果或数字数量计分。",
        ]
        suggestions = [
            gaps[0].suggestion if gaps else "保留直接回应本题的原话，压缩重复说明。",
            depth, f"按“{structure}”组织，不必套用 STAR。", impact,
        ]
        return [DimensionFeedback(
            dimension=field, score=getattr(evaluation, field),
            level=_level(getattr(evaluation, field)), evidence=dimension_evidence,
            suggestion=suggestion,
        ) for field, suggestion, dimension_evidence in zip(DIMENSION_LABELS, suggestions, evidence_by_dimension, strict=True)]
    covered = [item for item in analysis.question_coverage if item.status == "covered"]
    missing = [item for item in analysis.question_coverage if item.status == "missing"]
    step_labels = "、".join(item.label for item in analysis.semantic_steps[:6]) or "尚未提取清晰步骤"
    filler_total = sum(analysis.filler_counts.values())
    personal = next(
        (
            item
            for item in analysis.question_coverage
            if item.requirement == "明确个人职责与关键决策"
        ),
        None,
    )
    outcome = next(
        (item for item in analysis.question_coverage if item.requirement == "给出结果与验证方式"),
        None,
    )
    tradeoff = next(
        (
            item
            for item in analysis.question_coverage
            if item.requirement == "说明备选方案、权衡标准与最终选择"
        ),
        None,
    )
    method = next(
        (item for item in analysis.question_coverage if item.requirement == "说明方法或制定过程"),
        None,
    )
    metric_requirement = next(
        (
            item
            for item in analysis.question_coverage
            if item.requirement == "说明指标、口径与决策关系"
        ),
        None,
    )
    contract_complete = bool(analysis.question_coverage) and not any(
        item.status in {"missing", "partial"} for item in analysis.question_coverage
    )
    grounded_tradeoff = bool(
        (tradeoff and tradeoff.status == "covered")
        or (
            re.search(r"比较|对比|备选|方案|alternative|compared", clean, re.IGNORECASE)
            and re.search(
                r"我.{0,160}(?:决定|选择)|最终选择|i .{0,160}(?:decided|chose)",
                clean,
                re.IGNORECASE,
            )
            and re.search(r"依据|考虑|因为|约束|成本|风险|based on", clean, re.IGNORECASE)
        )
    )

    specs = [
        (
            "content",
            evaluation.content,
            (
                f"问题要求覆盖 {len(analysis.question_coverage)} 项，已明确覆盖 {len(covered)} 项；"
                f"主要步骤：{step_labels}。"
                if clean
                else f"回答为空，尚未提供可直接支持{competency_text}的行为证据。"
            ),
            (
                f"{target}优先补齐：{missing[0].requirement}。"
                if missing
                else "回答已直接覆盖题目核心；进一步提升时可压缩次要背景，保留最有区分度的证据。"
            ),
        ),
        (
            "technical_depth",
            evaluation.technical_depth,
            (
                f"个人职责与决策：{personal.status}；证据：{personal.evidence or '未找到明确的“我负责/我决定”表述'}。"
                if personal
                else f"识别到 {len(action_markers)} 个行动、决策或权衡表达。"
            ),
            (
                "已说明备选方案与选择依据；进一步提升时可用一句话概括被放弃方案的适用边界。"
                if grounded_tradeoff
                else "指标、口径与触发动作已经对应；进一步提升时可简述阈值来源或误判成本。"
                if metric_requirement and metric_requirement.status == "covered"
                else "补充你亲自做出的关键决定、至少一个备选方案，以及选择当前方案的约束和理由。"
            ),
        ),
        (
            "structure",
            evaluation.structure,
            f"提取到 {len(analysis.semantic_steps)} 个语义步骤；填充词约 {filler_total} 处，重复修正 {analysis.repetition_count} 处。",
            (
                "方法顺序已经清楚；口头表达时可把每一步压缩为“动作＋判断依据”。"
                if method and method.status == "covered" and len(analysis.semantic_steps) >= 3
                else "回答结构与本题匹配；保持“指标 → 口径 → 阈值触发动作”的短链路即可。"
                if contract_complete and metric_requirement
                else "按“情境/目标 → 你的任务 → 关键行动 → 结果 → 复盘”重排，每段只承担一个信息功能。"
            ),
        ),
        (
            "impact",
            evaluation.impact,
            (
                f"结果覆盖：{outcome.status}；证据：{outcome.evidence or '未找到实际结果'}；"
                f"数值线索 {len(metrics)} 个。"
                if outcome
                else "本题未要求项目结果；该维度只观察回答是否说明指标会触发什么动作。"
                if metric_requirement
                else f"识别到 {len(impact_markers)} 个结果或复盘表达、{len(metrics)} 个数值线索。"
            ),
            (
                "结果证据已经明确；进一步提升时区分预测值、基线、目标值和实际结果，并说明观察周期。"
                if outcome and outcome.status == "covered"
                else "本题无需补讲完整项目结果；如要深化，只需说明这些阈值会触发扩容、降级或回滚中的哪一项。"
                if metric_requirement and outcome is None
                else "补充已核验的结果、指标口径和时间范围；没有数字时说明可观察变化及你从中学到什么。"
            ),
        ),
    ]
    return [
        DimensionFeedback(
            dimension=dimension,
            score=round(score, 4),
            level=_level(score),
            evidence=evidence,
            suggestion=suggestion,
        )
        for dimension, score, evidence, suggestion in specs
    ]


def apply_specific_feedback(
    evaluation: AnswerEvaluation,
    answer: str,
    *,
    question: str = "",
    competency: str = "",
) -> None:
    """Replace generic coaching prose with ranked, grounded next actions."""
    preserved = [
        item
        for item in evaluation.feedback
        if any(marker in item for marker in ("人工复核", "模型草稿", "规则评分"))
    ]
    evaluation.dimension_feedback = build_dimension_feedback(
        evaluation,
        answer,
        question=question,
        competency=competency,
    )
    details = _grounded_coaching(evaluation, answer)
    for detail in details:
        dimension = next(item for item in evaluation.dimension_feedback if item.dimension == detail["dimension"])
        dimension.evidence = f"你的原话：“{detail['quote']}”"
        dimension.suggestion = (
            f"AI 教练解读（不改变评分）：{detail['interpretation']}\n"
            f"具体修改：{detail['action']}\n练习追问：{detail['next_question']}"
        )
    if details:
        evaluation.observed_signals = [
            f"{item.requirement}：{item.evidence[:120]}"
            for item in evaluation.spoken_analysis.question_coverage
            if item.status == "covered" and item.evidence
        ][:4]
        evaluation.feedback = [*preserved, *[
            f"针对「{item['requirement']}」，你说：“{item['quote']}”。\n"
            f"AI 教练解读：{item['interpretation']}\n"
            f"具体修改：{item['action']}\n练习追问：{item['next_question']}"
            for item in details
        ]][:5]
        return
    coverage_gaps = sorted(
        (
            item
            for item in evaluation.spoken_analysis.question_coverage
            if item.status in {"missing", "partial"}
        ),
        key=lambda item: 0 if item.status == "missing" else 1,
    )
    covered_items = [
        item
        for item in evaluation.spoken_analysis.question_coverage
        if item.status == "covered" and item.evidence
    ]
    evaluation.observed_signals = [
        f"{item.requirement}：{item.evidence[:120]}" for item in covered_items[:4]
    ]
    priorities = []
    for item in coverage_gaps[:3]:
        observed = f"回答中只找到“{item.evidence[:80]}”" if item.evidence else "回答中没有找到对应内容"
        priorities.append(
            f"本题要求「{item.requirement}」：{observed}；重答时{item.suggestion}"
        )
    if len(priorities) < 3:
        weakest_first = sorted(evaluation.dimension_feedback, key=lambda item: item.score)
        priorities.extend(
            f"{DIMENSION_LABELS[item.dimension]}（{round(item.score * 100)}）：{item.suggestion}"
            for item in weakest_first
            if item.suggestion not in " ".join(priorities)
        )
    if not priorities:
        priorities.append("本题要求均已覆盖；保持事实准确，并继续压缩次要背景。")
    evaluation.feedback = list(dict.fromkeys([*preserved, *priorities]))[:5]
