"""Deterministic, behavior-anchored feedback for scored interview answers."""

from __future__ import annotations

import re

from interview_os.core.state import AnswerEvaluation, DimensionFeedback

DIMENSION_LABELS = {
    "content": "岗位相关证据",
    "technical_depth": "决策与专业深度",
    "structure": "表达结构",
    "impact": "结果与复盘",
}


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
    gap = evaluation.missing_signals[0] if evaluation.missing_signals else "缺少第二个独立证据点"
    target = f"围绕“{question[:60]}”" if question.strip() else "围绕当前问题"
    competency_text = f"“{competency}”" if competency.strip() else "目标能力"
    analysis = evaluation.spoken_analysis
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
            f"{target}优先补齐：{missing[0].requirement if missing else gap}。",
        ),
        (
            "technical_depth",
            evaluation.technical_depth,
            (
                f"个人职责与决策：{personal.status}；证据：{personal.evidence or '未找到明确的“我负责/我决定”表述'}。"
                if personal
                else f"识别到 {len(action_markers)} 个行动、决策或权衡表达。"
            ),
            "补充你亲自做出的关键决定、至少一个备选方案，以及选择当前方案的约束和理由。",
        ),
        (
            "structure",
            evaluation.structure,
            f"提取到 {len(analysis.semantic_steps)} 个语义步骤；填充词约 {filler_total} 处，重复修正 {analysis.repetition_count} 处。",
            "按“情境/目标 → 你的任务 → 关键行动 → 结果 → 复盘”重排，每段只承担一个信息功能。",
        ),
        (
            "impact",
            evaluation.impact,
            (
                f"结果覆盖：{outcome.status}；证据：{outcome.evidence or '未找到实际结果'}；"
                f"数值线索 {len(metrics)} 个。"
                if outcome
                else f"识别到 {len(impact_markers)} 个结果或复盘表达、{len(metrics)} 个数值线索。"
            ),
            "补充已核验的结果、指标口径和时间范围；没有数字时说明可观察变化及你从中学到什么。",
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
    coverage_gaps = sorted(
        (
            item
            for item in evaluation.spoken_analysis.question_coverage
            if item.status in {"missing", "partial"}
        ),
        key=lambda item: 0 if item.status == "missing" else 1,
    )
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
    evaluation.feedback = list(dict.fromkeys([*preserved, *priorities]))[:5]
