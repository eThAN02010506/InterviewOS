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
    structure_markers = re.findall(
        r"(?:首先|其次|然后|最后|最终|背景|目标|挑战|行动|结果|复盘|因为|因此)|"
        r"\b(?:first|then|finally|situation|task|action|result|because)\b",
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
    observed = "；".join(evaluation.observed_signals[:2])
    gap = evaluation.missing_signals[0] if evaluation.missing_signals else "缺少第二个独立证据点"
    target = f"围绕“{question[:60]}”" if question.strip() else "围绕当前问题"
    competency_text = f"“{competency}”" if competency.strip() else "目标能力"

    specs = [
        (
            "content",
            evaluation.content,
            (
                f"回答约 {len(clean)} 字；已识别证据：{observed}。"
                if observed
                else f"回答约 {len(clean)} 字，尚未识别出可直接支持{competency_text}的具体行为证据。"
            ),
            f"{target}补齐一个可核验案例，并优先解决：{gap}。",
        ),
        (
            "technical_depth",
            evaluation.technical_depth,
            f"识别到 {len(action_markers)} 个行动、决策或权衡表达。",
            "补充你亲自做出的关键决定、至少一个备选方案，以及选择当前方案的约束和理由。",
        ),
        (
            "structure",
            evaluation.structure,
            f"识别到 {len(structure_markers)} 个结构连接或 STAR(R) 阶段标记。",
            "按“情境/目标 → 你的任务 → 关键行动 → 结果 → 复盘”重排，每段只承担一个信息功能。",
        ),
        (
            "impact",
            evaluation.impact,
            f"识别到 {len(impact_markers)} 个结果或复盘表达、{len(metrics)} 个数值线索。",
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
    weakest_first = sorted(evaluation.dimension_feedback, key=lambda item: item.score)
    priorities = [
        f"{DIMENSION_LABELS[item.dimension]}（{round(item.score * 100)}）：{item.suggestion}"
        for item in weakest_first[:3]
    ]
    evaluation.feedback = list(dict.fromkeys([*preserved, *priorities]))[:5]
