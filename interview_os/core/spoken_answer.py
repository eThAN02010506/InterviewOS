"""Deterministic, source-grounded analysis for spoken interview answers."""

from __future__ import annotations

import re

from interview_os.core.state import (
    AnswerEvaluation,
    QuestionCoverageItem,
    SemanticAnswerStep,
    SpokenAnswerAnalysis,
)

_FILLERS = ("嗯", "啊", "呃", "就是", "那个", "然后")
_STEP_RULES = (
    ("业务背景与目标", ("业务", "战略", "发展阶段", "目标")),
    ("人才画像", ("画像", "背景", "能力", "文化", "软性", "硬性")),
    ("目标人才定位与寻访", ("目标人群", "市场", "行业", "企业", "寻访", "人才地图")),
    ("评估体系", ("评估", "考评", "考核点", "文化契合")),
    ("面试与决策机制", ("面试官", "反馈", "会议", "决定", "筛选")),
    ("项目治理与复盘", ("项目制", "阶段", "回顾", "复盘", "调整")),
    ("结果与业务影响", ("提升", "降低", "缩短", "增长", "达成", "结果")),
)


def clean_spoken_transcript(text: str) -> tuple[str, dict[str, int], int]:
    """Remove non-semantic fillers while preserving the original transcript separately."""
    filler_counts = {word: text.count(word) for word in _FILLERS if text.count(word)}
    cleaned = text
    for filler in ("嗯", "啊", "呃"):
        cleaned = re.sub(re.escape(filler) + r"+", "", cleaned)
    for spoken_repeat, repaired in {
        "此此外": "此外",
        "面面试": "面试",
        "寻寻访": "寻访",
        "考考评": "考评",
        "团面面试官": "团面试官",
    }.items():
        cleaned = cleaned.replace(spoken_repeat, repaired)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*([，。！？；：])\s*", r"\1", cleaned)
    cleaned = re.sub(r"([，。！？；：])\1+", r"\1", cleaned).strip(" ，")
    repetition_patterns = (
        r"首先.{0,12}首先",
        r"(?:此此外|面面试|寻寻访|考考评|团面面试)",
        r"([\u4e00-\u9fff]{2,6})[，、\s]+\1",
    )
    repetition_count = sum(len(re.findall(pattern, text)) for pattern in repetition_patterns)
    return cleaned, filler_counts, repetition_count


def _sentences(text: str) -> list[str]:
    return [item.strip(" ，。") for item in re.split(r"[。！？；]", text) if item.strip(" ，。")]


def _excerpt(sentences: list[str], keywords: tuple[str, ...]) -> str:
    for sentence in sentences:
        if any(keyword in sentence for keyword in keywords):
            return sentence[:160]
    return ""


def _answer_type(question: str) -> str:
    if re.search(r"请讲|举例|一次|经历|期间|你是如何", question):
        return "behavioral_example"
    if re.search(r"假如|如果|场景|会怎么", question):
        return "situational"
    if re.search(r"为什么|动机|选择", question):
        return "motivation"
    if re.search(r"如何|怎么|流程|方法", question):
        return "methodology"
    return "general"


def _coverage(question: str, answer: str, steps: list[SemanticAnswerStep]) -> list[QuestionCoverageItem]:
    sentences = _sentences(answer)
    requirements: list[tuple[str, str, str, str]] = []

    def add(name: str, status: str, evidence: str, suggestion: str) -> None:
        if not any(item[0] == name for item in requirements):
            requirements.append((name, status, evidence, suggestion))

    question_type = _answer_type(question)
    named_context = [
        name
        for name in re.findall(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b", question)
        if name.casefold()
        not in {"tell", "describe", "how", "what", "why", "when", "please", "could", "give"}
    ]
    context_evidence = next((name for name in named_context if name.casefold() in answer.casefold()), "")
    if named_context or "期间" in question:
        add(
            "明确具体公司/业务场景",
            "covered" if context_evidence else "missing",
            context_evidence,
            "说明当时的公司、业务阶段、具体岗位和招聘目标。",
        )
    if question_type == "behavioral_example":
        case_context = bool(
            re.search(r"当时|我负责|我主导|我推动|项目中|最终入职", answer)
        )
        case_specificity = bool(
            re.search(r"\d+(?:[.,]\d+)?\s*(?:%|％|人|天|周|月|年|倍)", answer)
            or context_evidence
        )
        case_status = "covered" if case_context and case_specificity else "partial" if case_context else "missing"
        add(
            "提供一个真实案例",
            case_status,
            _excerpt(sentences, ("当时", "我负责", "我主导", "项目", "入职"))
            if case_context
            else "",
            "选一个真实高管岗位，讲清背景、约束、行动和结果。",
        )
    if re.search(r"如何|怎么|制定|方法|流程", question):
        method_evidence = steps[0].evidence if len(steps) >= 2 else ""
        add(
            "说明方法或制定过程",
            "covered" if len(steps) >= 2 else "partial",
            method_evidence,
            "把方法压缩为三到五个按顺序执行的步骤。",
        )
    if "执行" in question or question_type in {"behavioral_example", "situational"}:
        execution = _excerpt(sentences, ("寻访", "筛选", "评估", "面试", "项目制", "调整"))
        add(
            "说明执行动作",
            "covered" if execution else "missing",
            execution,
            "说明你亲自推进的动作、参与者和交付物。",
        )
    personal = _excerpt(
        sentences,
        (
            "我负责",
            "我主导",
            "我制定",
            "我决定",
            "我选择",
            "我推动",
            "我建立",
            "我根据",
            "我调整",
            "我对比",
            "我比较",
            "I decided",
            "I chose",
            "I compared",
            "I led",
            "I designed",
            "I implemented",
            "I owned",
        ),
    )
    add(
        "明确个人职责与关键决策",
        "covered" if personal else "missing",
        personal,
        "用“我负责/我决定/我推动”区分个人贡献与通用流程。",
    )
    outcome = _excerpt(
        sentences,
        (
            "提升",
            "降低",
            "缩短",
            "增长",
            "最终实现",
            "最终完成",
            "最终支持",
            "交付",
            "实现了",
            "完成了",
            "improved",
            "reduced",
            "delivered",
        ),
    )
    metric = re.search(r"\d+(?:[.,]\d+)?\s*(?:%|％|人|天|周|月|年|倍)", answer)
    add(
        "给出结果与验证方式",
        "covered" if outcome and metric else "partial" if outcome else "missing",
        outcome,
        "补充已核验结果、指标口径和时间范围；无数字时说明可观察变化。",
    )
    return [
        QuestionCoverageItem(
            requirement=name,
            status=status,
            evidence=evidence,
            suggestion=suggestion,
        )
        for name, status, evidence, suggestion in requirements
    ]


def analyze_spoken_answer(question: str, answer: str) -> SpokenAnswerAnalysis:
    cleaned, filler_counts, repetition_count = clean_spoken_transcript(answer)
    sentences = _sentences(cleaned)
    steps = []
    for label, keywords in _STEP_RULES:
        evidence = _excerpt(sentences, keywords)
        if evidence:
            steps.append(SemanticAnswerStep(label=label, evidence=evidence))
    coverage = _coverage(question, cleaned, steps)
    return SpokenAnswerAnalysis(
        raw_transcript=answer,
        cleaned_transcript=cleaned,
        answer_type=_answer_type(question),
        filler_counts=filler_counts,
        repetition_count=repetition_count,
        semantic_steps=steps,
        question_coverage=coverage,
    )


def calibrate_evaluation(evaluation: AnswerEvaluation, analysis: SpokenAnswerAnalysis) -> None:
    """Apply evidence caps so a score cannot exceed what the answer demonstrates."""
    statuses = {item.requirement: item.status for item in analysis.question_coverage}
    notes: list[str] = []

    def cap(field: str, maximum: float, reason: str) -> None:
        current = float(getattr(evaluation, field))
        if current > maximum:
            setattr(evaluation, field, maximum)
            notes.append(f"{field} {current:.2f}→{maximum:.2f}：{reason}")

    if analysis.answer_type == "behavioral_example" and statuses.get("提供一个真实案例") != "covered":
        cap("content", 0.65, "问题要求真实案例，但回答主要是通用方法")
    if statuses.get("明确个人职责与关键决策") != "covered":
        cap("technical_depth", 0.60, "缺少可归属于本人的关键决策")
    filler_total = sum(analysis.filler_counts.values())
    raw_length = max(1, len(analysis.raw_transcript))
    if filler_total / raw_length >= 0.03 or analysis.repetition_count >= 2:
        cap("structure", 0.55, "填充词或重复修正影响表达清晰度")
    if statuses.get("给出结果与验证方式") == "missing":
        cap("impact", 0.35, "没有实际结果或验证方式")
    elif statuses.get("给出结果与验证方式") == "partial":
        cap("impact", 0.55, "结果缺少可核验指标")
    analysis.calibration_notes = notes
    evaluation.spoken_analysis = analysis
