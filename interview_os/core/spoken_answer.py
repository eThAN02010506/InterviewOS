"""Deterministic, source-grounded analysis for interview answers."""

from __future__ import annotations

import re

from interview_os.core.state import (
    AnswerEvaluation,
    QuestionCoverageItem,
    SemanticAnswerStep,
    SpokenAnswerAnalysis,
)

RUBRIC_VERSION = "evidence-v2"
_STRONG_FILLERS = ("嗯", "啊", "呃")
_OBSERVED_DISCOURSE_WORDS = ("就是", "那个", "然后")
_SCORE_FIELDS = ("content", "technical_depth", "structure", "impact")
_METRIC_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|％|人|天|周|月|年|倍|ms|s|sec|seconds?|min|minutes?|"
    r"qps|rps|tps|gb|mb|kb|万元|元|美元|usd|cny|\$|¥)",
    re.IGNORECASE,
)

# Rules are cross-domain and bilingual. Recruitment-specific concepts remain as
# optional labels, while generic analysis/decision/action/result steps support
# engineering, product, sales, operations, and other roles.
_STEP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("业务背景与目标", ("业务", "战略", "发展阶段", "背景", "目标", "context", "goal")),
    ("人才画像", ("画像", "胜任力", "文化契合", "软性能力", "硬性能力")),
    ("目标人才定位与寻访", ("目标人群", "人才地图", "寻访", "sourcing")),
    ("分析与诊断", ("分析", "日志", "定位", "调研", "诊断", "analy", "diagnos", "investigat")),
    (
        "方案比较与决策",
        (
            "对比",
            "比较",
            "权衡",
            "取舍",
            "选择",
            "决定",
            "trade-off",
            "tradeoff",
            "compared",
            "chose",
            "decided",
        ),
    ),
    (
        "执行与推进",
        (
            "执行",
            "推动",
            "实施",
            "修改",
            "上线",
            "灰度",
            "回滚",
            "implemented",
            "launched",
            "rolled back",
            "restored",
        ),
    ),
    ("评估体系", ("评估", "考评", "考核点", "assessment")),
    ("协作与沟通", ("协作", "沟通", "面试官", "利益相关者", "stakeholder", "collaborat")),
    (
        "结果与验证",
        (
            "提升",
            "降低",
            "缩短",
            "增长",
            "达成",
            "结果",
            "压测",
            "p95",
            "improved",
            "reduced",
            "delivered",
            "result",
        ),
    ),
    (
        "项目治理与复盘",
        ("项目制", "回顾", "复盘", "调整", "reflection", "retrospective", "learned"),
    ),
)


def clean_spoken_transcript(text: str) -> tuple[str, dict[str, int], int]:
    """Create a semantic draft without overwriting the original transcript."""
    observed = (*_STRONG_FILLERS, *_OBSERVED_DISCOURSE_WORDS)
    filler_counts = {word: text.count(word) for word in observed if text.count(word)}
    cleaned = text
    for filler in _STRONG_FILLERS:
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
    cleaned = re.sub(r"\s*([,.!?;，。！？；：])\s*", r"\1", cleaned)
    cleaned = re.sub(r"([,.!?;，。！？；：])\1+", r"\1", cleaned).strip(" ,，")
    repetition_patterns = (
        r"首先.{0,12}首先",
        r"(?:此此外|面面试|寻寻访|考考评|团面面试)",
        r"([\u4e00-\u9fff]{2,6})[，、\s]+\1",
    )
    repetition_count = sum(len(re.findall(pattern, text)) for pattern in repetition_patterns)
    return cleaned, filler_counts, repetition_count


def _sentences(text: str) -> list[str]:
    return [
        item.strip(" ,，。")
        for item in re.split(r"[,.!?;，。！？；]", text)
        if item.strip(" ,，。")
    ]


def _excerpt(sentences: list[str], keywords: tuple[str, ...]) -> str:
    for sentence in sentences:
        folded = sentence.casefold()
        if any(keyword.casefold() in folded for keyword in keywords):
            return sentence[:200]
    return ""


def _answer_type(question: str) -> str:
    folded = question.casefold()
    if re.search(r"请讲|举例|一次|经历|期间|你是如何", question) or re.search(
        r"\b(?:tell me about a time|describe a time|give (?:me )?an example|experience where|when you)\b",
        folded,
    ):
        return "behavioral_example"
    if re.search(r"假如|如果|情景|会怎么", question) or re.search(
        r"\b(?:what would you|how would you|suppose|imagine|if you)\b", folded
    ):
        return "situational"
    if re.search(r"为什么|动机|选择", question) or re.search(
        r"\b(?:why do you|motivat|why this|why are you)\b", folded
    ):
        return "motivation"
    if re.search(r"如何|怎么|流程|方法|设计|步骤", question) or re.search(
        r"\b(?:how do you|how did you|approach|process|method|design)\b", folded
    ):
        return "methodology"
    return "general"


def _explicit_context(question: str) -> str:
    chinese = re.search(r"在\s*([A-Za-z][A-Za-z0-9&. -]{1,40}?)\s*(?:期间|公司)", question)
    if chinese:
        return chinese.group(1).strip()
    english = re.search(r"\b(?:at|for|with)\s+([A-Z][A-Za-z0-9&.-]{1,39})\b", question)
    return english.group(1).strip() if english else ""


def _semantic_steps(answer: str) -> list[SemanticAnswerStep]:
    sentences = _sentences(answer)
    found: list[tuple[int, SemanticAnswerStep]] = []
    used_evidence: set[str] = set()
    for label, keywords in _STEP_RULES:
        evidence = _excerpt(sentences, keywords)
        if evidence and evidence not in used_evidence:
            used_evidence.add(evidence)
            found.append(
                (
                    answer.casefold().find(evidence.casefold()),
                    SemanticAnswerStep(label=label, evidence=evidence),
                )
            )
    found.sort(key=lambda item: item[0] if item[0] >= 0 else len(answer))
    return [item for _, item in found[:8]]


def _coverage(
    question: str, answer: str, steps: list[SemanticAnswerStep]
) -> list[QuestionCoverageItem]:
    sentences = _sentences(answer)
    requirements: list[QuestionCoverageItem] = []

    def add(name: str, status: str, evidence: str, suggestion: str) -> None:
        requirements.append(
            QuestionCoverageItem(
                requirement=name, status=status, evidence=evidence, suggestion=suggestion
            )
        )

    question_type = _answer_type(question)
    folded_question = question.casefold()
    context_name = _explicit_context(question)
    context_evidence = context_name if context_name.casefold() in answer.casefold() else ""
    if context_name or "期间" in question:
        add(
            "明确具体公司/业务场景",
            "covered" if context_evidence else "missing",
            context_evidence,
            "说明当时的公司、业务阶段、具体任务和目标。",
        )

    execution = _excerpt(
        sentences,
        (
            "分析",
            "定位",
            "修改",
            "实施",
            "推动",
            "寻访",
            "筛选",
            "评估",
            "上线",
            "回滚",
            "analy",
            "diagnos",
            "implemented",
            "launched",
            "restored",
            "rolled back",
        ),
    )
    personal = _excerpt(
        sentences,
        (
            "我负责",
            "我主导",
            "我先",
            "我通过",
            "我分析",
            "我定位",
            "我修改",
            "我制定",
            "我决定",
            "我选择",
            "我推动",
            "我建立",
            "我根据",
            "我调整",
            "我对比",
            "我比较",
            "i led",
            "i owned",
            "i decided",
            "i chose",
            "i compared",
            "i analyzed",
            "i diagnosed",
            "i designed",
            "i implemented",
        ),
    )
    outcome = _excerpt(
        sentences,
        (
            "提升",
            "降低",
            "缩短",
            "增长",
            "达成",
            "恢复",
            "交付",
            "p95",
            "improved",
            "reduced",
            "restored",
            "returned",
            "resolved",
            "delivered",
            "increased",
        ),
    )
    metric = _METRIC_PATTERN.search(answer)

    if question_type == "behavioral_example":
        explicit_case = bool(
            re.search(r"当时|项目中|事故中|那次|最终入职", answer)
            or re.search(
                r"\b(?:during|in that project|at the time|incident)\b",
                answer.casefold(),
            )
        )
        case_context = bool(personal) or explicit_case
        # A first-person action can still be a generic method. Require a real
        # context marker or verifiable metric before calling it a specific case.
        case_specificity = bool(metric or context_evidence or explicit_case)
        status = (
            "covered"
            if case_context and case_specificity
            else "partial"
            if case_context
            else "missing"
        )
        add(
            "提供一个真实案例",
            status,
            personal or execution if case_context else "",
            "选一个真实案例，讲清背景、约束、行动和结果。",
        )

    asks_method = question_type in {"methodology", "situational"} or (
        question_type == "behavioral_example"
        and bool(re.search(r"如何|怎么|制定|方法|how|approach|process", folded_question))
    )
    if asks_method:
        method_evidence = steps[0].evidence if len(steps) >= 2 else execution
        add(
            "说明方法或制定过程",
            "covered" if len(steps) >= 2 else "partial" if method_evidence else "missing",
            method_evidence,
            "把方法压缩为三到五个按顺序执行的步骤。",
        )

    asks_execution = question_type == "behavioral_example" or bool(
        re.search(r"执行|推进|落地|解决|implement|execute|deliver|resolve", folded_question)
    )
    if asks_execution:
        add(
            "说明执行动作",
            "covered" if execution else "missing",
            execution,
            "说明你亲自推进的动作、参与者和交付物。",
        )

    asks_personal = question_type == "behavioral_example" or bool(
        re.search(
            r"你的职责|你做了什么|你如何|你是如何|your role|what did you|how did you",
            folded_question,
        )
    )
    if asks_personal:
        add(
            "明确个人职责与关键决策",
            "covered" if personal else "missing",
            personal,
            "用“我负责/我决定/我推动”区分个人贡献与团队行动。",
        )

    asks_result = (
        question_type == "behavioral_example"
        or bool(re.search(r"结果|效果|影响|收益|result|impact|outcome", folded_question))
        or bool(outcome)
    )
    if asks_result:
        add(
            "给出结果与验证方式",
            "covered" if outcome and metric else "partial" if outcome else "missing",
            outcome,
            "补充已核验结果、指标口径和时间范围；无数字时说明可观察变化。",
        )

    if question_type == "motivation":
        reason = _excerpt(sentences, ("因为", "吸引", "匹配", "希望", "because", "motiv", "align"))
        add(
            "说明动机与匹配关系",
            "covered" if reason else "missing",
            reason,
            "连接公司/岗位特点、你的经历和下一阶段目标。",
        )
    return requirements


def analyze_spoken_answer(
    question: str,
    answer: str,
    *,
    answer_modality: str = "typed",
) -> SpokenAnswerAnalysis:
    cleaned, filler_counts, repetition_count = clean_spoken_transcript(answer)
    steps = _semantic_steps(cleaned)
    return SpokenAnswerAnalysis(
        raw_transcript=answer,
        cleaned_transcript=cleaned,
        answer_modality=answer_modality,
        rubric_version=RUBRIC_VERSION,
        answer_type=_answer_type(question),
        filler_counts=filler_counts,
        repetition_count=repetition_count,
        semantic_steps=steps,
        question_coverage=_coverage(question, cleaned, steps),
    )


def calibrate_evaluation(evaluation: AnswerEvaluation, analysis: SpokenAnswerAnalysis) -> None:
    """Apply downward-only evidence caps and preserve the model's original scores."""
    statuses = {item.requirement: item.status for item in analysis.question_coverage}
    notes: list[str] = []
    analysis.pre_calibration_scores = {
        field: round(float(getattr(evaluation, field)), 4) for field in _SCORE_FIELDS
    }

    def cap(field: str, maximum: float, reason: str) -> None:
        current = float(getattr(evaluation, field))
        if current > maximum:
            setattr(evaluation, field, maximum)
            notes.append(f"{field} {current:.2f}→{maximum:.2f}：{reason}")

    if (
        analysis.answer_type == "behavioral_example"
        and statuses.get("提供一个真实案例") != "covered"
    ):
        cap("content", 0.65, "问题要求真实案例，但回答缺少具体案例证据")
    if statuses.get("明确个人职责与关键决策") == "missing":
        cap("technical_depth", 0.60, "缺少可归属于本人的关键决策")
    # Discourse connectors such as “然后” are observable but ambiguous. Only
    # strong ASR fillers and repeated repair patterns may cap spoken structure.
    if analysis.answer_modality in {"asr", "live_asr"}:
        strong_fillers = sum(analysis.filler_counts.get(word, 0) for word in _STRONG_FILLERS)
        raw_length = max(1, len(analysis.raw_transcript))
        if strong_fillers / raw_length >= 0.03 or analysis.repetition_count >= 2:
            cap("structure", 0.55, "明显填充词或重复修正影响表达清晰度")
    if statuses.get("给出结果与验证方式") == "missing":
        cap("impact", 0.35, "没有实际结果或验证方式")
    elif statuses.get("给出结果与验证方式") == "partial":
        cap("impact", 0.55, "结果缺少可核验指标")
    analysis.calibration_notes = notes
    evaluation.spoken_analysis = analysis
