"""Deterministic, source-grounded analysis for interview answers."""

from __future__ import annotations

import re

from interview_os.core.state import (
    AnswerEvaluation,
    QuestionCoverageItem,
    SemanticAnswerStep,
    SpokenAnswerAnalysis,
)

RUBRIC_VERSION = "evidence-v3"
_STRONG_FILLERS = ("嗯", "啊", "呃")
_OBSERVED_DISCOURSE_WORDS = ("就是", "那个", "然后")
_SCORE_FIELDS = ("content", "technical_depth", "structure", "impact")
_METRIC_PATTERN = re.compile(
    r"(?:"
    r"\d+(?:[.,]\d+)?\s*(?:%|％|人|天|周|月|年|倍|ms|s|sec|seconds?|min|minutes?|"
    r"qps|rps|tps|gb|mb|kb|万元|元|美元|usd|cny|\$|¥)"
    r"|百分之\s*[零〇一二两三四五六七八九十百千万\d.]+"
    r"|[零〇一二两三四五六七八九十百千万]+(?:个)?(?:人|天|周|月|年|倍|万元|元|美元)"
    r")",
    re.IGNORECASE,
)
_NAMED_METRIC_PATTERN = re.compile(
    r"(?:p\d{2}|qps|rps|tps|cpu|内存|连接池|命中率|错误率|超时率|通过率|转化率|"
    r"接受率|满意度|留存率|续费率|使用频率|积压|吞吐|延迟|周期|成本|"
    r"不一致率|完整率|安全余量)",
    re.IGNORECASE,
)
_FOCUS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("容量规划", ("容量", "qps", "吞吐", "扩容", "资源水位", "安全余量")),
    ("稳定性", ("稳定性", "故障", "错误率", "延迟", "可观测", "恢复", "降级")),
    ("技术权衡", ("权衡", "取舍", "备选", "方案", "选择", "决策", "决定", "约束")),
    (
        "指标",
        (
            "p95", "p99", "qps", "cpu", "指标", "数据", "crm", "错误率",
            "转化率", "留存率", "续费率", "使用频率", "吞吐",
        ),
    ),
    (
        "产品与客户结果",
        ("产品", "客户", "留存", "续费", "路线图", "商业化", "增长", "订阅"),
    ),
    ("团队管理", ("团队", "辅导", "管理", "一对一", "晋升", "培养")),
    ("招聘", ("招聘", "人才", "候选人", "岗位画像", "寻访", "面试")),
    (
        "跨团队协作",
        (
            "跨团队", "跨部门", "协作", "利益相关者", "业务团队",
            "业务", "研发", "运维", "分歧", "诉求", "达成一致",
        ),
    ),
)

_SUPPORTING_FOCUS_GROUPS = {"技术权衡", "指标"}


def _question_requests_focus_group(
    label: str, terms: tuple[str, ...], question: str
) -> bool:
    folded = question.casefold()
    if label == "团队管理":
        # “带领团队完成容量规划” describes ownership of the technical work;
        # it is not by itself a request for a people-management case.
        return bool(
            re.search(
                r"团队(?:管理|建设|成长|绩效|分工|冲突)|"
                r"(?:辅导|培养|晋升|一对一|授权|人员管理|领导力)",
                folded,
            )
        )
    if label == "跨团队协作":
        if any(
            marker in folded
            for marker in (
                "跨团队", "跨部门", "协作", "利益相关者", "业务团队",
                "分歧", "诉求", "达成一致",
            )
        ):
            return True
        # “业务场景” is context, not collaboration. Multiple named functions
        # are required before department nouns alone establish this focus.
        return sum(marker in folded for marker in ("业务", "研发", "运维")) >= 2
    return any(term.casefold() in folded for term in terms)

# Rules are cross-domain and bilingual. Recruitment-specific concepts remain as
# optional labels, while generic analysis/decision/action/result steps support
# engineering, product, sales, operations, and other roles.
_STEP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("业务背景与目标", ("业务", "战略", "发展阶段", "背景", "目标", "context", "goal")),
    ("人才画像", ("画像", "胜任力", "文化契合", "软性能力", "硬性能力")),
    ("目标人才定位与寻访", ("目标人群", "人才地图", "寻访", "sourcing")),
    (
        "分析与诊断",
        (
            "分析",
            "日志",
            "定位",
            "调研",
            "诊断",
            "拆成",
            "拆解",
            "采集",
            "建立基线",
            "统一口径",
            "analy",
            "diagnos",
            "investigat",
        ),
    ),
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
            "扩容",
            "预扩容",
            "接入",
            "增加",
            "拆出",
            "保留",
            "补充",
            "回放",
            "压测",
            "演练",
            "观察",
            "采集",
            "统一",
            "建立",
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
            "达成",
            "达到",
            "完成",
            "结果",
            "压测",
            "p95",
            "improved",
            "reduced",
            "delivered",
            "completed",
            "achieved",
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
        for item in re.split(r"(?<!\d)[.!?](?!\d)|[,;，。！？；]", text)
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
    if re.search(
        r"请讲|举例|一次|(?:谈谈|描述|说明).{0,30}经历|期间|已经发生|已经结束|具体项目|你是如何|"
        r"请(?:描述|说明)你在|(?:你)?在.{2,80}(?:时|中)[，,]?(?:你)?(?:是)?如何",
        question,
    ) or re.search(
        r"\b(?:tell me about a time|describe a time|give (?:me )?an example|experience where|when you)\b",
        folded,
    ):
        return "behavioral_example"
    if re.search(r"假如|如果|情景|会怎么", question) or re.search(
        r"\b(?:what would you|how would you|suppose|imagine|if you)\b", folded
    ):
        return "situational"
    if re.search(r"为什么|动机", question) or re.search(
        r"\b(?:why do you|motivat|why this|why are you)\b", folded
    ):
        return "motivation"
    if re.search(
        r"如何|怎么|流程|方法|设计|步骤|策略|"
        r"(?:哪些|什么)(?:证据|指标|标准|原则|机制|依据)|"
        r"用(?:哪些|什么)(?:证据|指标|标准|原则|机制|依据)",
        question,
    ) or re.search(
        r"\b(?:how do you|how did you|approach|process|method|design)\b", folded
    ):
        return "methodology"
    return "general"


def _explicit_context(question: str) -> str:
    chinese_company = re.search(
        r"在\s*([一-鿿A-Za-z0-9&.· -]{2,30}?)(?:负责|担任|工作|期间|时)",
        question,
    )
    if chinese_company:
        return chinese_company.group(1).strip()
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


def _context_evidence(sentences: list[str], *, behavioral: bool) -> str:
    if not behavioral:
        return ""
    return next(
        (
            sentence
            for sentence in sentences
            if re.search(
                r"(?:当时|那次|背景是|场景是|去年|前年|大促|双十一|618|活动前|事故中|"
                r"在.{0,36}(?:公司|企业|组织|部门|团队|项目|业务)|"
                r"在.{0,20}(?:软件|科技|集团|银行|大学)(?:[，,]|时|期间)|"
                r"在(?!每个|不同|各个).{2,36}(?:期间|阶段)|"
                r"\b(?:during|at the time|in (?:that|a|the) (?:project|team|company))\b)",
                sentence,
                re.IGNORECASE,
            )
        ),
        "",
    )


def _outcome_evidence(sentences: list[str]) -> str:
    """Return an observed result, never a forecast, target, or baseline metric."""
    forecast_terms = (
        "预计",
        "预测",
        "目标",
        "计划",
        "基线",
        "原来",
        "此前",
        "expected",
        "forecast",
        "target",
    )
    explicit_terms = (
        "最终结果",
        "最终",
        "最后",
        "结果是",
        "上线后",
        "实施后",
        "两周后",
        "活动实际",
        "实际峰值",
        "连续观察",
        "as a result",
        "after launch",
    )
    for index in range(len(sentences) - 1, -1, -1):
        sentence = sentences[index]
        folded = sentence.casefold()
        if any(term.casefold() in folded for term in explicit_terms) and not any(
            term.casefold() in folded for term in forecast_terms
        ):
            # ASR and our sentence splitter commonly separate a result lead-in
            # from the following metric clauses at commas. Keep the short result
            # window together so “实际峰值……，P99……，错误率……” remains auditable.
            return "，".join(sentences[index : index + 3])[:200]
    result_terms = (
        "最终",
        "最后",
        "提升",
        "降低",
        "降到",
        "下降",
        "缩短",
        "恢复到",
        "恢复至",
        "达成",
        "达到",
        "完成",
        "交付",
        "支持",
        "improved",
        "reduced",
        "restored to",
        "returned to",
        "returned below",
        "resolved",
        "achieved",
        "delivered",
    )
    # Answers usually state the observed outcome after explaining the baseline
    # and target. Prefer the latest eligible sentence to avoid citing a
    # pre-action metric as the result.
    for sentence in reversed(sentences):
        folded = sentence.casefold()
        if any(term.casefold() in folded for term in result_terms) and not any(
            term in sentence for term in forecast_terms
        ):
            return sentence[:200]
    return ""


def _asks_tradeoff_detail(question: str) -> bool:
    """Whether the question asks for alternatives and a decision, not just metrics.

    In a follow-up such as “做权衡时考虑了哪些指标”, “权衡” describes the
    setting; the requested payload is the metrics and their decision link. It
    should not silently expand into a second requirement to retell every option.
    """
    folded = question.casefold()
    explicit_decision = bool(
        re.search(r"取舍|比较|对比|备选|方案|最终选择|决定|决策|trade-?off|alternative", folded)
    )
    metric_led = bool(re.search(r"哪些指标|什么指标|指标.*(?:哪些|什么)|which metrics", folded))
    return explicit_decision or ("权衡" in folded and not metric_led)


def question_requirements(question: str) -> list[str]:
    """Build the single post-answer contract shared by questions and scoring."""
    folded = question.casefold()
    question_type = _answer_type(question)
    if question_type == "motivation":
        return ["说明动机与匹配关系"]
    requirements: list[str] = []
    if any(any(term.casefold() in folded for term in terms) for _, terms in _FOCUS_GROUPS):
        requirements.append("直接回应题目核心")
    if question_type == "behavioral_example":
        requirements.extend(
            [
                "提供一个真实案例",
                "明确具体公司/业务场景",
                "明确个人职责与关键决策",
                "说明执行动作",
            ]
        )
    asks_tradeoff = _asks_tradeoff_detail(question)
    asks_metrics = bool(
        re.search(r"指标|数据|口径|衡量|量化|qps|p\d{2}|cpu|metric|measure", folded)
    )
    # A narrow request such as “考虑了哪些指标” asks for metric detail, not a
    # new end-to-end case.  It is still a methodology-style question for analysis,
    # but its scoring contract must not silently grow into STAR requirements.
    metric_detail_follow_up = asks_metrics and bool(
        re.search(r"(?:哪些|什么)?指标|指标(?:有|是|包括)", question)
    ) and not bool(
        re.search(r"如何|怎么|制定|流程|步骤|方案|策略|机制|证据|how|approach|process", folded)
    )
    asks_method = (
        question_type in {"methodology", "situational"} and not metric_detail_follow_up
    ) or bool(
        re.search(r"如何|怎么|制定|方法|过程|步骤|方案|策略|how|approach|process", folded)
    )
    if asks_tradeoff:
        requirements.append("说明备选方案、权衡标准与最终选择")
    if asks_metrics:
        requirements.append("说明指标、口径与决策关系")
    if asks_method or (question_type == "behavioral_example" and asks_tradeoff):
        requirements.append("说明方法或制定过程")
    if question_type in {"methodology", "situational"} and not metric_detail_follow_up:
        requirements.append("说明执行动作")
    asks_result = question_type == "behavioral_example" or bool(
        re.search(r"结果|效果|成果|成效|影响|收益|验证|result|impact|outcome|verify", folded)
    )
    if question_type in {"methodology", "situational"} and not metric_detail_follow_up:
        asks_result = True
    if asks_result:
        requirements.append("给出结果与验证方式")
    return list(dict.fromkeys(requirements))


def question_answer_type(question: str) -> str:
    """Expose the shared answer taxonomy used by guidance and scoring."""
    return _answer_type(question)


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

    contract = question_requirements(question)
    question_type = _answer_type(question)
    context_name = _explicit_context(question)
    context_evidence = context_name if context_name.casefold() in answer.casefold() else ""
    if not context_evidence:
        context_evidence = _context_evidence(
            sentences, behavioral=question_type == "behavioral_example"
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
            "扩容",
            "预扩容",
            "接入",
            "增加",
            "拆出",
            "保留",
            "补充",
            "回放",
            "压测",
            "演练",
            "观察",
            "采集",
            "统一",
            "建立",
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
            "我用",
            "我把",
            "我和",
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
    outcome = _outcome_evidence(sentences)
    metric = _METRIC_PATTERN.search(answer)

    if "直接回应题目核心" in contract:
        requested_groups = [
            (label, terms)
            for label, terms in _FOCUS_GROUPS
            if _question_requests_focus_group(label, terms, question)
            and (label != "技术权衡" or _asks_tradeoff_detail(question))
        ]
        matched_groups = [
            (label, terms)
            for label, terms in requested_groups
            if any(term.casefold() in answer.casefold() for term in terms)
        ]
        requested_labels = {label for label, _ in requested_groups}
        matched_labels = {label for label, _ in matched_groups}
        primary_labels = requested_labels - _SUPPORTING_FOCUS_GROUPS
        if primary_labels and not (primary_labels & matched_labels):
            relevance_status = "missing"
        elif matched_labels >= requested_labels:
            relevance_status = "covered"
        elif matched_labels:
            relevance_status = "partial"
        else:
            relevance_status = "missing"
        focus_terms = tuple(term for _, terms in matched_groups for term in terms)
        focus_evidence = _excerpt(sentences, focus_terms) if focus_terms else ""
        add(
            "直接回应题目核心",
            relevance_status,
            focus_evidence,
            "先用一句话直接回答题目中的核心对象，再展开证据。",
        )

    if "明确具体公司/业务场景" in contract:
        add(
            "明确具体公司/业务场景",
            "covered" if context_evidence else "missing",
            context_evidence,
            "说明具体或匿名化的业务场景、阶段、任务和目标，不必披露公司名称。",
        )

    if "提供一个真实案例" in contract:
        explicit_case = bool(
            re.search(r"当时|项目中|事故中|那次|最终入职|去年|前年|大促|双十一|618", answer)
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

    if "说明备选方案、权衡标准与最终选择" in contract:
        tradeoff = _excerpt(
            sentences,
            (
                "比较",
                "对比",
                "权衡",
                "取舍",
                "备选",
                "方案",
                "虽然",
                "成本",
                "风险",
                "trade-off",
                "alternative",
            ),
        )
        decision = _excerpt(
            sentences,
            ("最终我决定", "我决定", "我选择", "因此我选择", "最终选择", "决定先"),
        )
        if not decision:
            decision = next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(r"我.{0,160}(?:决定|选择)", sentence)
                ),
                "",
            )
        criteria = bool(re.search(r"因为|考虑|约束|成本|风险|一致性|可逆|依据", answer))
        add(
            "说明备选方案、权衡标准与最终选择",
            "covered"
            if tradeoff and decision and criteria
            else "partial"
            if tradeoff or decision
            else "missing",
            "；".join(dict.fromkeys(item for item in (tradeoff, decision) if item)),
            "列出至少一个备选方案、选择标准及最终决定。",
        )

    if "说明指标、口径与决策关系" in contract:
        named_metrics = list(dict.fromkeys(_NAMED_METRIC_PATTERN.findall(answer)))
        metric_evidence = _excerpt(sentences, tuple(named_metrics)) if named_metrics else ""
        decision_link = bool(
            re.search(r"用于|依据|门槛|阈值|选择|决定|评估|验证|判断|观察|看四组|回滚", answer)
        )
        add(
            "说明指标、口径与决策关系",
            "covered"
            if len(named_metrics) >= 2 and decision_link and metric
            else "partial"
            if named_metrics or metric
            else "missing",
            metric_evidence,
            "说明至少两个与本题相关的指标、统计口径，以及它们如何影响决策。",
        )

    if "说明方法或制定过程" in contract:
        method_steps = [
            step.evidence
            for step in steps
            if step.label
            in {
                "分析与诊断",
                "方案比较与决策",
                "执行与推进",
                "评估体系",
                "项目治理与复盘",
            }
        ]
        method_evidence = "；".join(dict.fromkeys(method_steps[:2])) or execution
        has_ordered_actions = bool(
            re.search(r"(?:先|首先).{2,160}(?:再|然后|随后|其次)", answer)
            or re.search(r"\b(?:first|then|next|after that)\b", answer.casefold())
        )
        add(
            "说明方法或制定过程",
            "covered"
            if len(method_steps) >= 2 or (method_evidence and has_ordered_actions)
            else "partial"
            if method_evidence
            else "missing",
            method_evidence,
            "把方法压缩为三到五个按顺序执行的步骤。",
        )

    if "说明执行动作" in contract:
        add(
            "说明执行动作",
            "covered" if execution else "missing",
            execution,
            "说明你亲自推进的动作、参与者和交付物。",
        )

    if "明确个人职责与关键决策" in contract:
        add(
            "明确个人职责与关键决策",
            "covered" if personal else "missing",
            personal,
            "用“我负责/我决定/我推动”区分个人贡献与团队行动。",
        )

    if "给出结果与验证方式" in contract:
        outcome_metric = _METRIC_PATTERN.search(outcome) if outcome else None
        add(
            "给出结果与验证方式",
            "covered" if outcome and outcome_metric else "partial" if outcome else "missing",
            outcome,
            "补充已核验结果、指标口径和时间范围；无数字时说明可观察变化。",
        )

    if "说明动机与匹配关系" in contract:
        reason = _excerpt(sentences, ("因为", "吸引", "匹配", "希望", "because", "motiv", "align"))
        add(
            "说明动机与匹配关系",
            "covered" if reason else "missing",
            reason,
            "连接公司/岗位特点、你的经历和下一阶段目标。",
        )
    return requirements


def grounded_missing_signals(analysis: SpokenAnswerAnalysis) -> list[str]:
    """Derive persisted gaps only from the auditable coverage contract.

    The model may suggest useful coaching angles, but it must not persist a
    statement as missing when the deterministic analyzer has evidence that the
    same requirement was covered.
    """
    gaps: list[str] = []
    for item in analysis.question_coverage:
        if item.status == "missing":
            gaps.append(f"{item.requirement}：{item.suggestion}")
        elif item.status == "partial":
            gaps.append(f"{item.requirement}仍需补充：{item.suggestion}")
    return gaps


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
    """Align model scores with deterministic evidence caps and strong anchors."""
    statuses = {item.requirement: item.status for item in analysis.question_coverage}
    notes: list[str] = []
    analysis.pre_calibration_scores = {
        field: round(float(getattr(evaluation, field)), 4) for field in _SCORE_FIELDS
    }
    upper_bounds: dict[str, float] = {}

    def cap(field: str, maximum: float, reason: str) -> None:
        upper_bounds[field] = min(maximum, upper_bounds.get(field, 1.0))
        current = float(getattr(evaluation, field))
        if current > maximum:
            setattr(evaluation, field, maximum)
            notes.append(f"{field} {current:.2f}→{maximum:.2f}：{reason}")

    def floor(field: str, minimum: float, reason: str) -> None:
        minimum = min(minimum, upper_bounds.get(field, 1.0))
        current = float(getattr(evaluation, field))
        if current < minimum:
            setattr(evaluation, field, minimum)
            notes.append(f"{field} {current:.2f}→{minimum:.2f}：{reason}")

    relevance = statuses.get("直接回应题目核心")
    if relevance == "missing":
        cap("content", 0.30, "回答未直接回应题目的核心对象")
        cap("technical_depth", 0.30, "专业内容与本题要求不相关")
        cap("impact", 0.30, "结果属于另一主题，不能作为本题证据")
    elif relevance == "partial":
        cap("content", 0.55, "只回应了题目的一部分核心要求")

    for requirement in (
        "说明备选方案、权衡标准与最终选择",
        "说明指标、口径与决策关系",
    ):
        if statuses.get(requirement) == "missing":
            cap("technical_depth", 0.50, f"本题核心要求未覆盖：{requirement}")
        elif statuses.get(requirement) == "partial":
            cap("technical_depth", 0.70, f"本题核心要求仅部分覆盖：{requirement}")

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

    # The model contributes semantic nuance, but it may not assign a failing
    # score when the deterministic contract found strong, quoted evidence.
    # Floors apply only to unambiguous covered requirements and never override
    # the downward caps above for missing evidence.
    if relevance == "covered":
        floor("content", 0.65, "回答已直接覆盖题目核心")
        if statuses.get("提供一个真实案例") == "covered":
            floor("content", 0.72, "题目核心与具体案例均有证据")
    depth_requirements = (
        "明确个人职责与关键决策",
        "说明备选方案、权衡标准与最终选择",
        "说明指标、口径与决策关系",
        "说明方法或制定过程",
    )
    covered_depth = sum(statuses.get(item) == "covered" for item in depth_requirements)
    if covered_depth >= 2:
        floor("technical_depth", 0.70, "至少两项决策深度要求已有证据")
    elif covered_depth == 1:
        floor("technical_depth", 0.58, "至少一项决策深度要求已有证据")
    if len(analysis.semantic_steps) >= 3:
        floor("structure", 0.70, "回答已呈现三个以上可识别的语义步骤")
    elif len(analysis.semantic_steps) >= 2:
        floor("structure", 0.60, "回答已呈现多个有序语义步骤")
    if statuses.get("给出结果与验证方式") == "covered":
        floor("impact", 0.72, "已提供可核验的实际结果")
    analysis.calibration_notes = notes
    evaluation.spoken_analysis = analysis
