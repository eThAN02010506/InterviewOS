"""Deterministic interview-question interpretation and transfer guidance."""

from __future__ import annotations

import re

from interview_os.core.spoken_answer import question_answer_type, question_requirements
from interview_os.core.state import (
    CandidateStorySuggestion,
    QuestionAnswerLevel,
    QuestionProbeNode,
    QuestionUnderstanding,
)

_TYPE_LABELS = {
    "behavioral_example": "行为经历题",
    "methodology": "方法论题",
    "situational": "情境假设题",
    "motivation": "动机匹配题",
    "technical": "专业判断题",
    "case_analysis": "案例分析题",
    "compensation": "薪酬沟通题",
    "candidate_question": "候选人反问",
    "constraint": "现实条件题",
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
    folded = question.casefold()
    if any(term in folded for term in ("期望薪资", "薪资要求", "薪酬", "salary", "compensation")):
        return "compensation"
    if any(
        term in folded
        for term in ("反问面试官", "问面试官", "有什么想问", "questions for me")
    ):
        return "candidate_question"
    if any(
        term in folded
        for term in ("搬迁", "签证", "到岗时间", "空档", "gap", "relocation", "visa")
    ):
        return "constraint"
    answer_type = question_answer_type(question)
    if answer_type == "methodology" and any(
        term in folded for term in ("架构", "技术", "算法", "系统", "代码", "性能")
    ):
        return "technical"
    if any(term in folded for term in ("case study", "案例分析", "估算", "市场规模")):
        return "case_analysis"
    return answer_type if answer_type in _TYPE_LABELS else "general"


def deterministic_question_understanding(
    question: str,
    *,
    competency: str = "",
    job_title: str = "",
    explicit_job_requirements: list[str] | None = None,
    confirmed_claims: list[tuple[str, str]] | None = None,
) -> QuestionUnderstanding:
    """Create useful interpretation even when the local model is unavailable."""
    answer_type = _normalize_type(question)
    inferred_competency = _infer_competency(question, competency)
    explicit_job_requirements = explicit_job_requirements or []
    confirmed_claims = confirmed_claims or []
    requirements = question_requirements(question)
    boundaries = requirements or ["直接回应问题核心", "给出判断依据或真实证据"]

    if answer_type == "compensation":
        goal = "确认候选人的薪酬预期、依据和可协商边界，并判断是否与岗位预算匹配"
        boundaries = ["说明薪酬口径与预期范围", "说明期望的现实依据", "保留结合职责和整体包协商的空间"]
        mistakes = ["在了解岗位前只报一个刚性数字", "混淆税前税后、月薪年薪或现金与总包", "虚构当前薪资或其他 offer"]
        variants = ["你目前如何评估这个岗位的合理薪酬范围？", "除薪资外，哪些整体回报因素会影响你的决定？", "如果预算低于预期，你会如何评估？"]
        transfer = "保持薪酬口径和真实底线一致，根据面试阶段调整信息精度，不虚构竞争性 offer。"
        follow_ups = ["这个范围指税前年薪还是整体包？", "你的依据是岗位职责、市场数据还是当前总包？", "哪些条件会改变你的可接受范围？"]
    elif answer_type == "candidate_question":
        goal = "确认候选人能否根据面试官身份和当前对话，提出有利于双向判断的反问"
        boundaries = ["匹配面试官可回答的范围", "利用已有对话避免重复", "优先问成功标准、真实挑战和决策方式"]
        mistakes = ["向 HR 追问无法回答的技术细节", "询问官网已有的基本信息", "只为展示自己而设计过长问题"]
        variants = ["如果对方是用人经理，你最想确认哪个成功标准？", "如果对方是技术面试官，你会如何追问团队的真实挑战？", "前面哪句回答值得你继续深挖？"]
        transfer = "反问的核心是获得决策信息；随 HR、用人经理、专业面试官或创始人切换问题层级。"
        follow_ups = ["这个问题为什么适合由当前面试官回答？", "如果对方给出模糊回答，你会怎样追问？", "对方的哪种回答会成为你的风险信号？"]
    elif answer_type == "constraint":
        goal = "核对到岗、搬迁、签证或空档等现实条件，并判断时间线与岗位是否可行"
        boundaries = ["直接说明已确认的客观条件", "区分确定事实与待确认项", "给出可执行时间线或协商条件"]
        mistakes = ["为了显得配合而承诺不可行时间", "隐瞒签证或搬迁限制", "对空档过度辩解而不说明事实"]
        variants = ["你最早什么时候可以到岗？", "搬迁或远程安排中还有哪些条件待确认？", "请简要说明这段职业空档的事实和当前状态。"]
        transfer = "保持客观事实、时间线和底线前后一致，不因问法改变真实约束。"
        follow_ups = ["哪一项是已经确定的，哪一项还要与相关方确认？", "如果时间线无法满足，你的可行替代方案是什么？", "这个条件最晚什么时候能确认？"]
    elif answer_type == "motivation":
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
    role_relevance, role_relevance_source = _role_relevance(
        inferred_competency,
        job_title=job_title,
        explicit_job_requirements=explicit_job_requirements,
    )
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
        role_relevance=role_relevance,
        role_relevance_source=role_relevance_source,
        secondary_competencies=_secondary_competencies(answer_type, inferred_competency),
        decision_criteria=_decision_criteria(answer_type, inferred_competency),
        answer_levels=_answer_levels(answer_type, inferred_competency),
        candidate_story_options=_candidate_story_options(
            question, inferred_competency, confirmed_claims
        ),
        story_selection_guidance=_story_selection_guidance(answer_type),
        probe_tree=_probe_tree(answer_type, inferred_competency, follow_ups),
        analysis_source="rules",
    )


def reconcile_question_understanding(
    existing: QuestionUnderstanding | None,
    fallback: QuestionUnderstanding,
) -> QuestionUnderstanding:
    """Refresh persisted scoring contracts without losing useful model detail."""
    if existing is None or existing.analysis_source != "model":
        return fallback
    return existing.model_copy(
        update={
            "answer_type": fallback.answer_type,
            "answer_type_label": fallback.answer_type_label,
            "answer_boundary": fallback.answer_boundary,
            "transfer_principle": fallback.transfer_principle,
            "role_relevance": fallback.role_relevance,
            "role_relevance_source": fallback.role_relevance_source,
        }
    )


def _role_relevance(
    competency: str, *, job_title: str, explicit_job_requirements: list[str]
) -> tuple[str, str]:
    explicit = [item.strip() for item in explicit_job_requirements if item.strip()][:2]
    if explicit:
        role = job_title.strip() or "目标岗位"
        return (
            (
                f"{role}的明确 JD 包含“{'；'.join(explicit)}”。这道题用于验证候选人是否真正具备"
                f"支撑这些职责的{competency}，而不只是熟悉相关概念。"
            ),
            "explicit_jd",
        )
    if job_title.strip():
        return (
            (
                f"当前缺少可直接绑定的明确 JD 条款；仅根据“{job_title.strip()}”岗位名称推测，"
                f"面试官可能借此观察{competency}。该关联需要结合完整职责后重新确认。"
            ),
            "title_inference",
        )
    return (
        f"当前没有岗位职责可用于绑定；以下按通用面试逻辑分析{competency}，不代表目标岗位的明确要求。",
        "generic",
    )


def _secondary_competencies(answer_type: str, primary: str) -> list[str]:
    mapping = {
        "motivation": ["自我认知", "风险判断"],
        "behavioral_example": ["结果意识", "复盘能力"],
        "methodology": ["决策质量", "数据与结果验证"],
        "technical": ["权衡判断", "风险控制"],
        "case_analysis": ["问题拆解", "商业判断"],
        "situational": ["风险管理", "利益相关者管理"],
        "compensation": ["市场判断", "协商沟通"],
        "candidate_question": ["双向判断", "沟通深度"],
        "constraint": ["诚信度", "可执行性"],
        "general": ["逻辑表达", "证据意识"],
    }
    return [item for item in mapping.get(answer_type, mapping["general"]) if item != primary]


def _decision_criteria(answer_type: str, competency: str) -> list[str]:
    criteria = [
        f"是否直接证明{competency}，而不是回答相邻但不同的能力",
        "关键判断是否有清晰依据，并能说明本人承担的责任",
        "结论是否有可核验事实、结果或验证方法支撑",
    ]
    if answer_type in {"methodology", "technical", "case_analysis", "situational"}:
        criteria.insert(2, "是否说明前提、备选方案、决策门槛和风险边界")
    if answer_type == "motivation":
        criteria[1] = "动机是否来自具体岗位任务与真实经历，而不是泛化热情"
    return criteria[:4]


def _answer_levels(answer_type: str, competency: str) -> list[QuestionAnswerLevel]:
    evidence = "真实案例" if answer_type == "behavioral_example" else "具体依据"
    return [
        QuestionAnswerLevel(
            level="strong",
            description=f"直接回应{competency}，用{evidence}解释个人判断、行动、取舍和验证结果。",
            observable_signals=["边界全部覆盖", "个人贡献清晰", "结果可核验", "能说明限制与复盘"],
        ),
        QuestionAnswerLevel(
            level="acceptable",
            description="核心问题回答正确且有行动依据，但量化结果、取舍或岗位连接仍有一项不完整。",
            observable_signals=["核心结论明确", "至少一项具体证据", "没有明显事实矛盾"],
        ),
        QuestionAnswerLevel(
            level="risk",
            description="使用通用口号、团队成果或假设替代个人事实，且无法经追问验证关键判断。",
            observable_signals=["答非所问", "个人责任模糊", "目标冒充结果", "追问后证据仍不一致"],
        ),
    ]


def _story_selection_guidance(answer_type: str) -> list[str]:
    first = (
        "优先选择已经结束、本人做过关键决定且结果可核验的经历"
        if answer_type == "behavioral_example"
        else "优先选择能证明方法确实落地，而不只是知道概念的经历"
    )
    return [first, "优先匹配题目要求，而不是机械选择规模最大的项目", "无法确认的简历事实不要用于作答"]


def _candidate_story_options(
    question: str, competency: str, confirmed_claims: list[tuple[str, str]]
) -> list[CandidateStorySuggestion]:
    query = f"{question} {competency}".casefold()
    domain_terms = (
        "战略", "团队", "管理", "招聘", "人才", "数据", "指标", "业务", "项目", "产品",
        "技术", "架构", "系统", "客户", "市场", "协作", "跨部门", "成本", "增长", "风险",
    )
    ranked: list[tuple[int, str, str]] = []
    for claim_id, statement in confirmed_claims:
        folded = statement.casefold()
        score = sum(1 for term in domain_terms if term in query and term in folded)
        if score:
            ranked.append((score, claim_id, statement))
    ranked.sort(key=lambda item: (-item[0], item[2]))
    return [
        CandidateStorySuggestion(
            claim_id=claim_id,
            claim=statement[:500],
            fit_reason=f"该已确认事实与题目中的{competency}存在直接主题重合",
            adaptation_focus="只使用已确认事实；补充本人决定、执行动作和可验证结果，不得扩写简历未确认内容。",
        )
        for _, claim_id, statement in ranked[:2]
    ]


def _probe_tree(
    answer_type: str, competency: str, follow_ups: list[str]
) -> list[QuestionProbeNode]:
    foundation = (
        f"请先用一句话说明你处理{competency}问题时最核心的判断原则。"
        if answer_type != "behavioral_example"
        else f"请先明确这次{competency}经历中的目标、约束和你本人负责的部分。"
    )
    if answer_type == "behavioral_example":
        evidence_question = follow_ups[0]
    elif answer_type == "compensation":
        evidence_question = "你给出这个薪酬范围的具体口径和现实依据是什么？"
    elif answer_type == "candidate_question":
        evidence_question = "前面的对话中，哪一条具体信息使你决定追问这个问题？"
    elif answer_type == "constraint":
        evidence_question = "哪些客观条件已经确认，哪些还需要核对？"
    else:
        evidence_question = f"哪一项事实或数据最直接支持你的{competency}判断？你如何验证？"
    return [
        QuestionProbeNode(
            stage="foundation",
            question=foundation,
            purpose="确认候选人先直接回答核心，而不是从宽泛背景开始",
            entry_condition="回答尚未形成明确结论时",
        ),
        QuestionProbeNode(
            stage="evidence",
            question=evidence_question,
            purpose="把抽象主张落到个人事实或判断依据",
            entry_condition="结论清楚但缺少本人证据时",
        ),
        QuestionProbeNode(
            stage="tradeoff",
            question="你还考虑过哪些方案？当时用什么标准排除它们？",
            purpose="检验决策质量，而不只检验是否做过",
            entry_condition="已有行动和结果，但取舍过程不清楚时",
        ),
        QuestionProbeNode(
            stage="pressure",
            question="如果关键假设相反、资源减半或结果没有达到目标，你会怎样调整？",
            purpose="检验方法边界、风险意识和迁移能力",
            entry_condition="基础证据充分后再进入压力验证",
        ),
    ]


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
    secondary_competencies: list[str] | None = None,
    decision_criteria: list[str] | None = None,
    answer_levels: list[dict[str, object]] | None = None,
    candidate_story_options: list[dict[str, object]] | None = None,
    probe_tree: list[dict[str, object]] | None = None,
    confirmed_claims: list[tuple[str, str]] | None = None,
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
    selected_competency = (
        explicit_competency.strip() or competency.strip() or fallback.competency
    )[:100]
    return QuestionUnderstanding(
        answer_type=allowed_type,
        answer_type_label=_TYPE_LABELS[allowed_type],
        assessment_goal=(assessment_goal.strip() or fallback.assessment_goal)[:300],
        competency=selected_competency,
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
        # Role relevance must quote the deterministic JD boundary. Model prose
        # could otherwise turn an inferred responsibility into an explicit one.
        role_relevance=fallback.role_relevance,
        role_relevance_source=fallback.role_relevance_source,
        secondary_competencies=[
            item
            for item in _clean_list_items(secondary_competencies or [], limit=3)
            if item != selected_competency
        ]
        or fallback.secondary_competencies,
        decision_criteria=_clean_list_items(decision_criteria or [], limit=4)
        or fallback.decision_criteria,
        answer_levels=_merge_answer_levels(answer_levels or [], fallback.answer_levels),
        candidate_story_options=_merge_story_options(
            candidate_story_options or [],
            confirmed_claims or [],
            fallback.candidate_story_options,
        ),
        story_selection_guidance=fallback.story_selection_guidance,
        probe_tree=_merge_probe_tree(probe_tree or [], fallback.probe_tree),
        analysis_source="model",
    )


def _merge_answer_levels(
    model_items: list[dict[str, object]], fallback: list[QuestionAnswerLevel]
) -> list[QuestionAnswerLevel]:
    by_level: dict[str, QuestionAnswerLevel] = {}
    for item in model_items:
        level = str(item.get("level", "")).strip().casefold()
        description = str(item.get("description", "")).strip()
        raw_signals = item.get("observable_signals", [])
        if level not in {"strong", "acceptable", "risk"} or not description:
            continue
        signals = _clean_list_items(
            [str(signal) for signal in raw_signals] if isinstance(raw_signals, list) else [],
            limit=4,
        )
        by_level[level] = QuestionAnswerLevel(
            level=level,
            description=description[:400],
            observable_signals=signals,
        )
    fallback_by_level = {item.level: item for item in fallback}
    return [
        by_level.get(level) or fallback_by_level[level]
        for level in ("strong", "acceptable", "risk")
    ]


def _merge_story_options(
    model_items: list[dict[str, object]],
    confirmed_claims: list[tuple[str, str]],
    fallback: list[CandidateStorySuggestion],
) -> list[CandidateStorySuggestion]:
    output: list[CandidateStorySuggestion] = []
    for item in model_items:
        claim_number = item.get("claim_number")
        if not isinstance(claim_number, int) or not 1 <= claim_number <= len(confirmed_claims):
            continue
        claim_id, statement = confirmed_claims[claim_number - 1]
        if any(existing.claim_id == claim_id for existing in output):
            continue
        output.append(
            CandidateStorySuggestion(
                claim_id=claim_id,
                claim=statement[:500],
                fit_reason=str(item.get("fit_reason", "")).strip()[:300]
                or "模型从已确认简历事实中选择了该经历",
                adaptation_focus=str(item.get("adaptation_focus", "")).strip()[:300]
                or "围绕本题边界组织，不补充未经确认的事实。",
            )
        )
    return output[:2] or fallback


def _merge_probe_tree(
    model_items: list[dict[str, object]], fallback: list[QuestionProbeNode]
) -> list[QuestionProbeNode]:
    allowed = ("foundation", "evidence", "tradeoff", "pressure")
    by_stage: dict[str, QuestionProbeNode] = {}
    for item in model_items:
        stage = str(item.get("stage", "")).strip().casefold()
        question = str(item.get("question", "")).strip()
        if stage not in allowed or len(question) < 4:
            continue
        by_stage[stage] = QuestionProbeNode(
            stage=stage,
            question=re.sub(r"^\s*(?:\d{1,2}[.、)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*", "", question)[:300],
            purpose=str(item.get("purpose", "")).strip()[:300],
            entry_condition=str(item.get("entry_condition", "")).strip()[:300],
        )
    fallback_by_stage = {item.stage: item for item in fallback}
    return [by_stage.get(stage) or fallback_by_stage[stage] for stage in allowed]
