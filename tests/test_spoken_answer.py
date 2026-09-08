"""Evidence-grounded spoken-answer analysis and score calibration tests."""

from interview_os.agents.coach_agent import CoachAgent
from interview_os.core.answer_feedback import apply_specific_feedback
from interview_os.core.spoken_answer import (
    analyze_spoken_answer,
    calibrate_evaluation,
    grounded_missing_signals,
)
from interview_os.core.state import AnswerEvaluation

QUESTION = "请说说你在 Zuora 期间是如何制定并执行高管招聘策略的？"
ANSWER = (
    "好，首先是在呃公司的，首先在公司的业务层面了解整个业务它的策略，"
    "在业务的发展阶段，对应它对人才，特别是高级管理人才的这种啊需要的背景。"
    "具体到这个人的背景的画像、能力的画像以及人物的文化。"
    "根据这样的一个画像，我们可以去找到目标的这个人群，再去寻寻访。"
    "除此之外，我们要有一个非常好的一个评估系统，设置不同的考核点、考察官和考评团队。"
    "高管的职位招聘一定是有一个啊项目制来进行管理，在不同的阶段不停地回顾和调整策略。"
)


def test_spoken_answer_extracts_method_but_does_not_invent_a_case_or_result():
    analysis = analyze_spoken_answer(QUESTION, ANSWER, answer_modality="asr")
    statuses = {item.requirement: item.status for item in analysis.question_coverage}
    labels = {item.label for item in analysis.semantic_steps}

    assert analysis.raw_transcript == ANSWER
    assert len(analysis.cleaned_transcript) < len(ANSWER)
    assert "寻寻访" not in analysis.cleaned_transcript
    assert analysis.filler_counts["啊"] >= 2
    assert analysis.answer_type == "behavioral_example"
    assert {"人才画像", "目标人才定位与寻访", "评估体系", "项目治理与复盘"} <= labels
    assert statuses["提供一个真实案例"] == "missing"
    assert statuses["说明方法或制定过程"] == "covered"
    assert statuses["明确个人职责与关键决策"] == "missing"
    assert statuses["给出结果与验证方式"] == "missing"


def test_score_calibration_caps_unsupported_dimensions():
    analysis = analyze_spoken_answer(QUESTION, ANSWER, answer_modality="asr")
    evaluation = AnswerEvaluation(
        content=0.8,
        technical_depth=0.7,
        structure=0.6,
        impact=0.7,
    )

    calibrate_evaluation(evaluation, analysis)

    assert evaluation.content == 0.65
    assert evaluation.technical_depth == 0.6
    assert evaluation.structure == 0.55
    assert evaluation.impact == 0.35
    assert len(evaluation.spoken_analysis.calibration_notes) == 4


def test_english_behavioral_answer_is_classified_and_covered():
    analysis = analyze_spoken_answer(
        "Tell me about a time you handled a production incident.",
        (
            "During a checkout incident, I owned mitigation. I compared the logs, "
            "rolled back the release, and added an alert. Error rate returned below "
            "1% in 20 minutes."
        ),
        answer_modality="typed",
    )
    statuses = {item.requirement: item.status for item in analysis.question_coverage}

    assert analysis.answer_type == "behavioral_example"
    assert statuses["提供一个真实案例"] == "covered"
    assert statuses["明确个人职责与关键决策"] == "covered"
    assert statuses["给出结果与验证方式"] == "covered"


def test_chinese_technical_answer_recognizes_latency_metric_and_result():
    analysis = analyze_spoken_answer(
        "你如何定位并解决数据库延迟问题？",
        "我负责排查慢查询，先分析执行计划，再增加组合索引，P95 从 500ms 降到 120ms。",
        answer_modality="typed",
    )
    statuses = {item.requirement: item.status for item in analysis.question_coverage}

    assert analysis.answer_type == "methodology"
    assert statuses["说明执行动作"] == "covered"
    assert statuses["给出结果与验证方式"] == "covered"


def test_real_product_case_uses_observed_metrics_instead_of_later_support_phrase():
    question = (
        "请选一个最近三年你亲自负责、最能体现生成式 AI 产品管理的真实案例："
        "当时要解决什么业务问题，你做了什么关键决定，结果如何验证？"
    )
    answer = (
        "2025 年我负责企业知识助手从试点走向规模化。当时 5 家试点客户反馈回答质量不稳定，"
        "但销售希望立即扩大。我先和客户成功访谈 30 家目标客户，把问题拆成检索覆盖、答案可信度"
        "和使用习惯三类；备选方案是直接扩容，或先建立质量门槛。我选择后者，因为错误答案会放大"
        "信任风险。随后我与 20 人研发团队重构 RAG 工作流，建立离线评测集和线上采纳率、周活、"
        "成本监控，并设置分阶段准入与回滚条件。三个月内试点从 5 家增至 18 家，周活从 42% 到 "
        "61%，采纳率从 55% 到 73%。我的复盘是早期仍低估了客户成功的支持压力，之后把支持工时"
        "也纳入扩张门槛。"
    )

    analysis = analyze_spoken_answer(question, answer)
    coverage = {item.requirement: item for item in analysis.question_coverage}

    assert analysis.answer_type == "behavioral_example"
    assert coverage["说明执行动作"].status == "covered"
    assert "访谈 30 家" in coverage["说明执行动作"].evidence
    assert coverage["给出结果与验证方式"].status == "covered"
    assert "5 家增至 18 家" in coverage["给出结果与验证方式"].evidence
    assert "42% 到 61%" in coverage["给出结果与验证方式"].evidence
    assert "支持工时" not in coverage["给出结果与验证方式"].evidence


def test_past_context_question_uses_behavioral_contract_and_precise_evidence():
    analysis = analyze_spoken_answer(
        "请描述你在订单服务中，如何完成容量规划与故障恢复？",
        (
            "背景是订单服务高峰 P95 达到 420ms。"
            "我先用 tracing 和慢查询日志定位热点，再拆分批量写入并设置有界缓存。"
            "最终 P95 降到 180ms，高峰错误率下降 60%。"
        ),
    )
    coverage = {item.requirement: item for item in analysis.question_coverage}

    assert analysis.answer_type == "behavioral_example"
    assert coverage["明确具体公司/业务场景"].status == "covered"
    assert "tracing" in coverage["说明方法或制定过程"].evidence
    assert "最终 P95 降到 180ms" in coverage["给出结果与验证方式"].evidence
    assert "容量规划与故障恢复" not in coverage["给出结果与验证方式"].evidence


def test_result_evidence_does_not_confuse_target_with_achieved_outcome():
    analysis = analyze_spoken_answer(
        "请描述你如何降低订单服务延迟，并说明结果。",
        (
            "背景是订单服务促销高峰 P95 达到 420ms。"
            "我的任务是把 P95 降到 200ms 以下。"
            "我先分析慢查询，再增加索引。"
            "最终 P95 降到 180ms，高峰错误率下降 60%。"
        ),
    )
    coverage = {item.requirement: item for item in analysis.question_coverage}

    assert coverage["给出结果与验证方式"].evidence.startswith("最终 P95 降到 180ms")


def test_method_question_with_choice_is_not_misclassified_as_motivation():
    analysis = analyze_spoken_answer(
        "你如何选择项目架构方案？",
        "我会先比较容量、一致性和维护成本，再通过压力测试验证方案。",
        answer_modality="typed",
    )

    assert analysis.answer_type == "methodology"


def test_completed_project_prompt_is_classified_as_behavioral():
    analysis = analyze_spoken_answer(
        "请用一个已经结束的项目说明你如何处理跨部门分歧。",
        "在一次交付项目中，我负责协调产品和销售，最终按期上线。",
        answer_modality="typed",
    )

    assert analysis.answer_type == "behavioral_example"


def test_spoken_chinese_number_is_a_verifiable_result_metric():
    analysis = analyze_spoken_answer(
        "请讲一次你制定并执行招聘策略的经历。",
        (
            "在业务扩张阶段，我负责高管招聘，先确定画像，再建立分阶段评估。"
            "最终我们在八周内完成录用，试用期评估达到预设要求。"
        ),
        answer_modality="asr",
    )
    statuses = {item.requirement: item.status for item in analysis.question_coverage}

    assert statuses["提供一个真实案例"] == "covered"
    assert statuses["给出结果与验证方式"] == "covered"


def test_generic_behavioral_question_accepts_context_named_in_answer():
    analysis = analyze_spoken_answer(
        "请讲一个你亲自负责的真实案例，说明决定、行动和结果。",
        (
            "在一家进入新市场的企业中，业务要求八周内组建团队。"
            "我负责确定画像并调整筛选流程，最终按期完成关键岗位录用。"
        ),
    )
    coverage = {item.requirement: item for item in analysis.question_coverage}

    assert coverage["明确具体公司/业务场景"].status == "covered"
    assert "进入新市场" in coverage["明确具体公司/业务场景"].evidence


def test_generic_future_process_does_not_masquerade_as_a_past_case():
    analysis = analyze_spoken_answer(
        "请说说你在 Zuora 期间是如何制定并执行高管招聘策略的？",
        (
            "我会先理解业务战略，再定义人才画像，然后定点寻访。"
            "我会设计评估流程，并按项目制在每个阶段复盘和调整标准。"
        ),
    )
    coverage = {item.requirement: item.status for item in analysis.question_coverage}

    assert coverage["提供一个真实案例"] == "missing"
    assert coverage["明确具体公司/业务场景"] == "missing"
    assert coverage["给出结果与验证方式"] == "missing"


def test_typed_transition_words_do_not_trigger_asr_structure_penalty():
    evaluation = AnswerEvaluation(
        content=0.8,
        technical_depth=0.8,
        structure=0.8,
        impact=0.8,
    )
    analysis = analyze_spoken_answer(
        "请介绍你的方案。",
        "首先我比较约束，然后确定方案，就是这样完成了迁移。",
        answer_modality="typed",
    )

    calibrate_evaluation(evaluation, analysis)

    assert analysis.answer_modality == "typed"
    assert evaluation.spoken_analysis.pre_calibration_scores["structure"] == 0.8
    assert evaluation.structure == 0.8


def test_api_term_does_not_fabricate_company_context():
    analysis = analyze_spoken_answer(
        "请说明你如何设计 API 限流。",
        "我使用令牌桶，按租户设置配额，并监控拒绝率。",
        answer_modality="typed",
    )
    statuses = {item.requirement: item.status for item in analysis.question_coverage}

    assert "说明具体组织、项目或业务场景" not in statuses
    assert analysis.rubric_version == "evidence-v4"


def test_unrelated_polished_answer_is_capped_for_tradeoff_metric_followup():
    question = "你在做权衡时考虑了哪些指标？"
    answer = (
        "我长期辅导团队成员，每月做一对一沟通，并帮助两位同事晋升。"
        "后来团队满意度提升，协作氛围也更好了。"
    )
    analysis = analyze_spoken_answer(question, answer)
    evaluation = AnswerEvaluation(
        content=0.8,
        technical_depth=0.75,
        structure=0.8,
        impact=0.7,
    )

    calibrate_evaluation(evaluation, analysis)
    statuses = {item.requirement: item.status for item in analysis.question_coverage}

    assert statuses["直接回应题目核心"] == "missing"
    assert statuses["说明指标、口径与决策关系"] != "covered"
    assert evaluation.content <= 0.30
    assert evaluation.technical_depth <= 0.5


def test_relevant_tradeoff_answer_covers_metrics_and_decision_relationship():
    question = "你比较了哪些方案，做权衡时考虑了哪些指标，最终如何选择？"
    answer = (
        "我比较了全量预扩容和 HPA 自动扩容两个方案，重点看 P99、CPU 和错误率。"
        "考虑数据库扩容慢和成本约束，我决定数据库预扩容、无状态服务使用 HPA；"
        "当 P99 超过 300ms 或错误率超过 1% 时回滚。"
    )
    analysis = analyze_spoken_answer(question, answer)
    statuses = {item.requirement: item.status for item in analysis.question_coverage}

    assert statuses["直接回应题目核心"] == "covered"
    assert statuses["说明备选方案、权衡标准与最终选择"] == "covered"
    assert statuses["说明指标、口径与决策关系"] == "covered"


def test_metric_led_followup_does_not_invent_an_unasked_alternative_requirement():
    question = "你在做权衡时考虑了哪些指标？"
    answer = (
        "我看 P99、CPU 和错误率，并统一使用五分钟滚动窗口；"
        "P99 超过 300ms 或错误率超过 1% 就触发回滚。"
    )

    analysis = analyze_spoken_answer(question, answer)
    statuses = {item.requirement: item.status for item in analysis.question_coverage}

    assert statuses["直接回应题目核心"] == "covered"
    assert statuses["说明指标、口径与决策关系"] == "covered"
    assert "说明备选方案、权衡标准与最终选择" not in statuses

    evaluation = AnswerEvaluation(
        content=0.8,
        technical_depth=0.8,
        structure=0.8,
        impact=0.8,
        spoken_analysis=analysis,
    )
    apply_specific_feedback(evaluation, answer, question=question, competency="技术权衡")
    CoachAgent._build_grounded_improvement(evaluation, answer, question=question)
    feedback = " ".join(evaluation.feedback)
    assert "补充你亲自做出的关键决定" not in feedback
    assert "补充已核验的结果" not in feedback
    assert "阈值" in feedback
    assert "不必补讲一套新的 STAR 案例" in evaluation.improved_answer
    assert "[补充" not in evaluation.improved_answer


def test_motivation_feedback_does_not_demand_star_or_project_metrics():
    question = "为什么你想从成熟企业转到创业公司？请说明评估机会的具体标准。"
    answer = (
        "我希望对产品方向和商业结果承担完整责任。过去三年我负责企业知识助手的完整闭环。"
        "我评估机会主要看公司问题是否真实、产品是否有客户采用、团队是否用数据修正判断，"
        "以及岗位是否有明确决策责任。我会核验客户留存、交付成本和十二个月里程碑，"
        "入职前三个月再用客户访谈和质量基线验证匹配。"
    )
    analysis = analyze_spoken_answer(question, answer)
    evaluation = AnswerEvaluation(
        content=0.85,
        technical_depth=0.7,
        structure=0.8,
        impact=0.75,
        spoken_analysis=analysis,
    )

    apply_specific_feedback(evaluation, answer, question=question, competency="求职动机")
    CoachAgent._build_grounded_improvement(evaluation, answer, question=question)
    feedback = " ".join(evaluation.feedback)

    assert "至少一个备选方案" not in feedback
    assert "补充已核验的结果" not in feedback
    assert "一票否决" in feedback
    assert "不必套用 STAR" in evaluation.improved_answer


def test_motivation_keywords_do_not_override_model_scores():
    question = "为什么你想去创业公司？请说明你评估机会的具体标准。"
    answer = (
        "我希望承担完整的产品责任。我评估机会主要看客户问题、产品采用、团队判断方式和岗位责任。"
        "风险是资源有限，所以我会核验客户留存与成本，并在入职前三个月验证匹配。"
    )
    analysis = analyze_spoken_answer(question, answer)
    evaluation = AnswerEvaluation(
        content=0.4,
        technical_depth=0.4,
        structure=0.7,
        impact=0.4,
    )

    calibrate_evaluation(evaluation, analysis)

    assert evaluation.content == 0.4
    assert evaluation.technical_depth == 0.4
    assert evaluation.impact == 0.4


def test_capacity_answer_extracts_actions_and_observed_result_not_forecast():
    question = "请讲一次你如何完成容量规划，并说明选择依据和验证结果。"
    answer = (
        "去年大促前，业务预测峰值会增长六倍。我负责容量方案，先统一订单口径，"
        "再按网关、订单、库存和数据库拆解链路，建立 QPS、P99、CPU 和积压基线。"
        "我比较了全部预扩容和 HPA 两种方案，考虑数据库扩容耗时，决定数据库按预测峰值"
        "的 1.3 倍预扩容，无状态服务使用 HPA。上线前我回放历史流量并做 1.3 倍峰值压测。"
        "活动实际峰值为十一万 QPS，P99 为 240ms，错误率为 0.2%。"
    )
    analysis = analyze_spoken_answer(question, answer)
    coverage = {item.requirement: item for item in analysis.question_coverage}

    assert coverage["直接回应题目核心"].status == "covered"
    assert coverage["说明方法或制定过程"].status == "covered"
    assert coverage["说明执行动作"].status == "covered"
    assert coverage["给出结果与验证方式"].status == "covered"
    assert "活动实际峰值" in coverage["给出结果与验证方式"].evidence
    assert "预测" not in coverage["给出结果与验证方式"].evidence
    assert "0.2%" in analysis.cleaned_transcript


def test_capacity_question_does_not_turn_incidental_team_word_into_management_gap():
    question = (
        "在星河电商期间，你带领团队完成容量规划时，如何拆解模型、选择扩容方案，"
        "并验证实际运行结果？"
    )
    answer = (
        "我本人负责大促容量方案和上线决策，先统一订单量口径，再按网关、订单、库存、"
        "消息队列和数据库拆解容量模型。我比较全量预扩容与HPA，考虑扩容时延和回滚风险后，"
        "决定数据库与消息队列按预测峰值1.3倍预扩容，无状态服务使用HPA。上线前回放历史流量，"
        "完成1.3倍峰值压测。活动实际峰值11万QPS，P99为240ms，错误率0.2%。"
    )
    analysis = analyze_spoken_answer(question, answer)
    coverage = {item.requirement: item.status for item in analysis.question_coverage}
    evaluation = AnswerEvaluation(
        content=0.4, technical_depth=0.4, structure=0.4, impact=0.4
    )

    calibrate_evaluation(evaluation, analysis)

    assert coverage["直接回应题目核心"] == "covered"
    assert evaluation.content >= 0.72
    assert evaluation.technical_depth >= 0.70
    assert evaluation.impact >= 0.72


def test_polished_technical_incident_cannot_score_as_cross_department_collaboration():
    question = (
        "请讲一次你与业务、研发和运维对发布范围产生严重分歧的经历："
        "你如何理解各方诉求、建立决策机制并最终达成一致？请说明协作结果。"
    )
    answer = (
        "在一次交易系统故障中，我负责定位性能问题。我先查看链路追踪和数据库慢查询，"
        "发现连接池配置导致请求排队。我比较扩容、回滚配置和增加缓存三个方案，"
        "考虑恢复速度和一致性风险后决定先回滚，再分批恢复流量。十五分钟后P99从1.2秒"
        "恢复到220毫秒，错误率从4%下降到0.3%，随后我补充了变更检查和回滚演练。"
    )
    analysis = analyze_spoken_answer(question, answer)
    evaluation = AnswerEvaluation(
        content=0.8, technical_depth=0.8, structure=0.8, impact=0.8
    )

    calibrate_evaluation(evaluation, analysis)
    CoachAgent._build_grounded_improvement(evaluation, answer, question=question)
    coverage = {item.requirement: item.status for item in analysis.question_coverage}

    assert coverage["直接回应题目核心"] == "missing"
    assert evaluation.content <= 0.30
    assert evaluation.technical_depth <= 0.30
    assert evaluation.impact <= 0.30
    assert "不能直接重组成本题的示范答案" in evaluation.improved_answer
    assert "强行改写会虚构" in evaluation.improved_answer
    assert "我的核心做法是我负责定位性能问题" not in evaluation.improved_answer


def test_product_retention_answer_is_relevant_and_receives_evidence_floors():
    question = (
        "在北辰软件负责企业订阅计费产品时，面对续费率下降，你是如何定位问题并制定"
        "解决方案的？请说明业务场景、主要决策、关键行动、续费率结果和数据验证。"
    )
    answer = (
        "在北辰软件负责企业订阅计费产品时，我看到续费率连续下降。我本人负责定位原因"
        "和推动路线图调整。先把CRM中的120家流失客户按套餐、使用频率和流失原因分组，再结合客户"
        "成功团队的访谈记录。我比较了全面重做套餐与先建设预警看板两种方向，考虑影响范围"
        "和上线成本后，决定先上线用量预警与续费风险看板。之后每月按同一CRM口径验证，六个月内"
        "试点客户续费率从78%提升到86%。"
    )
    analysis = analyze_spoken_answer(question, answer)
    coverage = {item.requirement: item for item in analysis.question_coverage}
    evaluation = AnswerEvaluation(
        content=0.25,
        technical_depth=0.35,
        structure=0.3,
        impact=0.35,
    )

    calibrate_evaluation(evaluation, analysis)

    assert coverage["直接回应题目核心"].status == "covered"
    assert coverage["明确具体公司/业务场景"].status == "covered"
    assert coverage["给出结果与验证方式"].status == "covered"
    assert evaluation.content >= 0.72
    assert evaluation.technical_depth >= 0.70
    assert evaluation.structure >= 0.70
    assert evaluation.impact >= 0.72


def test_grounded_feedback_does_not_request_evidence_already_covered():
    question = "你在做容量权衡时考虑了哪些指标，最终结果如何？"
    answer = (
        "我比较了预扩容和 HPA，依据 P99、CPU、成本和错误率决定数据库预扩容、"
        "无状态服务使用 HPA。上线后实际峰值十一万 QPS，P99 为 240ms，错误率 0.2%。"
    )
    analysis = analyze_spoken_answer(question, answer)
    evaluation = AnswerEvaluation(
        content=0.85,
        technical_depth=0.85,
        structure=0.8,
        impact=0.85,
        spoken_analysis=analysis,
    )
    evaluation.missing_signals = grounded_missing_signals(analysis)
    calibrate_evaluation(evaluation, analysis)
    apply_specific_feedback(evaluation, answer, question=question, competency="容量规划")

    assert evaluation.observed_signals
    assert not any("补充你亲自做出的关键决定" in item for item in evaluation.feedback)
    assert not any("补充已核验的结果" in item for item in evaluation.feedback)
    assert not any(item.startswith("已覆盖「") for item in evaluation.feedback)
