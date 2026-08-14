"""Evidence-grounded spoken-answer analysis and score calibration tests."""

from interview_os.core.spoken_answer import (
    analyze_spoken_answer,
    calibrate_evaluation,
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
    assert analysis.rubric_version == "evidence-v2"
