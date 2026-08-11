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
