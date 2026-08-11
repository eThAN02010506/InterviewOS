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
    analysis = analyze_spoken_answer(QUESTION, ANSWER)
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
    analysis = analyze_spoken_answer(QUESTION, ANSWER)
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
