from interview_os.core.follow_up_planner import MAX_AUTOMATIC_FOLLOW_UPS, plan_follow_up
from interview_os.core.question_understanding import deterministic_question_understanding
from interview_os.core.state import (
    AnswerEvaluation,
    InterviewQuestion,
    MockAnswerRecord,
    QuestionCoverageItem,
    QuestionUnderstanding,
    SpokenAnswerAnalysis,
)


def _question() -> InterviewQuestion:
    text = "请讲一次你在资源有限时做关键取舍的真实经历。"
    return InterviewQuestion(
        question=text,
        competency="战略优先级",
        follow_ups=["你个人具体做了什么？"],
        understanding=deterministic_question_understanding(
            text,
            competency="战略优先级",
        ),
    )


def _response(
    question: InterviewQuestion,
    answer: str = "我先给出了判断，再根据数据验证。",
    *,
    incomplete: list[str] | None = None,
) -> MockAnswerRecord:
    coverage = [
        QuestionCoverageItem(
            requirement=item,
            status="missing",
            suggestion="补充对应事实。",
        )
        for item in (incomplete or [])
    ] if incomplete else [QuestionCoverageItem(
        requirement="提供一个真实案例", status="covered", evidence="去年我负责产品上线",
    )]
    return MockAnswerRecord(
        question_id=question.id,
        question=question.question,
        competency=question.competency,
        answer=answer,
        evaluation=AnswerEvaluation(
            content=0.6,
            technical_depth=0.6,
            structure=0.6,
            impact=0.6,
            spoken_analysis=SpokenAnswerAnalysis(question_coverage=coverage),
        ),
    )


def test_unable_answer_uses_non_fabricating_recovery_branch():
    question = _question()
    decision = plan_follow_up(question, _response(question, "我不知道。"))

    assert decision is not None
    assert decision.stage == "recovery"
    assert "不要编造" in decision.question
    assert "直接经历" in decision.question


def test_missing_direct_answer_uses_foundation_branch():
    question = _question()
    decision = plan_follow_up(
        question,
        _response(question, incomplete=["直接回应题目核心"]),
    )

    assert decision is not None
    assert decision.stage == "foundation"


def test_missing_personal_contribution_uses_evidence_branch():
    question = _question()
    decision = plan_follow_up(
        question,
        _response(question, incomplete=["明确个人职责与关键决策"]),
    )

    assert decision is not None
    assert decision.stage == "evidence"
    assert "证据" in decision.rationale or "事实" in decision.rationale
    assert any(term in decision.question for term in ("事实", "数据", "本人", "决定"))


def test_missing_tradeoff_uses_tradeoff_branch():
    question = _question()
    decision = plan_follow_up(
        question,
        _response(question, incomplete=["说明备选方案、权衡标准与最终选择"]),
    )

    assert decision is not None
    assert decision.stage == "tradeoff"


def test_complete_answer_enters_pressure_branch():
    question = _question()
    decision = plan_follow_up(question, _response(question))

    assert decision is not None
    assert decision.stage == "pressure"
    assert "假设" in decision.question or "资源减半" in decision.question


def test_used_pressure_branch_does_not_regress_to_a_shallower_probe():
    question = _question()
    pressure = next(
        item.question for item in question.understanding.probe_tree if item.stage == "pressure"
    )

    assert plan_follow_up(question, _response(question), offered_questions=[pressure]) is None


def test_automatic_followups_are_hard_bounded():
    question = _question()
    decision = plan_follow_up(
        question,
        _response(question, incomplete=["直接回应题目核心"]),
        offered_questions=[f"probe-{index}" for index in range(MAX_AUTOMATIC_FOLLOW_UPS)],
    )

    assert decision is None


def test_legacy_question_uses_authored_followup_when_tree_is_unavailable():
    question = InterviewQuestion(
        question="你如何做决策？",
        competency="决策质量",
        follow_ups=["你用什么事实验证了判断？"],
        understanding=QuestionUnderstanding(probe_tree=[]),
    )
    decision = plan_follow_up(
        question,
        _response(question, incomplete=["给出结果与验证方式"]),
    )

    assert decision is not None
    assert decision.question == "你用什么事实验证了判断？"
