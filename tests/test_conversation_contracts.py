"""Acceptance counterexamples for question-specific coaching and evidence."""

import pytest

from interview_os.agents.coach_agent import CoachAgent
from interview_os.core.answer_feedback import apply_specific_feedback
from interview_os.core.follow_up_planner import plan_follow_up
from interview_os.core.question_understanding import deterministic_question_understanding
from interview_os.core.spoken_answer import analyze_spoken_answer, calibrate_evaluation
from interview_os.core.state import AnswerEvaluation, InterviewQuestion, MockAnswerRecord
from interview_os.services.intelligence_service import review_job_description


def evaluate(question, answer, score=0.2):
    analysis = analyze_spoken_answer(question, answer)
    result = AnswerEvaluation(content=score, technical_depth=score, structure=score, impact=score)
    calibrate_evaluation(result, analysis)
    apply_specific_feedback(result, answer, question=question)
    CoachAgent._build_grounded_improvement(result, answer, question=question)
    return result


@pytest.mark.parametrize(
    "answer",
    [
        "因为我希望找工作。我没有评估公司，也没有考虑风险。",
        "我希望加入，但是我不清楚评估标准，不知道如何核验风险。",
        "I want a job. I have not evaluated the company and have never considered risk.",
    ],
)
def test_denial_is_not_positive_evidence_or_a_score_floor(answer):
    result = evaluate("为什么你想加入我们？请说明评估机会的标准。", answer)
    assert result.content == result.technical_depth == result.impact == 0.2
    assert any(item.status != "covered" for item in result.spoken_analysis.question_coverage)
    assert "已给出具体机会判断标准" not in " ".join(result.feedback)
    assert "已说明会如何核验" not in " ".join(result.feedback)


@pytest.mark.parametrize(
    ("question", "answer", "kind"),
    [
        (
            "你期望的薪酬是多少？",
            "我期望税前年薪六十万元，可结合岗位职责和整体薪酬协商。",
            "compensation",
        ),
        (
            "你有什么想问面试官的问题？",
            "这个岗位前三个月的成功标准是什么？目前最大的交付障碍是什么？",
            "candidate_question",
        ),
        ("如果需要搬迁，你的到岗时间是什么？", "我需要四周交接，搬迁安排尚待确认。", "constraint"),
        (
            "What is your expected salary?",
            "My expected annual base salary is $120000, negotiable.",
            "compensation",
        ),
        ("Do you have questions for me?", "What are this team's priorities?", "candidate_question"),
    ],
)
def test_special_questions_share_contract_and_do_not_penalize_valid_answers(question, answer, kind):
    result = evaluate(question, answer, 0.8)
    understanding = deterministic_question_understanding(question)
    assert understanding.answer_type == result.spoken_analysis.answer_type == kind
    assert understanding.answer_boundary == [
        i.requirement for i in result.spoken_analysis.question_coverage
    ]
    assert all(i.status == "covered" for i in result.spoken_analysis.question_coverage)
    assert result.content == 0.8
    assert "至少一个备选方案" not in " ".join(result.feedback)
    assert "[补充真实结果" not in result.improved_answer


def test_incomplete_motivation_gets_clarification_and_a_non_star_draft():
    question = InterviewQuestion(question="为什么你想加入我们？", competency="求职动机")
    result = evaluate(question.question, "工资高。")
    response = MockAnswerRecord(
        question_id=question.id,
        question=question.question,
        competency=question.competency,
        answer="工资高。",
        evaluation=result,
    )
    decision = plan_follow_up(question, response)
    assert decision.stage == "foundation"
    assert "已覆盖" not in decision.rationale
    assert "工资高" in result.improved_answer
    assert "最终结果是" not in result.improved_answer
    assert "仍需你补充" in result.improved_answer


def test_motivation_rewrite_reorders_existing_sentences_without_inventing_facts():
    question = "为什么你想加入我们？"
    answer = "入职前我会核验客户留存。我过去负责企业产品。我希望承担完整产品责任。"
    result = evaluate(question, answer, 0.8)
    draft = result.improved_answer.split("表达顺序：")[0]
    assert draft.index("我希望") < draft.index("我过去") < draft.index("入职前")
    for sentence in answer.split("。"):
        assert not sentence or sentence in draft


def test_mixed_jd_keeps_duties_before_first_heading():
    result = review_job_description(
        "负责企业 AI 产品路线图；带领六人团队。任职要求：七年以上产品经验。", []
    )
    assert [i.text for i in result.requirements] == [
        "负责企业 AI 产品路线图",
        "带领六人团队",
        "七年以上产品经验",
    ]


def test_empty_coverage_cannot_claim_answer_is_complete():
    question = InterviewQuestion(question="你怎么看这个机会？", competency="求职沟通")
    response = MockAnswerRecord(
        question_id=question.id,
        question=question.question,
        competency=question.competency,
        answer="再看看。",
        evaluation=AnswerEvaluation(content=0.5, technical_depth=0.5, structure=0.5, impact=0.5),
    )
    assert plan_follow_up(question, response).stage == "foundation"
