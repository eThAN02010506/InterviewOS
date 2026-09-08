from interview_os.core.answer_feedback import apply_specific_feedback
from interview_os.core.spoken_answer import analyze_spoken_answer
from interview_os.core.state import AnswerEvaluation


def evaluation_with_detail(answer, **overrides):
    analysis = analyze_spoken_answer("你如何制定高管招聘策略？", answer)
    detail = {
        "dimension": "technical_depth", "requirement": analysis.question_coverage[0].requirement,
        "quote": "初期由招聘团队严格筛选", "interpretation": "说明了筛选责任，但严格的含义还可以展开。",
        "action": "在严格筛选后补一句淘汰条件及所需证据，不要补造候选人案例。",
        "next_question": "什么证据会让你淘汰一位履历漂亮但不适合当前业务阶段的高管？",
    }
    detail.update(overrides)
    return AnswerEvaluation(content=.6, technical_depth=.5, structure=.7, impact=.5,
                            spoken_analysis=analysis, coaching_details=[detail])


def test_grounded_model_coaching_survives_without_changing_scores():
    answer = "初期由招聘团队严格筛选，再由业务负责人面试。"
    evaluation = evaluation_with_detail(answer)
    apply_specific_feedback(evaluation, answer, question="你如何制定高管招聘策略？")
    assert "淘汰一位" in evaluation.feedback[0]
    assert "初期由招聘团队严格筛选" in evaluation.dimension_feedback[1].evidence
    assert evaluation.technical_depth == .5
    assert "AI 教练解读" in evaluation.dimension_feedback[1].suggestion


def test_fabricated_quotes_or_foreign_requirements_fall_back():
    answer = "初期由招聘团队严格筛选，再由业务负责人面试。"
    for overrides in ({"quote": "我把周期缩短了50%"}, {"requirement": "必须解释组织裁员"},
                      {"dimension": "hiring"}, {"action": ""}):
        evaluation = evaluation_with_detail(answer, **overrides)
        apply_specific_feedback(evaluation, answer)
        assert not evaluation.coaching_details
        assert not any("淘汰一位" in item for item in evaluation.feedback)
        assert len(evaluation.dimension_feedback) == 4


def test_optional_malformed_notes_do_not_break_feedback():
    evaluation = evaluation_with_detail("初期由招聘团队严格筛选")
    evaluation.coaching_details = [None, "泛泛反馈", {"quote": 42}]
    apply_specific_feedback(evaluation, "初期由招聘团队严格筛选")
    assert not evaluation.coaching_details
    assert evaluation.feedback
