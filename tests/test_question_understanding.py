from interview_os.core.question_understanding import (
    deterministic_question_understanding,
    merge_model_understanding,
)


def test_behavioral_question_builds_transferable_question_family():
    analysis = deterministic_question_understanding(
        "请讲一次你处理跨部门冲突的真实经历。"
    )

    assert analysis.answer_type == "behavioral_example"
    assert analysis.competency == "跨部门协作"
    assert "个人贡献" in analysis.assessment_goal
    assert len(analysis.related_questions) == 3
    assert len(set(analysis.related_questions)) == 3
    assert all("真实经历" not in item or item != "请讲一次你处理跨部门冲突的真实经历。" for item in analysis.related_questions)


def test_situational_question_focuses_on_assumptions_and_risk():
    analysis = deterministic_question_understanding(
        "如果关键业务负责人不同意你的方案，你会怎么推进？",
        competency="影响力",
    )

    assert analysis.answer_type == "situational"
    assert analysis.competency == "影响力"
    assert "信息不完整" in analysis.assessment_goal
    assert "假设" in analysis.transfer_principle


def test_model_merge_cannot_replace_deterministic_answer_contract():
    fallback = deterministic_question_understanding("你如何设计容量规划流程？")
    merged = merge_model_understanding(
        fallback,
        answer_type="behavioral_example",
        answer_type_label="技术方法题",
        assessment_goal="验证容量规划的方法与判断标准",
        competency="模型猜测能力",
        answer_boundary=["不应覆盖确定性合同"],
        common_mistakes=["只列工具"],
        transfer_principle="保留方法，替换约束。",
        related_questions=["资源减半时如何调整容量方案？"],
        likely_follow_ups=["阈值如何确定？"],
        original_question="你如何设计容量规划流程？",
        explicit_competency="容量规划",
    )

    assert merged.analysis_source == "model"
    assert merged.answer_type == fallback.answer_type == "methodology"
    assert merged.competency == "容量规划"
    assert merged.answer_boundary == fallback.answer_boundary
    assert merged.related_questions[0] == "资源减半时如何调整容量方案？"
    assert len(merged.related_questions) == 3
    assert merged.answer_type_label == "方法论题"
    assert merged.transfer_principle == fallback.transfer_principle


def test_model_numbering_is_removed_from_teaching_lists():
    fallback = deterministic_question_understanding("你如何推动跨部门项目？")
    merged = merge_model_understanding(
        fallback,
        answer_type="behavioral_example",
        answer_type_label="行为题",
        assessment_goal="验证跨部门推进",
        competency="跨部门协作",
        answer_boundary=[],
        common_mistakes=["① 只开会不决策", "2. 没有验证结果"],
        transfer_principle="ignored",
        related_questions=["1. 资源减半时如何推进？", "② 分歧扩大时如何决策？"],
        likely_follow_ups=["① 谁做了最终决定？"],
        original_question="你如何推动跨部门项目？",
    )

    assert merged.common_mistakes[:2] == ["只开会不决策", "没有验证结果"]
    assert merged.related_questions[:2] == ["资源减半时如何推进？", "分歧扩大时如何决策？"]
    assert merged.likely_follow_ups == ["谁做了最终决定？"]
