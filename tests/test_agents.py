"""Tests for domain agents (with mock LLM)."""
import pytest

from interview_os.agents.candidate_agent import CandidateAgent
from interview_os.agents.coach_agent import CoachAgent
from interview_os.agents.company_agent import CompanyAgent
from interview_os.agents.evaluation_agent import EvaluationAgent
from interview_os.agents.feedback_agent import FeedbackAgent
from interview_os.agents.interview_strategy_agent import InterviewStrategyAgent
from interview_os.agents.live_interview_agent import LiveInterviewAgent
from interview_os.agents.mock_interview_agent import MockInterviewAgent
from interview_os.core.evidence import Evidence, EvidencePolarity
from interview_os.core.message import MessageType
from interview_os.core.spoken_answer import analyze_spoken_answer
from interview_os.core.state import (
    AnswerEvaluation,
    AnswerReviewStatus,
    AnswerScoringSource,
    CompetencyEvaluation,
    EvaluationReport,
    InterviewQuestion,
    InterviewState,
    JobDescription,
    JobDescriptionReview,
    JobRequirement,
    LiveInterviewRecord,
    MockAnswerRecord,
    QuestionSuggestion,
    QuestionSuggestionType,
    RequirementOrigin,
    ResumeClaim,
    ResumeClaimStatus,
    TranscriptSegment,
    TranscriptSpeaker,
)
from interview_os.models.structured import parse_model_output


class MockLLM:
    async def chat(self, messages, **kwargs):
        return '{"name": "Test", "strengths": ["Python"], "weaknesses": ["Deploy"]}'

    async def embed(self, text):
        return [0.1, 0.2, 0.3]


class DeepQuestionLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"answer_type":"methodology","answer_type_label":"方法题",'
            '"assessment_goal":"验证资源配置的决策质量",'
            '"competency":"战略优先级","answer_boundary":[], '
            '"common_mistakes":["只按声音大小分资源"],'
            '"transfer_principle":"复用方法",'
            '"related_questions":["资源减半时如何排序？"],'
            '"likely_follow_ups":["你使用了什么指标？"],'
            '"role_relevance":"明确 JD 要求跨部门确定研发投入，因此需要验证资源配置决策。",'
            '"secondary_competencies":["数据分析","影响力"],'
            '"decision_criteria":["指标是否可比较","机制是否透明"],'
            '"candidate_story_options":[{"claim_number":1,'
            '"fit_reason":"该事实包含跨部门资源决策",'
            '"adaptation_focus":"强调个人制定的排序标准"}]}'
        )

    async def embed(self, text):
        return []


class InvalidLLM:
    async def chat(self, messages, **kwargs):
        return "not json"


class EvaluationNarrativeLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"summary":"C1 现有回答显示候选人能够拆解招聘目标，但结果证据仍需核验。",'
            '"competency_reviews":['
            '{"competency_id":"C1","evidence_numbers":[1,2],'
            '"assessment":"两条证据分别说明了画像拆解和漏斗复盘；当前优势是方法清楚，缺口是结果尚未核验。",'
            '"next_probe":"请说明这套策略在多长时间内改善了哪个已核验指标？"},'
            '{"competency_id":"C2","evidence_numbers":[1],'
            '"assessment":"现有证据说明候选人推动了协作，但不足以判断长期辅导成效。",'
            '"next_probe":"请讲一个持续辅导团队成员并验证其成长结果的案例。"}'
            ']}'
        )


class ContradictorySingleEvidenceNarrativeLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"summary":"已有一条证据。","competency_reviews":['
            '{"competency_id":"C1","evidence_numbers":[1],'
            '"assessment":"回答说明了性能改进，但当前回答尚未提供证据。",'
            '"next_probe":"请再讲一个不同案例。"}]}'
        )


class FollowUpAsSecondCaseNarrativeLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"summary":"候选人已通过两次案例证明数据决策能力。",'
            '"competency_reviews":[{"competency_id":"C1","evidence_numbers":[1,2],'
            '"assessment":"第二个案例显示资源减半时仍能调整方案。",'
            '"next_probe":"请再讲一个不同产品线的真实案例。"}]}'
        )


class WrongEntityLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"name":"InterviewOS","industry":"Software","dna":"Test",'
            '"public_sources":[{"title":"Invented","url":null}]}'
        )


class EmptyPlanLLM:
    async def chat(self, messages, **kwargs):
        return '{"questions":[]}'


class FusedNumericPremiseLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"questions":[{"question":"在星河电商期间，你如何完成11万QPS的峰值压测？",'
            '"competency":"容量规划","rationale":"结合简历",'
            '"strong_signals":[],"follow_ups":[]}]}'
        )


class ScoreOnlyNarrativeLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"summary":": 0.6925, : 0.555",'
            '"competency_reviews":[{"competency_id":"C1","evidence_numbers":[1],'
            '"assessment":"回答提供了一项证据。","next_probe":"请补充第二个案例。"}]}'
        )


class RelevanceContradictingNarrativeLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"summary":"现有回答需要继续核验跨部门协作能力。",'
            '"competency_reviews":[{"competency_id":"C1","evidence_numbers":[1],'
            '"assessment":"回答体现了候选人在跨部门协作中快速定位并解决问题。",'
            '"next_probe":"请再说明各方诉求。"}]}'
        )


@pytest.mark.asyncio
async def test_mock_agent_without_llm_backfills_every_answer_framework():
    state = InterviewState()
    state.job.competencies = ["系统设计"]
    await MockInterviewAgent().execute(state)
    assert state.mock_interview.questions
    assert all(question.answer_framework for question in state.mock_interview.questions)


class HallucinatedCandidateLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"name":"John Doe","weaknesses":["Public speaking"],'
            '"achievements":["Employee of the Year 2021","Excellent Employee Award (2015)"]}'
        )

    async def embed(self, text):
        return []


class UnsupportedRiskLLM:
    async def chat(self, messages, **kwargs):
        return '{"summary":"准备策略","key_risks":["未提供团队规模"]}'

    async def embed(self, text):
        return []


class InventedMetricsLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"summary":"候选人有17年经验","answer_framework":'
            '["团队从10人扩展到200人，效率提升30%","说明2015年的真实奖项"]}'
        )

    async def embed(self, text):
        return []


class CapturingStrategyLLM:
    """Records the strategy prompt so tests can assert its content."""

    def __init__(self):
        self.last_prompt = ""

    async def chat(self, messages, **kwargs):
        self.last_prompt = messages[-1]["content"]
        return '{"summary":"准备策略","key_risks":[],"answer_framework":["先讲最近经历"]}'


class HallucinatingCoachLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"content":0.8,"technical_depth":0.7,"structure":0.75,"impact":0.7,'
            '"feedback":[],"observed_signals":["推动团队扩张"],'
            '"missing_signals":[],"improved_answer":'
            '"团队从80人增长到250人，交付周期缩短30%，收入增长18%。"}'
        )


class ContradictoryGapCoachLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"content":0.8,"technical_depth":0.8,"structure":0.8,"impact":0.8,'
            '"feedback":[],"observed_signals":[],'
            '"missing_signals":["验证结果：P95 降到 180ms","具体行动：增加索引"],'
            '"improved_answer":""}'
        )


class ForgedProvenanceCoachLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"content":0.8,"technical_depth":0.8,"structure":0.8,"impact":0.8,'
            '"feedback":[],"observed_signals":["说明了行动"],"missing_signals":[],'
            '"scoring_source":"human","review_status":"reviewed"}'
        )


def test_live_suggestion_migrates_legacy_main_question_type():
    suggestion = QuestionSuggestion.model_validate(
        {
            "suggested_question": "请说明架构权衡",
            "question_type": "main",
            "competency": "系统设计",
        }
    )
    assert suggestion.question_type == QuestionSuggestionType.NEXT_MAIN


@pytest.mark.asyncio
async def test_live_question_fallback_does_not_repeat_measured_results():
    agent = LiveInterviewAgent(llm_client=InvalidLLM())
    state = InterviewState(job=JobDescription(competencies=["架构设计"]))
    state.live_interview.segments.append(
        TranscriptSegment(
            sequence=1,
            speaker=TranscriptSpeaker.CANDIDATE,
            text="我定位连接池瓶颈并灰度上线，最终 P95 降低了 40%。",
        )
    )

    await agent.execute(state)

    suggestion = state.live_interview.suggestions[0]
    assert suggestion.competency == "架构设计"
    assert "替代方案" in suggestion.suggested_question
    assert "回滚" in suggestion.suggested_question


@pytest.mark.asyncio
async def test_candidate_agent():
    agent = CandidateAgent(llm_client=MockLLM())
    state = InterviewState()
    state.candidate.raw_resume_text = "Experienced Python developer"
    msg = await agent.execute(state)
    assert msg.type == MessageType.RESPONSE
    assert state.candidate.name == "Test"


def test_structured_parser_skips_unrelated_json_and_normalizes_scores():
    raw = 'Analysis metadata: {"step": 1}\nFinal: {"content": 8, "technical_depth": 75, "structure": 0.9, "impact": 7}'
    evaluation = parse_model_output(raw, AnswerEvaluation)
    assert evaluation.content == pytest.approx(0.8)
    assert evaluation.technical_depth == pytest.approx(0.75)
    assert evaluation.impact == pytest.approx(0.7)


def test_answer_evaluation_accepts_unambiguous_content_score_alias():
    evaluation = AnswerEvaluation.model_validate(
        {"content_score": 0.8, "technical_depth": 0.7, "structure": 0.9, "impact": 0.6}
    )
    assert evaluation.content == pytest.approx(0.8)
    assert evaluation.model_dump(mode="json", by_alias=True)["content"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_coach_degrades_to_reviewable_low_confidence_evidence():
    agent = CoachAgent(llm_client=InvalidLLM())
    state = InterviewState()
    message = await agent.execute(state, '{"question":"Q","answer":"A","competency":"Design"}')
    evaluation = AnswerEvaluation.model_validate_json(message.content)
    assert evaluation.overall_score() == pytest.approx(0.4)
    assert evaluation.spoken_analysis.pre_calibration_scores
    assert "人工复核" in evaluation.feedback[0]
    assert state.evidence[0].confidence == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_coach_deterministic_fallback_rewards_grounded_detail():
    agent = CoachAgent(llm_client=InvalidLLM())
    state = InterviewState()
    detailed = (
        "我先与业务负责人明确目标，再建立人才地图并每周复盘招聘漏斗。"
        "我根据转化率和交付周期调整渠道，最终支持了工程团队扩张；"
        "具体百分比仍需从 ATS 核对。"
    )

    message = await agent.execute(
        state,
        '{"question":"请说明招聘策略","answer":"'
        + detailed
        + '","competency":"招聘战略"}',
    )

    evaluation = AnswerEvaluation.model_validate_json(message.content)
    assert evaluation.content > 0.4
    assert evaluation.technical_depth >= 0.64
    assert evaluation.structure >= 0.56
    assert evaluation.impact > 0.4
    assert [item.dimension for item in evaluation.dimension_feedback] == [
        "content",
        "technical_depth",
        "structure",
        "impact",
    ]
    assert all(item.evidence and item.suggestion for item in evaluation.dimension_feedback)
    assert evaluation.spoken_analysis.semantic_steps
    assert evaluation.spoken_analysis.rubric_version == "evidence-v3"
    assert any("给出结果与验证方式仍需补充" in item for item in evaluation.missing_signals)
    assert detailed in evaluation.improved_answer


@pytest.mark.asyncio
async def test_coach_rejects_unsupported_metrics_in_improved_answer():
    agent = CoachAgent(llm_client=HallucinatingCoachLLM())
    state = InterviewState()
    original = "我组建了印度招聘团队，并支持了工程组织扩张。"

    message = await agent.execute(
        state,
        '{"question":"请说明招聘成果","answer":"'
        + original
        + '","competency":"招聘战略"}',
    )

    evaluation = AnswerEvaluation.model_validate_json(message.content)
    assert original in evaluation.improved_answer
    assert "80" not in evaluation.improved_answer
    assert "250" not in evaluation.improved_answer
    assert "基于你本次回答的重组示范" in evaluation.improved_answer
    assert "组建了印度招聘团队" in evaluation.improved_answer
    assert "[补充真实结果" in evaluation.improved_answer
    assert any("未提供的数字" in item for item in evaluation.feedback)
    assert "需要核验并补充真实量化结果" in evaluation.missing_signals
    # The model's invented observed signal remains a coaching hint only. The
    # persisted evidence must be the candidate's auditable original wording.
    assert state.evidence[0].signal == original
    assert state.evidence[0].polarity == EvidencePolarity.NEUTRAL
    assert all("推动团队扩张" not in item.evidence for item in evaluation.dimension_feedback)


@pytest.mark.asyncio
async def test_coach_discards_model_gaps_contradicted_by_coverage_evidence():
    agent = CoachAgent(llm_client=ContradictoryGapCoachLLM())
    state = InterviewState()
    answer = (
        "在订单服务高峰期间，我负责性能治理。"
        "我先分析 tracing 和慢查询，再增加索引并拆分批量写入。"
        "最终 P95 从 420ms 降到 180ms，错误率下降 60%。"
    )

    message = await agent.execute(
        state,
        '{"question":"请描述你在订单服务中如何解决性能问题？",'
        '"answer":"' + answer + '","competency":"系统性能"}',
    )

    evaluation = AnswerEvaluation.model_validate_json(message.content)
    assert evaluation.missing_signals == []
    assert state.evidence[0].notes == ""


@pytest.mark.asyncio
async def test_feedback_agent_keeps_hiring_voice_out_of_candidate_report():
    agent = FeedbackAgent()
    state = InterviewState(
        evaluation=EvaluationReport(
            competencies=[
                CompetencyEvaluation(
                    competency="招聘战略",
                    score=0.59,
                    confidence=0.6,
                    gaps=["需要更多独立回答交叉验证"],
                )
            ],
            overall_score=0.59,
        )
    )

    await agent.execute(state)

    assert "录用" not in state.feedback.overall
    assert "仅用于面试准备" in state.feedback.overall
    assert all("候选人" not in item for item in state.feedback.action_plan)
    assert all(item.startswith("准备并练习：") for item in state.feedback.action_plan)
    assert any("不同业务场景" in item for item in state.feedback.action_plan)
    assert "不能给出录用或不录用建议" in state.feedback.recommendation_reasoning


@pytest.mark.asyncio
async def test_evaluation_agent_calibrates_recommendation_from_evidence():
    agent = EvaluationAgent()
    state = InterviewState()
    state.evidence = [
        Evidence(
            competency="招聘战略",
            signal="说明了行动",
            confidence=0.61,
        )
        for _ in range(2)
    ] + [
        Evidence(
            competency="团队领导",
            signal="说明了协作",
            confidence=0.61,
        )
    ]

    await agent.execute(state)

    assert state.evaluation.overall_score == pytest.approx(0.61)
    assert state.evaluation.recommendation.value == "lean_hire"
    assert state.evaluation.summary == "已按已记录证据完成确定性聚合。"


@pytest.mark.asyncio
async def test_evaluation_agent_uses_only_persisted_evidence_aggregates():
    agent = EvaluationAgent()
    state = InterviewState()
    state.evidence = [
        Evidence(competency="招聘战略", signal="证据一", confidence=0.51),
        Evidence(competency="招聘战略", signal="证据二", confidence=0.51),
        Evidence(competency="团队领导", signal="证据三", confidence=0.51),
    ]

    await agent.execute(state)

    assert state.evaluation.overall_score == pytest.approx(0.51)
    assert state.evaluation.recommendation.value == "lean_no_hire"
    assert all(item.score == pytest.approx(0.51) for item in state.evaluation.competencies)
    assert all("模型声称高分" not in item.supporting_evidence for item in state.evaluation.competencies)
    assert state.evaluation.summary == "已按已记录证据完成确定性聚合。"


@pytest.mark.asyncio
async def test_evaluation_agent_adds_model_narrative_without_delegating_scores_or_gaps():
    agent = EvaluationAgent(llm_client=EvaluationNarrativeLLM())
    state = InterviewState()
    state.evidence = [
        Evidence(
            competency="招聘战略",
            signal="我先拆解岗位画像。",
            confidence=0.62,
            notes="缺少已核验结果",
        ),
        Evidence(
            competency="招聘战略",
            signal="我每周复盘招聘漏斗。",
            confidence=0.68,
            notes="缺少具体时间线",
        ),
        Evidence(
            competency="团队领导",
            signal="我组织业务和招聘团队对齐标准。",
            confidence=0.51,
            notes="缺少长期辅导成果",
        ),
    ]

    await agent.execute(state)

    results = {item.competency: item for item in state.evaluation.competencies}
    assert state.evaluation.narrative_source == "model"
    assert "候选人能够拆解招聘目标" in state.evaluation.summary
    assert "C1" not in state.evaluation.summary
    assert results["招聘战略"].score == pytest.approx(0.65)
    assert results["招聘战略"].gaps == ["缺少已核验结果", "缺少具体时间线"]
    assert len(results["招聘战略"].narrative_evidence_ids) == 2
    assert "结果尚未核验" in results["招聘战略"].assessment
    assert results["团队领导"].gaps == ["缺少长期辅导成果", "需要更多独立回答交叉验证"]
    assert results["团队领导"].next_probe.endswith("案例。")


@pytest.mark.asyncio
async def test_evaluation_agent_rejects_invalid_model_narrative_as_one_unit():
    agent = EvaluationAgent(llm_client=InvalidLLM())
    state = InterviewState()
    state.evidence = [
        Evidence(competency="招聘战略", signal="我复盘漏斗。", confidence=0.6)
    ]

    await agent.execute(state)

    assert state.evaluation.narrative_source == "deterministic"
    assert state.evaluation.competencies[0].assessment == ""
    assert "已按已记录证据" in state.evaluation.summary


@pytest.mark.asyncio
async def test_evaluation_narrative_does_not_deny_existing_single_evidence():
    agent = EvaluationAgent(llm_client=ContradictorySingleEvidenceNarrativeLLM())
    state = InterviewState()
    state.evidence = [
        Evidence(
            competency="系统性能",
            signal="最终 P95 降到 180ms。",
            confidence=0.7,
            notes="缺少复盘",
        )
    ]

    await agent.execute(state)

    result = state.evaluation.competencies[0]
    assert state.evaluation.narrative_source == "model"
    assert "尚未提供证据" not in result.assessment
    assert "仍需要第二个独立案例交叉验证" in result.assessment


@pytest.mark.asyncio
async def test_evaluation_rejects_follow_up_described_as_independent_second_case():
    agent = EvaluationAgent(llm_client=FollowUpAsSecondCaseNarrativeLLM())
    state = InterviewState()
    question_id = InterviewQuestion(question="如何改善续费？", competency="数据决策").id
    main = MockAnswerRecord(
        question_id=question_id,
        question="如何改善续费？",
        competency="数据决策",
        answer="我分析流失客户并上线风险看板。",
        evaluation=AnswerEvaluation(
            content=0.8, technical_depth=0.8, structure=0.8, impact=0.8
        ),
    )
    follow_up = MockAnswerRecord(
        question_id=question_id,
        question="资源减半时如何调整？",
        competency="数据决策",
        answer="我先缩小试点范围。",
        evaluation=AnswerEvaluation(
            content=0.7, technical_depth=0.7, structure=0.7, impact=0.7
        ),
        is_follow_up=True,
    )
    state.mock_session.responses = [main, follow_up]
    state.evidence = [
        Evidence(
            competency="数据决策",
            signal=record.answer,
            confidence=0.75,
            source_record_id=record.id,
        )
        for record in (main, follow_up)
    ]

    await agent.execute(state)

    assert state.evaluation.narrative_source == "deterministic"
    assert "两次案例" not in state.evaluation.summary
    assert "需要更多独立回答交叉验证" in state.evaluation.competencies[0].gaps
    assert state.evaluation.competencies[0].confidence == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_coach_model_cannot_forge_human_review_provenance():
    agent = CoachAgent(llm_client=ForgedProvenanceCoachLLM())
    state = InterviewState()

    message = await agent.execute(
        state,
        '{"question":"问题","answer":"回答","competency":"沟通"}',
    )
    evaluation = AnswerEvaluation.model_validate_json(message.content)

    assert evaluation.scoring_source == AnswerScoringSource.MODEL
    assert evaluation.review_status == AnswerReviewStatus.NOT_REQUIRED


def test_legacy_rule_score_is_migrated_to_explicit_pending_review():
    evaluation = AnswerEvaluation.model_validate(
        {
            "content": 0.4,
            "technical_depth": 0.4,
            "structure": 0.4,
            "impact": 0.4,
            "feedback": ["自动评分输出无效，本回答需要人工复核。"],
        }
    )

    assert evaluation.scoring_source == AnswerScoringSource.DETERMINISTIC_RULE
    assert evaluation.review_status == AnswerReviewStatus.PENDING


def test_provisional_detection_uses_structured_score_provenance():
    evaluation = CoachAgent._deterministic_evaluation("我先分析问题，然后推动解决。")
    state = InterviewState(
        live_interview_records=[
            LiveInterviewRecord(
                question="请介绍案例",
                answer="我先分析问题，然后推动解决。",
                competency="问题解决",
                evaluation=evaluation,
                scoring_status="scored",
            )
        ]
    )

    assert evaluation.scoring_source == AnswerScoringSource.DETERMINISTIC_RULE
    assert evaluation.review_status == AnswerReviewStatus.PENDING
    assert EvaluationAgent._has_provisional_scores(state)

    evaluation.review_status = AnswerReviewStatus.REVIEWED
    assert not EvaluationAgent._has_provisional_scores(state)


def test_evaluation_calibration_blocks_decision_for_provisional_rule_scores():
    report = EvaluationReport(
        competencies=[
            CompetencyEvaluation(
                competency="招聘战略",
                score=0.8,
                confidence=0.8,
            )
        ],
        overall_score=0.8,
        recommendation="hire",
    )

    EvaluationAgent._calibrate_recommendation(report, provisional_scoring=True)

    assert report.recommendation.value == "insufficient_evidence"
    assert any("尚未人工复核" in item for item in report.risks)


@pytest.mark.asyncio
async def test_negative_evidence_cannot_improve_hiring_recommendation():
    state = InterviewState(
        evidence=[
            Evidence(
                competency="诚信",
                signal="承认伪造数据",
                confidence=0.9,
                polarity=EvidencePolarity.NEGATIVE,
            ),
            Evidence(
                competency="诚信",
                signal="再次承认伪造",
                confidence=0.9,
                polarity=EvidencePolarity.NEGATIVE,
            ),
            Evidence(
                competency="沟通",
                signal="清晰解释过程",
                confidence=0.9,
                polarity=EvidencePolarity.POSITIVE,
            ),
        ]
    )

    await EvaluationAgent().execute(state)

    integrity = next(
        item for item in state.evaluation.competencies if item.competency == "诚信"
    )
    assert integrity.score == pytest.approx(0.1)
    assert state.evaluation.recommendation.value in {
        "lean_no_hire",
        "no_hire",
        "insufficient_evidence",
    }


def test_feedback_agent_only_emits_human_classified_negative_evidence():
    state = InterviewState()
    evidence = Evidence(
        competency="诚信",
        signal="候选人明确承认准备不足",
        confidence=0.1,
    )
    state.evidence = [evidence]
    state.missing_signals = ["回答缺乏项目时间线"]

    FeedbackAgent._enforce_candidate_voice(state)

    assert all(not item.startswith("负面证据：") for item in state.feedback.interviewer_notes)
    assert any(item == "仍待核验：回答缺乏项目时间线" for item in state.feedback.interviewer_notes)

    evidence.polarity = EvidencePolarity.NEGATIVE
    FeedbackAgent._enforce_candidate_voice(state)

    assert any(
        item == f"负面证据：候选人明确承认准备不足（Evidence {evidence.id}）"
        for item in state.feedback.interviewer_notes
    )
    assert all("伪造材料" not in item for item in state.feedback.interviewer_notes)
    assert "缺失信号不是负面证据" in state.feedback.recommendation_reasoning


@pytest.mark.asyncio
async def test_candidate_agent_extracts_explicit_name_when_model_output_fails():
    agent = CandidateAgent(llm_client=InvalidLLM())
    state = InterviewState()
    await agent.execute(state, "Experience Name ：JL Led recruiting across APAC")
    assert state.candidate.name == "JL"


@pytest.mark.asyncio
async def test_candidate_agent_preserves_session_name_when_model_omits_identity():
    agent = CandidateAgent(llm_client=InvalidLLM())
    state = InterviewState()
    state.candidate.name = "JLO"

    await agent.execute(state, "Talent acquisition leader across APAC")

    assert state.candidate.name == "JLO"


@pytest.mark.asyncio
async def test_candidate_agent_grounds_identity_weaknesses_and_achievements():
    agent = CandidateAgent(llm_client=HallucinatedCandidateLLM())
    state = InterviewState()
    await agent.execute(
        state,
        "Name ：JL Led recruiting. Recognition Excellent Employee Award (2015)",
    )
    assert state.candidate.name == "JL"
    assert state.candidate.weaknesses == []
    assert state.candidate.achievements == ["Excellent Employee Award (2015)"]


@pytest.mark.asyncio
async def test_strategy_does_not_create_risks_from_job_title_only():
    agent = InterviewStrategyAgent(llm_client=UnsupportedRiskLLM())
    state = InterviewState(
        job=JobDescription(
            title="Senior Recruiting Manager",
            raw_description="Senior Recruiting Manager",
        )
    )
    await agent.execute(state)
    assert state.strategy.key_risks == []


@pytest.mark.asyncio
async def test_strategy_removes_metrics_absent_from_evidence():
    agent = InterviewStrategyAgent(llm_client=InventedMetricsLLM())
    state = InterviewState()
    state.candidate.raw_resume_text = "17 years experience. Excellent award in 2015."
    await agent.execute(state)
    assert state.strategy.summary == "候选人有17年经验"
    assert state.strategy.answer_framework == ["说明2015年的真实奖项"]


@pytest.mark.asyncio
async def test_strategy_prompt_contains_recency_instruction_and_employer_block():
    llm = CapturingStrategyLLM()
    agent = InterviewStrategyAgent(llm_client=llm)
    state = InterviewState()
    state.candidate.raw_resume_text = "2018-07 to ZUORA 2022-12\nSenior Recruiting Manager"
    state.past_employer_sources = [
        {
            "title": "About Zuora",
            "url": "https://zuora.com/about",
            "snippet": "Zuora is a subscription platform.",
            "is_official": True,
            "source_quality": "official",
        }
    ]
    await agent.execute(state)
    # Recency/size weighting instruction is present.
    assert "最近的雇主" in llm.last_prompt
    assert "Fortune 500" in llm.last_prompt
    # The full resume (recent employer) and the past-employer research block are present.
    assert "ZUORA" in llm.last_prompt
    assert "Zuora is a subscription platform" in llm.last_prompt
    assert "不得据此推断候选人的职责、技能、业绩" in llm.last_prompt


@pytest.mark.asyncio
async def test_full_jd_strategy_does_not_turn_adjacent_numbers_into_fit_claims():
    agent = InterviewStrategyAgent(llm_client=InventedMetricsLLM())
    state = InterviewState(
        job=JobDescription(
            title="高级产品负责人",
            raw_description=(
                "岗位职责：负责B2B SaaS产品战略和路线图\n"
                "任职要求：7年以上产品经验，3年以上团队管理经验\n"
                "团队背景：产品团队5人，重点提升续费率"
            ),
        ),
        job_review=JobDescriptionReview(
            is_title_only=False,
            requirements=[
                JobRequirement(
                    text="7年以上产品经验，3年以上团队管理经验",
                    origin=RequirementOrigin.EXPLICIT,
                ),
                JobRequirement(
                    text="产品团队5人，重点提升续费率",
                    origin=RequirementOrigin.EXPLICIT,
                ),
            ],
        ),
    )
    state.candidate.raw_resume_text = (
        "带领3名产品经理，建立月度复盘机制。\n"
        "六个月内试点客户续费率从78%提升到86%。"
    )

    await agent.execute(state)

    rendered = state.strategy.model_dump_json()
    assert "不代表系统已判定候选人完全匹配" in rendered
    assert "管理经验不足" not in rendered
    assert "能带领5人" not in rendered
    assert "快速学习" not in rendered
    assert "候选人简历自述（待确认）" in rendered


@pytest.mark.asyncio
async def test_full_jd_strategy_matches_platform_evidence_aliases_without_false_risks():
    state = InterviewState(
        job=JobDescription(
            title="高级平台工程负责人",
            raw_description=(
                "负责核心交易平台架构、容量规划和稳定性。\n"
                "在业务快速增长阶段保障重大活动稳定交付。\n"
                "能够说明架构取舍、风险、回滚方案和实际运行指标。"
            ),
        ),
        job_review=JobDescriptionReview(
            is_title_only=False,
            requirements=[
                JobRequirement(
                    text="在业务快速增长阶段保障重大活动稳定交付",
                    origin=RequirementOrigin.EXPLICIT,
                ),
                JobRequirement(
                    text="说明架构取舍、风险、回滚方案和实际运行指标",
                    origin=RequirementOrigin.EXPLICIT,
                ),
            ],
        ),
    )
    state.candidate.raw_resume_text = (
        "负责交易链路容量和稳定性；大促前拆解容量模型并完成峰值压测。\n"
        "活动实际峰值11万QPS，P99为240ms，错误率0.2%，设置回滚线并完成故障演练。"
    )

    await InterviewStrategyAgent(llm_client=InvalidLLM()).execute(state)

    assert not any("重大活动" in risk for risk in state.strategy.key_risks)
    assert not any("运行指标" in risk for risk in state.strategy.key_risks)


def test_mock_framework_quotes_resume_without_inventing_actions_or_constraints():
    state = InterviewState()
    state.candidate.raw_resume_text = (
        "主导产品、销售、客户成功和研发共同调整路线图。\n"
        "六个月内试点客户续费率从78%提升到86%。"
    )

    framework = MockInterviewAgent.deterministic_framework(state, "产品战略与续费增长")

    assert "续费率从78%提升到86%" in framework
    assert "候选人简历自述（使用前确认）" in framework
    assert "资源有限" not in framework
    assert "安排测试" not in framework
    assert "客户访谈确认" not in framework


@pytest.mark.asyncio
async def test_mock_agent_degrades_to_current_job_competency_questions():
    agent = MockInterviewAgent(llm_client=InvalidLLM())
    state = InterviewState(job=JobDescription(competencies=["招聘策略", "团队领导力"]))
    await agent.execute(state)
    questions = state.mock_interview.questions
    # The pool is padded to MOCK_POOL_TARGET with unique competency questions.
    assert len(questions) == 5
    assert {q.competency for q in questions} == {"招聘策略", "团队领导力"}
    assert len({q.question for q in questions}) == 5  # all distinct
    assert all(q.answer_framework for q in questions)
    assert all(q.question_requirements for q in questions)
    assert all(q.example_answer for q in questions)
    assert all("结果" in q.question or "验证" in q.question for q in questions)


def test_mock_teaching_examples_are_specific_and_explicitly_fictional():
    question = InterviewQuestion(question="谈谈你的招聘经验", competency="招聘战略")

    MockInterviewAgent.enrich_question(question)

    assert "具体且已经发生的案例" in question.question
    assert "八周" in question.example_answer
    assert "虚构场景" in question.example_answer_note
    assert "明确个人职责与关键决策" in question.question_requirements


def test_mock_quality_contract_preserves_motivation_question_type():
    question = InterviewQuestion(question="你为什么选择这个岗位？", competency="求职动机")

    MockInterviewAgent.enrich_question(question)

    assert "岗位最吸引你的具体要素" in question.question
    assert "具体且已经发生的案例" not in question.question
    assert question.question_requirements == ["说明动机与匹配关系"]


@pytest.mark.asyncio
async def test_custom_question_deep_analysis_maps_only_confirmed_story_claims():
    claim = ResumeClaim(
        category="achievement",
        statement="推动产品与研发统一资源排序机制并完成核心版本上线",
        status=ResumeClaimStatus.CONFIRMED,
    )
    state = InterviewState(
        job=JobDescription(
            title="产品负责人",
            raw_description="负责跨部门产品战略和研发资源配置",
        ),
        job_review=JobDescriptionReview(
            requirements=[
                JobRequirement(
                    text="负责跨部门确定研发投入优先级",
                    origin=RequirementOrigin.EXPLICIT,
                )
            ]
        ),
    )
    state.resume_review.claims = [claim]
    agent = MockInterviewAgent(llm_client=DeepQuestionLLM())

    analysis = await agent.analyze_custom_question(
        state,
        "当团队争夺资源时，你会用哪些指标和机制确定优先级？",
        competency="战略优先级",
    )

    assert analysis.analysis_source == "model"
    assert analysis.answer_type == "methodology"
    assert analysis.role_relevance_source == "explicit_jd"
    assert "明确 JD" in analysis.role_relevance
    assert analysis.candidate_story_options[0].claim_id == str(claim.id)
    assert analysis.candidate_story_options[0].claim == claim.statement
    assert [item.level for item in analysis.answer_levels] == [
        "strong",
        "acceptable",
        "risk",
    ]
    assert [item.stage for item in analysis.probe_tree] == [
        "foundation",
        "evidence",
        "tradeoff",
        "pressure",
    ]


def test_mock_motivation_support_answers_the_actual_question():
    state = InterviewState()
    framework = MockInterviewAgent.deterministic_framework(state, "求职动机")
    example = MockInterviewAgent.teaching_example(
        "求职动机", "为什么你想从成熟的大公司转到早期创业公司？"
    )

    assert "主动选择什么" in framework
    assert "目标公司阶段" in framework
    assert "不是因为否定成熟平台" in example
    assert "入职前三个月" in example
    assert "跨团队项目" not in example


def test_mock_quality_contract_keeps_method_question_as_methodology():
    question = InterviewQuestion(
        question="你如何衡量平台健康度？", competency="系统设计"
    )

    MockInterviewAgent.enrich_question(question)

    assert "按顺序执行的步骤" in question.question
    assert "具体且已经发生的案例" not in question.question
    assert "说明指标、口径与决策关系" in question.question_requirements
    assert "说明方法或制定过程" in question.question_requirements
    assert "说明执行动作" in question.question_requirements
    assert "给出结果与验证方式" in question.question_requirements


def test_mock_quality_contract_does_not_treat_project_method_as_past_case():
    question = InterviewQuestion(
        question="你如何设计高管招聘项目的评估流程？", competency="高管招聘"
    )

    MockInterviewAgent.enrich_question(question)

    assert "按顺序执行的步骤" in question.question
    assert "具体且已经发生的案例" not in question.question
    assert "直接回应题目核心" in question.question_requirements
    assert "说明方法或制定过程" in question.question_requirements
    assert "说明执行动作" in question.question_requirements
    assert "给出结果与验证方式" in question.question_requirements


def test_capacity_teaching_example_passes_its_own_question_contract():
    question = InterviewQuestion(
        question="请讲一次你如何完成容量规划，并说明选择依据和验证结果。",
        competency="容量规划",
    )

    MockInterviewAgent.enrich_question(question)
    analysis = analyze_spoken_answer(question.question, question.example_answer)

    assert "QPS" in question.example_answer
    assert "HPA" in question.example_answer
    assert "压测" in question.example_answer
    assert all(item.status != "missing" for item in analysis.question_coverage)


def test_mock_quality_contract_recognizes_past_context_behavioral_question():
    question = InterviewQuestion(
        question="请描述你在订单服务中，如何完成容量规划与故障恢复？",
        competency="可靠性",
    )

    MockInterviewAgent.enrich_question(question)

    assert "提供一个真实案例" in question.question_requirements
    assert "明确个人职责与关键决策" in question.question_requirements
    assert "说明方法或制定过程" in question.question_requirements
    assert "给出结果与验证方式" in question.question_requirements


def test_mock_quality_contract_recognizes_past_context_with_shi_ruhe():
    question = InterviewQuestion(
        question="在带领 4 人小组做容量压测时，你是如何规划测试场景的？",
        competency="容量规划",
    )

    MockInterviewAgent.enrich_question(question)

    assert "提供一个真实案例" in question.question_requirements
    assert "明确个人职责与关键决策" in question.question_requirements


@pytest.mark.asyncio
async def test_mock_agent_degrades_when_model_returns_empty_valid_plan():
    agent = MockInterviewAgent(llm_client=EmptyPlanLLM())
    state = InterviewState(job=JobDescription(competencies=["招聘策略"]))
    await agent.execute(state)
    assert state.mock_interview.questions[0].competency == "招聘策略"


@pytest.mark.asyncio
async def test_mock_agent_rewrites_numeric_premise_not_supported_by_explicit_jd():
    state = InterviewState(
        job=JobDescription(
            title="高级平台工程负责人",
            raw_description="负责交易平台容量规划、稳定性和成本治理。",
            competencies=["容量规划"],
        )
    )
    state.candidate.raw_resume_text = (
        "完成1.3倍峰值压测；活动实际峰值11万QPS，P99为240ms。"
    )

    await MockInterviewAgent(llm_client=FusedNumericPremiseLLM()).execute(state)

    question = state.mock_interview.questions[0]
    assert "11万QPS的峰值压测" not in question.question
    assert "不预设事实" in question.rationale


@pytest.mark.asyncio
async def test_evaluation_rejects_score_only_model_summary():
    state = InterviewState()
    state.evidence = [
        Evidence(competency="容量规划", signal="完成容量压测。", confidence=0.7)
    ]

    await EvaluationAgent(llm_client=ScoreOnlyNarrativeLLM()).execute(state)

    assert state.evaluation.narrative_source == "deterministic"
    assert "已按已记录证据" in state.evaluation.summary


@pytest.mark.asyncio
async def test_evaluation_rejects_narrative_that_praises_locked_off_topic_answer():
    state = InterviewState()
    state.evidence = [
        Evidence(
            competency="跨部门协作",
            signal="我定位连接池问题并回滚配置。",
            confidence=0.3,
            notes="直接回应题目核心：missing",
        )
    ]

    await EvaluationAgent(llm_client=RelevanceContradictingNarrativeLLM()).execute(state)

    assert state.evaluation.narrative_source == "deterministic"


@pytest.mark.asyncio
async def test_company_agent_preserves_authoritative_company_name():
    agent = CompanyAgent(llm_client=WrongEntityLLM())
    state = InterviewState()
    state.company.name = "芯世界"
    await agent.execute(state)
    assert state.company.name == "芯世界"
    assert state.company.public_sources == []
