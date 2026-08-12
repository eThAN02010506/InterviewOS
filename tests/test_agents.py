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
from interview_os.core.state import (
    AnswerEvaluation,
    AnswerReviewStatus,
    AnswerScoringSource,
    CompetencyEvaluation,
    EvaluationReport,
    InterviewState,
    JobDescription,
    LiveInterviewRecord,
    QuestionSuggestion,
    QuestionSuggestionType,
    TranscriptSegment,
    TranscriptSpeaker,
)
from interview_os.models.structured import parse_model_output


class MockLLM:
    async def chat(self, messages, **kwargs):
        return '{"name": "Test", "strengths": ["Python"], "weaknesses": ["Deploy"]}'

    async def embed(self, text):
        return [0.1, 0.2, 0.3]


class InvalidLLM:
    async def chat(self, messages, **kwargs):
        return "not json"


class WrongEntityLLM:
    async def chat(self, messages, **kwargs):
        return (
            '{"name":"InterviewOS","industry":"Software","dna":"Test",'
            '"public_sources":[{"title":"Invented","url":null}]}'
        )


class EmptyPlanLLM:
    async def chat(self, messages, **kwargs):
        return '{"questions":[]}'


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
    assert evaluation.spoken_analysis.rubric_version == "evidence-v2"
    assert "缺少已核验的量化结果" in evaluation.missing_signals
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
    assert "STAR 补充框架" in evaluation.improved_answer
    assert any("未提供的数字" in item for item in evaluation.feedback)
    assert "需要核验并补充真实量化结果" in evaluation.missing_signals
    # The model's invented observed signal remains a coaching hint only. The
    # persisted evidence must be the candidate's auditable original wording.
    assert state.evidence[0].signal == original
    assert state.evidence[0].polarity == EvidencePolarity.NEUTRAL
    assert all("推动团队扩张" not in item.evidence for item in evaluation.dimension_feedback)


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


@pytest.mark.asyncio
async def test_mock_agent_degrades_when_model_returns_empty_valid_plan():
    agent = MockInterviewAgent(llm_client=EmptyPlanLLM())
    state = InterviewState(job=JobDescription(competencies=["招聘策略"]))
    await agent.execute(state)
    assert state.mock_interview.questions[0].competency == "招聘策略"


@pytest.mark.asyncio
async def test_company_agent_preserves_authoritative_company_name():
    agent = CompanyAgent(llm_client=WrongEntityLLM())
    state = InterviewState()
    state.company.name = "芯世界"
    await agent.execute(state)
    assert state.company.name == "芯世界"
    assert state.company.public_sources == []
