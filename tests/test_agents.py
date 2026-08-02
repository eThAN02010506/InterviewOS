"""Tests for domain agents (with mock LLM)."""
import pytest

from interview_os.agents.candidate_agent import CandidateAgent
from interview_os.agents.coach_agent import CoachAgent
from interview_os.agents.company_agent import CompanyAgent
from interview_os.agents.interview_strategy_agent import InterviewStrategyAgent
from interview_os.agents.live_interview_agent import LiveInterviewAgent
from interview_os.agents.mock_interview_agent import MockInterviewAgent
from interview_os.core.message import MessageType
from interview_os.core.state import (
    AnswerEvaluation,
    InterviewState,
    JobDescription,
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
    assert "人工复核" in evaluation.feedback[0]
    assert state.evidence[0].confidence == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_candidate_agent_extracts_explicit_name_when_model_output_fails():
    agent = CandidateAgent(llm_client=InvalidLLM())
    state = InterviewState()
    await agent.execute(state, "Experience Name ：JL Led recruiting across APAC")
    assert state.candidate.name == "JL"


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
async def test_mock_agent_degrades_to_current_job_competency_questions():
    agent = MockInterviewAgent(llm_client=InvalidLLM())
    state = InterviewState(job=JobDescription(competencies=["招聘策略", "团队领导力"]))
    await agent.execute(state)
    assert [question.competency for question in state.mock_interview.questions] == [
        "招聘策略",
        "团队领导力",
    ]


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
