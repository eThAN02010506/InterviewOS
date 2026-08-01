"""Tests for domain agents (with mock LLM)."""
import pytest

from interview_os.agents.candidate_agent import CandidateAgent
from interview_os.agents.coach_agent import CoachAgent
from interview_os.core.message import MessageType
from interview_os.core.state import AnswerEvaluation, InterviewState
from interview_os.models.structured import parse_model_output


class MockLLM:
    async def chat(self, messages, **kwargs):
        return '{"name": "Test", "strengths": ["Python"], "weaknesses": ["Deploy"]}'

    async def embed(self, text):
        return [0.1, 0.2, 0.3]


class InvalidLLM:
    async def chat(self, messages, **kwargs):
        return "not json"


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
