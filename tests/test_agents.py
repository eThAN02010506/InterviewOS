"""Tests for domain agents (with mock LLM)."""
import pytest

from interview_os.agents.candidate_agent import CandidateAgent
from interview_os.core.message import MessageType
from interview_os.core.state import InterviewState


class MockLLM:
    async def chat(self, messages, **kwargs):
        return '{"name": "Test", "strengths": ["Python"], "weaknesses": ["Deploy"]}'

    async def embed(self, text):
        return [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_candidate_agent():
    agent = CandidateAgent(llm_client=MockLLM())
    state = InterviewState()
    state.candidate.raw_resume_text = "Experienced Python developer"
    msg = await agent.execute(state)
    assert msg.type == MessageType.RESPONSE
    assert state.candidate.name == "Test"
