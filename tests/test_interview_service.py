from uuid import uuid4

import pytest

from interview_os.database.storage import Storage
from interview_os.services.interview_service import (
    InterviewService,
    MockInterviewStateError,
    SessionNotFoundError,
    WorkflowExecutionError,
)
from interview_os.tools.web_search import SearchProvider, SearchResult


class MockLLM:
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "resume" in prompt.lower():
            return '```json\n{"name":"Ada","skills":["Python"],"strengths":["Systems"]}\n```'
        if "job description" in prompt.lower():
            return '{"title":"Platform Engineer","required_skills":["Python"],"competencies":["Design"]}'
        return '{"name":"Grace","career_pattern":"technical leader"}'

    async def embed(self, text):
        return []


class FakeSearchProvider(SearchProvider):
    async def search(self, query: str, limit: int = 5):
        return [
            SearchResult(
                title="Public profile",
                url="https://example.com/profile",
                snippet="Engineering leadership and systems thinking",
                source="fake",
            )
        ]


class WorkflowMockLLM:
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        lowered = prompt.lower()
        if "structured candidate profile" in lowered:
            return '{"name":"Ada","skills":["Python"],"strengths":["Systems"]}'
        if "job description" in lowered:
            return '{"title":"Platform Engineer","required_skills":["Python"],"competencies":["System Design"]}'
        if "company dna" in lowered:
            return '{"name":"Example","culture":"Engineering","preferences":["Ownership"],"dna":"Builder culture"}'
        if "fuse three inputs" in lowered:
            return '{"summary":"Lead with impact","key_risks":["Business context"],"answer_framework":["Problem","Impact","Solution"],"topics_to_emphasize":["Ownership"],"topics_to_avoid":[],"likely_questions":["Why this trade-off?"]}'
        if "interviewer profile" in lowered:
            return '{"name":"Grace","position":"CTO","likely_preferences":["First principles"]}'
        if "mock interview plan" in lowered:
            return '{"questions":[{"question":"Explain the architecture","competency":"System Design","rationale":"Tests depth","strong_signals":["Trade-offs"],"follow_ups":["How does it scale?"]}]}'
        if "analyze this interview answer" in lowered:
            return '{"content":0.8,"technical_depth":0.9,"structure":0.7,"impact":0.6,"feedback":["Add metrics"],"improved_answer":"Improved","observed_signals":["Explained trade-offs"],"missing_signals":["Business impact"]}'
        if "interview blueprint" in lowered:
            return '{"position":"Platform Engineer","rounds":[{"name":"Technical","goal":"Depth","questions":[{"question":"Design the system","competency":"System Design","rationale":"Core requirement","strong_signals":["Trade-offs"],"follow_ups":[]}],"evaluation_criteria":["Clear reasoning"]}]}'
        raise AssertionError(f"Unexpected prompt: {prompt}")

    async def embed(self, text):
        return []


@pytest.fixture
async def service(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await storage.init_db()
    yield InterviewService(storage, MockLLM())
    await storage.close()


@pytest.mark.asyncio
async def test_session_analysis_is_persisted(service):
    session_id, _ = await service.create_session(candidate_name="Initial")
    await service.analyze_resume(session_id, "Experienced Python resume")

    restored = InterviewService(service.storage, MockLLM())
    state = await restored.get_state(session_id)
    assert state.candidate.name == "Ada"
    assert state.candidate.skills == ["Python"]
    assert state.candidate.raw_resume_text == "Experienced Python resume"


@pytest.mark.asyncio
async def test_missing_session_raises(service):
    with pytest.raises(SessionNotFoundError):
        await service.get_state("missing")


@pytest.mark.asyncio
async def test_interviewer_research_sources_are_persisted(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'research.db'}")
    await storage.init_db()
    service = InterviewService(storage, MockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.analyze_interviewer(session_id, "Grace", "CTO", "Example")
    state = await service.get_state(session_id)
    assert state.interviewer.public_expressions[0]["url"] == "https://example.com/profile"
    await storage.close()


@pytest.mark.asyncio
async def test_candidate_prep_workflow_persists_structured_results(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'prep.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    state = await service.run_candidate_prep(
        session_id,
        resume_text="Python systems engineer",
        job_description="Platform engineer owning distributed systems",
        company_name="Example",
        interviewer_name="Grace",
        interviewer_position="CTO",
    )
    assert state.workflow.status.value == "completed"
    assert state.workflow.completed_steps == 6
    assert state.strategy.summary == "Lead with impact"
    assert state.mock_interview.questions[0].competency == "System Design"

    restored = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    restored_state = await restored.get_state(session_id)
    assert restored_state.strategy.summary == "Lead with impact"
    assert restored_state.next_action == "Start mock interview"
    await storage.close()


@pytest.mark.asyncio
async def test_enterprise_design_workflow_builds_rounds(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'design.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    state = await service.run_enterprise_design(
        session_id,
        resume_text="Python systems engineer",
        job_description="Platform engineer owning distributed systems",
        company_name="Example",
    )
    assert state.workflow.status.value == "completed"
    assert state.blueprint.rounds[0].questions[0].competency == "System Design"
    await storage.close()


@pytest.mark.asyncio
async def test_failed_workflow_records_progress(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'failed.db'}")
    await storage.init_db()
    service = InterviewService(storage, MockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    with pytest.raises(WorkflowExecutionError):
        await service.run_enterprise_design(
            session_id,
            resume_text="Python engineer",
            job_description="Platform role",
            company_name="Example",
        )
    state = await service.get_state(session_id)
    assert state.workflow.status.value == "failed"
    assert state.workflow.current_step == "interview_design_agent"
    assert "did not return any rounds" in state.workflow.error
    await storage.close()


@pytest.mark.asyncio
async def test_mock_interview_answer_creates_scored_evidence(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'mock.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python systems engineer",
        job_description="Platform engineer",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    assert question is not None

    state = await service.submit_mock_answer(
        session_id, question.id, "I compared options and explained the trade-offs."
    )
    assert state.mock_session.status.value == "completed"
    assert state.mock_session.responses[0].evaluation.overall_score() == pytest.approx(0.75)
    assert state.evidence[-1].competency == "System Design"
    assert state.evidence[-1].confidence == pytest.approx(0.75)
    assert state.next_action == "Generate evidence-based evaluation"
    await storage.close()


@pytest.mark.asyncio
async def test_mock_interview_rejects_out_of_order_answer(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'wrong-question.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python engineer",
        job_description="Platform engineer",
        company_name="Example",
    )
    await service.start_mock_interview(session_id)
    with pytest.raises(MockInterviewStateError, match="current question"):
        await service.submit_mock_answer(session_id, uuid4(), "Answer for a different question")
    state = await service.get_state(session_id)
    assert state.mock_session.current_question_index == 0
    assert state.mock_session.responses == []
    await storage.close()
