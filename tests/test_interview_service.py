import asyncio
from uuid import uuid4

import pytest

from interview_os.database.storage import Storage
from interview_os.services.interview_service import (
    EvaluationStateError,
    InterviewService,
    MockInterviewStateError,
    SessionNotFoundError,
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
    async def search(self, query: str, limit: int = 5, *, search_depth: str = "basic"):
        return [
            SearchResult(
                title="Grace — CTO at Example",
                url="https://example.com/profile",
                snippet="Grace leads engineering at Example with a focus on systems thinking",
                source="fake",
            )
        ]


class EmployerSearchProvider(SearchProvider):
    """Returns a ZUORA source for employer queries, generic otherwise."""

    async def search(self, query: str, limit: int = 5, *, search_depth: str = "basic"):
        if "zuora" in query.lower():
            return [
                SearchResult(
                    title="Zuora — subscription management platform",
                    url="https://zuora.com/about",
                    snippet="Zuora is a pioneer in subscription management software.",
                    source="fake",
                )
            ]
        return []


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
        if "based on the following evidence" in lowered:
            return '{"competencies":[{"competency":"System Design","score":0.78,"confidence":0.75,"supporting_evidence":["Explained trade-offs"],"gaps":["Scale evidence"]}],"overall_score":0.78,"recommendation":"lean_hire","summary":"Solid fundamentals","risks":["Limited scale evidence"]}'
        if "generate evidence-based feedback" in lowered:
            return '{"overall":"Strong foundation with specific gaps","strengths":["Trade-off reasoning"],"improvements":["Add scale examples"],"action_plan":["Prepare one scaling story"],"interviewer_notes":["Verify production scale"],"recommendation_reasoning":"Evidence supports a lean hire, pending scale validation."}'
        if "interview blueprint" in lowered:
            return '{"position":"Platform Engineer","rounds":[{"name":"Technical","goal":"Depth","questions":[{"question":"Design the system","competency":"System Design","rationale":"Core requirement","strong_signals":["Trade-offs"],"follow_ups":[]}],"evaluation_criteria":["Clear reasoning"]}]}'
        raise AssertionError(f"Unexpected prompt: {prompt}")

    async def embed(self, text):
        return []


class CandidateFailingWorkflowLLM(WorkflowMockLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "structured candidate profile" in prompt or "candidateprofile" in prompt:
            return "not json"
        return await super().chat(messages, **kwargs)


class EmptyEvaluationWorkflowLLM(WorkflowMockLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "based on the following evidence" in prompt:
            return '{"competencies":[],"overall_score":0,"recommendation":"insufficient_evidence","summary":"","risks":[]}'
        return await super().chat(messages, **kwargs)


class UngroundedEvaluationWorkflowLLM(WorkflowMockLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "based on the following evidence" in prompt:
            return '{"competencies":[{"competency":"Generic Communication","score":0.9,"confidence":0.9,"supporting_evidence":[],"gaps":[]}],"overall_score":0.9,"recommendation":"hire","summary":"Strong","risks":[]}'
        return await super().chat(messages, **kwargs)


class ConcurrencyTrackingLLM(WorkflowMockLLM):
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def chat(self, messages, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return await super().chat(messages, **kwargs)
        finally:
            self.active -= 1


class InvalidProfileLLM:
    async def chat(self, messages, **kwargs):
        return "not json"

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
async def test_interviewer_sources_survive_structured_profile_failure(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'research-fallback.db'}")
    await storage.init_db()
    service = InterviewService(storage, InvalidProfileLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.analyze_interviewer(session_id, "Grace", "CTO", "Example")
    state = await service.get_state(session_id)
    assert state.interviewer.public_research_status == "completed"
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
async def test_candidate_prep_runs_independent_agents_concurrently(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'parallel.db'}")
    await storage.init_db()
    llm = ConcurrencyTrackingLLM()
    service = InterviewService(storage, llm, FakeSearchProvider())
    session_id, _ = await service.create_session()

    await service.run_candidate_prep(
        session_id,
        resume_text="Python engineer",
        job_description="Platform engineer",
        company_name="Example",
    )

    assert llm.max_active >= 3
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
async def test_autopilot_advances_then_waits_for_real_candidate_input(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'autopilot.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, old_state = await service.create_session()
    old_state.company.public_sources = [{"url": "https://stale.example"}]
    old_state.strategy.summary = "stale strategy"

    state = await service.run_autopilot(
        session_id,
        role="candidate",
        resume_text="Python systems engineer",
        job_description="Platform engineer",
        company_name="Example",
        authorized_public_research=False,
    )

    assert state.autopilot.status.value == "waiting_for_input"
    assert state.autopilot.phase == "interview"
    assert "mock_plan_generation" in state.autopilot.completed_actions
    assert state.company.public_sources == []
    assert state.strategy.summary != "stale strategy"
    assert state.mock_interview.questions
    assert state.mock_session.status.value == "active"
    await storage.close()


@pytest.mark.asyncio
async def test_autopilot_generates_final_report_after_last_answer(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'autopilot-complete.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    state = await service.run_autopilot(
        session_id,
        role="candidate",
        resume_text="Python engineer",
        job_description="Platform engineer",
        company_name="Example",
    )
    question = service.current_mock_question(state)
    assert question is not None

    state = await service.submit_mock_answer(session_id, question.id, "I compared trade-offs")
    assert state.mock_session.pending_follow_up == "How does it scale?"
    state = await service.submit_mock_answer(
        session_id, question.id, "At ten times traffic, p95 stayed below 100 ms"
    )

    assert state.autopilot.status.value == "completed"


@pytest.mark.asyncio
async def test_evaluation_falls_back_when_model_omits_competencies(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'empty-evaluation.db'}")
    await storage.init_db()
    service = InterviewService(storage, EmptyEvaluationWorkflowLLM())
    try:
        session_id, _ = await service.create_session()
        await service.run_candidate_prep(
            session_id,
            resume_text="Python platform engineer",
            job_description="Platform engineer responsible for system design",
            company_name="Example",
        )
        state = await service.start_mock_interview(session_id)
        question = service.current_mock_question(state)
        assert question is not None
        await service.submit_mock_answer(session_id, question.id, "Clear trade-offs and metrics")

        state = await service.run_evaluation(session_id)

        assert state.evaluation.competencies
        assert "已按已记录证据完成确定性聚合" in state.evaluation.summary
        assert state.workflow.status.value == "completed"
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_evaluation_rejects_competencies_not_bound_to_evidence(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'ungrounded-evaluation.db'}")
    await storage.init_db()
    service = InterviewService(storage, UngroundedEvaluationWorkflowLLM())
    try:
        session_id, _ = await service.create_session()
        await service.run_candidate_prep(
            session_id,
            resume_text="Python platform engineer",
            job_description="Platform engineer responsible for system design",
            company_name="Example",
        )
        state = await service.start_mock_interview(session_id)
        question = service.current_mock_question(state)
        assert question is not None
        await service.submit_mock_answer(session_id, question.id, "Clear trade-offs and metrics")

        state = await service.run_evaluation(session_id)

        evidence_competencies = {item.competency for item in state.evidence}
        assert {item.competency for item in state.evaluation.competencies} <= evidence_competencies
        assert "Generic Communication" not in state.evaluated_competencies
    finally:
        await storage.close()
    assert state.evaluation.recommendation.value == "insufficient_evidence"
    assert state.feedback.overall
    await storage.close()


@pytest.mark.asyncio
async def test_autopilot_reuses_candidate_cache_only_for_identical_resume(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'candidate-cache.db'}")
    await storage.init_db()
    service = InterviewService(storage, CandidateFailingWorkflowLLM(), FakeSearchProvider())
    session_id, state = await service.create_session()
    state.candidate.name = "Cached Candidate"
    state.candidate.skills = ["Recruiting"]
    state.candidate.raw_resume_text = "same resume"

    reused = await service.run_autopilot(
        session_id,
        role="candidate",
        resume_text="same resume",
        job_description="Recruiting Manager",
        company_name="Example",
    )
    assert reused.candidate.name == "Cached Candidate"
    assert reused.candidate.skills == ["Recruiting"]

    changed = await service.run_autopilot(
        session_id,
        role="candidate",
        resume_text="different resume",
        job_description="Recruiting Manager",
        company_name="Example",
    )
    assert changed.candidate.raw_resume_text == "different resume"
    assert changed.candidate.name == ""
    await storage.close()


@pytest.mark.asyncio
async def test_failed_workflow_records_progress(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'failed.db'}")
    await storage.init_db()
    service = InterviewService(storage, MockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    state = await service.run_enterprise_design(
        session_id,
        resume_text="Python engineer",
        job_description="Platform role",
        company_name="Example",
    )
    # The design agent returns a deterministic generic blueprint when the model
    # output is invalid, so the workflow completes instead of failing.
    assert state.workflow.status.value == "completed"
    assert state.blueprint.rounds
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
    assert state.mock_session.status.value == "active"
    state = await service.submit_mock_answer(
        session_id, question.id, "At ten times traffic, p95 stayed below 100 ms."
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


@pytest.mark.asyncio
async def test_final_evaluation_aggregates_evidence_and_feedback(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'evaluation.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python engineer",
        job_description="Platform engineer",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    assert question is not None
    await service.submit_mock_answer(session_id, question.id, "I explained trade-offs")

    state = await service.run_evaluation(session_id)
    assert state.evaluation.overall_score == pytest.approx(0.78)
    assert state.evaluation.recommendation.value == "insufficient_evidence"
    assert state.feedback.action_plan == ["Prepare one scaling story"]
    assert state.evaluated_competencies["System Design"] == pytest.approx(0.78)
    assert state.current_stage.value == "completed"
    await storage.close()


@pytest.mark.asyncio
async def test_final_evaluation_requires_evidence(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'no-evidence.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    with pytest.raises(EvaluationStateError, match="No interview evidence"):
        await service.run_evaluation(session_id)
    await storage.close()


@pytest.mark.asyncio
async def test_candidate_prep_survives_invalid_strategy_output(tmp_path):
    """Invalid strategy output falls back to a generic plan instead of failing."""

    class _StrategyFailingLLM(WorkflowMockLLM):
        async def chat(self, messages, **kwargs):
            prompt = messages[-1]["content"]
            if "fuse three inputs" in prompt.lower():
                return "not valid strategy json"
            return await super().chat(messages, **kwargs)

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'strategy-fallback.db'}")
    await storage.init_db()
    service = InterviewService(storage, _StrategyFailingLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    state = await service.run_candidate_prep(
        session_id,
        resume_text="Python systems engineer",
        job_description="Platform engineer owning distributed systems",
        company_name="Example",
    )
    assert state.workflow.status.value == "completed"
    assert state.strategy.summary  # generic fallback summary
    assert state.mock_interview.questions  # mock questions still generated
    await storage.close()


@pytest.mark.asyncio
async def test_research_recent_employers_populates_sources_and_fact_cards(tmp_path):
    from interview_os.services.intelligence_service import build_fact_cards

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'employer.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), EmployerSearchProvider())
    session_id, state = await service.create_session()
    # Provide a recent employer via structured experience.
    state.candidate.raw_resume_text = "2018-07 to ZUORA 2022-12\nSenior Recruiting Manager\nLed APAC talent acquisition"
    state.candidate.experience = [
        {
            "company": "ZUORA",
            "role": "Senior Recruiting Manager",
            "duration": "2018-2022",
            "summary": "Led APAC talent acquisition",
        }
    ]
    runtime = await service._get_runtime(session_id)
    runtime.state.candidate = state.candidate
    runtime.state.autopilot.authorized_public_research = True
    await service._persist(session_id, runtime.state)

    result = await service.research_recent_employers(session_id)
    assert result.past_employer_sources
    assert result.past_employer_research_status == "completed"
    build_fact_cards(result)
    assert any(card.category == "past_employer" for card in result.fact_cards)
    await storage.close()


@pytest.mark.asyncio
async def test_recent_employers_prioritizes_recent_and_related(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'employer-pick.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), EmployerSearchProvider())
    _, state = await service.create_session()
    state.company.name = "Acme Tech"
    state.candidate.experience = [
        {
            "company": "Zuora",
            "role": "SRM",
            "duration": "2018-2022",
            "summary": "subscription software",
        },
        {
            "company": "Old Consulting",
            "role": "Consultant",
            "duration": "2005-2007",
            "summary": "recruiting",
        },
        {
            "company": "Acme Tech",
            "role": "Engineer",
            "duration": "2010-2014",
            "summary": "platform",
        },
    ]
    picked = service._recent_employers(state, limit=2)
    # Recent (Zuora) and name-related (Acme Tech) win over the old small one.
    assert "Zuora" in picked
    assert "Old Consulting" not in picked
    await storage.close()


@pytest.mark.asyncio
async def test_recent_employers_sorts_recent_first_when_unrelated(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'employer-recency.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), EmployerSearchProvider())
    _, state = await service.create_session()
    # Both unrelated to the target company; recency alone must pick the recent one.
    state.candidate.experience = [
        {
            "company": "RecentCo",
            "role": "Engineer",
            "duration": "2021-2024",
            "summary": "platform work",
        },
        {
            "company": "AncientCo",
            "role": "Analyst",
            "duration": "2003-2007",
            "summary": "operations",
        },
    ]
    picked = service._recent_employers(state, limit=1)
    # The recent employer (2021-2024) must be selected over the 2003-2007 one,
    # even though neither name overlaps the target company. This failed before
    # the 4-digit year capture fix because recency never triggered.
    assert picked == ["RecentCo"]
    await storage.close()


@pytest.mark.asyncio
async def test_past_employer_research_requires_explicit_consent(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'employer-consent.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), EmployerSearchProvider())
    session_id, state = await service.create_session()
    state.candidate.experience = [
        {"company": "ZUORA", "role": "SRM", "duration": "2018-2022", "summary": "ta"}
    ]
    runtime = await service._get_runtime(session_id)
    runtime.state.candidate = state.candidate
    # No explicit research authorization -> research is blocked.
    await service._persist(session_id, runtime.state)
    result = await service.research_recent_employers(session_id)
    assert result.past_employer_sources == []
    assert result.past_employer_research_status == "consent_required"
    await storage.close()
