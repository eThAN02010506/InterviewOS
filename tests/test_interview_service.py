import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from interview_os.core.state import (
    InterviewQuestion,
    LiveInterviewStatus,
    MockSessionStatus,
    TranscriptSegment,
)
from interview_os.database.storage import Storage
from interview_os.services.interview_service import (
    CandidateSessionStateError,
    EvaluationStateError,
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
    def __init__(self):
        self.queries: list[str] = []

    async def search(self, query: str, limit: int = 5, *, search_depth: str = "basic"):
        self.queries.append(query)
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

    def __init__(self):
        self.queries: list[str] = []

    async def search(self, query: str, limit: int = 5, *, search_depth: str = "basic"):
        self.queries.append(query)
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


class JobDescriptionSearchProvider(SearchProvider):
    def __init__(self):
        self.queries = []

    async def search(self, query: str, limit: int = 5, *, search_depth: str = "basic"):
        self.queries.append(query)
        return [
            SearchResult(
                title="Example Platform Engineer 招聘",
                url="https://careers.example.com/platform-engineer",
                snippet="岗位职责：设计分布式平台和 SLO。任职要求：熟悉 Python 与可观测性。",
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
        if "more mock interview questions" in lowered:
            return '{"questions":[{"question":"Describe a second incident","competency":"System Design","rationale":"Refill","strong_signals":["Metrics"],"follow_ups":[],"answer_framework":"Reference the 2024 incident"},{"question":"How do you measure platform health","competency":"System Design","rationale":"Refill","strong_signals":["SLOs"],"follow_ups":[],"answer_framework":"Use the SLO story"}]}'
        if "参考答案提示框架" in lowered:
            return (
                '{"frameworks":['
                '{"question_index":0,"answer_framework":"用 STAR 讲 2024 事故复盘"},'
                '{"question_index":1,"answer_framework":"用 SLO 故事量化平台健康"},'
                '{"question_index":2,"answer_framework":"讲清取舍并给量化结果"},'
                '{"question_index":3,"answer_framework":"引用 ZUORA 账单平台扩展经历"},'
                '{"question_index":4,"answer_framework":"突出团队从 4 人扩展到 15 人的领导力"}]}'
            )
        if "mock interview plan" in lowered:
            return '{"questions":[{"question":"Explain the architecture","competency":"System Design","rationale":"Tests depth","strong_signals":["Trade-offs"],"follow_ups":["How does it scale?"],"answer_framework":"讲清取舍并给出量化结果"}]}'
        if "你是面试问题分析器" in prompt:
            return '{"answer_type":"motivation","answer_type_label":"动机匹配题","assessment_goal":"验证职业选择是否具体且稳定","competency":"求职动机","answer_boundary":["说明动机与匹配关系"],"common_mistakes":["只谈离职原因","没有岗位证据"],"transfer_principle":"保留选择证据，按问法切换决策标准和长期目标。","related_questions":["这个岗位最吸引你的工作内容是什么？","你会用哪些标准决定是否加入？","这次选择与三年目标有什么关系？"],"likely_follow_ups":["什么经历支持这个判断？","什么情况会改变你的决定？"]}'
        if "analyze this interview answer" in lowered:
            return '{"content":0.8,"technical_depth":0.9,"structure":0.7,"impact":0.6,"feedback":["Add metrics"],"improved_answer":"Improved","observed_signals":["Explained trade-offs"],"missing_signals":["Business impact"]}'
        if "structured final interview evaluation" in lowered:
            return (
                '{"summary":"现有证据显示了系统设计取舍，但业务结果仍需补充。",'
                '"competency_reviews":[{"competency_id":"C1",'
                '"evidence_numbers":[1],"assessment":"回答说明了方案取舍；当前分数受结果证据不足限制。",'
                '"next_probe":"请补充该方案上线后的已核验业务或稳定性结果。"}]}'
            )
        if "based on the following evidence" in lowered:
            return '{"competencies":[{"competency":"System Design","score":0.78,"confidence":0.75,"supporting_evidence":["Explained trade-offs"],"gaps":["Scale evidence"]}],"overall_score":0.78,"recommendation":"lean_hire","summary":"Solid fundamentals","risks":["Limited scale evidence"]}'
        if "generate evidence-based feedback" in lowered:
            return '{"overall":"Strong foundation with specific gaps","strengths":["Trade-off reasoning"],"improvements":["Add scale examples"],"action_plan":["Prepare one scaling story"],"interviewer_notes":["Verify production scale"],"recommendation_reasoning":"Evidence supports a lean hire, pending scale validation."}'
        if "interview blueprint" in lowered:
            return '{"position":"Platform Engineer","rounds":[{"name":"Technical","goal":"Depth","questions":[{"question":"Design the system","competency":"System Design","rationale":"Core requirement","strong_signals":["Trade-offs"],"follow_ups":[]}],"evaluation_criteria":["Clear reasoning"]}]}'
        raise AssertionError(f"Unexpected prompt: {prompt}")

    async def embed(self, text):
        return []


class JobPromptCapturingLLM(WorkflowMockLLM):
    def __init__(self):
        self.job_prompt = ""

    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "job description" in prompt.lower():
            self.job_prompt = prompt
        return await super().chat(messages, **kwargs)


class CandidateFailingWorkflowLLM(WorkflowMockLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "structured candidate profile" in prompt or "candidateprofile" in prompt:
            return "not json"
        return await super().chat(messages, **kwargs)


class IdentityOmittingWorkflowLLM(WorkflowMockLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "structured candidate profile" in prompt or "candidateprofile" in prompt:
            return '{"name":"","skills":["Recruiting"],"strengths":["APAC hiring"]}'
        return await super().chat(messages, **kwargs)


class EmptyEvaluationWorkflowLLM(WorkflowMockLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "structured final interview evaluation" in prompt:
            return '{"summary":"","competency_reviews":[]}'
        if "based on the following evidence" in prompt:
            return '{"competencies":[],"overall_score":0,"recommendation":"insufficient_evidence","summary":"","risks":[]}'
        return await super().chat(messages, **kwargs)


class UngroundedEvaluationWorkflowLLM(WorkflowMockLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "structured final interview evaluation" in prompt:
            return (
                '{"summary":"Strong","competency_reviews":['
                '{"competency_id":"C99","evidence_numbers":[1],'
                '"assessment":"Strong","next_probe":"More?"}]}'
            )
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
async def test_title_only_job_is_researched_before_question_generation(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'job-research.db'}")
    await storage.init_db()
    search = JobDescriptionSearchProvider()
    llm = JobPromptCapturingLLM()
    service = InterviewService(storage, llm, search)
    session_id, _ = await service.create_session()

    state = await service.run_candidate_prep(
        session_id,
        resume_text="Python systems engineer",
        job_description="Platform Engineer",
        company_name="Example",
        authorized_public_research=True,
    )

    assert search.queries and "岗位职责" in search.queries[0]
    assert "设计分布式平台" in llm.job_prompt
    assert state.job.raw_description == "Platform Engineer"
    assert state.job_review.public_research_status == "completed"
    assert state.job_review.public_sources[0]["url"].startswith("https://careers.example.com")
    assert state.job_review.requirements
    assert all(item.origin.value == "inferred" for item in state.job_review.requirements)
    assert any("待确认" in warning for warning in state.job_review.warnings)
    await storage.close()


@pytest.mark.asyncio
async def test_title_only_job_research_provenance_is_restored_when_workflow_fails(
    tmp_path, monkeypatch
):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'job-research-failure.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), JobDescriptionSearchProvider())
    session_id, _ = await service.create_session()

    async def fail_after_job_agent(_session_id, runtime, *_args, **_kwargs):
        # Reproduce a later Agent failure after the internal source-enriched
        # prompt has temporarily reached state.job.raw_description.
        runtime.state.job.raw_description = "INTERNAL SOURCE-ENRICHED PROMPT"
        raise WorkflowExecutionError("strategy unavailable")

    monkeypatch.setattr(service, "_execute_workflow", fail_after_job_agent)
    with pytest.raises(WorkflowExecutionError, match="strategy unavailable"):
        await service.run_candidate_prep(
            session_id,
            resume_text="Python systems engineer",
            job_description="Platform Engineer",
            company_name="Example",
            authorized_public_research=True,
        )

    restored = await service.get_state(session_id)
    assert restored.job.raw_description == "Platform Engineer"
    assert restored.job_review.public_research_status == "completed"
    assert restored.job_review.public_sources
    assert "INTERNAL SOURCE-ENRICHED PROMPT" not in restored.model_dump_json()
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
async def test_autopilot_forwards_public_research_consent_to_candidate_workflow(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'autopilot-research.db'}")
    await storage.init_db()
    search = FakeSearchProvider()
    service = InterviewService(storage, IdentityOmittingWorkflowLLM(), search)
    session_id, _ = await service.create_session(candidate_name="JLO")

    state = await service.run_autopilot(
        session_id,
        role="candidate",
        resume_text="Talent acquisition leader",
        job_description="Platform engineer",
        company_name="Example",
        interviewer_name="Grace",
        interviewer_position="CTO",
        authorized_public_research=True,
    )

    assert state.autopilot.authorized_public_research is True
    assert search.queries
    assert state.company.public_research_status == "completed"
    assert state.company.public_sources
    assert state.interviewer.public_research_status == "completed"
    assert state.interviewer.public_expressions
    assert state.candidate.name == "JLO"
    await storage.close()


@pytest.mark.asyncio
async def test_autopilot_without_consent_does_not_call_public_search(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'autopilot-no-research.db'}")
    await storage.init_db()
    search = FakeSearchProvider()
    service = InterviewService(storage, WorkflowMockLLM(), search)
    session_id, _ = await service.create_session()

    state = await service.run_autopilot(
        session_id,
        role="candidate",
        resume_text="Python systems engineer",
        job_description="Platform engineer",
        company_name="Example",
        interviewer_name="Grace",
        authorized_public_research=False,
    )

    assert search.queries == []
    assert state.company.public_sources == []
    assert state.interviewer.public_expressions == []
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
    assert state.mock_session.status.value == "active"
    state = await service.advance_mock_interview(session_id)
    assert state.mock_session.pending_follow_up
    assert state.mock_session.pending_follow_up_stage == "foundation"
    assert state.mock_session.pending_follow_up_rationale
    follow_question = service.current_mock_question(state)
    state = await service.submit_mock_answer(
        session_id, follow_question.id, "At ten times traffic, p95 stayed below 100 ms"
    )
    # Autopilot completes only when the user ends the interview.
    assert state.autopilot.status.value != "completed"
    state = await service.finish_mock_interview(session_id)
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

        state = await service.finish_mock_interview(session_id)

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

        state = await service.finish_mock_interview(session_id)

        evidence_competencies = {item.competency for item in state.evidence}
        assert {item.competency for item in state.evaluation.competencies} <= evidence_competencies
        assert "Generic Communication" not in state.evaluated_competencies
    finally:
        await storage.close()
    assert state.evaluation.recommendation.value == "insufficient_evidence"
    assert state.feedback.overall
    await storage.close()


@pytest.mark.asyncio
async def test_autopilot_reuses_analysis_only_for_identical_resume_but_keeps_name(tmp_path):
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
    assert changed.candidate.name == "Cached Candidate"
    assert changed.candidate.skills == []
    await storage.close()


@pytest.mark.asyncio
async def test_autopilot_clears_review_and_employer_data_when_resume_changes(tmp_path):
    from interview_os.core.state import ResumeClaim, ResumeClaimStatus

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'resume-switch.db'}")
    await storage.init_db()
    service = InterviewService(storage, CandidateFailingWorkflowLLM(), FakeSearchProvider())
    session_id, state = await service.create_session()
    state.candidate.raw_resume_text = "candidate A"
    state.resume_review.claims = [
        ResumeClaim(
            category="employment",
            statement="Candidate A worked at SecretCo",
            status=ResumeClaimStatus.CONFIRMED,
        )
    ]
    state.past_employer_sources = [{"title": "SecretCo"}]
    state.past_employer_research_status = "completed"
    state.past_employer_block = "SecretCo public profile"
    await service._persist(session_id, state)

    changed = await service.run_autopilot(
        session_id,
        role="candidate",
        resume_text="candidate B",
        job_description="Recruiting Manager",
        company_name="Example",
    )

    assert changed.resume_review.claims == []
    assert changed.past_employer_sources == []
    assert changed.past_employer_block == ""
    assert "Candidate A" not in changed.candidate_evidence_context()
    await storage.close()


@pytest.mark.asyncio
async def test_workflow_rerun_is_blocked_after_real_interview_activity(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'candidate-lifecycle.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    state = await service.run_candidate_prep(
        session_id, resume_text="Candidate A", job_description="Platform", company_name="Example"
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    state = await service.submit_mock_answer(session_id, question.id, "Preserved answer")
    response_id = state.mock_session.responses[0].id

    with pytest.raises(CandidateSessionStateError, match="请新建会话"):
        await service.run_candidate_prep(
            session_id,
            resume_text="Candidate B",
            job_description="Different role",
            company_name="Other Company",
        )

    preserved = await service.get_state(session_id)
    assert preserved.mock_session.responses[0].id == response_id
    assert preserved.evidence
    assert preserved.candidate.raw_resume_text == "Candidate A"
    await storage.close()


@pytest.mark.asyncio
async def test_resume_upload_rejects_active_session_before_parsing(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'resume-upload-preflight.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, state = await service.create_session()
    state.live_interview.segments.append(
        TranscriptSegment(sequence=1, speaker="candidate", text="Real interview answer")
    )
    await service._persist(session_id, state)

    def unexpected_parse(filename, content):
        raise AssertionError("resume parser must not run for an active interview session")

    service.resume_processor.process = unexpected_parse
    with pytest.raises(CandidateSessionStateError, match="请新建会话"):
        await service.upload_resume(session_id, "replacement.pdf", b"private replacement")

    await storage.close()


@pytest.mark.asyncio
async def test_resume_transition_invalidates_stale_outputs_before_answers(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'candidate-reset.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, state = await service.create_session()
    state.candidate.raw_resume_text = "Candidate A"
    state.strategy.summary = "Old strategy"
    state.mock_interview.questions = [InterviewQuestion(question="Old question")]
    state.evaluation.summary = "Old report"
    await service._persist(session_id, state)
    runtime = await service._get_runtime(session_id)

    await service._prepare_resume_transition(session_id, runtime, "Candidate B")

    reset = await service.get_state(session_id)
    assert reset.candidate.raw_resume_text == "Candidate B"
    assert reset.strategy.summary == ""
    assert reset.mock_interview.questions == []
    assert reset.evaluation.summary == ""
    assert reset.resume_review.claims == []
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
    # The interview does not auto-complete; the user advances, which may present
    # a follow-up (missing signals) before the next main question.
    state = await service.advance_mock_interview(session_id)
    assert state.mock_session.pending_follow_up  # follow-up offered
    follow_question = service.current_mock_question(state)
    assert follow_question is not None
    state = await service.submit_mock_answer(
        session_id, follow_question.id, "At ten times traffic, p95 stayed below 100 ms."
    )
    assert state.mock_session.status.value == "active"
    # User ends the interview manually; evaluation runs because answers exist.
    state = await service.finish_mock_interview(session_id)
    assert state.mock_session.status.value == "completed"
    first_evaluation = state.mock_session.responses[0].evaluation
    # The broad model question is tightened into a behavioral evidence question.
    # Trade-off language alone still lacks a concrete situation and result.
    assert first_evaluation.overall_score() == pytest.approx(0.4125)
    assert first_evaluation.spoken_analysis.calibration_notes
    assert state.mock_session.responses[0].evaluation.spoken_analysis.pre_calibration_scores
    assert state.evidence[-1].competency == "System Design"
    assert state.evidence[-1].confidence == pytest.approx(0.75)
    assert "evaluation" in state.next_action.lower()
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

    state = await service.finish_mock_interview(session_id)
    # The answer omitted the tightened question's case, personal-decision and
    # result requirements, so evidence calibration lowers the raw 0.75 average.
    assert state.evaluation.overall_score == pytest.approx(0.4125)
    assert state.evaluation.recommendation.value == "insufficient_evidence"
    assert "Business impact" not in state.feedback.action_plan
    assert any("直接回应题目核心" in item for item in state.feedback.action_plan)
    assert any("明确具体公司/业务场景" in item for item in state.feedback.action_plan)
    assert "需要更多独立回答交叉验证" in state.evaluation.competencies[0].gaps
    assert state.evaluated_competencies["System Design"] == pytest.approx(0.4125)
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
    provider = EmployerSearchProvider()
    service = InterviewService(storage, WorkflowMockLLM(), provider)
    session_id, state = await service.create_session()
    # Provide a recent employer via structured experience.
    state.candidate.raw_resume_text = (
        "2018-07 to ZUORA 2022-12\nSenior Recruiting Manager\nLed APAC talent acquisition"
    )
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
    assert all("招聘" not in query and "hiring" not in query.lower() for query in provider.queries)
    assert "不是候选人经历证据" in result.past_employer_block
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


@pytest.mark.asyncio
async def test_recent_employer_research_uses_only_confirmed_review_claims(tmp_path):
    from interview_os.core.state import ResumeClaim, ResumeClaimStatus

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'employer-review.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), EmployerSearchProvider())
    _, state = await service.create_session()
    state.candidate.experience = [
        {"company": "ZUORA", "duration": "2018-2022", "role": "Manager"},
        {"company": "UnconfirmedCo", "duration": "2022-2025", "role": "Director"},
    ]
    state.resume_review.claims = [
        ResumeClaim(
            category="employment",
            statement="2018-2022 ZUORA Manager",
            status=ResumeClaimStatus.CONFIRMED,
        ),
        ResumeClaim(category="employment", statement="2022-2025 UnconfirmedCo Director"),
    ]
    assert service._recent_employers(state, limit=5) == ["ZUORA"]
    await storage.close()


@pytest.mark.asyncio
async def test_mock_interview_pool_includes_likely_and_competency(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'pool.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    state = await service.run_candidate_prep(
        session_id,
        resume_text="Python systems engineer",
        job_description="Platform engineer",
        company_name="Example",
    )
    sources = {q.source for q in state.mock_interview.questions}
    assert "likely" in sources or "competency" in sources
    assert all(q.answer_framework for q in state.mock_interview.questions)
    # Every framework is source-bounded instead of accepting free-form model
    # additions as candidate facts.
    assert all("系统不会替你生成经历" in q.answer_framework for q in state.mock_interview.questions)
    assert len({q.question for q in state.mock_interview.questions}) == len(
        state.mock_interview.questions
    )
    await storage.close()


@pytest.mark.asyncio
async def test_mock_interview_does_not_auto_complete(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'no-complete.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    state = await service.submit_mock_answer(session_id, question.id, "A clear answer")
    assert state.mock_session.status.value == "active"
    state = await service.advance_mock_interview(session_id)
    assert state.mock_session.status.value == "active"
    # No auto-complete: finish is required.
    state = await service.finish_mock_interview(session_id)
    assert state.mock_session.status.value == "completed"
    await storage.close()


@pytest.mark.asyncio
async def test_mock_interview_retry_replaces_without_duplicate_evidence(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'retry.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    state = await service.submit_mock_answer(session_id, question.id, "First attempt")
    evidence_count = len(state.evidence)
    # Retry the same question: previous response + its evidence are replaced.
    state = await service.submit_mock_answer(
        session_id, question.id, "Improved attempt", retry=True
    )
    assert len(state.mock_session.responses) == 1
    assert state.mock_session.responses[0].answer == "Improved attempt"
    assert len(state.mock_session.attempt_history) == 1
    assert state.mock_session.attempt_history[0].answer == "First attempt"
    assert len(state.evidence) == evidence_count  # no duplicate evidence
    await storage.close()


@pytest.mark.asyncio
async def test_mock_audio_cleanup_retains_archived_retry_recording(tmp_path):
    recordings_dir = tmp_path / "recordings"
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'audio-cleanup.db'}")
    await storage.init_db()
    service = InterviewService(
        storage,
        WorkflowMockLLM(),
        FakeSearchProvider(),
        recordings_dir=recordings_dir,
    )
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    recording_id = uuid4()
    archived_filename = await service.save_mock_answer_audio(
        session_id, recording_id, b"RIFF-recording", extension="wav"
    )
    await service.submit_mock_answer(
        session_id,
        question.id,
        "Recorded attempt",
        recording_id=recording_id,
    )
    state = await service.submit_mock_answer(
        session_id,
        question.id,
        "Typed replacement",
        retry=True,
    )
    orphan = recordings_dir / f"mock-{session_id}-{uuid4()}.wav"
    orphan.write_bytes(b"orphan")

    removed = await service.cleanup_orphaned_mock_audio(max_age_seconds=0)

    assert removed == 1
    assert not orphan.exists()
    assert (recordings_dir / archived_filename).exists()
    assert state.mock_session.attempt_history[0].audio_file == archived_filename
    await storage.close()


@pytest.mark.asyncio
async def test_mock_retry_failure_preserves_previous_answer_and_evidence(tmp_path, monkeypatch):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'retry-atomic.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    state = await service.submit_mock_answer(session_id, question.id, "First attempt")
    original_response = state.mock_session.responses[0].model_copy(deep=True)
    original_evidence = state.evidence[-1].model_copy(deep=True)
    runtime = await service._get_runtime(session_id)
    coach = runtime.get_agent("coach_agent")

    async def fail_before_commit(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(coach, "execute", fail_before_commit)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await service.submit_mock_answer(
            session_id,
            question.id,
            "Attempt that cannot be scored",
            retry=True,
            retry_response_id=original_response.id,
        )

    persisted = await service.get_state(session_id)
    assert persisted.mock_session.responses == [original_response]
    assert persisted.mock_session.attempt_history == []
    assert persisted.evidence[-1] == original_evidence
    await storage.close()


@pytest.mark.asyncio
async def test_current_mock_question_includes_answer_framework(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'fw.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    assert question is not None
    assert question.answer_framework
    await storage.close()


@pytest.mark.asyncio
async def test_advance_can_skip_unanswered_question(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'advance-skip.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    # Advancing before answering is allowed (the interviewer stays in control):
    # the index moves to the next question without an answer.
    before = service.current_mock_question(state)
    state = await service.advance_mock_interview(session_id)
    after = service.current_mock_question(state)
    assert before is not None and after is not None
    assert state.mock_session.current_question_index == 1
    await storage.close()


@pytest.mark.asyncio
async def test_follow_up_answer_can_deepen_then_user_can_skip_and_advance(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'followup-clear.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    # Answer main -> advance offers follow-up.
    await service.submit_mock_answer(session_id, question.id, "Main answer")
    state = await service.advance_mock_interview(session_id)
    assert state.mock_session.pending_follow_up
    follow = service.current_mock_question(state)
    # Answer follow-up -> pending cleared, so the UI shows actions.
    state = await service.submit_mock_answer(session_id, follow.id, "Follow-up answer")
    assert not state.mock_session.pending_follow_up
    # The next advance uses the follow-up answer to choose a deeper branch.
    state = await service.advance_mock_interview(session_id)
    assert state.mock_session.pending_follow_up
    assert state.mock_session.pending_follow_up_stage == "pressure"
    assert len(state.mock_session.follow_up_history[str(question.id)]) == 2
    # The existing control contract remains: another advance skips a pending
    # probe and moves to the next main question.
    state = await service.advance_mock_interview(session_id)
    assert not state.mock_session.pending_follow_up
    assert state.mock_session.current_question_index >= 1
    await storage.close()


@pytest.mark.asyncio
async def test_follow_up_retry_targets_explicit_response(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'followup-retry.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id, resume_text="Python", job_description="Platform", company_name="Example"
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    await service.submit_mock_answer(session_id, question.id, "Main answer")
    state = await service.advance_mock_interview(session_id)
    follow = service.current_mock_question(state)
    state = await service.submit_mock_answer(session_id, follow.id, "First follow-up")
    follow_record = state.mock_session.responses[-1]

    state = await service.submit_mock_answer(
        session_id,
        question.id,
        "Improved follow-up",
        retry=True,
        retry_response_id=follow_record.id,
    )

    matching = [item for item in state.mock_session.responses if item.is_follow_up]
    assert len(matching) == 1
    assert matching[0].answer == "Improved follow-up"
    assert state.mock_session.responses[0].answer == "Main answer"
    await storage.close()


@pytest.mark.asyncio
async def test_retrying_main_answer_removes_dependent_follow_up_evidence(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'main-retry-children.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id, resume_text="Python", job_description="Platform", company_name="Example"
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    state = await service.submit_mock_answer(session_id, question.id, "Old main answer")
    main_record = state.mock_session.responses[-1]
    state = await service.advance_mock_interview(session_id)
    follow = service.current_mock_question(state)
    state = await service.submit_mock_answer(session_id, follow.id, "Old follow-up answer")
    assert len(state.mock_session.responses) == 2
    assert len(state.evidence) == 2

    state = await service.submit_mock_answer(
        session_id,
        question.id,
        "New main answer",
        retry=True,
        retry_response_id=main_record.id,
    )

    assert len(state.mock_session.responses) == 1
    assert state.mock_session.responses[0].answer == "New main answer"
    assert not state.mock_session.responses[0].is_follow_up
    assert len(state.evidence) == 1
    assert state.evidence[0].source_record_id == state.mock_session.responses[0].id
    await storage.close()


@pytest.mark.asyncio
async def test_finish_mock_interview_recovers_when_evaluation_fails(tmp_path, monkeypatch):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'finish-retry.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id, resume_text="Python", job_description="Platform", company_name="Example"
    )
    state = await service.start_mock_interview(session_id)
    question = service.current_mock_question(state)
    await service.submit_mock_answer(session_id, question.id, "Answer")
    monkeypatch.setattr(
        service,
        "run_evaluation",
        AsyncMock(side_effect=WorkflowExecutionError("model unavailable")),
    )

    with pytest.raises(WorkflowExecutionError, match="model unavailable"):
        await service.finish_mock_interview(session_id)
    recovered = await service.get_state(session_id)
    assert recovered.mock_session.status == MockSessionStatus.ACTIVE
    assert recovered.mock_session.responses

    monkeypatch.setattr(service, "run_evaluation", AsyncMock(return_value=recovered))
    completed = await service.finish_mock_interview(session_id)
    assert completed.mock_session.status == MockSessionStatus.COMPLETED
    await storage.close()


@pytest.mark.asyncio
async def test_previous_mock_question_navigates_back(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'prev.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    # Advance to the second question.
    state = await service.advance_mock_interview(session_id)
    assert state.mock_session.current_question_index == 1
    # Go back to the first.
    state = await service.previous_mock_question(session_id)
    assert state.mock_session.current_question_index == 0
    # Going back at the first question raises.
    with pytest.raises(MockInterviewStateError):
        await service.previous_mock_question(session_id)
    await storage.close()


@pytest.mark.asyncio
async def test_answered_previous_question_requires_explicit_retry(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'prev-retry.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    first = service.current_mock_question(state)
    await service.submit_mock_answer(session_id, first.id, "First answer")
    # First /next offers a follow-up; second /next skips it and reaches question 2.
    await service.advance_mock_interview(session_id)
    state = await service.advance_mock_interview(session_id)
    second = service.current_mock_question(state)
    await service.submit_mock_answer(session_id, second.id, "Second answer")
    state = await service.previous_mock_question(session_id)
    assert service.current_mock_question(state).id == first.id

    with pytest.raises(MockInterviewStateError, match="retry=true"):
        await service.submit_mock_answer(session_id, first.id, "Duplicate answer")
    state = await service.submit_mock_answer(session_id, first.id, "Replacement answer", retry=True)
    first_responses = [
        item for item in state.mock_session.responses if item.question_id == first.id
    ]
    assert len(first_responses) == 1
    assert first_responses[0].answer == "Replacement answer"
    await storage.close()


@pytest.mark.asyncio
async def test_exhausted_mock_pool_gets_immediate_unique_fallback(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'pool-boundary.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.run_candidate_prep(
        session_id,
        resume_text="Python",
        job_description="Platform",
        company_name="Example",
    )
    state = await service.start_mock_interview(session_id)
    original_count = len(state.mock_interview.questions)
    state.mock_session.current_question_index = original_count - 1
    await service._persist(session_id, state)

    state = await service.advance_mock_interview(session_id)
    assert state.mock_session.current_question_index == original_count
    fallback = service.current_mock_question(state)
    assert fallback is not None
    assert fallback.answer_framework
    assert f"第 {original_count + 1} 个" in fallback.question
    await service._background.flush()
    await storage.close()


@pytest.mark.asyncio
async def test_runtime_reload_clears_orphaned_mock_refill_flag(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'refill-restart.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, state = await service.create_session()
    state.mock_session.refill_in_flight = True
    await service._persist(session_id, state)

    restored = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    loaded = await restored.get_state(session_id)
    assert loaded.mock_session.refill_in_flight is False
    await storage.close()


@pytest.mark.asyncio
async def test_runtime_reload_upgrades_legacy_mock_questions(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'question-upgrade.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, state = await service.create_session()
    state.mock_interview.questions = [
        InterviewQuestion(question="谈谈你的经验", competency="招聘战略")
    ]
    await service._persist(session_id, state)

    restored = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    loaded = await restored.get_state(session_id)
    question = loaded.mock_interview.questions[0]

    assert "具体且已经发生的案例" in question.question
    assert question.question_requirements
    assert question.example_answer
    await storage.close()


@pytest.mark.asyncio
async def test_runtime_reload_recovers_interrupted_mock_evaluation(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'evaluation-restart.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, state = await service.create_session()
    state.mock_session.status = MockSessionStatus.EVALUATING
    await service._persist(session_id, state)

    restored = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    loaded = await restored.get_state(session_id)

    assert loaded.mock_session.status == MockSessionStatus.ACTIVE
    assert "retry" in loaded.next_action.lower()
    await storage.close()


@pytest.mark.asyncio
async def test_custom_mock_question_starts_practice_with_full_support(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'custom-question.db'}")
    await storage.init_db()
    service = InterviewService(storage)
    session_id, _ = await service.create_session(job_title="产品负责人")

    original = "为什么你想从大公司转到创业公司？"
    state = await service.add_custom_mock_question(
        session_id,
        original,
        competency="求职动机",
    )

    question = service.current_mock_question(state)
    assert state.mock_session.status == MockSessionStatus.ACTIVE
    assert question is not None
    assert question.question == original
    assert question.competency == "求职动机"
    assert question.source == "custom"
    assert question.question_requirements
    assert question.answer_framework
    assert question.example_answer
    assert question.understanding is not None
    assert question.understanding.answer_type == "motivation"
    assert len(question.understanding.related_questions) == 3
    assert question.understanding.role_relevance_source == "title_inference"
    assert len(question.understanding.answer_levels) == 3
    assert len(question.understanding.probe_tree) == 4
    assert question.follow_ups == question.understanding.likely_follow_ups[:3]

    persisted = await storage.get_session_state(session_id, owner_id="local")
    assert persisted is not None
    assert persisted["mock_interview"]["questions"][0]["source"] == "custom"

    restored = InterviewService(storage)
    loaded = await restored.get_state(session_id)
    restored_question = restored.current_mock_question(loaded)
    assert restored_question is not None
    assert restored_question.question == original
    assert restored_question.understanding is not None
    assert restored_question.understanding.decision_criteria
    await storage.close()


@pytest.mark.asyncio
async def test_custom_mock_question_inserts_after_active_question_and_deduplicates(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'custom-question-order.db'}")
    await storage.init_db()
    service = InterviewService(storage)
    session_id, state = await service.create_session()
    state.mock_interview.questions = [
        InterviewQuestion(question="原来的第一题", competency="沟通"),
        InterviewQuestion(question="原来的第二题", competency="协作"),
    ]
    await service._persist(session_id, state)
    await service.start_mock_interview(session_id)

    state = await service.add_custom_mock_question(session_id, "我最担心被问的问题？")
    assert state.mock_session.current_question_index == 1
    assert state.mock_interview.questions[0].question == "原来的第一题"
    assert state.mock_interview.questions[1].source == "custom"
    assert state.mock_interview.questions[2].question == "原来的第二题"

    state = await service.add_custom_mock_question(
        session_id, "  我最担心被问的问题？  "
    )
    assert len(state.mock_interview.questions) == 3
    assert state.mock_session.current_question_index == 1
    await storage.close()


@pytest.mark.asyncio
async def test_custom_mock_question_rejects_completed_session(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'custom-question-complete.db'}")
    await storage.init_db()
    service = InterviewService(storage)
    session_id, _ = await service.create_session()
    await service.add_custom_mock_question(session_id, "先练习这个问题")
    await service.finish_mock_interview(session_id)

    with pytest.raises(MockInterviewStateError, match="本轮已结束"):
        await service.add_custom_mock_question(session_id, "再添加一个问题")
    await storage.close()


@pytest.mark.asyncio
async def test_audio_suggestion_is_not_injected_after_live_session_ends(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'stale-live-suggestion.db'}")
    await storage.init_db()
    service = InterviewService(storage, WorkflowMockLLM(), FakeSearchProvider())
    session_id, _ = await service.create_session()
    await service.start_live_interview(session_id, consent_confirmed=True)
    runtime = await service._get_runtime(session_id)
    runtime.state.live_interview.status = LiveInterviewStatus.COMPLETED

    await service._inject_audio_direct_suggestion(session_id, runtime, "Stale question")

    assert runtime.state.live_interview.suggestions == []
    await storage.close()
