"""Use-case layer coordinating runtimes and persistence."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel
from interview_os.core.factory import create_runtime
from interview_os.core.message import Message
from interview_os.core.question_understanding import deterministic_question_understanding
from interview_os.core.request_context import current_owner
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import (
    AnswerEvaluation,
    AutopilotState,
    AutopilotStatus,
    CandidateProfile,
    CompanyInfo,
    EntityResolution,
    EvaluationReport,
    FeedbackReport,
    InterviewBlueprint,
    InterviewerProfile,
    InterviewStage,
    InterviewState,
    InterviewStrategy,
    JobDescription,
    JobDescriptionReview,
    LiveInterviewRecord,
    LiveInterviewSession,
    LiveInterviewStatus,
    MockInterviewPlan,
    MockInterviewSession,
    MockSessionStatus,
    ResumeReview,
    SpeechDeliveryFeedback,
    WorkflowProgress,
    WorkflowStatus,
)
from interview_os.database.storage import Storage
from interview_os.services.background import BackgroundTaskManager
from interview_os.services.evaluation_service import EvaluationServiceMixin
from interview_os.services.intelligence_service import (
    build_fact_cards,
    decide_fact_card,
    resolve_entity,
    review_job_description,
    sync_entity_resolutions,
)
from interview_os.services.live_interview_service import LiveInterviewServiceMixin
from interview_os.services.media_service import MediaServiceMixin
from interview_os.services.mock_interview_service import MockInterviewServiceMixin
from interview_os.services.preparation_service import PreparationServiceMixin
from interview_os.services.resume_service import (
    ResumeProcessor,
)
from interview_os.services.service_contracts import (
    CandidateSessionStateError,
    EvaluationStateError,
    LiveInterviewStateError,
    MockInterviewStateError,
    ResumeReviewStateError,
    SessionNotFoundError,
    WorkflowExecutionError,
)
from interview_os.tools.asr import ASRClient
from interview_os.tools.tts import TTSClient
from interview_os.tools.web_search import (
    SearchProvider,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateSessionStateError",
    "EvaluationStateError",
    "InterviewService",
    "LiveInterviewStateError",
    "MockInterviewStateError",
    "ResumeReviewStateError",
    "SessionNotFoundError",
    "WorkflowExecutionError",
]


class InterviewService(
    PreparationServiceMixin,
    MockInterviewServiceMixin,
    EvaluationServiceMixin,
    LiveInterviewServiceMixin,
    MediaServiceMixin,
):
    """Owns session-scoped runtimes and serializes mutations per session."""

    def __init__(
        self,
        storage: Storage,
        llm_client: Any = None,
        search_provider: SearchProvider | None = None,
        debug_events: DebugEventStore | None = None,
        asr_client: ASRClient | None = None,
        background: BackgroundTaskManager | None = None,
        omni_client: Any = None,
        tts_client: TTSClient | None = None,
        resume_llm_client: Any = None,
        recordings_dir: Path | None = None,
    ) -> None:
        self.storage = storage
        self.llm_client = llm_client
        self.search_provider = search_provider
        self.debug_events = debug_events
        self.asr_client = asr_client
        self.omni_client = omni_client
        self.tts_client = tts_client
        self.resume_llm_client = resume_llm_client
        self.live_audio_mode = "asr_text"  # "asr_text" | "audio_direct"
        self._background = background or BackgroundTaskManager(debug_events=debug_events)
        self._runtimes: dict[tuple[str, str], AgentRuntime] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._recordings_dir = recordings_dir or Path("data/recordings")
        self._mock_speech_feedback: dict[
            tuple[str, str, UUID], tuple[str, SpeechDeliveryFeedback, float]
        ] = {}
        self.resume_processor = ResumeProcessor()

    async def create_session(
        self, candidate_name: str = "", job_title: str = "", company_name: str = ""
    ) -> tuple[str, InterviewState]:
        owner = current_owner()
        session_id = str(uuid4())
        runtime = create_runtime(
            self.llm_client, self.search_provider, self.debug_events, session_id
        )
        runtime.state.candidate.name = candidate_name
        runtime.state.job.title = job_title
        runtime.state.company.name = company_name
        self._runtimes[(owner, session_id)] = runtime
        self._locks[(owner, session_id)] = asyncio.Lock()
        await self._persist(session_id, runtime.state)
        self._record_debug("session_created", session_id)
        return session_id, runtime.state

    async def get_state(self, session_id: str) -> InterviewState:
        return (await self._get_runtime(session_id)).state

    async def list_sessions(self) -> list[dict[str, Any]]:
        return await self.storage.list_sessions(owner_id=current_owner())

    async def run_autopilot(
        self,
        session_id: str,
        *,
        role: str,
        resume_text: str,
        job_description: str,
        company_name: str,
        company_context: str = "",
        interviewer_name: str = "",
        interviewer_position: str = "",
        authorized_public_research: bool = False,
    ) -> InterviewState:
        """Advance every safe deterministic/agent step and pause at real-world input."""
        runtime = await self._get_runtime(session_id)
        now = datetime.now(timezone.utc)
        async with self._lock_for(session_id):
            self._assert_candidate_reset_allowed(runtime.state)
            previous_candidate = runtime.state.candidate
            same_resume = previous_candidate.raw_resume_text.strip() == resume_text.strip()
            runtime.state.candidate = (
                previous_candidate.model_copy(deep=True)
                if same_resume
                else CandidateProfile(
                    # The session label is user-controlled identity metadata, not
                    # a derived resume artifact. Preserve it while clearing every
                    # analyzed skill/claim from the prior document.
                    name=previous_candidate.name,
                    raw_resume_text=resume_text,
                )
            )
            if not same_resume:
                # Confirmations belong to a specific resume. Reusing them after
                # the document changes can leak one candidate's facts into the
                # next candidate's prompts and public-employer research.
                runtime.state.resume_review = ResumeReview()
            runtime.state.candidate.raw_resume_text = resume_text
            runtime.state.job = JobDescription()
            runtime.state.job_review = JobDescriptionReview()
            runtime.state.company = CompanyInfo(
                name=company_name,
                context=company_context.strip(),
            )
            runtime.state.interviewer = InterviewerProfile(
                name=interviewer_name,
                position=interviewer_position,
                company=company_name,
            )
            runtime.state.entity_resolutions = []
            runtime.state.fact_cards = []
            runtime.state.past_employer_sources = []
            runtime.state.past_employer_research_status = "not_requested"
            runtime.state.past_employer_block = ""
            runtime.state.strategy = InterviewStrategy()
            runtime.state.blueprint = InterviewBlueprint()
            runtime.state.mock_interview = MockInterviewPlan()
            runtime.state.mock_session = MockInterviewSession()
            runtime.state.evaluation = EvaluationReport()
            runtime.state.feedback = FeedbackReport()
            runtime.state.live_interview_records = []
            runtime.state.live_interview = LiveInterviewSession()
            runtime.state.evidence.clear()
            runtime.state.evaluated_competencies.clear()
            runtime.state.missing_signals.clear()
            runtime.state.conversation_history.clear()
            runtime.state.current_stage = InterviewStage.NOT_STARTED
            runtime.state.autopilot = AutopilotState(
                enabled=True,
                status=AutopilotStatus.RUNNING,
                phase="intelligence",
                authorized_public_research=authorized_public_research,
                started_at=now,
                updated_at=now,
            )
            await self._persist(session_id, runtime.state)
        self._record_debug("autopilot_started", session_id, detail=role)
        try:
            if role == "candidate":
                state = await self.run_candidate_prep(
                    session_id,
                    resume_text=resume_text,
                    job_description=job_description,
                    company_name=company_name,
                    company_context=company_context,
                    interviewer_name=interviewer_name,
                    interviewer_position=interviewer_position,
                    authorized_public_research=authorized_public_research,
                    prepare_resume_transition=False,
                )
                state = await self.start_mock_interview(session_id)
                state.autopilot.completed_actions = [
                    "resume_analysis",
                    "job_analysis",
                    "company_research",
                    "strategy_generation",
                    "mock_plan_generation",
                ]
                state.autopilot.phase = "interview"
                state.autopilot.status = AutopilotStatus.WAITING_FOR_INPUT
                state.autopilot.pause_reason = "等待候选人回答当前问题"
                state.next_action = "Answer the current AI-led interview question"
            else:
                state = await self.run_enterprise_design(
                    session_id,
                    resume_text=resume_text,
                    job_description=job_description,
                    company_name=company_name,
                    company_context=company_context,
                    authorized_public_research=authorized_public_research,
                    prepare_resume_transition=False,
                )
                state.autopilot.completed_actions = [
                    "resume_analysis",
                    "job_analysis",
                    "company_research",
                    "interview_blueprint_generation",
                ]
                state.autopilot.phase = "interview_execution"
                state.autopilot.status = AutopilotStatus.WAITING_FOR_INPUT
                state.autopilot.pause_reason = "等待真实面试回答或面试记录，不能由 AI 代造证据"
                state.next_action = "Conduct the structured interview and collect evidence"
            state.autopilot.updated_at = datetime.now(timezone.utc)
            await self._persist(session_id, state)
            self._record_debug("autopilot_paused", session_id, detail=state.autopilot.pause_reason)
            return state
        except Exception:
            runtime.state.autopilot.status = AutopilotStatus.FAILED
            runtime.state.autopilot.phase = "error"
            runtime.state.autopilot.updated_at = datetime.now(timezone.utc)
            await self._persist(session_id, runtime.state)
            raise

    async def import_interview_transcript(
        self,
        session_id: str,
        entries: list[dict[str, str]],
        *,
        auto_evaluate: bool = True,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            for entry in entries[:50]:
                question = entry.get("question", "").strip()
                answer = entry.get("answer", "").strip()
                competency = entry.get("competency", "").strip() or "综合能力"
                if not question or not answer:
                    raise EvaluationStateError("Transcript entries require question and answer")
                evidence_count_before = len(runtime.state.evidence)
                message = await runtime.run(
                    "coach_agent",
                    json.dumps(
                        {
                            "question": question,
                            "answer": answer,
                            "competency": competency,
                            "evidence_source": "live_interview",
                            "answer_modality": "typed",
                        },
                        ensure_ascii=False,
                    ),
                )
                try:
                    evaluation = AnswerEvaluation.model_validate_json(message.content)
                except (ValueError, TypeError) as exc:
                    raise EvaluationStateError("Transcript answer evaluation failed") from exc
                record = LiveInterviewRecord(
                    question=question,
                    answer=answer,
                    competency=competency,
                    evaluation=evaluation,
                    source="live_interview",
                    scoring_status="scored",
                )
                runtime.state.live_interview_records.append(record)
                for evidence in runtime.state.evidence[evidence_count_before:]:
                    if evidence.source.value == "live_interview":
                        evidence.source_record_id = record.id
            runtime.state.next_action = "Generate evidence-based hiring evaluation"
            self._invalidate_final_reports(
                runtime.state, "Transcript evidence changed; regenerate evaluation"
            )
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "interview_transcript_imported",
                session_id,
                detail=f"{min(len(entries), 50)} entries",
            )
        if auto_evaluate:
            return await self.run_evaluation(session_id)
        return runtime.state

    async def _execute_workflow(
        self,
        session_id: str,
        runtime: AgentRuntime,
        name: str,
        steps: list[tuple[str, str]],
        company_name: str | None = None,
        company_context: str | None = None,
        interviewer: InterviewerProfile | None = None,
        parallel_prefix: int = 0,
        authorized_public_research: bool = False,
    ) -> InterviewState:
        workflow_started = perf_counter()
        async with self._lock_for(session_id):
            if name == "evaluation" and not runtime.state.evidence:
                raise EvaluationStateError(
                    "No interview evidence is available; complete a mock or live interview first"
                )
            if name == "evaluation":
                if runtime.state.mock_session.status == MockSessionStatus.ACTIVE:
                    raise EvaluationStateError(
                        "Finish the active mock interview before generating the final report"
                    )
                if runtime.state.live_interview.status in {
                    LiveInterviewStatus.ACTIVE,
                    LiveInterviewStatus.PAUSED,
                }:
                    raise EvaluationStateError(
                        "Complete the live interview before generating the final report"
                    )
                if any(
                    record.scoring_status != "scored"
                    for record in runtime.state.live_interview_records
                ):
                    raise EvaluationStateError(
                        "Wait for live answer scoring or complete human review before evaluation"
                    )
            if company_name is not None:
                runtime.state.company.name = company_name
            if company_context is not None:
                runtime.state.company.context = company_context.strip()
            if interviewer is not None:
                runtime.state.interviewer = interviewer
            # Record explicit public-research authorization so the past-employer
            # search respects it even in the non-autopilot workflow paths.
            runtime.state.autopilot.authorized_public_research = authorized_public_research
            runtime.state.workflow = WorkflowProgress(
                name=name,
                status=WorkflowStatus.RUNNING,
                total_steps=len(steps),
            )
            self._record_debug("workflow_started", session_id, detail=name)
            await self._persist(session_id, runtime.state)
            try:
                start_index = 0
                if parallel_prefix:
                    initial = steps[:parallel_prefix]
                    runtime.state.workflow.current_step = " + ".join(
                        agent_name for agent_name, _ in initial
                    )
                    await self._persist(session_id, runtime.state)
                    await asyncio.gather(
                        *(
                            runtime.run(agent_name, instruction)
                            for agent_name, instruction in initial
                        )
                    )
                    start_index = len(initial)
                    runtime.state.workflow.completed_steps = start_index
                    self._sync_intelligence(runtime.state)
                    self._refresh_live_question_usage(runtime.state)
                    await self._persist(session_id, runtime.state)
                    # After candidate/job/company analysis, research the
                    # candidate's recent/important past employers so strategy and
                    # design agents can reference what those companies do.
                    if any(agent_name == "candidate_agent" for agent_name, _ in initial):
                        block = await self._research_employers_inline(runtime.state)
                        if block:
                            runtime.state.past_employer_block = block
                            await self._persist(session_id, runtime.state)
                for index, (agent_name, instruction) in enumerate(
                    steps[start_index:], start=start_index + 1
                ):
                    runtime.state.workflow.current_step = agent_name
                    await self._persist(session_id, runtime.state)
                    await runtime.run(agent_name, instruction)
                    runtime.state.workflow.completed_steps = index
                    self._sync_intelligence(runtime.state)
                    self._refresh_live_question_usage(runtime.state)
                    await self._persist(session_id, runtime.state)
                self._validate_workflow_result(name, runtime.state)
            except Exception as exc:
                runtime.state.workflow.status = WorkflowStatus.FAILED
                runtime.state.workflow.error = str(exc)
                await self._persist(session_id, runtime.state)
                self._record_debug(
                    "workflow_failed",
                    session_id,
                    level=DebugLevel.ERROR,
                    detail=f"{name}: {exc}",
                )
                raise WorkflowExecutionError(str(exc)) from exc
            runtime.state.workflow.status = WorkflowStatus.COMPLETED
            runtime.state.workflow.current_step = ""
            runtime.state.next_action = (
                "Start mock interview"
                if name == "candidate_prep"
                else "Execute interview blueprint"
            )
            if name == "evaluation":
                runtime.state.current_stage = InterviewStage.COMPLETED
                runtime.state.next_action = "Review the final evaluation and feedback"
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "workflow_completed",
                session_id,
                detail=name,
                duration_ms=(perf_counter() - workflow_started) * 1000,
            )
            return runtime.state

    async def resolve_entity_candidate(
        self,
        session_id: str,
        resolution_id: UUID,
        *,
        accept: bool,
        proposed_name: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            try:
                resolution: EntityResolution = resolve_entity(
                    runtime.state,
                    resolution_id,
                    accept=accept,
                    proposed_name=proposed_name,
                )
            except LookupError as exc:
                raise ResumeReviewStateError(str(exc)) from exc
            self._sync_intelligence(runtime.state)
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "entity_resolution_updated",
                session_id,
                detail=f"{resolution.input_name} -> {resolution.proposed_name}: {resolution.status.value}",
            )
            return runtime.state

    async def decide_fact_card(
        self, session_id: str, card_id: UUID, *, action: str, note: str = ""
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            try:
                card = decide_fact_card(runtime.state, card_id, action=action, note=note)
            except (LookupError, ValueError) as exc:
                raise ResumeReviewStateError(str(exc)) from exc
            runtime.state.next_action = (
                "Use confirmed fact cards or continue public research review"
            )
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "fact_card_decided",
                session_id,
                detail=f"{card.category}: {card.status.value}",
            )
            return runtime.state

    @staticmethod
    def _sync_intelligence(state: InterviewState) -> None:
        sync_entity_resolutions(state)
        build_fact_cards(state)

    @staticmethod
    def _validate_workflow_result(name: str, state: InterviewState) -> None:
        if name == "candidate_prep":
            if not state.strategy.summary and not state.strategy.key_risks:
                raise ValueError("Strategy agent did not return a valid structured result")
            if not state.mock_interview.questions:
                raise ValueError("Mock interview agent did not return any questions")
        elif name == "enterprise_design" and not state.blueprint.rounds:
            raise ValueError("Interview design agent did not return any rounds")
        elif name == "evaluation":
            if not state.evaluation.competencies:
                raise ValueError("Evaluation agent did not return competency results")
            if not state.feedback.overall:
                raise ValueError("Feedback agent did not return a valid report")

    async def _run(self, session_id: str, agent_name: str, instruction: str) -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            message = await runtime.run(agent_name, instruction)
            await self._persist(session_id, runtime.state)
            return message

    async def _get_runtime(self, session_id: str) -> AgentRuntime:
        owner = current_owner()
        cached = self._runtimes.get((owner, session_id))
        if cached is not None:
            return cached
        state = await self.storage.get_session_state(session_id, owner_id=owner)
        if state is None:
            raise SessionNotFoundError(session_id)
        runtime = create_runtime(
            self.llm_client, self.search_provider, self.debug_events, session_id
        )
        runtime.state = InterviewState.model_validate(state)
        # Upgrade persisted pre-quality-contract questions in place. This keeps
        # existing user sessions usable after deployment instead of requiring a
        # new candidate workflow solely to obtain bounded questions/examples.
        mock_agent = runtime.get_agent("mock_interview_agent")
        if mock_agent is not None and hasattr(mock_agent, "enrich_question"):
            for question in runtime.state.mock_interview.questions:
                # User-supplied wording is itself part of the practice contract.
                # It already receives requirements/framework/example at insert
                # time and must not be rewritten by legacy-question migration.
                if question.source == "custom":
                    explicit_requirements = [
                        item.text
                        for item in runtime.state.job_review.requirements
                        if item.origin.value == "explicit"
                    ]
                    deep_fallback = deterministic_question_understanding(
                        question.question,
                        competency=(
                            "" if question.competency == "自定义问题" else question.competency
                        ),
                        job_title=runtime.state.job.title,
                        explicit_job_requirements=explicit_requirements,
                        confirmed_claims=[
                            (str(claim.id), claim.statement)
                            for claim in runtime.state.resume_review.claims
                            if claim.status.value in {"confirmed", "modified"}
                        ],
                    )
                    if question.understanding is None:
                        question.understanding = deep_fallback
                    elif not question.understanding.decision_criteria:
                        # Preserve the older model-enhanced shallow analysis and
                        # backfill only the newly introduced deep layer.
                        question.understanding.role_relevance = deep_fallback.role_relevance
                        question.understanding.role_relevance_source = (
                            deep_fallback.role_relevance_source
                        )
                        question.understanding.secondary_competencies = (
                            deep_fallback.secondary_competencies
                        )
                        question.understanding.decision_criteria = deep_fallback.decision_criteria
                        question.understanding.answer_levels = deep_fallback.answer_levels
                        question.understanding.candidate_story_options = (
                            deep_fallback.candidate_story_options
                        )
                        question.understanding.story_selection_guidance = (
                            deep_fallback.story_selection_guidance
                        )
                        question.understanding.probe_tree = deep_fallback.probe_tree
                elif question.source != "custom":
                    mock_agent.enrich_question(  # type: ignore[union-attr]
                        question, runtime.state
                    )
        # In-process refill tasks do not survive a service restart.
        runtime.state.mock_session.refill_in_flight = False
        if runtime.state.mock_session.status == MockSessionStatus.EVALUATING:
            # The evaluation coroutine was process-local. Restore a retryable
            # state rather than leaving the session permanently in-flight.
            runtime.state.mock_session.status = MockSessionStatus.ACTIVE
            runtime.state.mock_session.completed_at = None
            runtime.state.next_action = "Evaluation was interrupted; retry ending the interview"
        if not runtime.state.job_review.requirements and (
            runtime.state.job.raw_description or runtime.state.job.title
        ):
            inferred = [
                *runtime.state.job.required_skills,
                *runtime.state.job.preferred_skills,
                *runtime.state.job.competencies,
            ]
            runtime.state.job_review = review_job_description(
                runtime.state.job.raw_description or runtime.state.job.title,
                list(dict.fromkeys(inferred)),
            )
        self._sync_intelligence(runtime.state)
        self._refresh_live_coverage_guidance(runtime.state)
        self._refresh_live_question_usage(runtime.state)
        if runtime.state.evaluation.finalized_at:
            runtime.state.enforce_evaluation_evidence_floor()
        self._runtimes[(owner, session_id)] = runtime
        self._locks.setdefault((owner, session_id), asyncio.Lock())
        await self._persist(session_id, runtime.state)
        return runtime

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        owner = current_owner()
        return self._locks.setdefault((owner, session_id), asyncio.Lock())

    async def _persist(self, session_id: str, state: InterviewState) -> None:
        self._refresh_live_action_card(state)
        await self.storage.save_session(
            session_id, state.model_dump(mode="json"), owner_id=current_owner()
        )

    def get_cached_runtime(
        self, session_id: str, *, owner_id: str | None = None
    ) -> AgentRuntime | None:
        """Return a cached runtime for a session.

        The debug console (localhost-only) inspects any session, so by default
        this scans all owners. Owner-scoped callers can pass ``owner_id``.
        """
        if owner_id is not None:
            return self._runtimes.get((owner_id, session_id))
        for (owner, sid), runtime in self._runtimes.items():
            if sid == session_id:
                return runtime
        return None

    def _record_debug(
        self,
        action: str,
        session_id: str,
        *,
        level: DebugLevel = DebugLevel.INFO,
        detail: str = "",
        duration_ms: float | None = None,
    ) -> None:
        if self.debug_events is not None:
            self.debug_events.record(
                DebugEvent(
                    level=level,
                    category="service",
                    action=action,
                    session_id=session_id,
                    duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
                    detail=detail[:1000],
                )
            )
