"""Use-case layer coordinating runtimes and persistence."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel
from interview_os.core.factory import create_runtime
from interview_os.core.message import Message
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import (
    AnswerEvaluation,
    InterviewerProfile,
    InterviewState,
    MockAnswerRecord,
    MockInterviewSession,
    MockSessionStatus,
    WorkflowProgress,
    WorkflowStatus,
)
from interview_os.database.storage import Storage
from interview_os.tools.web_search import SearchProvider


class SessionNotFoundError(LookupError):
    pass


class WorkflowExecutionError(RuntimeError):
    pass


class MockInterviewStateError(RuntimeError):
    pass


class InterviewService:
    """Owns session-scoped runtimes and serializes mutations per session."""

    def __init__(
        self,
        storage: Storage,
        llm_client: Any = None,
        search_provider: SearchProvider | None = None,
        debug_events: DebugEventStore | None = None,
    ) -> None:
        self.storage = storage
        self.llm_client = llm_client
        self.search_provider = search_provider
        self.debug_events = debug_events
        self._runtimes: dict[str, AgentRuntime] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def create_session(
        self, candidate_name: str = "", job_title: str = "", company_name: str = ""
    ) -> tuple[str, InterviewState]:
        session_id = str(uuid4())
        runtime = create_runtime(
            self.llm_client, self.search_provider, self.debug_events, session_id
        )
        runtime.state.candidate.name = candidate_name
        runtime.state.job.title = job_title
        runtime.state.company.name = company_name
        self._runtimes[session_id] = runtime
        self._locks[session_id] = asyncio.Lock()
        await self._persist(session_id, runtime.state)
        self._record_debug("session_created", session_id)
        return session_id, runtime.state

    async def get_state(self, session_id: str) -> InterviewState:
        return (await self._get_runtime(session_id)).state

    async def analyze_resume(self, session_id: str, text: str) -> Message:
        return await self._run(session_id, "candidate_agent", text)

    async def analyze_job(self, session_id: str, text: str) -> Message:
        return await self._run(session_id, "job_agent", text)

    async def analyze_company(self, session_id: str, name: str, context: str = "") -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            runtime.state.company.name = name
            message = await runtime.run("company_agent", context)
            await self._persist(session_id, runtime.state)
            return message

    async def analyze_interviewer(
        self, session_id: str, name: str, position: str = "", company: str = "", public_info: str = ""
    ) -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            runtime.state.interviewer = InterviewerProfile(
                name=name,
                position=position,
                company=company,
                public_expressions=[{"text": public_info}] if public_info else [],
            )
            message = await runtime.run("interviewer_agent")
            await self._persist(session_id, runtime.state)
            return message

    async def run_candidate_prep(
        self,
        session_id: str,
        *,
        resume_text: str,
        job_description: str,
        company_name: str,
        company_context: str = "",
        interviewer_name: str = "",
        interviewer_position: str = "",
        interviewer_public_info: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        steps: list[tuple[str, str]] = [
            ("candidate_agent", resume_text),
            ("job_agent", job_description),
            ("company_agent", company_context),
        ]
        interviewer = None
        if interviewer_name:
            interviewer = InterviewerProfile(
                name=interviewer_name,
                position=interviewer_position,
                company=company_name,
                public_expressions=(
                    [{"text": interviewer_public_info}] if interviewer_public_info else []
                ),
            )
            steps.append(("interviewer_agent", ""))
        steps.extend(
            [
                ("interview_strategy_agent", "Generate personalized interview strategy"),
                ("mock_interview_agent", "Generate personalized mock questions"),
            ]
        )
        return await self._execute_workflow(
            session_id,
            runtime,
            "candidate_prep",
            steps,
            company_name=company_name,
            interviewer=interviewer,
        )

    async def run_enterprise_design(
        self,
        session_id: str,
        *,
        resume_text: str,
        job_description: str,
        company_name: str,
        company_context: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        steps = [
            ("candidate_agent", resume_text),
            ("job_agent", job_description),
            ("company_agent", company_context),
            ("interview_design_agent", "Design an evidence-based interview blueprint"),
        ]
        return await self._execute_workflow(
            session_id,
            runtime,
            "enterprise_design",
            steps,
            company_name=company_name,
        )

    async def start_mock_interview(self, session_id: str) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            if not runtime.state.mock_interview.questions:
                raise MockInterviewStateError(
                    "No mock questions are available; run candidate preparation first"
                )
            if runtime.state.mock_session.status == MockSessionStatus.ACTIVE:
                return runtime.state
            if runtime.state.mock_session.status == MockSessionStatus.COMPLETED:
                raise MockInterviewStateError("Mock interview is already completed")
            runtime.state.mock_session = MockInterviewSession(
                status=MockSessionStatus.ACTIVE,
                started_at=datetime.now(timezone.utc),
            )
            runtime.state.next_action = "Answer the current mock interview question"
            await self._persist(session_id, runtime.state)
            return runtime.state

    async def submit_mock_answer(
        self, session_id: str, question_id: UUID, answer: str
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            questions = runtime.state.mock_interview.questions
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            if mock_session.current_question_index >= len(questions):
                raise MockInterviewStateError("Mock interview has no remaining questions")
            question = questions[mock_session.current_question_index]
            if question.id != question_id:
                raise MockInterviewStateError("Answer does not match the current question")

            payload = json.dumps(
                {
                    "question": question.question,
                    "answer": answer,
                    "competency": question.competency or "Answer Quality",
                }
            )
            message = await runtime.run("coach_agent", payload)
            try:
                evaluation = AnswerEvaluation.model_validate_json(message.content)
            except (ValueError, TypeError) as exc:
                raise MockInterviewStateError(
                    "Coach agent did not return a valid answer evaluation"
                ) from exc

            mock_session.responses.append(
                MockAnswerRecord(
                    question_id=question.id,
                    question=question.question,
                    competency=question.competency,
                    answer=answer,
                    evaluation=evaluation,
                )
            )
            mock_session.current_question_index += 1
            if mock_session.current_question_index == len(questions):
                mock_session.status = MockSessionStatus.COMPLETED
                mock_session.completed_at = datetime.now(timezone.utc)
                runtime.state.next_action = "Generate evidence-based evaluation"
            else:
                runtime.state.next_action = "Answer the next mock interview question"
            await self._persist(session_id, runtime.state)
            return runtime.state

    @staticmethod
    def current_mock_question(state: InterviewState):
        session = state.mock_session
        questions = state.mock_interview.questions
        if session.status != MockSessionStatus.ACTIVE or session.current_question_index >= len(
            questions
        ):
            return None
        return questions[session.current_question_index]

    async def _execute_workflow(
        self,
        session_id: str,
        runtime: AgentRuntime,
        name: str,
        steps: list[tuple[str, str]],
        company_name: str,
        interviewer: InterviewerProfile | None = None,
    ) -> InterviewState:
        async with self._lock_for(session_id):
            runtime.state.company.name = company_name
            if interviewer is not None:
                runtime.state.interviewer = interviewer
            runtime.state.workflow = WorkflowProgress(
                name=name,
                status=WorkflowStatus.RUNNING,
                total_steps=len(steps),
            )
            self._record_debug("workflow_started", session_id, detail=name)
            await self._persist(session_id, runtime.state)
            try:
                for index, (agent_name, instruction) in enumerate(steps, start=1):
                    runtime.state.workflow.current_step = agent_name
                    await self._persist(session_id, runtime.state)
                    await runtime.run(agent_name, instruction)
                    runtime.state.workflow.completed_steps = index
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
                "Start mock interview" if name == "candidate_prep" else "Execute interview blueprint"
            )
            await self._persist(session_id, runtime.state)
            self._record_debug("workflow_completed", session_id, detail=name)
            return runtime.state

    @staticmethod
    def _validate_workflow_result(name: str, state: InterviewState) -> None:
        if name == "candidate_prep":
            if not state.strategy.summary and not state.strategy.key_risks:
                raise ValueError("Strategy agent did not return a valid structured result")
            if not state.mock_interview.questions:
                raise ValueError("Mock interview agent did not return any questions")
        elif name == "enterprise_design" and not state.blueprint.rounds:
            raise ValueError("Interview design agent did not return any rounds")

    async def _run(self, session_id: str, agent_name: str, instruction: str) -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            message = await runtime.run(agent_name, instruction)
            await self._persist(session_id, runtime.state)
            return message

    async def _get_runtime(self, session_id: str) -> AgentRuntime:
        cached = self._runtimes.get(session_id)
        if cached is not None:
            return cached
        state = await self.storage.get_session_state(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)
        runtime = create_runtime(
            self.llm_client, self.search_provider, self.debug_events, session_id
        )
        runtime.state = InterviewState.model_validate(state)
        self._runtimes[session_id] = runtime
        self._locks.setdefault(session_id, asyncio.Lock())
        return runtime

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def _persist(self, session_id: str, state: InterviewState) -> None:
        await self.storage.save_session(session_id, state.model_dump(mode="json"))

    def get_cached_runtime(self, session_id: str) -> AgentRuntime | None:
        return self._runtimes.get(session_id)

    def _record_debug(
        self, action: str, session_id: str, *, level: DebugLevel = DebugLevel.INFO, detail: str = ""
    ) -> None:
        if self.debug_events is not None:
            self.debug_events.record(
                DebugEvent(
                    level=level,
                    category="service",
                    action=action,
                    session_id=session_id,
                    detail=detail[:1000],
                )
            )
