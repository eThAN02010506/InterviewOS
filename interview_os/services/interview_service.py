"""Use-case layer coordinating runtimes and persistence."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel
from interview_os.core.factory import create_runtime
from interview_os.core.message import Message
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
    MockAnswerRecord,
    MockInterviewPlan,
    MockInterviewSession,
    MockSessionStatus,
    QuestionSuggestionStatus,
    RequirementOrigin,
    ResumeClaimStatus,
    TranscriptSegment,
    TranscriptSpeaker,
    WorkflowProgress,
    WorkflowStatus,
)
from interview_os.database.storage import Storage
from interview_os.services.intelligence_service import (
    build_fact_cards,
    resolve_entity,
    review_job_description,
    sync_entity_resolutions,
)
from interview_os.services.resume_service import ResumeProcessor
from interview_os.tools.asr import ASRClient, ASRError
from interview_os.tools.web_search import SearchProvider


class SessionNotFoundError(LookupError):
    pass


class WorkflowExecutionError(RuntimeError):
    pass


class MockInterviewStateError(RuntimeError):
    pass


class EvaluationStateError(RuntimeError):
    pass


class ResumeReviewStateError(RuntimeError):
    pass


class LiveInterviewStateError(RuntimeError):
    pass


class InterviewService:
    """Owns session-scoped runtimes and serializes mutations per session."""

    def __init__(
        self,
        storage: Storage,
        llm_client: Any = None,
        search_provider: SearchProvider | None = None,
        debug_events: DebugEventStore | None = None,
        asr_client: ASRClient | None = None,
    ) -> None:
        self.storage = storage
        self.llm_client = llm_client
        self.search_provider = search_provider
        self.debug_events = debug_events
        self.asr_client = asr_client
        self._runtimes: dict[str, AgentRuntime] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.resume_processor = ResumeProcessor()

    async def start_live_interview(
        self, session_id: str, *, consent_confirmed: bool
    ) -> InterviewState:
        if not consent_confirmed:
            raise LiveInterviewStateError("开始监听前必须确认候选人已知情并同意转写")
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if live.status == LiveInterviewStatus.COMPLETED:
                runtime.state.live_interview = LiveInterviewSession()
                live = runtime.state.live_interview
            live.status = LiveInterviewStatus.ACTIVE
            live.consent_confirmed = True
            live.started_at = live.started_at or datetime.now(timezone.utc)
            live.completed_at = None
            runtime.state.next_action = "Listen to the interview and prepare the next question"
            await self._persist(session_id, runtime.state)
        self._record_debug("live_interview_started", session_id)
        return runtime.state

    async def set_live_interview_status(
        self, session_id: str, status: LiveInterviewStatus
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if not live.consent_confirmed:
                raise LiveInterviewStateError("Live interview has not been started with consent")
            if status not in {
                LiveInterviewStatus.ACTIVE,
                LiveInterviewStatus.PAUSED,
                LiveInterviewStatus.COMPLETED,
            }:
                raise LiveInterviewStateError("Unsupported live interview status")
            live.status = status
            if status == LiveInterviewStatus.COMPLETED:
                live.completed_at = datetime.now(timezone.utc)
                runtime.state.next_action = "Review the transcript before final evaluation"
            await self._persist(session_id, runtime.state)
        self._record_debug(f"live_interview_{status.value}", session_id)
        return runtime.state

    async def append_live_transcript(
        self,
        session_id: str,
        *,
        text: str,
        speaker: TranscriptSpeaker,
        source: str = "manual",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        clean_text = text.strip()
        if not clean_text:
            raise LiveInterviewStateError("Transcript text cannot be empty")
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if live.status != LiveInterviewStatus.ACTIVE:
                raise LiveInterviewStateError("Live interview must be active before adding transcript")
            live.current_speaker = speaker
            live.segments.append(
                TranscriptSegment(
                    sequence=len(live.segments) + 1,
                    speaker=speaker,
                    text=clean_text,
                    source=source,
                )
            )
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_transcript_added",
            session_id,
            detail=f"speaker={speaker.value}; source={source}; chars={len(clean_text)}",
        )
        return runtime.state

    async def update_live_transcript_segment(
        self,
        session_id: str,
        segment_id: UUID,
        *,
        text: str | None = None,
        speaker: TranscriptSpeaker | None = None,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        clean_text = text.strip() if text is not None else None
        if text is not None and not clean_text:
            raise LiveInterviewStateError("Transcript text cannot be empty")
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if not live.consent_confirmed:
                raise LiveInterviewStateError("Live interview has not been started with consent")
            segment = next((item for item in live.segments if item.id == segment_id), None)
            if segment is None:
                raise LiveInterviewStateError("Transcript segment was not found")
            if any(segment_id in record.transcript_segment_ids for record in runtime.state.live_interview_records):
                raise LiveInterviewStateError("Confirmed evidence segments cannot be edited")
            if clean_text is not None:
                segment.text = clean_text
            if speaker is not None:
                segment.speaker = speaker
                live.current_speaker = speaker
            segment.stable = True
            segment.confirmed = True
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_transcript_updated",
            session_id,
            detail=(
                f"speaker={segment.speaker.value}; chars={len(segment.text)}"
            ),
        )
        return runtime.state

    async def transcribe_live_audio(
        self,
        session_id: str,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        speaker: TranscriptSpeaker,
        language: str = "zh",
    ) -> tuple[InterviewState, str]:
        if self.asr_client is None:
            raise LiveInterviewStateError("ASR client is not configured")
        if len(content) > 25 * 1024 * 1024:
            raise LiveInterviewStateError("Audio chunk exceeds the 25 MB limit")
        runtime = await self._get_runtime(session_id)
        if runtime.state.live_interview.status != LiveInterviewStatus.ACTIVE:
            raise LiveInterviewStateError("Live interview must be active before transcribing audio")
        started = perf_counter()
        try:
            transcript = await self.asr_client.transcribe(
                content,
                filename=filename,
                content_type=content_type,
                language=language,
            )
        except ASRError as exc:
            self._record_debug(
                "asr_transcription_failed",
                session_id,
                level=DebugLevel.ERROR,
                detail=str(exc),
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise WorkflowExecutionError(str(exc)) from exc
        state = await self.append_live_transcript(
            session_id, text=transcript, speaker=speaker, source="asr"
        )
        self._record_debug(
            "asr_transcription_completed",
            session_id,
            detail=f"speaker={speaker.value}; bytes={len(content)}; chars={len(transcript)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return state, transcript

    async def plan_live_next_question(self, session_id: str) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if live.status != LiveInterviewStatus.ACTIVE:
                raise LiveInterviewStateError("Live interview must be active to plan a question")
            candidate_segments = [
                item
                for item in live.segments
                if item.speaker == TranscriptSpeaker.CANDIDATE and item.stable and item.confirmed
            ]
            if not candidate_segments:
                raise LiveInterviewStateError("需要至少一段候选人回答后才能准备下一问题")
            await runtime.run(
                "live_interview_agent",
                "根据最新候选人回答准备一个问题；优先补齐关键证据，然后推进未覆盖能力。",
            )
            runtime.state.next_action = "Interviewer reviews the suggested next question"
            await self._persist(session_id, runtime.state)
        self._record_debug("live_question_planned", session_id)
        return runtime.state

    async def decide_live_suggestion(
        self,
        session_id: str,
        suggestion_id: UUID,
        *,
        status: QuestionSuggestionStatus,
        final_question: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            suggestion = next(
                (
                    item
                    for item in runtime.state.live_interview.suggestions
                    if item.id == suggestion_id
                ),
                None,
            )
            if suggestion is None:
                raise LiveInterviewStateError("Question suggestion was not found")
            if suggestion.status != QuestionSuggestionStatus.PENDING:
                raise LiveInterviewStateError("Question suggestion has already been decided")
            suggestion.status = status
            suggestion.final_question = (
                final_question.strip() or suggestion.suggested_question
                if status in {QuestionSuggestionStatus.ADOPTED, QuestionSuggestionStatus.EDITED}
                else ""
            )
            if status in {QuestionSuggestionStatus.ADOPTED, QuestionSuggestionStatus.EDITED}:
                source_id = suggestion.source_question_id.strip()
                if source_id and source_id not in runtime.state.live_interview.used_question_ids:
                    runtime.state.live_interview.used_question_ids.append(source_id)
                runtime.state.live_interview.segments.append(
                    TranscriptSegment(
                        sequence=len(runtime.state.live_interview.segments) + 1,
                        speaker=TranscriptSpeaker.INTERVIEWER,
                        text=suggestion.final_question,
                        source="copilot",
                    )
                )
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_question_decided", session_id, detail=f"status={status.value}"
        )
        return runtime.state

    async def confirm_live_answer(
        self,
        session_id: str,
        answer_segment_id: UUID,
        *,
        question_segment_id: UUID | None = None,
        question: str = "",
        competency: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if not live.consent_confirmed:
                raise LiveInterviewStateError("Live interview has not been started with consent")
            answer_segment = next(
                (item for item in live.segments if item.id == answer_segment_id), None
            )
            if answer_segment is None:
                raise LiveInterviewStateError("Candidate answer segment was not found")
            if answer_segment.speaker != TranscriptSpeaker.CANDIDATE:
                raise LiveInterviewStateError("Only candidate transcript segments can become evidence")
            if not answer_segment.stable or not answer_segment.confirmed:
                raise LiveInterviewStateError("Transcript segment must be stable and confirmed")
            if any(
                answer_segment_id in record.transcript_segment_ids
                for record in runtime.state.live_interview_records
            ):
                raise LiveInterviewStateError("This answer has already been confirmed as evidence")

            question_segment = self._resolve_live_question_segment(
                live.segments, answer_segment, question_segment_id
            )
            clean_question = question.strip() or (question_segment.text if question_segment else "")
            if not clean_question:
                clean_question = "现场面试问题"
            clean_competency = (
                competency.strip()
                or self._infer_live_competency(runtime.state, clean_question)
                or "综合能力"
            )
            evidence_count_before = len(runtime.state.evidence)
            message = await runtime.run(
                "coach_agent",
                json.dumps(
                    {
                        "question": clean_question,
                        "answer": answer_segment.text,
                        "competency": clean_competency,
                        "evidence_source": "live_interview",
                    },
                    ensure_ascii=False,
                ),
            )
            try:
                evaluation = AnswerEvaluation.model_validate_json(message.content)
            except (ValueError, TypeError) as exc:
                raise EvaluationStateError("Live answer evaluation failed") from exc
            segment_ids = [answer_segment.id]
            if question_segment is not None:
                segment_ids.insert(0, question_segment.id)
            record = LiveInterviewRecord(
                question=clean_question,
                answer=answer_segment.text,
                competency=clean_competency,
                evaluation=evaluation,
                source="live_interview",
                transcript_segment_ids=segment_ids,
            )
            runtime.state.live_interview_records.append(record)
            for evidence in runtime.state.evidence[evidence_count_before:]:
                if evidence.source.value == "live_interview":
                    evidence.source_record_id = record.id
            runtime.state.next_action = "Review more live evidence or generate evaluation"
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_answer_confirmed",
            session_id,
            detail=f"competency={clean_competency}; chars={len(answer_segment.text)}",
        )
        return runtime.state

    async def revoke_live_evidence(self, session_id: str, record_id: UUID) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            record = next(
                (item for item in runtime.state.live_interview_records if item.id == record_id),
                None,
            )
            if record is None:
                raise LiveInterviewStateError("Live evidence record was not found")
            if record.source != "live_interview":
                raise LiveInterviewStateError("Only live interview evidence can be revoked here")
            runtime.state.live_interview_records = [
                item for item in runtime.state.live_interview_records if item.id != record_id
            ]
            runtime.state.evidence = [
                item for item in runtime.state.evidence if item.source_record_id != record_id
            ]
            runtime.state.next_action = "Review corrected live evidence before evaluation"
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_evidence_revoked",
            session_id,
            detail=f"record={str(record_id)[:8]}; competency={record.competency}",
        )
        return runtime.state

    async def confirm_pending_live_answers(
        self, session_id: str, *, competency: str = ""
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        recorded_segment_ids = {
            segment_id
            for record in runtime.state.live_interview_records
            for segment_id in record.transcript_segment_ids
        }
        pending_ids = [
            item.id
            for item in runtime.state.live_interview.segments
            if item.speaker == TranscriptSpeaker.CANDIDATE
            and item.stable
            and item.confirmed
            and item.id not in recorded_segment_ids
        ]
        if not pending_ids:
            raise LiveInterviewStateError("No pending candidate answers require evidence confirmation")
        state = runtime.state
        for segment_id in pending_ids:
            state = await self.confirm_live_answer(
                session_id,
                segment_id,
                competency=competency,
            )
        self._record_debug(
            "live_answers_batch_confirmed",
            session_id,
            detail=f"count={len(pending_ids)}",
        )
        return state

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

    async def upload_resume(self, session_id: str, filename: str, content: bytes) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            text, review = await asyncio.to_thread(self.resume_processor.process, filename, content)
            runtime.state.candidate.raw_resume_text = text
            runtime.state.resume_review = review
            runtime.state.next_action = "Review resume checks, then continue the interview workflow"
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "resume_processed",
                session_id,
                detail=f"{review.metadata.file_type}, {review.metadata.character_count} chars, "
                f"{len(review.issues)} issues, {len(review.claims)} claims",
            )
            return runtime.state

    async def update_resume_claim(
        self,
        session_id: str,
        claim_id: UUID,
        status: ResumeClaimStatus,
        note: str = "",
        statement: str | None = None,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            claim = next(
                (item for item in runtime.state.resume_review.claims if item.id == claim_id), None
            )
            if claim is None:
                raise ResumeReviewStateError("Resume claim not found")
            claim.status = status
            claim.note = note.strip()[:500]
            if statement is not None:
                revised = statement.strip()[:2000]
                if not revised:
                    raise ResumeReviewStateError("Resume claim statement cannot be empty")
                claim.original_statement = claim.original_statement or claim.statement
                claim.statement = revised
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "resume_claim_updated", session_id, detail=f"{claim.category}: {status}"
            )
            return runtime.state

    async def analyze_job(self, session_id: str, text: str) -> Message:
        return await self._run(session_id, "job_agent", text)

    async def update_job_requirement(
        self,
        session_id: str,
        index: int,
        *,
        action: str,
        text: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        clean_text = text.strip()
        async with self._lock_for(session_id):
            requirements = runtime.state.job_review.requirements
            if index < 0 or index >= len(requirements):
                raise ResumeReviewStateError("Job requirement not found")
            if action == "delete":
                requirements.pop(index)
            elif action == "confirm":
                requirements[index].origin = RequirementOrigin.EXPLICIT
            elif action == "edit":
                if not clean_text:
                    raise ResumeReviewStateError("Job requirement text cannot be empty")
                requirements[index].text = clean_text
                requirements[index].origin = RequirementOrigin.EXPLICIT
            else:
                raise ResumeReviewStateError("Unsupported job requirement action")
            explicit_count = sum(
                1 for item in requirements if item.origin == RequirementOrigin.EXPLICIT
            )
            if explicit_count:
                runtime.state.job_review.is_title_only = False
                runtime.state.job_review.completeness_score = max(
                    runtime.state.job_review.completeness_score, min(1.0, 0.45 + explicit_count * 0.05)
                )
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "job_requirement_updated",
                session_id,
                detail=f"index={index}; action={action}",
            )
            return runtime.state

    async def analyze_company(self, session_id: str, name: str, context: str = "") -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            runtime.state.company.name = name
            message = await runtime.run("company_agent", context)
            self._sync_intelligence(runtime.state)
            await self._persist(session_id, runtime.state)
            return message

    async def analyze_interviewer(
        self,
        session_id: str,
        name: str,
        position: str = "",
        company: str = "",
        public_info: str = "",
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
            self._sync_intelligence(runtime.state)
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
            parallel_prefix=4 if interviewer else 3,
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
            parallel_prefix=3,
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
        should_auto_evaluate = False
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            questions = runtime.state.mock_interview.questions
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            if mock_session.current_question_index >= len(questions):
                raise MockInterviewStateError("Mock interview has no remaining questions")
            question = questions[mock_session.current_question_index]
            is_follow_up = bool(mock_session.pending_follow_up)
            expected_id = mock_session.pending_parent_question_id or question.id
            if expected_id != question_id:
                raise MockInterviewStateError("Answer does not match the current question")
            asked_question = mock_session.pending_follow_up or question.question

            payload = json.dumps(
                {
                    "question": asked_question,
                    "answer": answer,
                    "competency": question.competency or "Answer Quality",
                    "evidence_source": "mock_interview",
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
                    question=asked_question,
                    competency=question.competency,
                    answer=answer,
                    evaluation=evaluation,
                    is_follow_up=is_follow_up,
                )
            )
            if not is_follow_up and evaluation.missing_signals and question.follow_ups:
                mock_session.pending_follow_up = question.follow_ups[0]
                mock_session.pending_parent_question_id = question.id
                runtime.state.next_action = "Answer the evidence-seeking follow-up question"
                await self._persist(session_id, runtime.state)
                return runtime.state
            mock_session.pending_follow_up = ""
            mock_session.pending_parent_question_id = None
            mock_session.current_question_index += 1
            if mock_session.current_question_index == len(questions):
                mock_session.status = MockSessionStatus.COMPLETED
                mock_session.completed_at = datetime.now(timezone.utc)
                runtime.state.next_action = "Generate evidence-based evaluation"
                should_auto_evaluate = runtime.state.autopilot.enabled
            else:
                runtime.state.next_action = "Answer the next mock interview question"
            await self._persist(session_id, runtime.state)
        if should_auto_evaluate:
            return await self.run_evaluation(session_id)
        return runtime.state

    async def run_evaluation(self, session_id: str) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        state = await self._execute_workflow(
            session_id,
            runtime,
            "evaluation",
            [
                ("evaluation_agent", "Aggregate evidence and evaluate competencies"),
                ("feedback_agent", "Generate candidate and interviewer feedback"),
            ],
        )
        if state.autopilot.enabled:
            state.autopilot.status = AutopilotStatus.COMPLETED
            state.autopilot.phase = "completed"
            state.autopilot.pause_reason = ""
            state.autopilot.completed_actions.extend(
                action
                for action in ("answer_evaluation", "final_evaluation", "feedback_generation")
                if action not in state.autopilot.completed_actions
            )
            state.autopilot.updated_at = datetime.now(timezone.utc)
            await self._persist(session_id, state)
            self._record_debug("autopilot_completed", session_id)
        return state

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
            previous_candidate = runtime.state.candidate
            same_resume = previous_candidate.raw_resume_text.strip() == resume_text.strip()
            runtime.state.candidate = (
                previous_candidate.model_copy(deep=True)
                if same_resume
                else CandidateProfile(raw_resume_text=resume_text)
            )
            runtime.state.candidate.raw_resume_text = resume_text
            runtime.state.job = JobDescription()
            runtime.state.job_review = JobDescriptionReview()
            runtime.state.company = CompanyInfo(name=company_name)
            runtime.state.interviewer = InterviewerProfile(
                name=interviewer_name,
                position=interviewer_position,
                company=company_name,
            )
            runtime.state.entity_resolutions = []
            runtime.state.fact_cards = []
            runtime.state.strategy = InterviewStrategy()
            runtime.state.blueprint = InterviewBlueprint()
            runtime.state.mock_interview = MockInterviewPlan()
            runtime.state.mock_session = MockInterviewSession()
            runtime.state.evaluation = EvaluationReport()
            runtime.state.feedback = FeedbackReport()
            runtime.state.evidence.clear()
            runtime.state.evaluated_competencies.clear()
            runtime.state.missing_signals.clear()
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

    @staticmethod
    def current_mock_question(state: InterviewState):
        session = state.mock_session
        questions = state.mock_interview.questions
        if session.status != MockSessionStatus.ACTIVE or session.current_question_index >= len(
            questions
        ):
            return None
        question = questions[session.current_question_index]
        if session.pending_follow_up:
            return question.model_copy(
                update={
                    "question": session.pending_follow_up,
                    "rationale": "根据上一回答中缺失的证据进行追问",
                }
            )
        return question

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
                )
                runtime.state.live_interview_records.append(record)
                for evidence in runtime.state.evidence[evidence_count_before:]:
                    if evidence.source.value == "live_interview":
                        evidence.source_record_id = record.id
            runtime.state.next_action = "Generate evidence-based hiring evaluation"
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
        interviewer: InterviewerProfile | None = None,
        parallel_prefix: int = 0,
    ) -> InterviewState:
        workflow_started = perf_counter()
        async with self._lock_for(session_id):
            if name == "evaluation" and not runtime.state.evidence:
                raise EvaluationStateError(
                    "No interview evidence is available; complete a mock or live interview first"
                )
            if company_name is not None:
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
                    await self._persist(session_id, runtime.state)
                for index, (agent_name, instruction) in enumerate(
                    steps[start_index:], start=start_index + 1
                ):
                    runtime.state.workflow.current_step = agent_name
                    await self._persist(session_id, runtime.state)
                    await runtime.run(agent_name, instruction)
                    runtime.state.workflow.completed_steps = index
                    self._sync_intelligence(runtime.state)
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

    @staticmethod
    def _sync_intelligence(state: InterviewState) -> None:
        sync_entity_resolutions(state)
        build_fact_cards(state)

    @staticmethod
    def _resolve_live_question_segment(
        segments: list[TranscriptSegment],
        answer_segment: TranscriptSegment,
        question_segment_id: UUID | None,
    ) -> TranscriptSegment | None:
        if question_segment_id is not None:
            explicit = next((item for item in segments if item.id == question_segment_id), None)
            if explicit is None:
                raise LiveInterviewStateError("Question segment was not found")
            if explicit.speaker != TranscriptSpeaker.INTERVIEWER:
                raise LiveInterviewStateError("Question segment must belong to the interviewer")
            return explicit
        prior_questions = [
            item
            for item in segments
            if item.sequence < answer_segment.sequence
            and item.speaker == TranscriptSpeaker.INTERVIEWER
            and item.stable
            and item.confirmed
        ]
        return prior_questions[-1] if prior_questions else None

    @staticmethod
    def _infer_live_competency(state: InterviewState, question: str) -> str:
        for suggestion in reversed(state.live_interview.suggestions):
            final_question = suggestion.final_question or suggestion.suggested_question
            if final_question and (
                final_question == question
                or final_question[:80] in question
                or question[:80] in final_question
            ):
                return suggestion.competency
        for round_item in state.blueprint.rounds:
            for item in round_item.questions:
                if item.question and (
                    item.question == question
                    or item.question[:80] in question
                    or question[:80] in item.question
                ):
                    return item.competency
        if state.job.competencies:
            return state.job.competencies[0]
        if state.mock_interview.questions:
            return state.mock_interview.questions[0].competency
        return ""

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
        if runtime.state.evaluation.finalized_at:
            runtime.state.enforce_evaluation_evidence_floor()
        self._runtimes[session_id] = runtime
        self._locks.setdefault(session_id, asyncio.Lock())
        await self._persist(session_id, runtime.state)
        return runtime

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def _persist(self, session_id: str, state: InterviewState) -> None:
        await self.storage.save_session(session_id, state.model_dump(mode="json"))

    def get_cached_runtime(self, session_id: str) -> AgentRuntime | None:
        return self._runtimes.get(session_id)

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
