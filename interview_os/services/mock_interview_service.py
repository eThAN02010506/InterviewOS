"""Mock interview lifecycle, adaptive questions, and evaluation operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from interview_os.core.evidence import Evidence, EvidencePolarity, EvidenceSource
from interview_os.core.follow_up_planner import plan_follow_up
from interview_os.core.question_understanding import deterministic_question_understanding
from interview_os.core.request_context import current_owner
from interview_os.core.state import (
    EVIDENCE_SIGNAL_CHAR_LIMIT,
    MOCK_CACHE_MIN,
    AnswerEvaluation,
    InterviewQuestion,
    InterviewState,
    MockAnswerRecord,
    MockInterviewSession,
    MockSessionStatus,
    SpeechDeliveryFeedback,
)

logger = logging.getLogger(__name__)

from interview_os.services.service_contracts import (
    MockInterviewStateError,
)
from interview_os.services.service_mixin import InterviewServiceMixin


class MockInterviewServiceMixin(InterviewServiceMixin):
    """Mock interview lifecycle, adaptive questions, and evaluation operations."""

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
            if runtime.state.mock_session.status != MockSessionStatus.IDLE:
                raise MockInterviewStateError("Mock interview cannot restart while evaluating")
            runtime.state.mock_session = MockInterviewSession(
                status=MockSessionStatus.ACTIVE,
                started_at=datetime.now(timezone.utc),
            )
            runtime.state.next_action = "Answer the current mock interview question"
            await self._persist(session_id, runtime.state)
            return runtime.state

    async def add_custom_mock_question(
        self,
        session_id: str,
        question_text: str,
        *,
        competency: str = "",
        practice_now: bool = True,
    ) -> InterviewState:
        """Interpret and persist a user-supplied question in the mock pipeline.

        The semantic model call runs outside the session mutation lock. A
        deterministic interpretation is always available, and the original
        wording stays untouched regardless of model output.
        """
        runtime = await self._get_runtime(session_id)
        normalized_text = " ".join(question_text.split()).strip()
        normalized_competency = " ".join(competency.split()).strip()
        if len(normalized_text) < 2:
            raise MockInterviewStateError("自定义问题至少需要 2 个字符")

        # Analyze from a stable snapshot without blocking answers/navigation.
        # The write phase below revalidates status and duplicates after the
        # optional model call completes.
        async with self._lock_for(session_id):
            state = runtime.state
            if state.mock_session.status in {
                MockSessionStatus.EVALUATING,
                MockSessionStatus.COMPLETED,
            }:
                raise MockInterviewStateError("本轮已结束，请新建练习会话后添加问题")
            state_snapshot = state.model_copy(deep=True)
        agent = runtime.get_agent("mock_interview_agent")
        if agent is not None and hasattr(agent, "analyze_custom_question"):
            understanding = await agent.analyze_custom_question(  # type: ignore[union-attr]
                state_snapshot,
                normalized_text,
                competency=normalized_competency,
            )
        else:
            explicit_requirements = [
                item.text
                for item in state_snapshot.job_review.requirements
                if item.origin.value == "explicit"
            ]
            understanding = deterministic_question_understanding(
                normalized_text,
                competency=normalized_competency,
                job_title=state_snapshot.job.title,
                explicit_job_requirements=explicit_requirements,
                confirmed_claims=[
                    (str(claim.id), claim.statement)
                    for claim in state_snapshot.resume_review.claims
                    if claim.status.value in {"confirmed", "modified"}
                ],
            )

        async with self._lock_for(session_id):
            state = runtime.state
            mock_session = state.mock_session
            if mock_session.status in {
                MockSessionStatus.EVALUATING,
                MockSessionStatus.COMPLETED,
            }:
                raise MockInterviewStateError("本轮已结束，请新建练习会话后添加问题")

            questions = state.mock_interview.questions
            comparison = normalized_text.casefold()
            existing_index = next(
                (
                    index
                    for index, item in enumerate(questions)
                    if " ".join(item.question.split()).strip().casefold() == comparison
                ),
                None,
            )
            if existing_index is None:
                selected_competency = normalized_competency or understanding.competency
                requirements = (
                    agent.question_requirements(normalized_text)  # type: ignore[union-attr]
                    if agent is not None and hasattr(agent, "question_requirements")
                    else []
                )
                framework = (
                    agent.deterministic_framework(  # type: ignore[union-attr]
                        state, selected_competency
                    )
                    if agent is not None and hasattr(agent, "deterministic_framework")
                    else "建议按：直接结论 → 真实案例 → 个人行动 → 可验证结果 → 复盘来组织。"
                )
                example = (
                    agent.teaching_example(  # type: ignore[union-attr]
                        selected_competency, normalized_text
                    )
                    if agent is not None and hasattr(agent, "teaching_example")
                    else ""
                )
                custom_question = InterviewQuestion(
                    question=normalized_text,
                    competency=selected_competency,
                    rationale="用户认为面试中可能出现的问题",
                    strong_signals=requirements,
                    follow_ups=understanding.likely_follow_ups[:3],
                    answer_framework=framework,
                    question_requirements=requirements,
                    example_answer=example,
                    understanding=understanding,
                    source="custom",
                )
                if practice_now:
                    insertion_index = (
                        mock_session.current_question_index + 1
                        if mock_session.status == MockSessionStatus.ACTIVE and questions
                        else 0
                    )
                    questions.insert(insertion_index, custom_question)
                    target_index = insertion_index
                else:
                    questions.append(custom_question)
                    target_index = len(questions) - 1
            else:
                target_index = existing_index

            if practice_now:
                if mock_session.status == MockSessionStatus.IDLE:
                    state.mock_session = MockInterviewSession(
                        status=MockSessionStatus.ACTIVE,
                        started_at=datetime.now(timezone.utc),
                        current_question_index=target_index,
                    )
                    mock_session = state.mock_session
                else:
                    mock_session.pending_follow_up = ""
                    mock_session.pending_parent_question_id = None
                    mock_session.pending_follow_up_stage = ""
                    mock_session.pending_follow_up_rationale = ""
                    mock_session.current_question_index = target_index
                state.next_action = "Answer the current custom mock interview question"
            else:
                state.next_action = "Review the custom question in the mock interview pool"
            await self._persist(session_id, state)
            return state

    async def submit_mock_answer(
        self,
        session_id: str,
        question_id: UUID,
        answer: str,
        *,
        retry: bool = False,
        retry_response_id: UUID | None = None,
        recording_id: UUID | None = None,
    ) -> InterviewState:
        """Evaluate one mock answer. Does NOT advance the interview — the caller
        (frontend) chooses 重新来 / 下一题 / 结束面试 afterwards."""
        runtime = await self._get_runtime(session_id)
        staged_audio: Path | None = None
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            if recording_id is not None:
                staged_audio = self._pending_mock_audio_path(session_id, recording_id)
                if staged_audio is None:
                    raise MockInterviewStateError("Recording does not belong to this session")
            if staged_audio is not None and any(
                item.audio_file == staged_audio.name
                for item in (
                    *mock_session.responses,
                    *mock_session.attempt_history,
                )
            ):
                raise MockInterviewStateError("Recording has already been submitted")
            questions = runtime.state.mock_interview.questions
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            if mock_session.current_question_index >= len(questions):
                raise MockInterviewStateError("Mock interview has no remaining questions")
            question = questions[mock_session.current_question_index]
            if retry_response_id is not None and not retry:
                raise MockInterviewStateError("retry_response_id requires retry=true")
            retry_target = (
                next(
                    (
                        item
                        for item in mock_session.responses
                        if item.id == retry_response_id and item.question_id == question.id
                    ),
                    None,
                )
                if retry_response_id is not None
                else None
            )
            if retry_response_id is not None and retry_target is None:
                raise MockInterviewStateError(
                    "Retry response does not belong to the current question"
                )
            is_follow_up = (
                retry_target.is_follow_up
                if retry_target is not None
                else bool(mock_session.pending_follow_up)
            )
            expected_id = mock_session.pending_parent_question_id or question.id
            if expected_id != question_id:
                raise MockInterviewStateError("Answer does not match the current question")
            asked_question = (
                retry_target.question
                if retry_target is not None
                else mock_session.pending_follow_up or question.question
            )
            follow_up_stage = (
                retry_target.follow_up_stage
                if retry_target is not None
                else mock_session.pending_follow_up_stage
            )

            prior = retry_target or next(
                (
                    item
                    for item in reversed(mock_session.responses)
                    if item.question_id == question.id and item.question == asked_question
                ),
                None,
            )
            if prior is not None and not retry:
                raise MockInterviewStateError(
                    "This question was already answered; submit with retry=true to replace it"
                )

            replaced_records: list[MockAnswerRecord] = []
            if retry and prior is not None:
                # Identify the replacement set before model work, but do not
                # mutate live state until a valid new evaluation exists.
                replaced_records = [prior]
                if prior is not None and not prior.is_follow_up:
                    # Follow-up answers were elicited from the old main answer;
                    # retaining them would leave stale evidence in the report.
                    replaced_records.extend(
                        item
                        for item in mock_session.responses
                        if item.question_id == question.id and item.is_follow_up
                    )

            record = MockAnswerRecord(
                question_id=question.id,
                question=asked_question,
                competency=question.competency,
                answer=answer,
                evaluation=AnswerEvaluation(
                    content=0.0, technical_depth=0.0, structure=0.0, impact=0.0
                ),
                answer_modality="asr" if staged_audio is not None else "typed",
                is_follow_up=is_follow_up,
                follow_up_stage=follow_up_stage if is_follow_up else "",
                audio_file=staged_audio.name if staged_audio is not None else "",
                speech_delivery=(
                    self._mock_speech_feedback[(current_owner(), session_id, recording_id)][1]
                    if recording_id is not None
                    and (current_owner(), session_id, recording_id) in self._mock_speech_feedback
                    else SpeechDeliveryFeedback()
                ),
            )
            payload = json.dumps(
                {
                    "question": asked_question,
                    "answer": answer,
                    "competency": question.competency or "Answer Quality",
                    "evidence_source": "mock_interview",
                    "record_id": str(record.id),
                    "persist_evidence": False,
                    "answer_modality": record.answer_modality,
                }
            )
            message = await runtime.run("coach_agent", payload)
            try:
                evaluation = AnswerEvaluation.model_validate_json(message.content)
            except (ValueError, TypeError) as exc:
                raise MockInterviewStateError(
                    "Coach agent did not return a valid answer evaluation"
                ) from exc
            record.evaluation = evaluation
            if replaced_records:
                replaced_ids = {item.id for item in replaced_records}
                mock_session.attempt_history.extend(
                    item.model_copy(deep=True) for item in replaced_records
                )
                mock_session.responses = [
                    item for item in mock_session.responses if item.id not in replaced_ids
                ]
                runtime.state.evidence = [
                    item
                    for item in runtime.state.evidence
                    if not (
                        item.source == EvidenceSource.MOCK_INTERVIEW
                        and item.source_record_id in replaced_ids
                    )
                ]
                if prior is not None and not prior.is_follow_up:
                    mock_session.pending_follow_up = ""
                    mock_session.pending_parent_question_id = None
                    mock_session.pending_follow_up_stage = ""
                    mock_session.pending_follow_up_rationale = ""
                    # Dependent probes belonged to the replaced main answer.
                    # Re-plan from the new evidence instead of suppressing a
                    # branch merely because the old attempt already saw it.
                    mock_session.follow_up_history.pop(str(question.id), None)
            mock_session.responses.append(record)
            runtime.state.evidence.append(
                Evidence(
                    competency=record.competency or "Answer Quality",
                    signal=answer[:EVIDENCE_SIGNAL_CHAR_LIMIT],
                    confidence=evaluation.overall_score(),
                    source=EvidenceSource.MOCK_INTERVIEW,
                    source_record_id=record.id,
                    polarity=EvidencePolarity.NEUTRAL,
                    notes="; ".join(evaluation.missing_signals),
                )
            )
            self._invalidate_final_reports(
                runtime.state, "Mock answer evidence changed; regenerate evaluation"
            )
            if is_follow_up:
                # Clear only the pending offer.  A later /next asks the planner
                # whether this answer warrants a deeper, unused branch.
                mock_session.pending_follow_up = ""
                mock_session.pending_parent_question_id = None
                mock_session.pending_follow_up_stage = ""
                mock_session.pending_follow_up_rationale = ""
            runtime.state.next_action = "回答已评价：请选择 重新来 / 下一题 / 结束面试"
            await self._persist(session_id, runtime.state)
        return runtime.state

    async def advance_mock_interview(self, session_id: str) -> InterviewState:
        """Advance to the next question, or offer a follow-up when the latest
        main answer left missing signals. Advancing may skip the current
        question (the interviewer stays in control); refills when low."""
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            questions = runtime.state.mock_interview.questions
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            if mock_session.current_question_index >= len(questions):
                raise MockInterviewStateError("Mock interview has no remaining questions")
            question = questions[mock_session.current_question_index]
            if mock_session.pending_follow_up:
                # A follow-up is pending; advancing skips it and moves on.
                mock_session.pending_follow_up = ""
                mock_session.pending_parent_question_id = None
                mock_session.pending_follow_up_stage = ""
                mock_session.pending_follow_up_rationale = ""
                mock_session.current_question_index += 1
            else:
                last_response = next(
                    (
                        item
                        for item in reversed(mock_session.responses)
                        if item.question_id == question.id
                    ),
                    None,
                )
                history_key = str(question.id)
                decision = (
                    plan_follow_up(
                        question,
                        last_response,
                        offered_questions=mock_session.follow_up_history.get(history_key, []),
                    )
                    if last_response is not None
                    else None
                )
                if decision is not None:
                    # Each /next makes one inspectable decision from the latest
                    # answer. The user remains in control and can skip the offer
                    # by advancing again.
                    mock_session.pending_follow_up = decision.question
                    mock_session.pending_parent_question_id = question.id
                    mock_session.pending_follow_up_stage = decision.stage
                    mock_session.pending_follow_up_rationale = decision.rationale
                    mock_session.follow_up_history.setdefault(history_key, []).append(
                        decision.question
                    )
                    runtime.state.next_action = (
                        f"Answer the {decision.stage} follow-up question: {decision.rationale}"
                    )
                    await self._persist(session_id, runtime.state)
                    return runtime.state
                mock_session.current_question_index += 1
            if mock_session.current_question_index >= len(questions):
                # The model refill stays asynchronous, but an active session must
                # never expose an empty current question at the pool boundary.
                agent = runtime.get_agent("mock_interview_agent")
                if agent is not None and hasattr(agent, "deterministic_refill_question"):
                    questions.append(
                        agent.deterministic_refill_question(  # type: ignore[union-attr]
                            runtime.state, ordinal=len(questions) + 1
                        )
                    )
            runtime.state.next_action = "Answer the next mock interview question"
            await self._maybe_refill_mock_questions(session_id, runtime, mock_session, questions)
            await self._persist(session_id, runtime.state)
        return runtime.state

    async def previous_mock_question(self, session_id: str) -> InterviewState:
        """Go back to the previous question (no-op at the first question)."""
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            if mock_session.current_question_index <= 0:
                raise MockInterviewStateError("已回到第一题")
            # Clear any follow-up state so the previous question renders cleanly.
            mock_session.pending_follow_up = ""
            mock_session.pending_parent_question_id = None
            mock_session.pending_follow_up_stage = ""
            mock_session.pending_follow_up_rationale = ""
            mock_session.current_question_index -= 1
            runtime.state.next_action = "Answer the current mock interview question"
            await self._persist(session_id, runtime.state)
        return runtime.state

    async def finish_mock_interview(self, session_id: str) -> InterviewState:
        """End the mock interview manually and run the evaluation when answers exist."""
        runtime = await self._get_runtime(session_id)
        should_evaluate = False
        async with self._lock_for(session_id):
            mock_session = runtime.state.mock_session
            if mock_session.status == MockSessionStatus.EVALUATING:
                raise MockInterviewStateError("Mock interview evaluation is already running")
            if mock_session.status != MockSessionStatus.ACTIVE:
                raise MockInterviewStateError("Mock interview is not active")
            should_evaluate = bool(mock_session.responses)
            mock_session.pending_follow_up = ""
            mock_session.pending_parent_question_id = None
            mock_session.pending_follow_up_stage = ""
            mock_session.pending_follow_up_rationale = ""
            if should_evaluate:
                mock_session.status = MockSessionStatus.EVALUATING
                runtime.state.next_action = "Generate evidence-based evaluation"
            else:
                mock_session.status = MockSessionStatus.COMPLETED
                mock_session.completed_at = datetime.now(timezone.utc)
                runtime.state.next_action = "Mock interview completed without answers"
            await self._persist(session_id, runtime.state)
        if should_evaluate:
            try:
                state = await self.run_evaluation(session_id)
            except Exception:
                # Keep all answers and make the normal finish action retryable.
                async with self._lock_for(session_id):
                    runtime.state.mock_session.status = MockSessionStatus.ACTIVE
                    runtime.state.mock_session.completed_at = None
                    runtime.state.next_action = (
                        "Final evaluation failed; retry ending the interview"
                    )
                    await self._persist(session_id, runtime.state)
                raise
            async with self._lock_for(session_id):
                state.mock_session.status = MockSessionStatus.COMPLETED
                state.mock_session.completed_at = datetime.now(timezone.utc)
                await self._persist(session_id, state)
            return state
        return runtime.state

    async def _maybe_refill_mock_questions(
        self, session_id: str, runtime, mock_session, questions
    ) -> None:
        pending = len(questions) - mock_session.current_question_index
        if (
            pending < MOCK_CACHE_MIN
            and not mock_session.refill_in_flight
            and mock_session.status == MockSessionStatus.ACTIVE
        ):
            mock_session.refill_in_flight = True
            self._record_debug("mock_refill_scheduled", session_id)
            owner = current_owner()
            self._background.schedule(self._refill_mock_questions_task(session_id, owner=owner))

    async def _refill_mock_questions_task(self, session_id: str, *, owner: str) -> None:
        """Background: generate more mock questions when the cache runs low."""
        runtime = self._runtimes.get((owner, session_id))
        if runtime is None:
            return
        agent = runtime.get_agent("mock_interview_agent")
        recent = [
            f"{item.competency}：{item.answer[:120]}"
            for item in runtime.state.mock_session.responses
        ]
        new_questions: list[Any] = []
        try:
            if agent is not None and hasattr(agent, "generate_mock_questions"):
                new_questions = await agent.generate_mock_questions(  # type: ignore[union-attr]
                    runtime.state, recent_answers=recent
                )
        except Exception as exc:  # noqa: BLE001 - background boundary
            logger.error("Mock question refill failed for %s (%s)", session_id, type(exc).__name__)
            new_questions = []
        async with self._lock_for(session_id):
            current = self._runtimes.get((owner, session_id))
            if current is None:
                return
            mock_session = current.state.mock_session
            mock_session.refill_in_flight = False
            if mock_session.status == MockSessionStatus.ACTIVE and new_questions:
                existing_ids = {q.id for q in current.state.mock_interview.questions}
                existing_texts = {
                    q.question.strip() for q in current.state.mock_interview.questions
                }
                added = 0
                for q in new_questions:
                    if q.id not in existing_ids and q.question.strip() not in existing_texts:
                        current.state.mock_interview.questions.append(q)
                        existing_texts.add(q.question.strip())
                        added += 1
                self._record_debug("mock_refill_completed", session_id, detail=f"added={added}")
            await self._persist(session_id, current.state)

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
                    "rationale": session.pending_follow_up_rationale
                    or "根据上一回答选择下一个验证分支",
                }
            )
        return question
