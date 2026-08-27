"""Live question planning, suggestion decisions, and interview maps."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from uuid import UUID

from interview_os.core.state import (
    ACTION_CARD_SOURCE_REFS_MAX,
    BOUNDARY_MERGE_THRESHOLD,
    CROSS_VALIDATION_EVIDENCE_COUNT,
    MIN_COMPETENCY_COVERAGE,
    MIN_EVIDENCE_COUNT,
    QUESTION_USAGE_MAX_ITEMS,
    InterviewQuestion,
    InterviewState,
    LiveActionCard,
    LiveInterviewStatus,
    LiveQuestionUsage,
    QuestionSuggestion,
    QuestionSuggestionStatus,
    TranscriptSegment,
    TranscriptSpeaker,
)

logger = logging.getLogger(__name__)

from interview_os.services.service_contracts import (
    LiveInterviewStateError,
)
from interview_os.services.service_mixin import InterviewServiceMixin


class LivePlanningServiceMixin(InterviewServiceMixin):
    """Live question planning, suggestion decisions, and interview maps."""

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
            self._refresh_live_coverage_guidance(runtime.state)
            self._refresh_live_question_usage(runtime.state)
            await runtime.run(
                "live_interview_agent",
                "根据最新候选人回答准备一个问题；优先补齐关键证据，然后推进未覆盖能力。",
            )
            self._refresh_live_question_usage(runtime.state)
            runtime.state.next_action = "Interviewer reviews the suggested next question"
            await self._persist(session_id, runtime.state)
        self._record_debug(
            "live_question_planned",
            session_id,
            detail=(
                f"summary_until={runtime.state.live_interview.summarized_until_sequence}; "
                f"recent_segments={min(12, len(runtime.state.live_interview.segments))}; "
                f"duplicates_dropped={runtime.state.live_interview.duplicate_segments_dropped}; "
                f"top_gap={runtime.state.live_interview.coverage_guidance[0].competency if runtime.state.live_interview.coverage_guidance else 'none'}; "
                f"pending_blueprint={sum(1 for item in runtime.state.live_interview.question_usage if item.status == 'pending')}; "
                f"top_boundary_confidence={runtime.state.live_interview.answer_boundary_suggestions[-1].confidence if runtime.state.live_interview.answer_boundary_suggestions else 0}; "
                f"action={runtime.state.live_interview.action_card.action_type}"
            ),
        )
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
            self._refresh_live_question_usage(runtime.state)
            await self._persist(session_id, runtime.state)
        self._record_debug("live_question_decided", session_id, detail=f"status={status.value}")
        return runtime.state

    async def _maybe_plan_live_next_question(self, session_id: str) -> None:
        """Auto-trigger next-question planning after evidence is confirmed.

        Fires only when the live session is active, no suggestion is still
        pending (so repeated confirms do not stack LLM calls), and there is
        something to plan against: pending candidate segments or live evidence.
        """
        runtime = await self._get_runtime(session_id)
        live = runtime.state.live_interview
        if live.status != LiveInterviewStatus.ACTIVE:
            return
        if any(item.status == QuestionSuggestionStatus.PENDING for item in live.suggestions):
            return
        live_evidence = [
            item for item in runtime.state.live_interview_records if item.source == "live_interview"
        ]
        recorded_segment_ids = {
            segment_id for record in live_evidence for segment_id in record.transcript_segment_ids
        }
        has_pending_candidate = any(
            item.speaker == TranscriptSpeaker.CANDIDATE
            and item.stable
            and item.confirmed
            and item.id not in recorded_segment_ids
            for item in live.segments
        )
        if not has_pending_candidate and not live_evidence:
            return
        await self.plan_live_next_question(session_id)

    async def stream_live_suggestion(self, session_id: str) -> AsyncIterator[dict[str, str]]:
        """Stream a next-question suggestion token by token.

        Yields each text chunk as the model generates it, then persists the
        completed suggestion into the review queue so the flow is the same as
        a non-streamed plan. Falls back to a static suggestion if streaming is
        not possible. Callers should guard with :meth:`assert_live_active`
        first so the 409 is raised before streaming begins.
        """
        runtime = await self._get_runtime(session_id)
        state = runtime.state
        self.assert_live_active(state)
        context = self._live_audio_context(state)
        text = ""
        stream_failed = False
        if self.llm_client is not None and hasattr(self.llm_client, "chat_stream"):
            prompt = (
                "你是面试官助手。根据面试上下文，给出一个聚焦证据缺口的下一问追问。"
                "必须直接追问上下文中明确标记的最新候选人回答，不得重新追问较早回答。"
                "直接输出问题本身，中文，简洁，不超过两句话。不要输出 JSON。\n\n"
                f"面试上下文：\n{context}"
            )
            messages = [
                {"role": "system", "content": "你是面试官助手，根据上下文给出下一问追问。"},
                {"role": "user", "content": prompt},
            ]
            try:
                async for piece in self.llm_client.chat_stream(
                    messages, temperature=0.4, max_tokens=200
                ):
                    text += piece
                    yield {"type": "append", "text": piece}
            except Exception as exc:  # noqa: BLE001 - model adapter boundary
                stream_failed = True
                logger.warning("Live suggestion stream degraded: %s", type(exc).__name__)
        if stream_failed or not text.strip():
            text = "请再补充说明一下你刚才提到的方案权衡与量化结果。"
            yield {"type": "replace", "text": text}
        await self._inject_audio_direct_suggestion(session_id, runtime, text)

    def _refresh_live_question_usage(self, state: InterviewState) -> None:
        questions = self._live_blueprint_questions(state)
        if not questions:
            state.live_interview.question_usage = []
            return
        by_question_id: dict[str, list[QuestionSuggestion]] = {item[0]: [] for item in questions}
        for suggestion in state.live_interview.suggestions:
            source_id = suggestion.source_question_id.strip()
            if source_id in by_question_id:
                by_question_id[source_id].append(suggestion)
        usage: list[LiveQuestionUsage] = []
        used_ids = set(state.live_interview.used_question_ids)
        for question_id, round_name, question in questions:
            related = by_question_id.get(question_id, [])
            latest = related[-1] if related else None
            status = "used" if question_id in used_ids else "suggested" if related else "pending"
            usage.append(
                LiveQuestionUsage(
                    question_id=question_id,
                    round_name=round_name,
                    question=question.question,
                    competency=question.competency,
                    status=status,
                    suggested_count=len(related),
                    last_suggestion_status=latest.status.value if latest else "",
                    last_decided_at=latest.created_at if latest and status == "used" else None,
                )
            )
        status_order = {"pending": 0, "suggested": 1, "used": 2}
        usage.sort(
            key=lambda item: (status_order.get(item.status, 3), item.round_name, item.question)
        )
        state.live_interview.question_usage = usage[:QUESTION_USAGE_MAX_ITEMS]

    def _refresh_live_action_card(self, state: InterviewState) -> None:
        live = state.live_interview
        live_evidence = [
            item for item in state.live_interview_records if item.source == "live_interview"
        ]
        recorded_segment_ids = {
            segment_id for record in live_evidence for segment_id in record.transcript_segment_ids
        }
        pending_candidate_segments = [
            item
            for item in live.segments
            if item.speaker == TranscriptSpeaker.CANDIDATE
            and item.stable
            and item.confirmed
            and item.id not in recorded_segment_ids
        ]
        unknown_segments = [
            item
            for item in live.segments
            if item.speaker == TranscriptSpeaker.UNKNOWN and item.stable and item.confirmed
        ]
        pending_suggestion = next(
            (
                item
                for item in reversed(live.suggestions)
                if item.status == QuestionSuggestionStatus.PENDING
            ),
            None,
        )
        boundary = max(
            live.answer_boundary_suggestions,
            key=lambda item: item.confidence,
            default=None,
        )
        total_evidence = len(state.evidence)
        covered_competencies = {
            item.competency for item in state.evidence if item.competency.strip()
        }
        evidence_status = f"{total_evidence} 条证据 · {len(covered_competencies)} 个能力覆盖"
        top_gap = live.coverage_guidance[0] if live.coverage_guidance else None
        source_refs = [
            f"segments={len(live.segments)}",
            f"pending_candidate={len(pending_candidate_segments)}",
            f"live_evidence={len(live_evidence)}",
        ]

        def card(
            action_type: str,
            priority: str,
            title: str,
            detail: str,
            primary_cta: str,
            secondary_cta: str = "",
            refs: list[str] | None = None,
        ) -> LiveActionCard:
            return LiveActionCard(
                action_type=action_type,
                priority=priority,
                title=title,
                detail=detail,
                primary_cta=primary_cta,
                secondary_cta=secondary_cta,
                evidence_status=evidence_status,
                source_refs=[*source_refs, *(refs or [])][:ACTION_CARD_SOURCE_REFS_MAX],
            )

        if live.status == LiveInterviewStatus.IDLE:
            live.action_card = card(
                "start",
                "medium",
                "开始实时面试",
                "确认候选人已知情并同意转写后，启动监听并记录第一轮问题。",
                "开始实时会话",
            )
            return
        if live.status == LiveInterviewStatus.PAUSED:
            live.action_card = card(
                "resume",
                "medium",
                "实时面试已暂停",
                "恢复监听前，可以先审阅已产生的转写片段和证据归档状态。",
                "恢复监听",
                "审阅证据",
            )
            return
        if live.status == LiveInterviewStatus.COMPLETED:
            ready = (
                total_evidence >= MIN_EVIDENCE_COUNT
                and len(covered_competencies) >= MIN_COMPETENCY_COVERAGE
            )
            live.action_card = card(
                "evaluate" if ready else "review_evidence",
                "high" if not ready else "medium",
                "生成最终评价" if ready else "先补齐证据再评价",
                (
                    "证据数量和能力覆盖已达到招聘评价门槛，可以聚合证据生成报告。"
                    if ready
                    else "实时会话已结束，但证据数量或能力覆盖不足；请先确认候选人回答或导入记录。"
                ),
                "聚合证据并评估" if ready else "审阅待确认回答",
                refs=["completed=true"],
            )
            return
        if unknown_segments:
            live.action_card = card(
                "review_speaker",
                "high",
                "先确认待识别说话人",
                "连续监听产生了待确认片段；确认说话人与文本后，后续 Agent 才能安全使用这些事实。",
                "审阅转写片段",
                refs=[f"unknown={len(unknown_segments)}"],
            )
            return
        if boundary is not None and boundary.confidence >= BOUNDARY_MERGE_THRESHOLD:
            live.action_card = card(
                "merge_boundary",
                "high",
                "合并连续候选人回答",
                (
                    f"检测到 {len(boundary.answer_segment_ids)} 段连续候选人发言，"
                    f"边界置信度 {round(boundary.confidence * 100)}%；建议先合并确认成一条证据。"
                ),
                "按边界建议合并",
                "逐条确认",
                refs=[f"boundary={round(boundary.confidence, 2)}"],
            )
            return
        if pending_candidate_segments:
            live.action_card = card(
                "confirm_evidence",
                "high",
                "归档候选人回答为证据",
                (
                    f"还有 {len(pending_candidate_segments)} 段候选人回答未进入证据链；"
                    "先确认能力维度，再让下一问题建议基于已确认事实。"
                ),
                "确认为证据",
                "生成下一问题",
            )
            return
        if pending_suggestion is not None:
            live.action_card = card(
                "decide_question",
                "medium",
                "处理下一问题建议",
                (
                    f"AI 已准备一个聚焦“{pending_suggestion.competency}”的问题；"
                    "采用、编辑或跳过后，蓝图问题使用图会同步更新。"
                ),
                "采用或编辑问题",
                "跳过建议",
                refs=[f"suggestion={str(pending_suggestion.id)[:8]}"],
            )
            return
        if top_gap is not None and (
            total_evidence < MIN_EVIDENCE_COUNT
            or len(covered_competencies) < MIN_COMPETENCY_COVERAGE
            or top_gap.priority in {"high", "medium"}
        ):
            live.action_card = card(
                "plan_gap_question",
                "medium" if total_evidence >= CROSS_VALIDATION_EVIDENCE_COUNT else "high",
                f"补齐“{top_gap.competency}”证据",
                f"{top_gap.reason} 建议下一问：{top_gap.sample_question}",
                "根据回答生成",
                "继续提问",
                refs=[f"gap={top_gap.competency}", f"priority={top_gap.priority}"],
            )
            return
        if (
            total_evidence < MIN_EVIDENCE_COUNT
            or len(covered_competencies) < MIN_COMPETENCY_COVERAGE
        ):
            live.action_card = card(
                "plan_gap_question",
                "high",
                "继续补齐招聘评价证据",
                "当前证据数量或能力覆盖仍不足；如果 JD 能力维度尚未完善，请先补充岗位职责、任职要求和团队背景。",
                "根据回答生成",
                "补充 JD 信息",
                refs=["gap=generic"],
            )
            return
        live.action_card = card(
            "ready_to_evaluate",
            "low",
            "证据链已基本可用",
            "当前证据数量和能力覆盖达到基础招聘评价门槛；可以继续深挖，也可以结束后生成评价。",
            "结束并生成评价",
            "继续追问",
        )

    @staticmethod
    def _live_blueprint_questions(
        state: InterviewState,
    ) -> list[tuple[str, str, InterviewQuestion]]:
        return [
            (str(question.id), interview_round.name, question)
            for interview_round in state.blueprint.rounds
            for question in interview_round.questions
            if question.question.strip()
        ]

    @staticmethod
    def _live_target_competencies(state: InterviewState) -> list[str]:
        values = [
            *state.job.competencies,
            *[
                question.competency
                for interview_round in state.blueprint.rounds
                for question in interview_round.questions
            ],
            *[question.competency for question in state.mock_interview.questions],
        ]
        return list(dict.fromkeys(item.strip() for item in values if item.strip()))

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
