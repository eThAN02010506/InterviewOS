"""Unit tests for Live Copilot internal service logic.

These tests exercise pure state-transition and scoring helpers directly,
without an LLM, so formula regressions are caught before they surface in
API-level end-to-end tests.
"""

from __future__ import annotations

import pytest

from interview_os.core.evidence import Evidence, EvidencePolarity, EvidenceSource
from interview_os.core.state import (
    AnswerEvaluation,
    AnswerReviewStatus,
    AnswerScoringSource,
    InterviewBlueprint,
    InterviewQuestion,
    InterviewRound,
    InterviewState,
    JobDescription,
    LiveInterviewRecord,
    LiveInterviewStatus,
    QuestionSuggestion,
    QuestionSuggestionStatus,
    TranscriptSegment,
    TranscriptSpeaker,
)
from interview_os.database.storage import Storage
from interview_os.services.background import BackgroundTaskManager
from interview_os.services.interview_service import InterviewService


class _CoachLLM:
    """Minimal LLM stub returning a valid coach answer evaluation."""

    async def chat(self, messages, **kwargs):
        return (
            '{"content":0.8,"technical_depth":0.8,"structure":0.8,"impact":0.8,'
            '"feedback":[],"improved_answer":"better","observed_signals":["Clear design"],'
            '"missing_signals":[]}'
        )

    async def embed(self, text):
        return []


@pytest.fixture
async def service(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-service.db'}")
    await storage.init_db()
    yield InterviewService(storage, llm_client=None)
    await storage.close()


@pytest.fixture
async def scoring_service(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-scoring.db'}")
    await storage.init_db()
    background = BackgroundTaskManager()
    yield InterviewService(storage, llm_client=_CoachLLM(), background=background)
    await background.close()
    await storage.close()


def _segment(
    sequence: int,
    text: str,
    speaker: TranscriptSpeaker = TranscriptSpeaker.CANDIDATE,
    *,
    source: str = "manual",
    stable: bool = True,
    confirmed: bool = True,
) -> TranscriptSegment:
    return TranscriptSegment(
        sequence=sequence,
        speaker=speaker,
        text=text,
        source=source,
        stable=stable,
        confirmed=confirmed,
    )


def _interviewer_segment(sequence: int, text: str) -> TranscriptSegment:
    return _segment(
        sequence, text, speaker=TranscriptSpeaker.INTERVIEWER, source="copilot"
    )


# ---------------------------------------------------------------------------
# _normalize_live_segment_text
# ---------------------------------------------------------------------------


def test_normalize_strips_punctuation_and_casing(service):
    normalized = service._normalize_live_segment_text("Hello, World! #2")
    assert normalized == "helloworld2"


def test_normalize_handles_cjk_mixed_text(service):
    normalized = service._normalize_live_segment_text("我用  P95  验证了方案。")
    assert normalized == "我用p95验证了方案"


# ---------------------------------------------------------------------------
# _find_duplicate_live_segment
# ---------------------------------------------------------------------------


def test_find_duplicate_returns_prior_stable_segment(service):
    existing = _segment(1, "我先梳理瓶颈并设计分层缓存。")
    segments = [existing]
    duplicate = service._find_duplicate_live_segment(
        segments, "我先梳理瓶颈并设计分层缓存。", TranscriptSpeaker.CANDIDATE
    )
    assert duplicate is not None
    assert duplicate.sequence == 1


def test_find_duplicate_ignores_short_normalized_text(service):
    segments = [_segment(1, "好的"), _segment(2, "好的")]
    duplicate = service._find_duplicate_live_segment(
        segments, "好的", TranscriptSpeaker.CANDIDATE
    )
    assert duplicate is None


def test_find_duplicate_skips_unstable_and_unconfirmed(service):
    unstable = _segment(1, "我先梳理瓶颈并设计分层缓存。", stable=False)
    unconfirmed = _segment(2, "我先梳理瓶颈并设计分层缓存。", confirmed=False)
    segments = [unstable, unconfirmed]
    duplicate = service._find_duplicate_live_segment(
        segments, "我先梳理瓶颈并设计分层缓存。", TranscriptSpeaker.CANDIDATE
    )
    assert duplicate is None


def test_find_duplicate_ignores_different_text(service):
    segments = [_segment(1, "第一段回答。")]
    duplicate = service._find_duplicate_live_segment(
        segments, "完全不同的第二段回答。", TranscriptSpeaker.CANDIDATE
    )
    assert duplicate is None


# ---------------------------------------------------------------------------
# _score_live_boundary
# ---------------------------------------------------------------------------


def test_boundary_scores_high_when_linked_to_question(service):
    question = _interviewer_segment(1, "请讲一次系统设计经历。")
    run = [
        _segment(2, "我先梳理瓶颈并设计分层缓存方案，用压测指标验证了缓存命中率和读写延时的权衡。"),
        _segment(3, "最终灰度上线到全量，P95 延迟降低了 40%，回滚预案也已准备，以上就是这次经历。"),
    ]
    score, factors = service._score_live_boundary(run, question)
    assert score >= 0.75
    assert any("已关联最近面试官问题" in factor for factor in factors)
    assert any("结束信号" in factor for factor in factors)


def test_boundary_scores_lower_without_question(service):
    run = [
        _segment(1, "第一段回答内容。"),
        _segment(2, "第二段补充回答内容。"),
    ]
    score, factors = service._score_live_boundary(run, None)
    assert score < 0.5
    assert any("需人工确认" in factor for factor in factors)


def test_boundary_punishes_short_answer(service):
    run = [_segment(1, "是的。"), _segment(2, "对的。")]
    score, _ = service._score_live_boundary(run, None)
    assert score < 0.5


def test_boundary_flags_asr_segments_for_review(service):
    run = [
        _segment(1, "第一段 ASR 回答。", source="asr"),
        _segment(2, "第二段 ASR 回答。", source="asr"),
    ]
    _, factors = service._score_live_boundary(run, None)
    assert any("ASR" in factor for factor in factors)


def test_boundary_score_is_clamped_to_valid_range(service):
    long_text = "回答内容" * 100
    run = [
        _segment(1, long_text),
        _segment(2, long_text + "以上就是"),
    ]
    question = _interviewer_segment(1, "问题")
    score, _ = service._score_live_boundary(run, question)
    assert 0.2 <= score <= 0.92


# ---------------------------------------------------------------------------
# _refresh_live_coverage_guidance
# ---------------------------------------------------------------------------


def test_coverage_guidance_empty_without_competencies(service):
    state = InterviewState()
    service._refresh_live_coverage_guidance(state)
    assert state.live_interview.coverage_guidance == []


def test_coverage_guidance_marks_missing_competency_high_priority(service):
    state = InterviewState(job=JobDescription(competencies=["系统设计"]))
    service._refresh_live_coverage_guidance(state)
    guidance = state.live_interview.coverage_guidance[0]
    assert guidance.competency == "系统设计"
    assert guidance.priority == "high"
    assert guidance.evidence_count == 0


def test_coverage_guidance_medium_after_single_evidence(service):
    state = InterviewState(job=JobDescription(competencies=["系统设计"]))
    state.evidence.append(
        Evidence(
            competency="系统设计",
            signal="设计方案",
            confidence=0.8,
            source=EvidenceSource.LIVE_INTERVIEW,
        )
    )
    service._refresh_live_coverage_guidance(state)
    guidance = state.live_interview.coverage_guidance[0]
    assert guidance.priority == "medium"
    assert guidance.evidence_count == 1


def test_coverage_guidance_low_after_strong_multi_evidence(service):
    state = InterviewState(job=JobDescription(competencies=["系统设计"]))
    for _ in range(2):
        state.evidence.append(
            Evidence(
                competency="系统设计",
                signal="设计方案",
                confidence=0.9,
                source=EvidenceSource.LIVE_INTERVIEW,
            )
        )
    service._refresh_live_coverage_guidance(state)
    assert state.live_interview.coverage_guidance[0].priority == "low"


def test_coverage_guidance_sorts_by_priority(service):
    state = InterviewState(
        job=JobDescription(competencies=["高优先级能力", "已覆盖能力"])
    )
    state.evidence.append(
        Evidence(
            competency="已覆盖能力",
            signal="充分证据",
            confidence=0.9,
            source=EvidenceSource.LIVE_INTERVIEW,
        )
    )
    state.evidence.append(
        Evidence(
            competency="已覆盖能力",
            signal="充分证据",
            confidence=0.9,
            source=EvidenceSource.LIVE_INTERVIEW,
        )
    )
    service._refresh_live_coverage_guidance(state)
    priorities = [item.competency for item in state.live_interview.coverage_guidance]
    assert priorities == ["高优先级能力", "已覆盖能力"]


# ---------------------------------------------------------------------------
# _refresh_live_answer_boundaries
# ---------------------------------------------------------------------------


def test_answer_boundaries_merge_consecutive_candidate_segments(service):
    state = InterviewState()
    state.live_interview.segments = [
        _interviewer_segment(1, "请讲一次系统设计经历。"),
        _segment(2, "第一段回答内容。"),
        _segment(3, "第二段补充回答内容。"),
    ]
    service._refresh_live_answer_boundaries(state)
    suggestions = state.live_interview.answer_boundary_suggestions
    assert len(suggestions) == 1
    assert suggestions[0].question_segment_id is not None
    assert len(suggestions[0].answer_segment_ids) == 2


def test_answer_boundaries_skip_single_candidate_segment(service):
    state = InterviewState()
    state.live_interview.segments = [
        _interviewer_segment(1, "请讲一次系统设计经历。"),
        _segment(2, "单段回答。"),
    ]
    service._refresh_live_answer_boundaries(state)
    assert state.live_interview.answer_boundary_suggestions == []


def test_answer_boundaries_exclude_archived_segments(service):
    state = InterviewState()
    first = _segment(2, "第一段回答内容。")
    second = _segment(3, "第二段补充回答内容。")
    state.live_interview.segments = [
        _interviewer_segment(1, "请讲一次系统设计经历。"),
        first,
        second,
    ]
    state.live_interview_records = [
        LiveInterviewRecord(
            question="问题",
            answer="回答",
            competency="系统设计",
            evaluation=AnswerEvaluation(
                content=0.8, technical_depth=0.8, structure=0.8, impact=0.8
            ),
            transcript_segment_ids=[first.id, second.id],
        )
    ]
    service._refresh_live_answer_boundaries(state)
    assert state.live_interview.answer_boundary_suggestions == []


def test_answer_boundaries_ignore_unconfirmed_segments(service):
    state = InterviewState()
    state.live_interview.segments = [
        _interviewer_segment(1, "请讲一次系统设计经历。"),
        _segment(2, "未确认的第一段。", confirmed=False),
        _segment(3, "未确认的第二段。", stable=False),
    ]
    service._refresh_live_answer_boundaries(state)
    assert state.live_interview.answer_boundary_suggestions == []


# ---------------------------------------------------------------------------
# _refresh_live_action_card (priority chain)
# ---------------------------------------------------------------------------


def test_action_card_idle_shows_start(service):
    state = InterviewState()
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "start"


def test_action_card_paused_shows_resume(service):
    state = InterviewState()
    state.live_interview.status = LiveInterviewStatus.PAUSED
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "resume"


def test_action_card_unknown_speaker_takes_priority(service):
    state = InterviewState()
    state.live_interview.status = LiveInterviewStatus.ACTIVE
    state.live_interview.segments.append(
        _segment(1, "无法识别说话人的内容", speaker=TranscriptSpeaker.UNKNOWN)
    )
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "review_speaker"


def test_action_card_merge_boundary_when_confident(service):
    state = InterviewState()
    state.live_interview.status = LiveInterviewStatus.ACTIVE
    state.live_interview.segments = [
        _interviewer_segment(1, "请讲一次系统设计经历。"),
        _segment(2, "我先梳理瓶颈并设计分层缓存方案，用压测指标验证了缓存命中率和读写延时。"),
        _segment(3, "最终灰度上线到全量，P95 延迟降低了 40%，以上就是这次经历。"),
    ]
    service._refresh_live_answer_boundaries(state)
    assert state.live_interview.answer_boundary_suggestions
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "merge_boundary"


def test_action_card_confirm_evidence_when_candidate_answer_pending(service):
    state = InterviewState()
    state.live_interview.status = LiveInterviewStatus.ACTIVE
    state.live_interview.segments = [
        _interviewer_segment(1, "请讲一次系统设计经历。"),
        _segment(2, "一段完整回答，涉及权衡与结果。"),
    ]
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "confirm_evidence"


def test_action_card_decide_question_when_suggestion_pending(service):
    state = InterviewState()
    state.live_interview.status = LiveInterviewStatus.ACTIVE
    state.evidence.append(
        Evidence(
            competency="系统设计",
            signal="完整回答",
            confidence=0.9,
            source=EvidenceSource.LIVE_INTERVIEW,
        )
    )
    state.evidence.append(
        Evidence(
            competency="系统设计",
            signal="完整回答",
            confidence=0.9,
            source=EvidenceSource.LIVE_INTERVIEW,
        )
    )
    state.evidence.append(
        Evidence(
            competency="团队协作",
            signal="完整回答",
            confidence=0.9,
            source=EvidenceSource.LIVE_INTERVIEW,
        )
    )
    state.live_interview.suggestions.append(
        QuestionSuggestion(
            suggested_question="追问内容",
            competency="系统设计",
            status=QuestionSuggestionStatus.PENDING,
        )
    )
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "decide_question"


def test_action_card_plan_gap_when_evidence_insufficient(service):
    state = InterviewState()
    state.live_interview.status = LiveInterviewStatus.ACTIVE
    state.job.competencies = ["系统设计"]
    state.evidence.append(
        Evidence(
            competency="系统设计",
            signal="单条回答",
            confidence=0.8,
            source=EvidenceSource.LIVE_INTERVIEW,
        )
    )
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "plan_gap_question"


def test_action_card_completed_insufficient_shows_review(service):
    state = InterviewState()
    state.live_interview.status = LiveInterviewStatus.COMPLETED
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "review_evidence"


def test_action_card_completed_sufficient_shows_evaluate(service):
    state = InterviewState()
    state.live_interview.status = LiveInterviewStatus.COMPLETED
    for competency in ("系统设计", "团队协作"):
        for _ in range(2):
            state.evidence.append(
                Evidence(
                    competency=competency,
                    signal="完整回答",
                    confidence=0.9,
                    source=EvidenceSource.LIVE_INTERVIEW,
                )
            )
    service._refresh_live_action_card(state)
    assert state.live_interview.action_card.action_type == "evaluate"


# ---------------------------------------------------------------------------
# Async live scoring: placeholder evidence at confirm, backfill in background
# ---------------------------------------------------------------------------


async def test_confirm_creates_placeholder_and_scores_in_background(scoring_service):
    session_id, state = await scoring_service.create_session()
    await scoring_service.start_live_interview(session_id, consent_confirmed=True)
    state = await scoring_service.append_live_transcript(
        session_id, text="请讲一次架构决策。", speaker=TranscriptSpeaker.INTERVIEWER
    )
    state = await scoring_service.append_live_transcript(
        session_id, text="我对比了缓存与数据库扩容方案。", speaker=TranscriptSpeaker.CANDIDATE
    )
    answer = state.live_interview.segments[-1]
    confirmed = await scoring_service.confirm_live_answer(
        session_id,
        answer.id,
        question="请讲一次架构决策。",
        competency="系统设计",
    )
    # Confirm returns with exactly one evidence row linked to the record; with a
    # fast model the background coach may already have finished, so the row may
    # be a placeholder (confidence 0) or already backfilled.
    assert len(confirmed.evidence) == 1
    assert confirmed.evidence[0].source_record_id == confirmed.live_interview_records[0].id
    assert confirmed.live_interview_records[0].scoring_status in {"scoring", "scored"}

    await scoring_service._background.flush()
    refreshed = await scoring_service.get_state(session_id)
    record = refreshed.live_interview_records[0]
    assert record.scoring_status == "scored"
    assert record.evaluation.overall_score() == 0.8
    matching = [
        item
        for item in refreshed.evidence
        if item.source_record_id == record.id
    ]
    assert len(matching) == 1
    assert matching[0].confidence == 0.8
    assert matching[0].signal == "Clear design"


async def test_confirm_scores_in_background_without_blocking(tmp_path):
    """Confirm must return before the background scoring completes."""
    import asyncio as _asyncio

    class _SlowCoach:
        async def chat(self, messages, **kwargs):
            await _asyncio.sleep(0.001)
            return (
                '{"content":0.8,"technical_depth":0.8,"structure":0.8,"impact":0.8,'
                '"feedback":[],"improved_answer":"better","observed_signals":["Clear design"],'
                '"missing_signals":[]}'
            )

        async def embed(self, text):
            return []

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-slow-scoring.db'}")
    await storage.init_db()
    background = BackgroundTaskManager()
    service = InterviewService(storage, llm_client=_SlowCoach(), background=background)
    session_id, _ = await service.create_session()
    await service.start_live_interview(session_id, consent_confirmed=True)
    state = await service.append_live_transcript(
        session_id, text="问题", speaker=TranscriptSpeaker.INTERVIEWER
    )
    state = await service.append_live_transcript(
        session_id, text="一段完整回答。", speaker=TranscriptSpeaker.CANDIDATE
    )
    await service.set_live_interview_status(session_id, LiveInterviewStatus.PAUSED)
    answer = state.live_interview.segments[-1]
    confirmed = await service.confirm_live_answer(
        session_id, answer.id, question="问题", competency="系统设计"
    )
    # The coach coroutine is still pending when confirm returns, so the record
    # must still be marked as scoring.
    assert confirmed.live_interview_records[0].scoring_status == "scoring"
    await background.flush()
    refreshed = await service.get_state(session_id)
    assert refreshed.live_interview_records[0].scoring_status == "scored"
    await background.close()
    await storage.close()


async def test_human_review_wins_race_with_background_live_scoring(tmp_path):
    import asyncio as _asyncio

    started = _asyncio.Event()
    release = _asyncio.Event()

    class _BlockedCoach:
        async def chat(self, messages, **kwargs):
            started.set()
            await release.wait()
            return (
                '{"content":0.2,"technical_depth":0.2,"structure":0.2,"impact":0.2,'
                '"feedback":[],"observed_signals":["AI score"],"missing_signals":[]}'
            )

        async def embed(self, text):
            return []

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-human-race.db'}")
    await storage.init_db()
    background = BackgroundTaskManager()
    service = InterviewService(storage, llm_client=_BlockedCoach(), background=background)
    session_id, _ = await service.create_session()
    await service.start_live_interview(session_id, consent_confirmed=True)
    await service.append_live_transcript(
        session_id, text="问题", speaker=TranscriptSpeaker.INTERVIEWER
    )
    state = await service.append_live_transcript(
        session_id, text="一段完整回答。", speaker=TranscriptSpeaker.CANDIDATE
    )
    # Isolate the scoring race: pausing disables the separate auto-planning
    # task, which uses the same deliberately blocked test client.
    await service.set_live_interview_status(session_id, LiveInterviewStatus.PAUSED)
    answer = state.live_interview.segments[-1]
    confirmed = await service.confirm_live_answer(
        session_id, answer.id, question="问题", competency="系统设计"
    )
    await started.wait()
    record_id = confirmed.live_interview_records[0].id

    await service.review_answer_evaluation(
        session_id,
        record_id,
        content=0.9,
        technical_depth=0.8,
        structure=0.7,
        impact=0.6,
        evidence_polarity=EvidencePolarity.POSITIVE,
    )
    release.set()
    await background.flush()
    refreshed = await service.get_state(session_id)
    record = refreshed.live_interview_records[0]

    assert record.evaluation.scoring_source == AnswerScoringSource.HUMAN
    assert record.evaluation.review_status == AnswerReviewStatus.REVIEWED
    assert record.evaluation.overall_score() == pytest.approx(0.75)
    assert refreshed.evidence[0].confidence == pytest.approx(0.75)
    assert refreshed.evidence[0].polarity == EvidencePolarity.POSITIVE
    await background.close()
    await storage.close()


async def test_auto_plan_skips_when_suggestion_pending(scoring_service):
    session_id, _ = await scoring_service.create_session()
    await scoring_service.start_live_interview(session_id, consent_confirmed=True)
    state = await scoring_service.append_live_transcript(
        session_id, text="问题", speaker=TranscriptSpeaker.INTERVIEWER
    )
    state = await scoring_service.append_live_transcript(
        session_id, text="一段完整回答。", speaker=TranscriptSpeaker.CANDIDATE
    )
    answer = state.live_interview.segments[-1]
    # A pending suggestion already exists: auto-plan must not stack a second one.
    state.live_interview.suggestions.append(
        QuestionSuggestion(
            suggested_question="追问内容",
            competency="系统设计",
            status=QuestionSuggestionStatus.PENDING,
        )
    )
    confirmed = await scoring_service.confirm_live_answer(
        session_id, answer.id, question="问题", competency="系统设计"
    )
    pending = [
        item
        for item in confirmed.live_interview.suggestions
        if item.status == QuestionSuggestionStatus.PENDING
    ]
    assert len(pending) == 1
    await scoring_service._background.flush()


async def test_revoke_while_scoring_drops_orphaned_placeholder(scoring_service):
    session_id, _ = await scoring_service.create_session()
    await scoring_service.start_live_interview(session_id, consent_confirmed=True)
    state = await scoring_service.append_live_transcript(
        session_id, text="问题", speaker=TranscriptSpeaker.INTERVIEWER
    )
    state = await scoring_service.append_live_transcript(
        session_id, text="一段完整回答。", speaker=TranscriptSpeaker.CANDIDATE
    )
    answer = state.live_interview.segments[-1]
    confirmed = await scoring_service.confirm_live_answer(
        session_id, answer.id, question="问题", competency="系统设计"
    )
    record_id = confirmed.live_interview_records[0].id
    await scoring_service._background.flush()
    revoked = await scoring_service.revoke_live_evidence(session_id, record_id)
    assert revoked.live_interview_records == []
    assert all(item.source_record_id != record_id for item in revoked.evidence)


# ---------------------------------------------------------------------------
# Planner context slimming: unused questions only, max_tokens, single call
# ---------------------------------------------------------------------------


class _CapturingPlannerLLM:
    """Records the last planner prompt and chat kwargs, returns valid JSON."""

    def __init__(self):
        self.last_prompt = ""
        self.last_kwargs = {}
        self.chat_calls = 0
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.chat_calls += 1
        self.calls.append((messages, kwargs))
        self.last_prompt = "\n".join(
            message.get("content", "") for message in messages
        )
        self.last_kwargs = kwargs
        return (
            '{"suggested_question":"追问一下验证方法。","question_type":"follow_up",'
            '"competency":"系统设计","rationale":"需要补充验证","evidence_gap":"量化验证",'
            '"expected_signals":["指标","压测"],"confidence":0.8,"alternatives":["回滚"]}'
        )

    async def embed(self, text):
        return []


def _planner_state() -> InterviewState:
    state = InterviewState()
    state.blueprint = InterviewBlueprint(
        rounds=[
            InterviewRound(
                name="技术面",
                goal="验证",
                questions=[
                    InterviewQuestion(
                        question="已用的系统设计题",
                        competency="系统设计",
                        rationale="",
                    ),
                    InterviewQuestion(
                        question="未用的事故复盘题",
                        competency="Incident Review",
                        rationale="",
                    ),
                ],
            )
        ]
    )
    state.live_interview.used_question_ids = [
        str(state.blueprint.rounds[0].questions[0].id)
    ]
    state.live_interview.segments = [
        _interviewer_segment(1, "请讲一次系统设计经历。"),
        _segment(2, "我对比了缓存方案并用压测验证。"),
    ]
    return state


async def test_planner_excludes_used_questions_and_passes_max_tokens(tmp_path):
    llm = _CapturingPlannerLLM()
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'planner-slim.db'}")
    await storage.init_db()
    try:
        from interview_os.agents.live_interview_agent import LiveInterviewAgent

        state = _planner_state()
        agent = LiveInterviewAgent(llm_client=llm)
        await agent.execute(state)
    finally:
        await storage.close()
    # The used question must not appear in the planner prompt.
    assert "已用的系统设计题" not in llm.last_prompt
    assert "未用的事故复盘题" in llm.last_prompt
    # max_tokens must be forwarded to the LLM client.
    assert llm.last_kwargs.get("max_tokens") == 512


async def test_think_structured_invalid_output_makes_single_call(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'planner-bad.db'}")
    await storage.init_db()
    try:
        from interview_os.agents.live_interview_agent import LiveInterviewAgent

        calls = []

        class _CountingBadLLM:
            async def chat(self, messages, **kwargs):
                calls.append(1)
                return "not json"

            async def embed(self, text):
                return []

        state = _planner_state()
        counting_agent = LiveInterviewAgent(llm_client=_CountingBadLLM())
        await counting_agent.execute(state)
    finally:
        await storage.close()
    # Invalid output triggers one lightweight retry (same prompt, no added
    # context), then the planner falls back deterministically instead of a
    # second repair call re-prefilling the whole context.
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# _live_audio_context (bounded full context for audio-direct mode)
# ---------------------------------------------------------------------------


def test_live_audio_context_includes_blueprint_transcript_evidence(service):
    state = InterviewState()
    state.job.competencies = ["系统设计", "稳定性治理"]
    state.blueprint = InterviewBlueprint(
        rounds=[
            InterviewRound(
                name="技术面",
                goal="验证",
                questions=[
                    InterviewQuestion(
                        question="请讲一次系统架构权衡。",
                        competency="系统设计",
                        rationale="",
                    ),
                    InterviewQuestion(
                        question="请讲一次事故复盘。",
                        competency="稳定性治理",
                        rationale="",
                    ),
                ],
            )
        ]
    )
    state.live_interview.segments = [
        _interviewer_segment(1, "请讲一次系统架构权衡。"),
        _segment(2, "我对比了缓存和数据库方案。"),
    ]
    state.live_interview.rolling_summary = "历史摘要内容"
    state.evidence.append(
        Evidence(
            competency="系统设计",
            signal="缓存方案选型与压测验证",
            confidence=0.8,
            source=EvidenceSource.LIVE_INTERVIEW,
        )
    )
    context = service._live_audio_context(state)
    # Focused mode keeps the anchor, recent transcript, and evidence...
    assert "岗位能力：系统设计、稳定性治理" in context
    assert "请讲一次系统架构权衡" in context
    assert "缓存方案选型与压测验证" in context
    assert "系统设计" in context
    # ...and intentionally excludes advancement-only info.
    assert "历史摘要" not in context
    assert "待问蓝图题" not in context
    assert "覆盖引导" not in context


def test_live_audio_context_keeps_anchor_when_over_limit(service):
    state = InterviewState(job=JobDescription(title="岗位", competencies=["能力"]))
    # Long transcript far exceeding the cap; the anchor must survive.
    state.live_interview.segments = [
        _segment(i, f"第 {i} 段回答：{'内容' * 200}", confirmed=True)
        for i in range(1, 13)
    ]
    context = service._live_audio_context(state)
    assert len(context) <= 2600  # OMNI_CONTEXT_CHAR_LIMIT + slack
    assert "岗位能力：能力" in context  # anchor is never dropped


# ---------------------------------------------------------------------------
# diarize: parse + merge
# ---------------------------------------------------------------------------


def test_parse_diarized_response_maps_speakers():
    from interview_os.models.omni_client import _parse_diarized_response

    raw = '''```json
    {"segments":[{"speaker":"面试官","text":"请讲一次系统设计。"},{"speaker":"候选人","text":"我对比了方案。"}]}
    ```'''
    segments = _parse_diarized_response(raw)
    assert segments == [
        {"speaker": "interviewer", "text": "请讲一次系统设计。"},
        {"speaker": "candidate", "text": "我对比了方案。"},
    ]


def test_parse_diarized_response_invalid_returns_empty():
    from interview_os.models.omni_client import _parse_diarized_response

    assert _parse_diarized_response("not json") == []
    assert _parse_diarized_response("") == []


def test_merge_diarized_segments_combines_consecutive_speakers():
    merged = InterviewService._merge_diarized_segments(
        [
            {"speaker": "interviewer", "text": "问题一"},
            {"speaker": "candidate", "text": "回答第一句"},
            {"speaker": "candidate", "text": "回答第二句"},
            {"speaker": "interviewer", "text": "追问"},
        ]
    )
    assert len(merged) == 3
    assert merged[1]["speaker"] == "candidate"
    assert "回答第一句" in merged[1]["text"]
    assert "回答第二句" in merged[1]["text"]


def test_merge_diarized_segments_skips_empty():
    merged = InterviewService._merge_diarized_segments(
        [
            {"speaker": "interviewer", "text": "   "},
            {"speaker": "candidate", "text": "有效回答"},
        ]
    )
    assert len(merged) == 1
    assert merged[0]["speaker"] == "candidate"
