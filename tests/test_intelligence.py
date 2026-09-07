import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from interview_os.core.debug import DebugEvent, DebugEventStore
from interview_os.core.state import (
    InterviewState,
    ResumeClaim,
    ResumeClaimStatus,
    ResumeStructuredSection,
)
from interview_os.models.local_llm import LocalLLMClient
from interview_os.services.intelligence_service import (
    build_fact_cards,
    decide_fact_card,
    resolve_entity,
    review_job_description,
    sync_entity_resolutions,
)
from interview_os.tools.web_search import (
    MAX_PERSISTED_SEARCH_METRIC,
    SearchProviderManager,
    SearchResult,
    TavilySearchProvider,
)


def test_title_only_jd_separates_inferred_requirements():
    review = review_job_description("高级 Python 工程师", ["熟悉分布式系统"])
    assert review.is_title_only is True
    assert review.missing_sections == ["岗位职责", "任职要求", "团队背景"]
    assert review.requirements[0].origin.value == "inferred"


def test_full_jd_preserves_meaningful_leading_experience_number():
    review = review_job_description(
        "岗位职责：\n1. 负责产品路线图\n任职要求：\n- 7年以上企业软件产品经验",
        [],
    )

    texts = [item.text for item in review.requirements]
    assert "负责产品路线图" in texts
    assert "7年以上企业软件产品经验" in texts


def test_inline_full_jd_is_split_into_concise_explicit_requirements():
    review = review_job_description(
        (
            "高级 AI 产品经理。岗位职责：制定企业级生成式 AI 产品战略与年度路线图；"
            "与算法、工程和销售团队协作交付。任职要求：7 年以上 B2B 产品经验；"
            "有 RAG 或 Agent 评测实践。团队背景：向产品副总裁汇报，协作 25 人平台团队。"
        ),
        [],
    )

    texts = [item.text for item in review.requirements]
    assert "高级 AI 产品经理" not in texts
    assert "制定企业级生成式 AI 产品战略与年度路线图" in texts
    assert "与算法、工程和销售团队协作交付" in texts
    assert "7 年以上 B2B 产品经验" in texts
    assert "有 RAG 或 Agent 评测实践" in texts
    assert "向产品副总裁汇报，协作 25 人平台团队" in texts
    assert all("岗位职责" not in item and "任职要求" not in item for item in texts)


def test_entity_resolution_requires_acceptance_before_renaming():
    state = InterviewState()
    state.company.name = "芯世界"
    state.company.public_sources = [
        {
            "identity_match": "corroborated_alias",
            "input_identity": "芯世界",
            "matched_identity": "芯视界",
            "url": "https://example.com/company",
            "snippet": "芯视界公开资料",
        }
    ]
    sync_entity_resolutions(state)
    resolution = state.entity_resolutions[0]
    assert state.company.name == "芯世界"
    resolve_entity(state, resolution.id, accept=True)
    assert state.company.name == "芯视界"


def test_entity_resolution_accepts_user_corrected_name():
    state = InterviewState()
    state.company.name = "芯世界"
    state.company.public_sources = [
        {
            "identity_match": "corroborated_alias",
            "input_identity": "芯世界",
            "matched_identity": "芯视界",
            "url": "https://example.com/company",
            "snippet": "芯视界公开资料",
        }
    ]
    sync_entity_resolutions(state)
    resolution = state.entity_resolutions[0]

    resolve_entity(state, resolution.id, accept=True, proposed_name="芯视界科技")

    assert state.company.name == "芯视界科技"
    assert resolution.proposed_name == "芯视界科技"
    assert resolution.status.value == "accepted"


def test_fact_cards_preserve_source_and_status():
    state = InterviewState()
    state.company.name = "Example"
    state.company.public_sources = [
        {
            "url": "https://example.com/about",
            "snippet": "Example develops AI systems.",
            "source_quality": "official",
            "is_official": True,
        }
    ]
    build_fact_cards(state)
    assert state.fact_cards[0].status.value == "verified"
    assert state.fact_cards[0].source_urls == ["https://example.com/about"]
    assert state.fact_cards[0].source_quality == "official"


def test_fact_cards_ignore_image_captions_even_from_official_pages():
    state = InterviewState()
    state.company.name = "Zuora"
    state.company.public_sources = [
        {
            "url": "https://www.zuora.com/careers",
            "title": "Careers at Zuora",
            "snippet": (
                "A woman with long brown hair wearing a black top is smiling "
                "with her arms crossed against a plain light background."
            ),
            "source_quality": "official",
            "is_official": True,
        }
    ]

    build_fact_cards(state)

    assert state.fact_cards == []


def test_fact_cards_include_past_employer_sources():
    state = InterviewState()
    state.past_employer_sources = [
        {
            "url": "https://zuora.com/about",
            "snippet": "Zuora is a subscription management platform.",
            "source_quality": "official",
            "is_official": True,
        }
    ]
    build_fact_cards(state)
    assert any(card.category == "past_employer" for card in state.fact_cards)
    past = next(card for card in state.fact_cards if card.category == "past_employer")
    assert past.status.value == "verified"
    assert "subscription management" in past.claim


def test_past_employer_cards_stay_grouped_despite_technology_keywords():
    state = InterviewState()
    state.past_employer_sources = [
        {
            "url": "https://hpe.com/about",
            "snippet": "HPE is an enterprise technology company with engineering depth.",
            "source_quality": "official",
            "is_official": True,
        }
    ]
    build_fact_cards(state)
    # Even though the snippet contains "technology"/"engineering", the card must
    # remain in the past_employer group, not drift to the technology group.
    assert any(card.category == "past_employer" for card in state.fact_cards)
    assert not any(card.category == "technology" for card in state.fact_cards)
    assert state.fact_cards[0].source_count == 1
    assert state.fact_cards[0].generated_at is not None


def test_fact_cards_merge_duplicate_sources_with_quality_metadata():
    state = InterviewState()
    state.company.name = "Example"
    claim = "Example announced an AI platform direction for enterprise hiring."
    state.company.public_sources = [
        {
            "url": "https://media.example.com/story",
            "snippet": claim,
            "source_quality": "secondary",
            "fetched_at": "2026-08-02T01:00:00+00:00",
            "filter_reason": "accepted: exact entity/context match",
        },
        {
            "url": "https://example.com/news",
            "snippet": claim,
            "source_quality": "official",
            "is_official": True,
            "fetched_at": "2026-08-02T02:00:00+00:00",
            "filter_reason": "accepted: exact entity/context match",
            "cache_hit": True,
        },
    ]
    build_fact_cards(state)
    card = state.fact_cards[0]
    assert card.status.value == "verified"
    assert card.source_quality == "official"
    assert card.source_count == 2
    assert card.source_fetched_at == datetime(2026, 8, 2, 2, tzinfo=timezone.utc)
    assert card.source_filter_reason == "accepted: exact entity/context match"
    assert card.cache_hit is True
    assert card.source_urls == ["https://media.example.com/story", "https://example.com/news"]
    assert "2 个公开来源交叉支持" in card.note


def test_fact_card_decision_flow_marks_accept_reject_and_reset():
    state = InterviewState()
    state.company.name = "Example"
    state.company.public_sources = [
        {
            "url": "https://example.com/news",
            "snippet": "Example announced an AI platform direction in a public article.",
            "source_quality": "secondary",
        }
    ]
    build_fact_cards(state)
    card = state.fact_cards[0]

    decide_fact_card(state, card.id, action="accept", note="官网外来源但与面试官说法一致")
    assert card.status.value == "accepted"
    assert card.confidence >= 0.85
    assert card.resolved_at is not None

    decide_fact_card(state, card.id, action="reject", note="来源过旧")
    assert card.status.value == "rejected"
    assert card.confidence <= 0.2
    assert card.note == "来源过旧"

    decide_fact_card(state, card.id, action="reset")
    assert card.status.value == "inferred"
    assert card.resolved_at is None


def test_only_confirmed_or_modified_resume_claims_reach_agents():
    state = InterviewState()
    state.candidate.raw_resume_text = "2018-07 to ZUORA 2022-12\nSenior Recruiting Manager\nAPAC talent acquisition"
    state.resume_review.claims = [
        ResumeClaim(category="achievement", statement="待核验 100% 增长"),
        ResumeClaim(
            category="experience",
            statement="确认后的 30% 增长",
            status=ResumeClaimStatus.MODIFIED,
        ),
    ]
    context = state.candidate_evidence_context()
    # Confirmed claim is present and labelled with priority.
    assert "确认后的 30% 增长" in context
    # Unconfirmed claim is not surfaced as a fact.
    assert "100% 增长" not in context
    # Once review exists, unconfirmed raw resume content is withheld.
    assert "ZUORA" not in context


def test_evidence_context_withholds_raw_resume_when_claims_unconfirmed():
    state = InterviewState()
    state.candidate.raw_resume_text = (
        "2018-07 to ZUORA 2022-12\n"
        "Senior Recruiting Manager\n"
        "2011-03 to Hewlett Packard Enterprise 2016-05\n"
        "Senior Recruiting Consultant"
    )
    state.resume_review.claims = [
        ResumeClaim(category="employment", statement="2018-07 to ZUORA"),
    ]
    context = state.candidate_evidence_context()
    assert "尚无已确认的简历事实" in context
    assert "ZUORA" not in context
    assert "Hewlett Packard Enterprise" not in context


def test_raw_resume_is_labelled_unverified_before_review_exists():
    state = InterviewState()
    state.candidate.raw_resume_text = "2020-01 to HPE 2023-06\nSenior Engineer"
    context = state.candidate_evidence_context()
    assert "未核验" in context
    assert "不得作为事实或评价证据" in context
    assert "HPE" in context


def test_evidence_context_includes_structured_employment_when_requested():
    state = InterviewState()
    state.candidate.raw_resume_text = "2020-01 to HPE 2023-06\nSenior Engineer"
    state.resume_review.structured = [
        ResumeStructuredSection(
            category="employment",
            institution="HPE",
            title="Senior Engineer",
            date_range="2020-2023",
            description="led platform team",
        )
    ]
    context = state.candidate_evidence_context(structure_required=True)
    assert "HPE" in context
    assert "led platform team" in context
    # Without structure_required, structured sections are omitted.
    assert "led platform team" not in state.candidate_evidence_context()


def test_debug_store_redacts_secrets_and_resume_contacts():
    store = DebugEventStore()
    store.record(
        DebugEvent(
            category="test",
            action="redaction",
            detail="api_key=secret-value ada@example.com 13800138000",
            metadata={"tavily_api_key": "tvly-secret-value"},
        )
    )
    event = store.list_events()[0]
    assert "secret-value" not in event.detail
    assert "ada@example.com" not in event.detail
    assert "13800138000" not in event.detail
    assert event.metadata["tavily_api_key"] == "[REDACTED_SECRET]"


def test_search_manager_uses_bounded_cache(monkeypatch):
    calls = 0

    async def fake_search(self, query, limit=5, *, search_depth="basic"):
        nonlocal calls
        calls += 1
        return [SearchResult(title="Result", url="https://example.com", snippet=query)]

    monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
    manager = SearchProviderManager(cache_capacity=2)
    manager.configure(provider="tavily", tavily_api_key="secret")
    asyncio.run(manager.search("Example"))
    cached = asyncio.run(manager.search("Example"))
    assert calls == 1
    assert cached[0].cache_hit is True
    assert cached[0].fetched_at is not None
    assert manager.status()["cache"]["hits"] == 1
    assert "source_filter_policy" in manager.status()


def test_concurrent_identical_searches_share_one_provider_request(monkeypatch):
    async def scenario():
        calls = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_search(self, query, limit=5, *, search_depth="basic"):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return [SearchResult(title="Result", url="https://example.com", snippet=query)]

        monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
        debug_events = DebugEventStore()
        manager = SearchProviderManager(debug_events=debug_events)
        manager.configure(provider="tavily", tavily_api_key="secret")
        leader = asyncio.create_task(manager.search("same query"))
        await started.wait()
        follower = asyncio.create_task(manager.search("same query"))
        await asyncio.sleep(0)
        release.set()
        first, second = await asyncio.gather(leader, follower)
        return calls, first, second, manager, debug_events

    calls, first, second, manager, debug_events = asyncio.run(scenario())
    assert calls == 1
    assert first[0].title == second[0].title == "Result"
    assert manager.status()["provider_requests"] == 1
    assert manager.status()["cache"]["misses"] == 1
    assert sorted(event.action for event in debug_events.list_events()) == [
        "inflight_join",
        "provider_request",
    ]


@pytest.mark.parametrize("cancel_leader", [True, False])
def test_search_caller_cancellation_does_not_cancel_shared_provider_work(
    monkeypatch, cancel_leader
):
    async def scenario():
        calls = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_search(self, query, limit=5, *, search_depth="basic"):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return [SearchResult(title="Result", url="https://example.com", snippet=query)]

        monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
        manager = SearchProviderManager()
        manager.configure(provider="tavily", tavily_api_key="secret")
        leader = asyncio.create_task(manager.search("shared cancellation query"))
        await started.wait()
        follower = asyncio.create_task(manager.search("shared cancellation query"))
        # Let the follower observe and join the manager-owned provider task.
        await asyncio.sleep(0)

        cancelled = leader if cancel_leader else follower
        survivor = follower if cancel_leader else leader
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        release.set()
        results = await survivor
        # Let the task completion callback retire the single-flight entry.
        await asyncio.sleep(0)
        return calls, results, manager

    calls, results, manager = asyncio.run(scenario())

    assert calls == 1
    assert results[0].title == "Result"
    assert manager.status()["provider_requests"] == 1
    assert manager.status()["cache"]["misses"] == 1
    assert manager.status()["cache"]["size"] == 1
    assert manager._in_flight == {}


def test_abandoned_failed_search_consumes_manager_task_exception(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        loop_errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

        async def fake_search(self, query, limit=5, *, search_depth="basic"):
            started.set()
            await release.wait()
            raise RuntimeError("provider failed after caller left")

        monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
        manager = SearchProviderManager()
        manager.configure(provider="tavily", tavily_api_key="secret")
        caller = asyncio.create_task(manager.search("abandoned query"))
        await started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

        release.set()
        while manager._in_flight:
            await asyncio.sleep(0)
        # Give the loop an additional turn in which it would report an
        # unretrieved task exception if the manager callback had not consumed it.
        await asyncio.sleep(0)
        return loop_errors, manager

    loop_errors, manager = asyncio.run(scenario())

    assert loop_errors == []
    assert manager._in_flight == {}


def test_search_cache_and_metrics_survive_restart(monkeypatch, tmp_path):
    calls = 0

    async def fake_search(self, query, limit=5, *, search_depth="basic"):
        nonlocal calls
        calls += 1
        return [SearchResult(title="Result", url="https://example.com", snippet=query)]

    monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
    cache_path = tmp_path / "search-cache.json"
    metrics_path = tmp_path / "search-metrics.json"
    first = SearchProviderManager(cache_path=cache_path, metrics_path=metrics_path)
    first.configure(
        provider="tavily", tavily_api_key="secret", search_request_cost_usd=0.01
    )
    asyncio.run(first.search("Persistent Example"))

    second = SearchProviderManager(cache_path=cache_path, metrics_path=metrics_path)
    second.configure(
        provider="tavily", tavily_api_key="secret", search_request_cost_usd=0.01
    )
    results = asyncio.run(second.search("Persistent Example"))

    assert results[0].title == "Result"
    assert calls == 1
    assert second.status()["cache"]["hits"] == 1
    assert second.status()["provider_requests"] == 1
    assert second.status()["provider_successes"] == 1
    assert second.status()["provider_failures"] == 0
    assert second.status()["estimated_cost_usd"] == 0.01
    assert "secret" not in cache_path.read_text(encoding="utf-8")


def test_search_cost_uses_the_price_captured_when_provider_work_starts(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_search(self, query, limit=5, *, search_depth="basic"):
            if query == "priced before reconfigure":
                started.set()
                await release.wait()
            return [SearchResult(title="Result", url=f"https://example.com/{query}")]

        monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
        manager = SearchProviderManager()
        manager.configure(
            provider="tavily",
            tavily_api_key="secret",
            search_request_cost_usd=0.01,
        )
        first = asyncio.create_task(manager.search("priced before reconfigure"))
        await started.wait()
        manager.configure(provider="tavily", search_request_cost_usd=2.0)
        # Changing the displayed/current unit price cannot revalue paid work
        # already started under the earlier configuration.
        assert manager.status()["estimated_cost_usd"] == 0.01
        assert manager.status()["provider_unresolved"] == 1
        release.set()
        await first
        assert manager.status()["estimated_cost_usd"] == 0.01
        assert manager.status()["provider_unresolved"] == 0
        await manager.search("priced after reconfigure")
        return manager.status()

    status = asyncio.run(scenario())

    assert status["provider_requests"] == 2
    assert status["provider_successes"] == 2
    assert status["provider_failures"] == 0
    assert status["estimated_cost_usd"] == 2.01


def test_failed_provider_attempt_records_outcome_cost_and_survives_restart(
    monkeypatch, tmp_path
):
    async def fake_search(self, query, limit=5, *, search_depth="basic"):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
    metrics_path = tmp_path / "search-metrics.json"
    first = SearchProviderManager(metrics_path=metrics_path)
    first.configure(
        provider="tavily",
        tavily_api_key="secret",
        search_request_cost_usd=0.25,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(first.search("failed paid request"))

    first_status = first.status()
    assert first_status["provider_requests"] == 1
    assert first_status["provider_successes"] == 0
    assert first_status["provider_failures"] == 1
    assert first_status["provider_unresolved"] == 0
    assert first_status["estimated_cost_usd"] == 0.25

    restored = SearchProviderManager(metrics_path=metrics_path)
    restored.configure(
        provider="tavily",
        tavily_api_key="secret",
        # A new price must not revalue the persisted failed attempt.
        search_request_cost_usd=9.0,
    )
    restored_status = restored.status()
    assert restored_status["provider_requests"] == 1
    assert restored_status["provider_successes"] == 0
    assert restored_status["provider_failures"] == 1
    assert restored_status["estimated_cost_usd"] == 0.25


def test_legacy_search_metrics_keep_outcomes_without_inventing_historical_cost(tmp_path):
    metrics_path = tmp_path / "search-metrics.json"
    metrics_path.write_text(
        '{"cache_hits":2,"cache_misses":3,"provider_requests":3}',
        encoding="utf-8",
    )

    manager = SearchProviderManager(metrics_path=metrics_path)
    manager.configure(
        provider="tavily",
        tavily_api_key="secret",
        search_request_cost_usd=7.0,
    )
    status = manager.status()

    # Old provider_requests counted only completed successful requests. Their
    # historical unit prices were never stored, so expose them as unpriced rather
    # than multiplying them by today's unrelated price.
    assert status["provider_requests"] == 3
    assert status["provider_successes"] == 3
    assert status["provider_failures"] == 0
    assert status["unpriced_provider_requests"] == 3
    assert status["estimated_cost_usd"] == 0.0


def test_search_results_survive_best_effort_persistence_failures(monkeypatch, tmp_path):
    calls = 0

    async def fake_search(self, query, limit=5, *, search_depth="basic"):
        nonlocal calls
        calls += 1
        return [SearchResult(title="Result", url="https://example.com", snippet=query)]

    monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
    manager = SearchProviderManager(
        cache_path=tmp_path / "search-cache.json",
        metrics_path=tmp_path / "search-metrics.json",
    )
    manager.configure(provider="tavily", tavily_api_key="secret")
    assert manager._cache_store is not None
    assert manager._metrics_store is not None

    def fail_cache_save(payload):
        raise ValueError("cache full")

    def fail_metrics_save(payload):
        raise OSError("disk unavailable")

    monkeypatch.setattr(manager._cache_store, "save", fail_cache_save)
    monkeypatch.setattr(manager._metrics_store, "save", fail_metrics_save)

    first = asyncio.run(manager.search("Persistence is optional"))
    cached = asyncio.run(manager.search("Persistence is optional"))

    assert first[0].title == cached[0].title == "Result"
    assert cached[0].cache_hit is True
    assert calls == 1


def test_search_cache_is_scoped_to_the_provider_credential(monkeypatch):
    calls = []

    async def fake_search(self, query, limit=5, *, search_depth="basic"):
        calls.append(self.api_key)
        return [SearchResult(title="Result", url="https://example.com", snippet=query)]

    monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
    manager = SearchProviderManager()
    manager.configure(provider="tavily", tavily_api_key="account-one")
    asyncio.run(manager.search("same query"))
    manager.configure(provider="tavily", tavily_api_key="account-two")
    asyncio.run(manager.search("same query"))

    assert calls == ["account-one", "account-two"]


def test_rejected_search_credential_cannot_partially_poison_runtime_configuration():
    manager = SearchProviderManager()
    manager.configure(provider="tavily", tavily_api_key="working-key")
    before = manager.secret_snapshot()
    generation = manager._configuration_generation

    with pytest.raises(ValueError, match="visible ASCII"):
        manager.configure(provider="tavily", tavily_api_key="\ud800")

    assert manager.secret_snapshot() == before
    assert manager._configuration_generation == generation
    manager.configure(provider="tavily", tavily_api_key="replacement-key")
    assert manager.secret_snapshot()["tavily_api_key"] == "replacement-key"


def test_invalid_environment_search_secret_fails_closed_and_remains_reconfigurable(
    monkeypatch,
):
    monkeypatch.setenv("TAVILY_API_KEY", "poison\nheader")
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    manager = SearchProviderManager()

    assert manager.selected == "none"
    manager.configure(provider="tavily", tavily_api_key="replacement-key")
    assert manager.selected == "tavily"


def test_search_request_from_retired_generation_cannot_repopulate_cache(monkeypatch):
    async def scenario():
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_search(self, query, limit=5, *, search_depth="basic"):
            calls.append(self.api_key)
            if len(calls) == 1:
                started.set()
                await release.wait()
            return [
                SearchResult(
                    title="Result", url="https://example.com", snippet=self.api_key
                )
            ]

        monkeypatch.setattr(TavilySearchProvider, "search", fake_search)
        manager = SearchProviderManager()
        manager.configure(provider="tavily", tavily_api_key="account-one")
        retired_request = asyncio.create_task(manager.search("same query"))
        await started.wait()
        manager.configure(provider="tavily", tavily_api_key="account-two")
        release.set()
        await retired_request

        # Returning to the old configuration must issue a fresh request: the
        # response that finished after its generation retired was never cached.
        manager.configure(provider="tavily", tavily_api_key="account-one")
        refreshed = await manager.search("same query")
        return calls, refreshed

    calls, refreshed = asyncio.run(scenario())
    assert calls == ["account-one", "account-one"]
    assert refreshed[0].cache_hit is False


def test_malformed_search_metrics_are_safely_normalized_and_bounded(tmp_path):
    metrics_path = tmp_path / "search-metrics.json"
    metrics_path.write_text(
        '{"cache_hits":{"bad":true},"cache_misses":-8,'
        '"provider_requests":999999999999999999999999999999}',
        encoding="utf-8",
    )

    manager = SearchProviderManager(metrics_path=metrics_path)
    status = manager.status()

    assert status["cache"]["hits"] == 0
    assert status["cache"]["misses"] == 0
    assert status["provider_requests"] == MAX_PERSISTED_SEARCH_METRIC


def test_debug_events_survive_restart_after_redaction(tmp_path):
    path = tmp_path / "events.json"
    first = DebugEventStore(path=path)
    first.record(
        DebugEvent(category="search", action="query", detail="api_key=secret ada@example.com")
    )

    restored = DebugEventStore(path=path)
    event = restored.list_events()[0]
    assert "secret" not in event.detail
    assert "ada@example.com" not in event.detail
    assert path.stat().st_mode & 0o777 == 0o600


def test_llm_metrics_and_estimated_cost_survive_restart(tmp_path):
    async def run_request():
        metrics_path = tmp_path / "llm-metrics.json"
        client = LocalLLMClient(
            base_url="http://local.test/v1",
            model="test",
            metrics_path=metrics_path,
            input_cost_per_million=2,
            output_cost_per_million=4,
        )
        await client._client.aclose()

        async def handler(request):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
                },
            )

        client._client = httpx.AsyncClient(
            base_url=client.base_url, transport=httpx.MockTransport(handler)
        )
        assert await client.chat([{"role": "user", "content": "hello"}]) == "ok"
        await client.close()
        restored = LocalLLMClient(
            base_url="http://local.test/v1",
            model="test",
            metrics_path=metrics_path,
            input_cost_per_million=2,
            output_cost_per_million=4,
        )
        status = restored.settings_status()
        await restored.close()
        return status

    status = asyncio.run(run_request())
    assert status["metrics"]["requests"] == 1
    assert status["metrics"]["estimated_cost_usd"] == 0.004


def test_llm_cost_uses_request_time_price_and_survives_reconfigure_and_restart(tmp_path):
    async def run_requests():
        gate = asyncio.Event()
        started = asyncio.Event()
        metrics_path = tmp_path / "llm-request-cost.json"

        async def handler(request):
            started.set()
            await gate.wait()
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1_000, "completion_tokens": 500},
                },
            )

        client = LocalLLMClient(
            base_url="http://local.test/v1",
            model="test",
            metrics_path=metrics_path,
            input_cost_per_million=2,
            output_cost_per_million=4,
            transport=httpx.MockTransport(handler),
        )
        first = asyncio.create_task(client.chat([{"role": "user", "content": "hello"}]))
        await started.wait()
        client.reconfigure_now(input_cost_per_million=200, output_cost_per_million=400)
        gate.set()
        assert await first == "ok"
        assert client.settings_status()["metrics"]["estimated_cost_usd"] == 0.004
        await client.close()

        restored = LocalLLMClient(
            base_url="http://local.test/v1",
            model="test",
            metrics_path=metrics_path,
            input_cost_per_million=999,
            output_cost_per_million=999,
        )
        status = restored.settings_status()["metrics"]
        await restored.close()
        return status

    status = asyncio.run(run_requests())
    assert status["estimated_cost_usd"] == 0.004
    assert status["unpriced_prompt_tokens"] == 0
    assert status["unpriced_completion_tokens"] == 0


def test_legacy_llm_tokens_are_reported_unpriced_instead_of_revalued(tmp_path):
    metrics_path = tmp_path / "legacy-llm-metrics.json"
    metrics_path.write_text(
        '{"requests":2,"failures":0,"prompt_tokens":1000,'
        '"completion_tokens":500,"total_latency_ms":5}',
        encoding="utf-8",
    )

    client = LocalLLMClient(
        base_url="http://local.test/v1",
        metrics_path=metrics_path,
        input_cost_per_million=500,
        output_cost_per_million=500,
    )
    metrics = client.settings_status()["metrics"]
    asyncio.run(client.close())

    assert metrics["estimated_cost_usd"] == 0.0
    assert metrics["unpriced_prompt_tokens"] == 1000
    assert metrics["unpriced_completion_tokens"] == 500
