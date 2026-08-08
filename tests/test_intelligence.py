import asyncio
from datetime import datetime, timezone

import httpx

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
    SearchProviderManager,
    SearchResult,
    TavilySearchProvider,
)


def test_title_only_jd_separates_inferred_requirements():
    review = review_job_description("高级 Python 工程师", ["熟悉分布式系统"])
    assert review.is_title_only is True
    assert review.missing_sections == ["岗位职责", "任职要求", "团队背景"]
    assert review.requirements[0].origin.value == "inferred"


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
    assert second.status()["estimated_cost_usd"] == 0.01


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
