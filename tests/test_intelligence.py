import asyncio

import httpx

from interview_os.core.debug import DebugEvent, DebugEventStore
from interview_os.core.state import InterviewState, ResumeClaim, ResumeClaimStatus
from interview_os.models.local_llm import LocalLLMClient
from interview_os.services.intelligence_service import (
    build_fact_cards,
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


def test_only_confirmed_or_modified_resume_claims_reach_agents():
    state = InterviewState()
    state.resume_review.claims = [
        ResumeClaim(category="achievement", statement="待核验 100% 增长"),
        ResumeClaim(
            category="experience",
            statement="确认后的 30% 增长",
            status=ResumeClaimStatus.MODIFIED,
        ),
    ]
    context = state.candidate_evidence_context()
    assert "30%" in context
    assert "100%" not in context


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
    asyncio.run(manager.search("Example"))
    assert calls == 1
    assert manager.status()["cache"]["hits"] == 1


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
