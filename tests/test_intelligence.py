import asyncio

from interview_os.core.debug import DebugEvent, DebugEventStore
from interview_os.core.state import InterviewState, ResumeClaim, ResumeClaimStatus
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
