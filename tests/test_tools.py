"""Tests for tool registry."""
import json

import httpx
import pytest

from interview_os.core.tool import Tool, ToolRegistry, ToolResult
from interview_os.tools.asr import ASRClient
from interview_os.tools.web_search import (
    SearchProvider,
    SearchResult,
    TavilySearchProvider,
    WebSearchTool,
    assess_source_quality,
    filter_entity_results,
    merge_search_results,
    search_provider_from_env,
)


class DummyTool(Tool):
    name = "dummy"
    description = "A test tool"

    async def execute(self, **kwargs):
        return ToolResult(success=True, data={"echo": kwargs})


@pytest.mark.asyncio
async def test_asr_client_uses_openai_multipart_contract_and_extracts_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["Authorization"] == "Bearer asr-secret"
        assert "multipart/form-data" in request.headers["Content-Type"]
        assert b'filename="answer.webm"' in request.content
        assert b'form-data; name="model"' in request.content
        return httpx.Response(200, json={"text": "这是候选人的回答"})

    client = ASRClient(
        base_url="http://asr.test:9001",
        api_key="asr-secret",
        transport=httpx.MockTransport(handler),
    )
    try:
        transcript = await client.transcribe(
            b"audio-bytes", filename="answer.webm", content_type="audio/webm"
        )
    finally:
        await client.close()
    assert transcript == "这是候选人的回答"


@pytest.mark.asyncio
async def test_asr_probe_skips_missing_health_endpoint():
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(404)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    client = ASRClient(
        base_url="http://asr.test:9001",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.probe()
    finally:
        await client.close()

    assert result == {"ok": True, "path": "/v1/models", "status_code": 200}
    assert requested_paths == ["/health", "/v1/models"]


@pytest.mark.asyncio
async def test_tool_registry_register():
    registry = ToolRegistry()
    registry.register(DummyTool())
    assert registry.get("dummy") is not None


@pytest.mark.asyncio
async def test_tool_registry_call():
    registry = ToolRegistry()
    registry.register(DummyTool())
    result = await registry.call("dummy", key="value")
    assert result.success
    assert result.data["echo"]["key"] == "value"


@pytest.mark.asyncio
async def test_tool_registry_missing():
    registry = ToolRegistry()
    result = await registry.call("nonexistent")
    assert not result.success
    assert "not found" in result.error


class FakeSearchProvider(SearchProvider):
    async def search(self, query: str, limit: int = 5, *, search_depth: str = "basic"):
        return [
            SearchResult(
                title="Example Engineering Blog",
                url="https://example.com/engineering",
                snippet=f"Public information for {query}",
                source="fake",
            )
        ][:limit]


@pytest.mark.asyncio
async def test_web_search_normalizes_provider_results():
    result = await WebSearchTool(FakeSearchProvider()).execute(query="Example CTO", num_results=3)
    assert result.success
    assert result.data["results"][0]["url"] == "https://example.com/engineering"
    assert result.data["results"][0]["source"] == "fake"


@pytest.mark.asyncio
async def test_tavily_provider_filters_low_relevance_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tvly-test"
        payload = json.loads(request.content)
        assert payload["search_depth"] == "basic"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Relevant", "url": "https://a.test", "content": "A", "score": 0.9},
                    {"title": "Noise", "url": "https://b.test", "content": "B", "score": 0.2},
                ]
            },
        )

    provider = TavilySearchProvider("tvly-test", transport=httpx.MockTransport(handler))
    results = await provider.search("Example person", limit=5)
    assert [result.title for result in results] == ["Relevant"]
    assert results[0].source == "tavily"


def test_tavily_has_provider_precedence(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080")
    assert isinstance(search_provider_from_env(), TavilySearchProvider)


def test_source_quality_marks_entity_domain_as_official():
    results = assess_source_quality(
        [
            SearchResult(title="Official", url="https://openai.com/careers"),
            SearchResult(title="Article", url="https://example.net/openai"),
        ],
        '"OpenAI" engineering culture',
    )
    assert results[0].is_official is True
    assert results[0].source_quality == "official"
    assert results[1].source_quality == "secondary"
    assert results[0].corroboration_count == 2


def test_search_result_merge_preserves_order_and_deduplicates_urls():
    merged = merge_search_results(
        [{"url": "https://a.example/", "title": "A"}],
        [
            {"url": "https://a.example", "title": "duplicate"},
            {"url": "https://b.example", "title": "B"},
        ],
    )
    assert [item["title"] for item in merged] == ["A", "B"]


def test_entity_filter_rejects_fuzzy_chinese_company_matches():
    results = filter_entity_results(
        [
            {"title": "芯视界创始人鲍捷", "url": "https://wrong.example"},
            {"title": "芯世界 CEO 鲍捷采访", "url": "https://right.example"},
        ],
        entity="鲍捷",
        required_context="芯世界",
    )
    assert [item["url"] for item in results] == ["https://right.example"]


def test_entity_filter_accepts_corroborated_one_character_company_alias():
    results = filter_entity_results(
        [
            {
                "title": "芯视界创始人鲍捷",
                "url": "https://qtetech.example/profile",
                "snippet": "鲍捷是芯视界创始人兼首席科学家",
            }
        ],
        entity="芯世界",
        corroborating_entity="鲍捷",
    )
    assert results[0]["identity_match"] == "corroborated_alias"
    assert results[0]["matched_identity"] == "芯视界"


def test_context_alias_requires_exact_person_identity():
    results = filter_entity_results(
        [
            {
                "title": "芯视界创始人鲍捷",
                "url": "https://qtetech.example/profile",
            }
        ],
        entity="鲍捷",
        required_context="芯世界",
        allow_context_alias=True,
    )
    assert results[0]["matched_identity"] == "芯视界"
