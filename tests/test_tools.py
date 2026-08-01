"""Tests for tool registry."""
import json

import httpx
import pytest

from interview_os.core.tool import Tool, ToolRegistry, ToolResult
from interview_os.tools.web_search import (
    SearchProvider,
    SearchResult,
    TavilySearchProvider,
    WebSearchTool,
    search_provider_from_env,
)


class DummyTool(Tool):
    name = "dummy"
    description = "A test tool"

    async def execute(self, **kwargs):
        return ToolResult(success=True, data={"echo": kwargs})


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
    async def search(self, query: str, limit: int = 5):
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
