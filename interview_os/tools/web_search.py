"""Provider-neutral web search for person and company research."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel

from interview_os.core.tool import Tool, ToolResult


class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...


class SearXNGProvider(SearchProvider):
    def __init__(self, base_url: str, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "language": "auto", "safesearch": 1},
            )
            response.raise_for_status()
        items = response.json().get("results", [])[:limit]
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source=item.get("engine", "searxng"),
            )
            for item in items
        ]


class BraveSearchProvider(SearchProvider):
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                self.endpoint,
                params={"q": query, "count": min(limit, 20)},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            )
            response.raise_for_status()
        items = response.json().get("web", {}).get("results", [])[:limit]
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source="brave",
            )
            for item in items
        ]


class TavilySearchProvider(SearchProvider):
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": min(limit, 20),
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
        items = response.json().get("results", [])[:limit]
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source="tavily",
            )
            for item in items
            if float(item.get("score", 1.0)) >= 0.5
        ]


class SearchProviderManager(SearchProvider):
    """Mutable provider router shared by all session runtimes."""

    def __init__(self) -> None:
        self._credentials = {
            "tavily": os.getenv("TAVILY_API_KEY", "").strip(),
            "searxng": os.getenv("SEARXNG_BASE_URL", "").strip(),
            "brave": os.getenv("BRAVE_SEARCH_API_KEY", "").strip(),
        }
        self.selected = self._default_provider()

    def _default_provider(self) -> str:
        return next((name for name in ("tavily", "searxng", "brave") if self._credentials[name]), "none")

    def configure(
        self,
        provider: str,
        *,
        tavily_api_key: str | None = None,
        searxng_base_url: str | None = None,
        brave_api_key: str | None = None,
    ) -> None:
        if provider not in {"none", "tavily", "searxng", "brave"}:
            raise ValueError(f"Unsupported search provider: {provider}")
        updates = {
            "tavily": tavily_api_key,
            "searxng": searxng_base_url,
            "brave": brave_api_key,
        }
        for name, value in updates.items():
            if value is not None:
                self._credentials[name] = value.strip()
        if provider != "none" and not self._credentials[provider]:
            raise ValueError(f"Credentials for {provider} are not configured")
        self.selected = provider

    def status(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "configured": {name: bool(value) for name, value in self._credentials.items()},
        }

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        value = self._credentials.get(self.selected, "")
        providers: dict[str, SearchProvider] = {
            "tavily": TavilySearchProvider(value),
            "searxng": SearXNGProvider(value),
            "brave": BraveSearchProvider(value),
        }
        if self.selected == "none":
            raise ValueError("Web search provider is disabled")
        return await providers[self.selected].search(query, limit)


def search_provider_from_env() -> SearchProvider | None:
    """Select a provider without leaking configuration into domain agents."""
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        return TavilySearchProvider(tavily_key)
    searxng_url = os.getenv("SEARXNG_BASE_URL", "").strip()
    if searxng_url:
        return SearXNGProvider(searxng_url)
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if brave_key:
        return BraveSearchProvider(brave_key)
    return None


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search public web sources for information about a person or company"
    parameters: ClassVar[dict[str, Any]] = {
        "query": {"type": "string", "description": "Search query"},
        "num_results": {"type": "int", "description": "Number of results", "default": 5},
    }

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self.provider = provider or search_provider_from_env()

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        limit = max(1, min(int(kwargs.get("num_results", 5)), 10))
        if not query:
            return ToolResult(success=False, error="Search query cannot be empty")
        if self.provider is None:
            return ToolResult(
                success=False,
                error=(
                    "Web search is not configured; set TAVILY_API_KEY, "
                    "SEARXNG_BASE_URL, or BRAVE_SEARCH_API_KEY"
                ),
            )
        try:
            results = await self.provider.search(query, limit)
            return ToolResult(
                success=True,
                data={"query": query, "results": [item.model_dump() for item in results]},
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return ToolResult(success=False, error=f"Web search failed: {exc}")


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Produce a compact, source-preserving block for an LLM prompt."""
    return "\n".join(
        f"[{index}] {item.get('title', '')}\n"
        f"URL: {item.get('url', '')}\n"
        f"Snippet: {item.get('snippet', '')}"
        for index, item in enumerate(results, start=1)
    )
