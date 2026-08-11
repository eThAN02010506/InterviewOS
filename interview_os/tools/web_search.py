"""Provider-neutral web search for person and company research."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel
from interview_os.core.tool import Tool, ToolResult
from interview_os.services.settings_service import PermissionRestrictedJsonStore


class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""
    source_quality: str = "unrated"
    source_quality_reason: str = ""
    filter_reason: str = ""
    fetched_at: datetime | None = None
    cache_hit: bool = False
    is_official: bool = False
    corroboration_count: int = 1


def assess_source_quality(results: list[SearchResult], query: str) -> list[SearchResult]:
    """Attach transparent heuristics; relevance never implies factual truth."""
    # Person research commonly quotes both the interviewer and their company.
    # A corporate press release is official for that query when its domain
    # matches either quoted entity, not only the first one.
    entities = {
        compact
        for quoted in re.findall(r'"([^"\n]+)"', query)
        if (compact := re.sub(r"[^a-z0-9]", "", quoted.lower()))
    }
    reputable_domains = ("reuters.com", "apnews.com", "bloomberg.com", "ft.com", "wsj.com")
    domains = [urlparse(item.url).hostname or "" for item in results]
    independent_domains = len(set(domains))
    for item, domain in zip(results, domains, strict=True):
        compact_domain = re.sub(r"[^a-z0-9]", "", domain.lower().removeprefix("www."))
        item.is_official = any(entity in compact_domain for entity in entities)
        if item.is_official:
            item.source_quality = "official"
            item.source_quality_reason = "URL domain appears to match the quoted entity"
        elif domain.endswith(reputable_domains):
            item.source_quality = "high"
            item.source_quality_reason = "URL domain is in the reputable news allowlist"
        else:
            item.source_quality = "secondary"
            item.source_quality_reason = (
                "Public search result; not an official domain or allowlisted news source"
            )
        item.corroboration_count = independent_domains
    return results


def merge_search_results(*batches: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    """Deduplicate ordered provider results by canonical URL."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in batches:
        for item in batch:
            url = str(item.get("url", "")).rstrip("/")
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def filter_entity_results(
    results: list[dict[str, Any]],
    *,
    entity: str,
    required_context: str = "",
    corroborating_entity: str = "",
    allow_context_alias: bool = False,
) -> list[dict[str, Any]]:
    """Keep exact identities plus tightly corroborated one-character Chinese aliases."""
    normalize = lambda value: re.sub(r"[^\w\u4e00-\u9fff]", "", value.lower())
    entity_key = normalize(entity)
    context_key = normalize(required_context)
    corroborator_key = normalize(corroborating_entity)
    filtered = []
    for item in results:
        haystack = normalize(
            " ".join(str(item.get(key, "")) for key in ("title", "snippet", "url"))
        )
        entity_exact = bool(entity_key and entity_key in haystack)
        entity_alias = "" if entity_exact else _one_character_alias(haystack, entity_key)
        corroborated = bool(corroborator_key and corroborator_key in haystack)
        if entity_key and not entity_exact and not (entity_alias and corroborated):
            continue
        context_exact = not context_key or context_key in haystack
        context_alias = "" if context_exact else _one_character_alias(haystack, context_key)
        if not context_exact and not (allow_context_alias and entity_exact and context_alias):
            continue
        accepted = dict(item)
        if entity_alias or context_alias:
            accepted["identity_match"] = "corroborated_alias"
            accepted["input_identity"] = entity if entity_alias else required_context
            accepted["matched_identity"] = entity_alias or context_alias
            accepted["filter_reason"] = (
                "accepted: one-character Chinese alias corroborated by the required entity"
            )
        else:
            accepted["identity_match"] = "exact"
            accepted["filter_reason"] = "accepted: exact entity/context match"
        filtered.append(accepted)
    return filtered


def _one_character_alias(haystack: str, needle: str) -> str:
    """Return a near Chinese name only for equal-length strings differing by one character."""
    if not 3 <= len(needle) <= 8 or not all("\u4e00" <= char <= "\u9fff" for char in needle):
        return ""
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        candidate = haystack[start : start + width]
        if (
            all("\u4e00" <= char <= "\u9fff" for char in candidate)
            and sum(left != right for left, right in zip(candidate, needle, strict=True)) == 1
        ):
            return candidate
    return ""


class SearchProvider(ABC):
    @abstractmethod
    async def search(
        self, query: str, limit: int = 5, *, search_depth: str = "basic"
    ) -> list[SearchResult]: ...


class SearXNGProvider(SearchProvider):
    def __init__(self, base_url: str, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def search(
        self, query: str, limit: int = 5, *, search_depth: str = "basic"
    ) -> list[SearchResult]:
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

    async def search(
        self, query: str, limit: int = 5, *, search_depth: str = "basic"
    ) -> list[SearchResult]:
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

    async def search(
        self, query: str, limit: int = 5, *, search_depth: str = "basic"
    ) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "search_depth": search_depth,
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

    def __init__(
        self,
        *,
        cache_ttl_seconds: float = 3600,
        cache_capacity: int = 128,
        debug_events: DebugEventStore | None = None,
        cache_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ) -> None:
        self._credentials = {
            "tavily": os.getenv("TAVILY_API_KEY", "").strip(),
            "searxng": os.getenv("SEARXNG_BASE_URL", "").strip(),
            "brave": os.getenv("BRAVE_SEARCH_API_KEY", "").strip(),
        }
        self.selected = self._default_provider()
        self._cache: OrderedDict[tuple[str, str, int, str], tuple[float, list[SearchResult]]] = (
            OrderedDict()
        )
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache_capacity = cache_capacity
        self._cache_store = (
            PermissionRestrictedJsonStore(cache_path) if cache_path is not None else None
        )
        self._metrics_store = (
            PermissionRestrictedJsonStore(metrics_path) if metrics_path is not None else None
        )
        persisted_metrics = self._metrics_store.load() if self._metrics_store else {}
        self._cache_hits = int(persisted_metrics.get("cache_hits", 0))
        self._cache_misses = int(persisted_metrics.get("cache_misses", 0))
        self._provider_requests = int(persisted_metrics.get("provider_requests", 0))
        self.search_request_cost_usd = 0.0
        self.debug_events = debug_events
        self._load_cache()

    def _default_provider(self) -> str:
        return next(
            (name for name in ("tavily", "searxng", "brave") if self._credentials[name]), "none"
        )

    def configure(
        self,
        provider: str,
        *,
        tavily_api_key: str | None = None,
        searxng_base_url: str | None = None,
        brave_api_key: str | None = None,
        search_request_cost_usd: float | None = None,
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
        if search_request_cost_usd is not None:
            self.search_request_cost_usd = max(0.0, search_request_cost_usd)

    def secret_snapshot(self) -> dict[str, Any]:
        return {
            "provider": self.selected,
            "tavily_api_key": self._credentials["tavily"],
            "searxng_base_url": self._credentials["searxng"],
            "brave_api_key": self._credentials["brave"],
            "search_request_cost_usd": self.search_request_cost_usd,
        }

    def status(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "configured": {name: bool(value) for name, value in self._credentials.items()},
            "cache": {
                "size": len(self._cache),
                "capacity": self._cache_capacity,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "persistent": self._cache_store is not None,
            },
            "provider_requests": self._provider_requests,
            "search_request_cost_usd": self.search_request_cost_usd,
            "estimated_cost_usd": round(
                self._provider_requests * self.search_request_cost_usd, 6
            ),
            "source_filter_policy": "exact entity/context match, or one-character Chinese alias with corroboration",
        }

    async def search(
        self, query: str, limit: int = 5, *, search_depth: str = "basic"
    ) -> list[SearchResult]:
        key = (self.selected, query.strip(), limit, search_depth)
        cached = self._cache.get(key)
        if cached and time() - cached[0] <= self._cache_ttl_seconds:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            self._persist_metrics()
            self._record_search(query, len(cached[1]), cache_hit=True)
            return [
                item.model_copy(deep=True, update={"cache_hit": True})
                for item in cached[1]
            ]
        if cached:
            self._cache.pop(key, None)
        self._cache_misses += 1
        value = self._credentials.get(self.selected, "")
        providers: dict[str, SearchProvider] = {
            "tavily": TavilySearchProvider(value),
            "searxng": SearXNGProvider(value),
            "brave": BraveSearchProvider(value),
        }
        if self.selected == "none":
            raise ValueError("Web search provider is disabled")
        fetched_at = datetime.now(timezone.utc)
        results = assess_source_quality(
            await providers[self.selected].search(query, limit, search_depth=search_depth), query
        )
        results = [
            item.model_copy(update={"fetched_at": fetched_at, "cache_hit": False})
            for item in results
        ]
        self._provider_requests += 1
        self._cache[key] = (time(), [item.model_copy(deep=True) for item in results])
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)
        self._persist_cache()
        self._persist_metrics()
        self._record_search(query, len(results), cache_hit=False)
        return results

    def _record_search(self, query: str, count: int, *, cache_hit: bool) -> None:
        if self.debug_events is None:
            return
        self.debug_events.record(
            DebugEvent(
                level=DebugLevel.INFO,
                category="search",
                action="cache_hit" if cache_hit else "provider_request",
                detail=query[:500],
                metadata={
                    "provider": self.selected,
                    "result_count": count,
                    "source_filter": "exact entity or one-character alias with corroboration",
                    "cache": "hit" if cache_hit else "miss",
                },
            )
        )

    def _load_cache(self) -> None:
        if self._cache_store is None:
            return
        entries = self._cache_store.load().get("entries", [])
        for entry in entries[-self._cache_capacity :]:
            try:
                key = (
                    str(entry["provider"]),
                    str(entry["query"]),
                    int(entry["limit"]),
                    str(entry["depth"]),
                )
                timestamp = float(entry["timestamp"])
                if time() - timestamp > self._cache_ttl_seconds:
                    continue
                results = [SearchResult.model_validate(item) for item in entry["results"]]
                self._cache[key] = (timestamp, results)
            except (KeyError, TypeError, ValueError):
                continue

    def _persist_cache(self) -> None:
        if self._cache_store is None:
            return
        entries = [
            {
                "provider": key[0],
                "query": key[1],
                "limit": key[2],
                "depth": key[3],
                "timestamp": timestamp,
                "results": [item.model_dump(mode="json") for item in results],
            }
            for key, (timestamp, results) in self._cache.items()
        ]
        self._cache_store.save({"entries": entries})

    def _persist_metrics(self) -> None:
        if self._metrics_store is not None:
            self._metrics_store.save(
                {
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "provider_requests": self._provider_requests,
                }
            )


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
        search_depth = str(kwargs.get("search_depth", "basic"))
        if search_depth not in {"basic", "advanced"}:
            return ToolResult(success=False, error="Search depth must be basic or advanced")
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
            results = await self.provider.search(query, limit, search_depth=search_depth)
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": [item.model_dump(mode="json") for item in results],
                },
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return ToolResult(success=False, error=f"Web search failed: {exc}")


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Produce a compact, source-preserving block for an LLM prompt."""
    return "\n".join(
        f"[{index}] {item.get('title', '')}\n"
        f"URL: {item.get('url', '')}\n"
        f"Quality: {item.get('source_quality', 'unrated')} | "
        f"Reason: {item.get('source_quality_reason', '')} | "
        f"Official: {item.get('is_official', False)} | "
        f"Independent domains: {item.get('corroboration_count', 1)} | "
        f"Fetched: {item.get('fetched_at', '')} | "
        f"Cache: {item.get('cache_hit', False)}\n"
        f"Snippet: {item.get('snippet', '')}"
        for index, item in enumerate(results, start=1)
    )
