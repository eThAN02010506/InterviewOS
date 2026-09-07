"""Provider-neutral web search for person and company research."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import time
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel
from interview_os.core.provider_config import (
    normalize_provider_api_key,
    provider_api_key_from_env,
)
from interview_os.core.tool import Tool, ToolResult
from interview_os.services.settings_service import (
    PermissionRestrictedJsonStore,
    is_valid_http_endpoint,
)

MAX_SEARCH_REQUEST_COST_USD = 10_000.0
MAX_PERSISTED_SEARCH_METRIC = 10**18
MAX_PERSISTED_SEARCH_COST_USD = (
    MAX_PERSISTED_SEARCH_METRIC * MAX_SEARCH_REQUEST_COST_USD
)
MAX_SEARCH_CACHE_FILE_BYTES = 8 * 1024 * 1024
logger = logging.getLogger(__name__)


def _normalized_search_credential(value: str, *, label: str) -> str:
    """Normalize one search credential using the shared HTTP-header contract."""

    return normalize_provider_api_key(value, label=label)


def _search_endpoint_from_env(name: str) -> str:
    """Read one optional endpoint without allowing it to poison startup."""

    value = os.getenv(name, "")
    if value and not is_valid_http_endpoint(value):
        logger.warning("Ignoring invalid %s configuration", name)
        return ""
    return value.rstrip("/")


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


_RECRUITMENT_HOST_MARKERS = (
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "liepin.com",
    "zhipin.com",
    "51job.com",
    "lagou.com",
)
_RECRUITMENT_PATH_MARKERS = (
    "/career",
    "/careers",
    "/job",
    "/jobs",
    "/join-us",
    "/joinus",
    "/recruit",
    "/招聘",
    "/职位",
)
_RECRUITMENT_TITLE_PATTERN = re.compile(
    r"(?:招聘|职位|岗位空缺|加入我们|job openings?|careers?|vacanc(?:y|ies)|we(?:'|’)re hiring)",
    re.IGNORECASE,
)


def is_recruitment_source(item: dict[str, Any]) -> bool:
    """Return whether a result is a vacancy/recruiting page, not company background."""
    url = str(item.get("url", "")).strip().lower()
    parsed = urlparse(url)
    host = (parsed.hostname or "").removeprefix("www.")
    path = parsed.path.lower()
    title = str(item.get("title", "")).strip()
    return bool(
        any(marker in host for marker in _RECRUITMENT_HOST_MARKERS)
        or any(marker in path for marker in _RECRUITMENT_PATH_MARKERS)
        or _RECRUITMENT_TITLE_PATTERN.search(title)
    )


def filter_employer_business_results(
    results: list[dict[str, Any]], *, entity: str
) -> list[dict[str, Any]]:
    """Keep company-background sources and explicitly exclude current vacancies.

    Official company pages are preferred when the provider or domain match can
    identify them. A safe secondary result is retained only when no official
    page is available, which helps companies whose brand and domain differ.
    """
    entity_key = re.sub(r"[^a-z0-9]", "", entity.lower())
    safe: list[dict[str, Any]] = []
    for item in results:
        if is_recruitment_source(item):
            continue
        accepted = dict(item)
        host_key = re.sub(
            r"[^a-z0-9]",
            "",
            (urlparse(str(item.get("url", ""))).hostname or "").lower().removeprefix("www."),
        )
        domain_matches = bool(entity_key and entity_key in host_key)
        accepted["is_official"] = bool(item.get("is_official")) or domain_matches
        if accepted["is_official"] and accepted.get("source_quality") in {None, "", "unrated"}:
            accepted["source_quality"] = "official"
            accepted["source_quality_reason"] = "URL domain matches the past employer"
        accepted["context_scope"] = "employer_business_background"
        prior_reason = str(accepted.get("filter_reason", "")).strip()
        boundary_reason = "accepted: company business background; recruiting pages excluded"
        accepted["filter_reason"] = prior_reason or boundary_reason
        if prior_reason and boundary_reason not in prior_reason:
            accepted["filter_reason"] = f"{prior_reason}; {boundary_reason}"
        safe.append(accepted)
    official = [item for item in safe if item.get("is_official")]
    # Preference is meaningful only while filtering one employer. Merged
    # multi-employer context must not drop a lesser-known employer merely because
    # another employer in the same list has an identifiable official domain.
    return official if entity_key and official else safe


def format_employer_business_context(results: list[dict[str, Any]]) -> str:
    """Build a prompt block whose provenance cannot be mistaken for candidate evidence."""
    safe = filter_employer_business_results(results, entity="")
    if not safe:
        return ""
    return (
        "过往雇主业务背景（仅用于理解公司所处行业、业务与产品，不是候选人经历证据）：\n"
        "边界规则：不得据此推断候选人的职责、技能、业绩或任职范围；不得把该公司的"
        "当前招聘职位、岗位要求或其他员工经历归因给候选人。\n"
        + format_search_results(safe)
    )


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
        if not is_valid_http_endpoint(base_url):
            raise ValueError("SearXNG base_url must be a valid http:// or https:// URL")
        self.base_url = base_url.strip().rstrip("/")
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
        self.api_key = _normalized_search_credential(api_key, label="Brave API key")
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
        self.api_key = _normalized_search_credential(api_key, label="Tavily API key")
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
        searxng_endpoint = _search_endpoint_from_env("SEARXNG_BASE_URL")
        self._credentials = {
            "tavily": provider_api_key_from_env(
                None,
                env_name="TAVILY_API_KEY",
                logger=logger,
            ),
            "searxng": searxng_endpoint,
            "brave": provider_api_key_from_env(
                None,
                env_name="BRAVE_SEARCH_API_KEY",
                logger=logger,
            ),
        }
        self.selected = self._default_provider()
        self._configuration_generation = 0
        self._cache: OrderedDict[
            tuple[str, str, str, int, str], tuple[float, list[SearchResult]]
        ] = OrderedDict()
        self._in_flight: dict[
            tuple[str, str, str, int, str], asyncio.Task[list[SearchResult]]
        ] = {}
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache_capacity = cache_capacity
        self._cache_store = (
            PermissionRestrictedJsonStore(
                cache_path,
                max_file_bytes=MAX_SEARCH_CACHE_FILE_BYTES,
            )
            if cache_path is not None
            else None
        )
        self._metrics_store = (
            PermissionRestrictedJsonStore(metrics_path) if metrics_path is not None else None
        )
        persisted_metrics = self._metrics_store.load() if self._metrics_store else {}
        self._cache_hits = self._safe_persisted_metric(
            persisted_metrics.get("cache_hits", 0)
        )
        self._cache_misses = self._safe_persisted_metric(
            persisted_metrics.get("cache_misses", 0)
        )
        self._provider_requests = self._safe_persisted_metric(
            persisted_metrics.get("provider_requests", 0)
        )
        metrics_have_outcomes = any(
            key in persisted_metrics
            for key in ("provider_successes", "provider_failures")
        )
        self._provider_failures = min(
            self._provider_requests,
            self._safe_persisted_metric(
                persisted_metrics.get("provider_failures", 0)
            ),
        )
        # Before outcome counters existed, provider_requests was incremented only
        # after a successful response. Preserve that history as successful rather
        # than turning every upgraded installation into an apparent failure.
        legacy_successes = self._provider_requests if not metrics_have_outcomes else 0
        self._provider_successes = min(
            self._provider_requests - self._provider_failures,
            self._safe_persisted_metric(
                persisted_metrics.get("provider_successes", legacy_successes)
            ),
        )
        self._accrued_cost_usd = self._safe_persisted_cost(
            persisted_metrics.get("accrued_cost_usd", 0.0)
        )
        # Old metric files did not retain the historical unit price. Keep those
        # attempts explicit instead of silently repricing them with today's rate.
        self._unpriced_provider_requests = min(
            self._provider_requests,
            self._safe_persisted_metric(
                persisted_metrics.get(
                    "unpriced_provider_requests",
                    self._provider_requests
                    if "accrued_cost_usd" not in persisted_metrics
                    else 0,
                )
            ),
        )
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
        previous_fingerprint = self._configuration_fingerprint()
        next_credentials = dict(self._credentials)
        api_key_updates = {
            "tavily": tavily_api_key,
            "brave": brave_api_key,
        }
        for name, value in api_key_updates.items():
            if value is not None:
                next_credentials[name] = _normalized_search_credential(
                    value, label=f"{name} API key"
                )
        if searxng_base_url is not None:
            if searxng_base_url and not is_valid_http_endpoint(searxng_base_url):
                raise ValueError(
                    "SearXNG base_url must be a valid http:// or https:// URL"
                )
            next_credentials["searxng"] = searxng_base_url.rstrip("/")
        if provider != "none" and not next_credentials[provider]:
            raise ValueError(f"Credentials for {provider} are not configured")
        next_cost = self.search_request_cost_usd
        if search_request_cost_usd is not None:
            try:
                cost = float(search_request_cost_usd)
            except (TypeError, ValueError, OverflowError):
                cost = 0.0
            next_cost = (
                min(MAX_SEARCH_REQUEST_COST_USD, max(0.0, cost))
                if math.isfinite(cost)
                else 0.0
            )
        next_fingerprint = self._fingerprint_for(provider, next_credentials)

        # Publish only after every normalization and validation step succeeds.
        # Settings rollback can therefore never encounter a half-applied,
        # unencodable credential from a rejected request.
        self._credentials = next_credentials
        self.selected = provider
        self.search_request_cost_usd = next_cost
        if next_fingerprint != previous_fingerprint:
            self._configuration_generation += 1

    @staticmethod
    def _safe_persisted_metric(value: Any) -> int:
        """Normalize untrusted counters without failing process startup."""

        if isinstance(value, bool):
            return 0
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return min(MAX_PERSISTED_SEARCH_METRIC, max(0, normalized))

    @staticmethod
    def _safe_persisted_cost(value: Any) -> float:
        """Normalize an untrusted cumulative monetary value."""

        if isinstance(value, bool):
            return 0.0
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(normalized):
            return 0.0
        return min(MAX_PERSISTED_SEARCH_COST_USD, max(0.0, normalized))

    def _configuration_fingerprint(self) -> str:
        """Identify result-affecting routing without exposing credentials."""

        return self._fingerprint_for(self.selected, self._credentials)

    @staticmethod
    def _fingerprint_for(provider: str, credentials: dict[str, str]) -> str:
        """Hash one complete candidate configuration before publishing it."""

        credential = credentials.get(provider, "")
        encoded = f"{provider}\0{credential}".encode()
        return hashlib.sha256(encoded).hexdigest()

    def _increment_metric(self, name: str) -> None:
        current = int(getattr(self, name))
        setattr(self, name, min(MAX_PERSISTED_SEARCH_METRIC, current + 1))

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
            # Compatibility: provider_requests remains the total number of
            # provider calls actually started. The outcome counters split that
            # total without changing the long-standing top-level field name.
            "provider_requests": self._provider_requests,
            "provider_successes": self._provider_successes,
            "provider_failures": self._provider_failures,
            "provider_unresolved": max(
                0,
                self._provider_requests
                - self._provider_successes
                - self._provider_failures,
            ),
            "unpriced_provider_requests": self._unpriced_provider_requests,
            "search_request_cost_usd": self.search_request_cost_usd,
            "estimated_cost_usd": round(self._accrued_cost_usd, 6),
            "source_filter_policy": "exact entity/context match, or one-character Chinese alias with corroboration",
        }

    async def search(
        self, query: str, limit: int = 5, *, search_depth: str = "basic"
    ) -> list[SearchResult]:
        provider_name = self.selected
        configuration_generation = self._configuration_generation
        configuration_fingerprint = self._configuration_fingerprint()
        key = (
            configuration_fingerprint,
            provider_name,
            query.strip(),
            limit,
            search_depth,
        )
        cached = self._cache.get(key)
        if cached and time() - cached[0] <= self._cache_ttl_seconds:
            self._increment_metric("_cache_hits")
            self._cache.move_to_end(key)
            self._persist_metrics()
            self._record_search(
                query, len(cached[1]), provider=provider_name, cache_hit=True
            )
            return [
                item.model_copy(deep=True, update={"cache_hit": True})
                for item in cached[1]
            ]
        if cached:
            self._cache.pop(key, None)
        in_flight = self._in_flight.get(key)
        coalesced = in_flight is not None
        if in_flight is None:
            self._increment_metric("_cache_misses")
            value = self._credentials.get(provider_name, "")
            if provider_name == "none":
                raise ValueError("Web search provider is disabled")
            provider: SearchProvider = (
                TavilySearchProvider(value)
                if provider_name == "tavily"
                else SearXNGProvider(value)
                if provider_name == "searxng"
                else BraveSearchProvider(value)
            )
            in_flight = asyncio.create_task(
                self._run_provider_search(
                    key=key,
                    query=query,
                    limit=limit,
                    search_depth=search_depth,
                    provider_name=provider_name,
                    provider=provider,
                    configuration_generation=configuration_generation,
                    configuration_fingerprint=configuration_fingerprint,
                    unit_cost_usd=self.search_request_cost_usd,
                ),
                name="interview-os-search-provider",
            )
            self._in_flight[key] = in_flight
            in_flight.add_done_callback(partial(self._finish_in_flight, key))

        # Every caller, including the one that created the manager-owned task,
        # awaits through a shield. Cancelling an HTTP request or navigation must
        # not cancel provider work that another coalesced caller still needs.
        results = await asyncio.shield(in_flight)
        if coalesced:
            self._record_search(
                query,
                len(results),
                provider=provider_name,
                cache_hit=False,
                coalesced=True,
            )
        return [item.model_copy(deep=True) for item in results]

    async def _run_provider_search(
        self,
        *,
        key: tuple[str, str, str, int, str],
        query: str,
        limit: int,
        search_depth: str,
        provider_name: str,
        provider: SearchProvider,
        configuration_generation: int,
        configuration_fingerprint: str,
        unit_cost_usd: float,
    ) -> list[SearchResult]:
        """Run shared provider work independently from any one HTTP caller."""

        fetched_at = datetime.now(timezone.utc)
        self._increment_metric("_provider_requests")
        self._accrued_cost_usd = min(
            MAX_PERSISTED_SEARCH_COST_USD,
            self._accrued_cost_usd + unit_cost_usd,
        )
        # Persist before the network await: a provider can bill a request even if
        # the process exits or the response never reaches this application.
        self._persist_metrics()
        try:
            raw_results = await provider.search(
                query, limit, search_depth=search_depth
            )
        except BaseException:
            self._increment_metric("_provider_failures")
            self._persist_metrics()
            raise
        self._increment_metric("_provider_successes")
        self._persist_metrics()
        results = assess_source_quality(raw_results, query)
        results = [
            item.model_copy(update={"fetched_at": fetched_at, "cache_hit": False})
            for item in results
        ]
        stale = (
            configuration_generation != self._configuration_generation
            or configuration_fingerprint != self._configuration_fingerprint()
        )
        if not stale:
            self._cache[key] = (
                time(),
                [item.model_copy(deep=True) for item in results],
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_capacity:
                self._cache.popitem(last=False)
            self._persist_cache()
        self._persist_metrics()
        self._record_search(
            query,
            len(results),
            provider=provider_name,
            cache_hit=False,
            stale=stale,
        )
        return results

    def _finish_in_flight(
        self,
        key: tuple[str, str, str, int, str],
        completed: asyncio.Future[list[SearchResult]],
    ) -> None:
        """Retire shared work and consume unobserved terminal exceptions."""

        if self._in_flight.get(key) is completed:
            self._in_flight.pop(key, None)
        try:
            completed.exception()
        except asyncio.CancelledError:
            # The event loop may cancel manager-owned work during shutdown.
            pass

    def _record_search(
        self,
        query: str,
        count: int,
        *,
        provider: str,
        cache_hit: bool,
        stale: bool = False,
        coalesced: bool = False,
    ) -> None:
        if self.debug_events is None:
            return
        self.debug_events.record(
            DebugEvent(
                level=DebugLevel.INFO,
                category="search",
                action=(
                    "inflight_join"
                    if coalesced
                    else "cache_hit"
                    if cache_hit
                    else "provider_request_stale"
                    if stale
                    else "provider_request"
                ),
                detail=query[:500],
                metadata={
                    "provider": provider,
                    "result_count": count,
                    "source_filter": "exact entity or one-character alias with corroboration",
                    "cache": (
                        "coalesced"
                        if coalesced
                        else "hit"
                        if cache_hit
                        else "stale_not_saved"
                        if stale
                        else "miss"
                    ),
                },
            )
        )

    def _load_cache(self) -> None:
        if self._cache_store is None:
            return
        entries = self._cache_store.load().get("entries", [])
        if not isinstance(entries, list):
            return
        for entry in entries[-self._cache_capacity :]:
            try:
                key = (
                    str(entry["configuration_fingerprint"]),
                    str(entry["provider"]),
                    str(entry["query"]),
                    int(entry["limit"]),
                    str(entry["depth"]),
                )
                if not re.fullmatch(r"[0-9a-f]{64}", key[0]):
                    continue
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
                "configuration_fingerprint": key[0],
                "provider": key[1],
                "query": key[2],
                "limit": key[3],
                "depth": key[4],
                "timestamp": timestamp,
                "results": [item.model_dump(mode="json") for item in results],
            }
            for key, (timestamp, results) in self._cache.items()
        ]
        try:
            self._cache_store.save({"entries": entries})
        except (OSError, ValueError, UnicodeError):
            # Search succeeded already. A best-effort local cache must never
            # turn a paid provider response into a failed user request.
            logger.warning("Could not persist the local search-result cache")

    def _persist_metrics(self) -> None:
        if self._metrics_store is not None:
            try:
                self._metrics_store.save(
                    {
                        "cache_hits": self._cache_hits,
                        "cache_misses": self._cache_misses,
                        "provider_requests": self._provider_requests,
                        "provider_successes": self._provider_successes,
                        "provider_failures": self._provider_failures,
                        "accrued_cost_usd": self._accrued_cost_usd,
                        "unpriced_provider_requests": self._unpriced_provider_requests,
                    }
                )
            except (OSError, ValueError, UnicodeError):
                # Metrics are diagnostic; serving search results has priority.
                logger.warning("Could not persist local search metrics")


def search_provider_from_env() -> SearchProvider | None:
    """Select a provider without leaking configuration into domain agents."""
    tavily_key = provider_api_key_from_env(
        None, env_name="TAVILY_API_KEY", logger=logger
    )
    if tavily_key:
        return TavilySearchProvider(tavily_key)
    searxng_url = _search_endpoint_from_env("SEARXNG_BASE_URL")
    if searxng_url:
        return SearXNGProvider(searxng_url)
    brave_key = provider_api_key_from_env(
        None, env_name="BRAVE_SEARCH_API_KEY", logger=logger
    )
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
            return ToolResult(
                success=False,
                error=f"Web search failed ({type(exc).__name__})",
            )


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
