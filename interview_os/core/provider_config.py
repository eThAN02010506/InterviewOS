"""Validation helpers for secret-bearing provider configuration.

Provider credentials are eventually placed in HTTP ``Authorization`` headers.
Keeping their accepted representation deliberately narrow prevents malformed
persisted values or environment variables from failing much later inside the
HTTP client, where they would otherwise surface as opaque 500 responses.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

MAX_PROVIDER_API_KEY_LENGTH = 8192


def normalize_provider_endpoint(
    value: str,
    *,
    label: str = "Provider endpoint",
    allow_empty: bool = False,
) -> str:
    """Return one normalized HTTP(S) endpoint or raise before publication."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    if not value:
        if allow_empty:
            return ""
        raise ValueError(f"{label} must not be empty")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    if value.partition(":")[0] not in {"http", "https"}:
        raise ValueError(f"{label} must use http:// or https://")
    if "?" in value or "#" in value:
        raise ValueError(f"{label} must not contain a query or fragment")
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in value
    ):
        raise ValueError(f"{label} contains invalid whitespace or control characters")
    try:
        value.encode("utf-8")
        parsed = urlsplit(value)
        _ = parsed.port
        hostname = parsed.hostname
        if hostname is not None:
            hostname.encode("idna")
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid HTTP endpoint") from exc
    if not (
        parsed.scheme in {"http", "https"}
        and hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    ):
        raise ValueError(f"{label} is not a valid HTTP endpoint")
    return value.rstrip("/")


def provider_endpoint_from_env(
    explicit: str | None,
    *,
    env_name: str,
    default: str,
    label: str,
    logger: logging.Logger | None = None,
) -> str:
    """Resolve an endpoint, ignoring malformed environment overrides."""

    if explicit is not None:
        return normalize_provider_endpoint(explicit, label=label)
    raw = os.getenv(env_name)
    if raw is None:
        return normalize_provider_endpoint(default, label=label)
    try:
        return normalize_provider_endpoint(raw, label=label)
    except (TypeError, ValueError):
        if logger is not None:
            logger.warning("Ignoring invalid %s provider endpoint", env_name)
        return normalize_provider_endpoint(default, label=label)


def normalize_provider_api_key(value: str, *, label: str = "API key") -> str:
    """Return a header-safe API key, allowing an empty value to clear a key."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    if len(value) > MAX_PROVIDER_API_KEY_LENGTH:
        raise ValueError(
            f"{label} must not exceed {MAX_PROVIDER_API_KEY_LENGTH} characters"
        )
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    if any(not 0x21 <= ord(character) <= 0x7E for character in value):
        raise ValueError(f"{label} must contain visible ASCII characters only")
    return value


def provider_api_key_from_env(
    explicit: str | None,
    *,
    env_name: str,
    default: str = "",
    logger: logging.Logger | None = None,
) -> str:
    """Resolve a provider key while failing closed on malformed environment data.

    Explicit constructor values are programmer/configuration input and therefore
    raise on invalid data.  Environment values are untrusted process input: an
    invalid one is ignored so the local app can still start in an unauthenticated
    provider mode, without ever logging the credential itself.
    """

    if explicit is not None:
        return normalize_provider_api_key(explicit)
    raw = os.getenv(env_name)
    if raw is None:
        return normalize_provider_api_key(default)
    try:
        return normalize_provider_api_key(raw)
    except ValueError:
        if logger is not None:
            logger.warning("Ignoring invalid %s provider credential", env_name)
        return normalize_provider_api_key(default)


__all__ = [
    "MAX_PROVIDER_API_KEY_LENGTH",
    "normalize_provider_api_key",
    "normalize_provider_endpoint",
    "provider_api_key_from_env",
    "provider_endpoint_from_env",
]
