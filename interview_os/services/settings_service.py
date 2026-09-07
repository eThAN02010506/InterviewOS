"""Permission-restricted, atomic storage for local provider settings."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any

from interview_os.core.provider_config import normalize_provider_endpoint

logger = logging.getLogger(__name__)
MAX_SETTINGS_FILE_BYTES = 1024 * 1024
MAX_SETTINGS_JSON_DEPTH = 64


def is_valid_http_endpoint(value: str, *, allow_empty: bool = False) -> bool:
    """Validate a configurable provider endpoint without making a request.

    This deliberately accepts LAN hosts and IP literals, while rejecting URL
    schemes that could make a provider setting read local files.  Accessing
    ``port`` is part of validation because ``urlsplit`` otherwise tolerates an
    invalid textual port until a caller later tries to use it.
    """

    try:
        normalize_provider_endpoint(
            value,
            label="Provider endpoint",
            allow_empty=allow_empty,
        )
    except (TypeError, ValueError):
        return False
    return True


def _api_key_digest(client: Any) -> str:
    """Return a non-reversible marker for one client's effective credential."""

    secret: Any = None
    snapshot = getattr(client, "secret_snapshot", None)
    if callable(snapshot):
        try:
            values = snapshot()
        except Exception:  # noqa: BLE001 - custom injected adapters are optional
            values = None
        if isinstance(values, dict):
            secret = values.get("api_key")
    if secret is None:
        secret = getattr(client, "api_key", "")
    encoded = str(secret or "").encode("utf-8", errors="backslashreplace")
    return sha256(encoded).hexdigest() if encoded else ""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"settings JSON contains invalid constant {value}")


def _validate_settings_json(value: Any, *, depth: int = 0) -> None:
    """Bound recursive JSON and reject numbers/strings unsafe for persistence."""

    if depth > MAX_SETTINGS_JSON_DEPTH:
        raise ValueError("settings JSON is nested too deeply")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("settings JSON contains a non-finite number")
    if isinstance(value, str) and any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise UnicodeError("settings JSON contains an unpaired surrogate")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_settings_json(key, depth=depth + 1)
            _validate_settings_json(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_settings_json(item, depth=depth + 1)


def audio_settings_fingerprint(asr: Any, omni: Any) -> str:
    """Fingerprint effective audio behavior without persisting API key material."""

    asr_status = asr.status()
    omni_status = omni.status() if omni is not None else {"enabled": False}
    canonical = {
        "asr": {
            key: asr_status.get(key)
            for key in (
                "base_url",
                "model",
                "transcription_path",
                "timeout_seconds",
                "api_key_configured",
            )
        }
        | {"api_key_digest": _api_key_digest(asr)},
        "live_audio": {
            key: omni_status.get(key)
            for key in (
                "enabled",
                "mode",
                "base_url",
                "model",
                "api_key_configured",
            )
        }
        | {"api_key_digest": _api_key_digest(omni) if omni is not None else ""},
    }
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class PermissionRestrictedJsonStore:
    def __init__(
        self,
        path: str | Path,
        *,
        max_file_bytes: int = MAX_SETTINGS_FILE_BYTES,
    ) -> None:
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        self.path = Path(path).expanduser()
        self.max_file_bytes = max_file_bytes
        self._lock = Lock()

    def load(self) -> dict[str, Any]:
        payload, _ = self.load_with_status()
        return payload

    def load_with_status(self) -> tuple[dict[str, Any], str]:
        """Return payload plus ``ok``, ``missing``, ``invalid``, or ``error``."""

        try:
            with self._lock:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with self._interprocess_lock():
                    self._cleanup_stale_temporary_files()
                    if not self.path.exists():
                        return {}, "missing"
                    if not stat.S_ISREG(self.path.lstat().st_mode):
                        return {}, "invalid"
                    flags = os.O_RDONLY
                    if hasattr(os, "O_NOFOLLOW"):
                        flags |= os.O_NOFOLLOW
                    descriptor = os.open(self.path, flags)
                    try:
                        metadata = os.fstat(descriptor)
                        if (
                            not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_size > self.max_file_bytes
                        ):
                            return {}, "invalid"
                        if hasattr(os, "fchmod"):
                            os.fchmod(descriptor, 0o600)
                        with os.fdopen(descriptor, encoding="utf-8") as stream:
                            descriptor = -1
                            payload = json.load(
                                stream,
                                parse_constant=_reject_json_constant,
                            )
                            _validate_settings_json(payload)
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
        except (RecursionError, UnicodeError, ValueError):
            return {}, "invalid"
        except OSError:
            return {}, "error"
        return (payload, "ok") if isinstance(payload, dict) else ({}, "invalid")

    def save(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with self._interprocess_lock():
                self._cleanup_stale_temporary_files()
                encoded = json.dumps(
                    payload, ensure_ascii=False, indent=2, allow_nan=False
                ).encode("utf-8")
                if len(encoded) > self.max_file_bytes:
                    limit_label = (
                        f"{self.max_file_bytes // (1024 * 1024)} MB"
                        if self.max_file_bytes % (1024 * 1024) == 0
                        else f"{self.max_file_bytes} bytes"
                    )
                    raise ValueError(
                        f"JSON store exceeds the {limit_label} limit"
                    )
                temporary: Path | None = None
                descriptor: int | None = None
                replaced = False
                try:
                    # A random, exclusive file prevents a predictable .tmp symlink
                    # from redirecting secrets. The restrictive mode applies at
                    # creation time, before any API key bytes are written.
                    for _ in range(8):
                        candidate = self.path.with_name(
                            f".{self.path.name}.{secrets.token_hex(8)}.tmp"
                        )
                        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                        if hasattr(os, "O_NOFOLLOW"):
                            flags |= os.O_NOFOLLOW
                        try:
                            descriptor = os.open(candidate, flags, 0o600)
                        except FileExistsError:
                            continue
                        temporary = candidate
                        break
                    if descriptor is None or temporary is None:
                        raise OSError("could not allocate a private settings file")

                    with os.fdopen(descriptor, "wb") as stream:
                        descriptor = None
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Rename is the transaction's final possible failure point so
                    # callers never see an exception after new settings are visible.
                    os.replace(temporary, self.path)
                    replaced = True
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
                    if temporary is not None and not replaced:
                        try:
                            temporary.unlink()
                        except FileNotFoundError:
                            pass

    @contextmanager
    def _interprocess_lock(self) -> Iterator[None]:
        """Serialize cleanup/replace across app windows and backend processes."""

        lock_path = self.path.with_name(f".{self.path.name}.lock")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                    os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(  # type: ignore[attr-defined]
                    descriptor, msvcrt.LK_LOCK, 1  # type: ignore[attr-defined]
                )
                try:
                    yield
                finally:
                    try:
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(  # type: ignore[attr-defined]
                            descriptor,
                            msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
                            1,
                        )
                    except OSError:
                        # The protected read/replace already reached its final
                        # outcome. A release diagnostic must not turn a durable
                        # settings commit into an apparent transaction failure.
                        logger.warning("Could not explicitly release settings file lock")
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        logger.warning("Could not explicitly release settings file lock")
        finally:
            try:
                os.close(descriptor)
            except OSError:
                logger.warning("Could not close settings lock descriptor")

    def _cleanup_stale_temporary_files(self) -> None:
        """Remove private crash remnants belonging only to this store path."""

        if not self.path.parent.exists():
            return
        pattern = re.compile(rf"\.{re.escape(self.path.name)}\.[0-9a-f]{{16}}\.tmp")
        try:
            entries = list(self.path.parent.iterdir())
        except OSError:
            return
        for candidate in entries:
            if not pattern.fullmatch(candidate.name):
                continue
            try:
                mode = candidate.lstat().st_mode
                if stat.S_ISREG(mode):
                    candidate.unlink()
            except (FileNotFoundError, OSError):
                continue


class LocalSettingsStore(PermissionRestrictedJsonStore):
    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv("INTERVIEW_OS_SETTINGS_PATH")
        super().__init__(
            Path(configured).expanduser()
            if configured
            else Path.home() / ".interview_os" / "settings.json"
        )
