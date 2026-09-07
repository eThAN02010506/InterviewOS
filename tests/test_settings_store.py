"""Security and failure-safety checks for local secret-bearing JSON files."""

from __future__ import annotations

import asyncio
import os
from threading import Event, Thread, current_thread

import pytest

from interview_os.core.provider_config import (
    MAX_PROVIDER_API_KEY_LENGTH,
    normalize_provider_api_key,
    normalize_provider_endpoint,
    provider_api_key_from_env,
)
from interview_os.models.local_llm import LocalLLMClient
from interview_os.models.omni_client import DEFAULT_OMNI_BASE_URL, OmniAudioClient
from interview_os.services.settings_service import (
    PermissionRestrictedJsonStore,
    audio_settings_fingerprint,
    is_valid_http_endpoint,
)
from interview_os.tools.asr import ASRClient
from interview_os.tools.tts import DEFAULT_TTS_BASE_URL, TTSClient


class _FingerprintClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def status(self):
        return {
            "enabled": True,
            "mode": "asr_text",
            "base_url": "http://provider.test/v1",
            "model": "model",
            "transcription_path": "/audio/transcriptions",
            "timeout_seconds": 30,
            "api_key_configured": bool(self.api_key),
        }

    def secret_snapshot(self):
        return {"api_key": self.api_key}


@pytest.mark.parametrize(
    "credential",
    (
        " leading",
        "trailing ",
        "embedded space",
        "line\nbreak",
        "中文密钥",
        "\ud800",
        "x" * (MAX_PROVIDER_API_KEY_LENGTH + 1),
    ),
)
def test_provider_api_keys_reject_values_unsafe_for_http_headers(credential):
    with pytest.raises(ValueError, match="API key"):
        normalize_provider_api_key(credential)


def test_invalid_provider_key_environment_fails_closed(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "poison\nheader")

    assert (
        provider_api_key_from_env(
            None,
            env_name="TEST_PROVIDER_KEY",
            default="safe-default",
        )
        == "safe-default"
    )
    assert normalize_provider_api_key("") == ""


@pytest.mark.parametrize(
    "factory",
    (
        lambda endpoint: LocalLLMClient(base_url=endpoint),
        lambda endpoint: ASRClient(base_url=endpoint),
        lambda endpoint: OmniAudioClient(base_url=endpoint),
        lambda endpoint: TTSClient(base_url=endpoint),
    ),
)
@pytest.mark.parametrize(
    "endpoint",
    ("file:///tmp/model", " http://provider.test/v1", "http://\ud800"),
)
def test_provider_clients_reject_invalid_explicit_endpoints(factory, endpoint):
    with pytest.raises(ValueError):
        factory(endpoint)


@pytest.mark.parametrize(
    ("env_name", "factory", "expected"),
    (
        (
            "LLM_BASE_URL",
            lambda: LocalLLMClient(),
            "http://localhost:11434/v1",
        ),
        ("OMNI_BASE_URL", lambda: OmniAudioClient(), DEFAULT_OMNI_BASE_URL),
        ("TTS_BASE_URL", lambda: TTSClient(), DEFAULT_TTS_BASE_URL),
    ),
)
def test_invalid_provider_endpoint_environment_falls_back_without_poisoning_startup(
    monkeypatch, env_name, factory, expected
):
    monkeypatch.setenv(env_name, "file:///tmp/provider")

    client = factory()

    assert client.base_url == normalize_provider_endpoint(expected)
    asyncio.run(client.close())


def test_audio_settings_fingerprint_changes_when_non_empty_key_rotates():
    first = audio_settings_fingerprint(
        _FingerprintClient("credential-one"), _FingerprintClient("omni-key")
    )
    second = audio_settings_fingerprint(
        _FingerprintClient("credential-two"), _FingerprintClient("omni-key")
    )

    assert first != second
    assert "credential-one" not in first
    assert "credential-two" not in second


@pytest.mark.parametrize(
    "endpoint",
    ("file:///tmp/model", "http://", "https:///missing-host", "http://host:bad"),
)
def test_http_endpoint_validation_rejects_unsafe_or_hostless_urls(endpoint):
    assert is_valid_http_endpoint(endpoint) is False


def test_http_endpoint_validation_accepts_loopback_and_lan_hosts():
    assert is_valid_http_endpoint("http://127.0.0.1:8001/v1") is True
    assert is_valid_http_endpoint("https://models.example.test/v1") is True
    assert is_valid_http_endpoint("") is False
    assert is_valid_http_endpoint("", allow_empty=True) is True


def test_http_endpoint_validation_rejects_ambiguous_or_secret_bearing_base_urls():
    for endpoint in (
        "HTTP://127.0.0.1:8001/v1",
        "https://user:password@example.test/v1",
        "https://example.test/v1?tenant=private",
        "https://example.test/v1#fragment",
        " https://example.test/v1",
        "https://example.test/v1 ",
        "https://example.test/v1\nignored",
        "http://\ud800",
        "http://exa\ud800mple.test/v1",
        "http://example.test/\ud800",
        "http://example.test?",
        "http://example.test#",
        "http://example.test/?",
        "http://example.test/#",
    ):
        assert is_valid_http_endpoint(endpoint) is False


def test_settings_store_does_not_follow_legacy_predictable_temp_symlink(tmp_path):
    target = tmp_path / "settings.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("leave-me-alone", encoding="utf-8")
    (tmp_path / "settings.json.tmp").symlink_to(victim)

    PermissionRestrictedJsonStore(target).save({"api_key": "secret"})

    assert victim.read_text(encoding="utf-8") == "leave-me-alone"
    assert target.stat().st_mode & 0o777 == 0o600


def test_settings_store_removes_private_temp_after_replace_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "settings.json"

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        PermissionRestrictedJsonStore(target).save({"api_key": "never-leak"})

    assert not target.exists()
    assert list(tmp_path.glob(".settings.json.*.tmp")) == []


def test_settings_store_load_removes_private_crash_remnant(tmp_path):
    target = tmp_path / "settings.json"
    crash_remnant = tmp_path / ".settings.json.0123456789abcdef.tmp"
    crash_remnant.write_text('{"api_key":"stale-secret"}', encoding="utf-8")
    crash_remnant.chmod(0o600)

    assert PermissionRestrictedJsonStore(target).load() == {}

    assert not crash_remnant.exists()


def test_settings_store_load_repairs_legacy_wide_permissions(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text('{"api_key":"legacy-secret"}', encoding="utf-8")
    target.chmod(0o644)

    assert PermissionRestrictedJsonStore(target).load() == {
        "api_key": "legacy-secret"
    }
    assert target.stat().st_mode & 0o777 == 0o600


def test_settings_store_does_not_change_existing_parent_permissions(tmp_path):
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o755)
    shared_parent.chmod(0o755)
    target = shared_parent / "settings.json"

    assert PermissionRestrictedJsonStore(target).load() == {}
    PermissionRestrictedJsonStore(target).save({"api_key": "private"})

    assert shared_parent.stat().st_mode & 0o777 == 0o755
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX fsync barrier test")
def test_settings_store_serializes_cleanup_across_store_instances(
    tmp_path, monkeypatch
):
    target = tmp_path / "settings.json"
    first = PermissionRestrictedJsonStore(target)
    second = PermissionRestrictedJsonStore(target)
    temp_ready = Event()
    allow_replace = Event()
    reader_finished = Event()
    errors = []
    observed = []
    original_fsync = os.fsync
    writer_thread: Thread | None = None

    def gated_fsync(descriptor):
        original_fsync(descriptor)
        if current_thread() is writer_thread:
            temp_ready.set()
            assert allow_replace.wait(timeout=2)

    monkeypatch.setattr(os, "fsync", gated_fsync)

    def write_settings():
        try:
            first.save({"api_key": "committed-secret"})
        except BaseException as exc:  # noqa: BLE001 - surfaced in the test thread
            errors.append(exc)

    def read_settings():
        try:
            observed.append(second.load())
        finally:
            reader_finished.set()

    writer_thread = Thread(target=write_settings)
    writer_thread.start()
    assert temp_ready.wait(timeout=2)
    reader_thread = Thread(target=read_settings)
    reader_thread.start()

    assert not reader_finished.wait(timeout=0.05)
    allow_replace.set()
    writer_thread.join(timeout=2)
    reader_thread.join(timeout=2)

    assert errors == []
    assert observed == [{"api_key": "committed-secret"}]


@pytest.mark.skipif(os.name == "nt", reason="POSIX lock fault injection")
def test_settings_store_does_not_report_failure_after_committed_unlock_error(
    tmp_path, monkeypatch
):
    import fcntl

    target = tmp_path / "settings.json"
    original_flock = fcntl.flock

    def fail_unlock(descriptor, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError("simulated unlock failure")
        return original_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", fail_unlock)

    PermissionRestrictedJsonStore(target).save({"api_key": "committed"})

    monkeypatch.setattr(fcntl, "flock", original_flock)
    assert PermissionRestrictedJsonStore(target).load() == {"api_key": "committed"}


def test_settings_store_distinguishes_invalid_json_from_missing_file(tmp_path):
    missing = PermissionRestrictedJsonStore(tmp_path / "missing.json")
    assert missing.load_with_status() == ({}, "missing")

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{broken", encoding="utf-8")
    invalid = PermissionRestrictedJsonStore(invalid_path)
    assert invalid.load_with_status() == ({}, "invalid")

    invalid_path.write_bytes(b"\xff")
    assert invalid.load_with_status() == ({}, "invalid")


def test_settings_store_rejects_non_finite_json_serialization(tmp_path):
    target = tmp_path / "settings.json"

    with pytest.raises(ValueError):
        PermissionRestrictedJsonStore(target).save({"cost": float("inf")})

    assert not target.exists()


@pytest.mark.parametrize("encoded", [b'{"cost":NaN}', b'{"cost":Infinity}', b'{"cost":1e309}'])
def test_settings_store_rejects_non_finite_json_on_load(tmp_path, encoded):
    target = tmp_path / "settings.json"
    target.write_bytes(encoded)

    assert PermissionRestrictedJsonStore(target).load_with_status() == ({}, "invalid")


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"asr":{"model":"\\ud800"}}',
        ("[" * 70 + "0" + "]" * 70).encode(),
        b'{"padding":"' + (b"x" * (1024 * 1024)) + b'"}',
    ],
    ids=("unpaired-surrogate", "excessive-depth", "oversized-payload"),
)
def test_settings_store_rejects_unsafe_or_unbounded_json_on_load(tmp_path, encoded):
    target = tmp_path / "settings.json"
    target.write_bytes(encoded)

    assert PermissionRestrictedJsonStore(target).load_with_status() == ({}, "invalid")


def test_settings_store_rejects_oversized_serialization_before_creating_file(tmp_path):
    target = tmp_path / "settings.json"

    with pytest.raises(ValueError, match="1 MB"):
        PermissionRestrictedJsonStore(target).save({"padding": "x" * (1024 * 1024)})

    assert not target.exists()
