import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import httpx
import pytest

from interview_os.desktop.sidecar import (
    BOOTSTRAP_STDIN_PREFIX,
    HANDSHAKE_PREFIX,
    SHUTDOWN_STDIN_COMMAND,
    ShutdownController,
    configure_desktop_environment,
    handshake_payload,
    read_bootstrap_token,
    reserve_loopback_socket,
)


def test_sidecar_reserves_random_loopback_port():
    listener = reserve_loopback_socket()
    try:
        host, port = listener.getsockname()
        assert host == "127.0.0.1"
        assert 0 < port <= 65535
        assert listener.get_inheritable() is False
    finally:
        listener.close()


def test_sidecar_rejects_invalid_port():
    with pytest.raises(ValueError, match="port"):
        reserve_loopback_socket(65536)


@pytest.mark.skipif(
    not hasattr(__import__("socket"), "SO_EXCLUSIVEADDRUSE"),
    reason="Windows-only exclusive bind behavior",
)
def test_windows_sidecar_port_cannot_be_rebound():
    import socket

    listener = reserve_loopback_socket()
    contender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        contender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            contender.bind(listener.getsockname())
    finally:
        contender.close()
        listener.close()


def test_sidecar_environment_is_configured_before_app_import(monkeypatch, tmp_path):
    # Register all three keys with monkeypatch before the production helper
    # mutates os.environ directly, so teardown restores the parent process.
    monkeypatch.setenv("INTERVIEW_OS_RUNTIME_MODE", "server")
    monkeypatch.setenv("INTERVIEW_OS_BOOTSTRAP_TOKEN", "previous-test-value")
    monkeypatch.setenv("INTERVIEW_OS_DATA_DIR", "previous-test-value")
    monkeypatch.setenv("INTERVIEW_OS_REQUIRE_AUTH", "0")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:////tmp/foreign.db")
    monkeypatch.setenv("INTERVIEW_OS_SETTINGS_PATH", "/tmp/foreign-settings.json")
    monkeypatch.setenv("INTERVIEW_OS_RECORDINGS_DIR", "/tmp/foreign-recordings")

    bootstrap_token = "private-token-that-is-long-enough-123"
    configure_desktop_environment(tmp_path / "app data", bootstrap_token)

    assert __import__("os").environ["INTERVIEW_OS_RUNTIME_MODE"] == "desktop"
    assert __import__("os").environ["INTERVIEW_OS_BOOTSTRAP_TOKEN"] == bootstrap_token
    assert Path(__import__("os").environ["INTERVIEW_OS_DATA_DIR"]) == tmp_path / "app data"
    assert __import__("os").environ["INTERVIEW_OS_REQUIRE_AUTH"] == "1"
    assert "DATABASE_URL" not in __import__("os").environ
    assert "INTERVIEW_OS_SETTINGS_PATH" not in __import__("os").environ
    assert "INTERVIEW_OS_RECORDINGS_DIR" not in __import__("os").environ


def test_shutdown_request_before_server_attach_is_not_lost():
    controller = ShutdownController()
    controller.request()
    server = type("FakeServer", (), {"should_exit": False})()

    controller.attach(server)

    assert server.should_exit is True
    controller.finish()


def test_sidecar_without_data_argument_clears_inherited_data_directory(monkeypatch):
    monkeypatch.setenv("INTERVIEW_OS_RUNTIME_MODE", "server")
    monkeypatch.setenv("INTERVIEW_OS_BOOTSTRAP_TOKEN", "previous-test-value")
    monkeypatch.setenv("INTERVIEW_OS_REQUIRE_AUTH", "0")
    monkeypatch.setenv("INTERVIEW_OS_DATA_DIR", "/tmp/inherited-desktop-data")

    configure_desktop_environment(None, "private-token-that-is-long-enough-123")

    assert "INTERVIEW_OS_DATA_DIR" not in os.environ


def test_sidecar_reads_parent_generated_token_from_stdin():
    token = "a" * 64
    assert read_bootstrap_token(StringIO(f"{BOOTSTRAP_STDIN_PREFIX}{token}\n")) == token
    with pytest.raises(ValueError, match="handshake"):
        read_bootstrap_token(StringIO("not-a-handshake\n"))
    with pytest.raises(ValueError, match="token"):
        read_bootstrap_token(StringIO(f"{BOOTSTRAP_STDIN_PREFIX}short\n"))


def test_sidecar_handshake_is_a_single_versioned_json_record(capsys):
    from interview_os.desktop.sidecar import emit_handshake

    emit_handshake(43123)
    line = capsys.readouterr().out.strip()

    assert line.startswith(HANDSHAKE_PREFIX)
    payload = json.loads(line.removeprefix(HANDSHAKE_PREFIX))
    assert payload == handshake_payload(43123)
    assert payload["protocol"] == 1
    assert payload["apiBaseUrl"] == "http://127.0.0.1:43123"
    assert "bootstrapToken" not in payload
    assert "data" not in payload


def test_sidecar_process_reports_ready_and_enforces_parent_token(tmp_path):
    token = "test-parent-token-which-is-at-least-thirty-two-bytes"
    environment = os.environ.copy()
    environment.pop("INTERVIEW_OS_BOOTSTRAP_TOKEN", None)
    environment.pop("DATABASE_URL", None)
    environment.pop("INTERVIEW_OS_SETTINGS_PATH", None)
    environment.pop("INTERVIEW_OS_RECORDINGS_DIR", None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "interview_os.desktop.sidecar",
            "--data-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
    )
    lines: Queue[str] = Queue()
    reader = Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True)
    reader.start()
    try:
        process.stdin.write(f"{BOOTSTRAP_STDIN_PREFIX}{token}\n")
        process.stdin.flush()
        try:
            line = lines.get(timeout=60)
        except Empty:
            pytest.fail("desktop sidecar did not report readiness within 60 seconds")
        assert line.startswith(HANDSHAKE_PREFIX)
        payload = json.loads(line.removeprefix(HANDSHAKE_PREFIX))
        assert payload["status"] == "ready"
        assert "bootstrapToken" not in payload

        assert httpx.get(f"{payload['apiBaseUrl']}/health", timeout=2).status_code == 403
        response = httpx.get(
            f"{payload['apiBaseUrl']}/health",
            headers={"X-InterviewOS-Bootstrap": token},
            timeout=2,
        )
        assert response.json() == {"status": "ok"}
        assert (tmp_path / "interview_os.db").is_file()
    finally:
        if process.poll() is None:
            process.stdin.write(f"{SHUTDOWN_STDIN_COMMAND}\n")
            process.stdin.flush()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
    assert process.returncode == 0
    stderr = process.stderr.read()
    assert token not in stderr
