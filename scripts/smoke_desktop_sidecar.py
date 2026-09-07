"""Smoke-test the frozen desktop sidecar handshake and shutdown contract."""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interview_os.desktop.sidecar import (
    BOOTSTRAP_STDIN_PREFIX,
    HANDSHAKE_PREFIX,
    SHUTDOWN_STDIN_COMMAND,
)
from scripts.build_desktop_sidecar import DEFAULT_OUTPUT, target_triple


def default_binary() -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return DEFAULT_OUTPUT / f"interview-os-sidecar-{target_triple()}{suffix}"


def smoke(binary: Path, timeout: float) -> None:
    if not binary.is_file():
        raise FileNotFoundError(f"desktop sidecar not found: {binary}")

    token = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix="interview-os-desktop-smoke-") as data_dir:
        process = subprocess.Popen(
            [str(binary), "--data-dir", data_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        stdin = process.stdin
        stdout = process.stdout
        stderr_pipe = process.stderr
        lines: Queue[str] = Queue()
        Thread(target=lambda: lines.put(stdout.readline()), daemon=True).start()
        try:
            stdin.write(f"{BOOTSTRAP_STDIN_PREFIX}{token}\n")
            stdin.flush()
            try:
                line = lines.get(timeout=timeout)
            except Empty as exc:
                raise RuntimeError("desktop sidecar readiness timed out") from exc
            if not line.startswith(HANDSHAKE_PREFIX):
                raise RuntimeError("desktop sidecar returned an invalid handshake")
            payload = json.loads(line.removeprefix(HANDSHAKE_PREFIX))
            base_url = payload["apiBaseUrl"]
            if httpx.get(f"{base_url}/health", timeout=5).status_code != 403:
                raise RuntimeError("desktop sidecar accepted a request without its parent token")
            authorized = httpx.get(
                f"{base_url}/health",
                headers={"X-InterviewOS-Bootstrap": token},
                timeout=5,
            )
            if authorized.json() != {"status": "ok"}:
                raise RuntimeError("desktop sidecar health check failed")
            index = httpx.get(f"{base_url}/", timeout=5)
            runtime_module = httpx.get(f"{base_url}/modules/runtime.js", timeout=5)
            if index.status_code != 200 or "InterviewOS" not in index.text:
                raise RuntimeError("desktop sidecar is missing the packaged web application")
            if runtime_module.status_code != 200 or "loadRuntimeConfig" not in runtime_module.text:
                raise RuntimeError("desktop sidecar is missing packaged frontend modules")
            if not (Path(data_dir) / "interview_os.db").is_file():
                raise RuntimeError("desktop sidecar did not use its private data directory")
        finally:
            if process.poll() is None:
                stdin.write(f"{SHUTDOWN_STDIN_COMMAND}\n")
                stdin.flush()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

        stderr = stderr_pipe.read()
        if process.returncode != 0:
            raise RuntimeError(f"desktop sidecar exited with code {process.returncode}")
        if token in stderr:
            raise RuntimeError("desktop sidecar leaked its bootstrap token")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    binary = args.binary or default_binary()
    smoke(binary.resolve(), args.timeout)
    print(f"Desktop sidecar smoke passed: {binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
