"""Run the FastAPI application as a private desktop sidecar."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import uvicorn

HANDSHAKE_PROTOCOL = 1
HANDSHAKE_PREFIX = "INTERVIEW_OS_HANDSHAKE "
BOOTSTRAP_STDIN_PREFIX = "INTERVIEW_OS_BOOTSTRAP "
SHUTDOWN_STDIN_COMMAND = "INTERVIEW_OS_SHUTDOWN 1"
DEFAULT_HOST = "127.0.0.1"
FORCED_EXIT_TIMEOUT_SECONDS = 20


class ShutdownController:
    """Coordinate stdin shutdown even while the application is importing."""

    def __init__(self) -> None:
        self.requested = Event()
        self.completed = Event()
        self._server: uvicorn.Server | None = None
        self._lock = Lock()

    def attach(self, server: uvicorn.Server) -> None:
        with self._lock:
            self._server = server
            if self.requested.is_set():
                server.should_exit = True

    def request(self) -> None:
        self.requested.set()
        with self._lock:
            if self._server is not None:
                self._server.should_exit = True

    def finish(self) -> None:
        self.completed.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private InterviewOS desktop service")
    parser.add_argument("--port", type=int, default=0, help="loopback port; 0 selects a free port")
    parser.add_argument("--data-dir", type=Path, help="override the desktop application data directory")
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info"),
        default="warning",
    )
    return parser


def reserve_loopback_socket(port: int = 0) -> socket.socket:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind((DEFAULT_HOST, port))
        listener.listen(2048)
        listener.set_inheritable(False)
    except Exception:
        listener.close()
        raise
    return listener


def read_bootstrap_token(stream: Any = None) -> str:
    """Read the parent-generated secret from a private stdin pipe."""

    stream = stream or sys.stdin
    line = stream.readline(256)
    if not line.startswith(BOOTSTRAP_STDIN_PREFIX):
        raise ValueError("desktop bootstrap handshake is missing")
    token = line.removeprefix(BOOTSTRAP_STDIN_PREFIX).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token):
        raise ValueError("desktop bootstrap token is invalid")
    return token


def configure_desktop_environment(data_dir: Path | None, bootstrap_token: str) -> None:
    """Set private process state before the FastAPI module is imported."""

    os.environ["INTERVIEW_OS_RUNTIME_MODE"] = "desktop"
    os.environ["INTERVIEW_OS_BOOTSTRAP_TOKEN"] = bootstrap_token
    # A packaged desktop process must not inherit a development-only auth
    # override from its parent shell.
    os.environ["INTERVIEW_OS_REQUIRE_AUTH"] = "1"
    # Server-mode path overrides must never redirect a packaged application's
    # private database, settings, or recordings outside its OS app directory.
    for variable in (
        "DATABASE_URL",
        "INTERVIEW_OS_DATA_DIR",
        "INTERVIEW_OS_SETTINGS_PATH",
        "INTERVIEW_OS_RECORDINGS_DIR",
    ):
        os.environ.pop(variable, None)
    if data_dir is not None:
        os.environ["INTERVIEW_OS_DATA_DIR"] = str(data_dir.expanduser().resolve())


def handshake_payload(port: int) -> dict[str, Any]:
    return {
        "protocol": HANDSHAKE_PROTOCOL,
        "status": "ready",
        "apiBaseUrl": f"http://{DEFAULT_HOST}:{port}",
        "mode": "desktop",
    }


def emit_handshake(port: int) -> None:
    # stdout is a private one-line parent-process protocol. Uvicorn logs stay
    # on stderr, and neither stream contains candidate data or provider keys.
    payload = json.dumps(handshake_payload(port), separators=(",", ":"))
    print(f"{HANDSHAKE_PREFIX}{payload}", flush=True)


def watch_parent(stream: Any, controller: ShutdownController) -> None:
    """Request shutdown on a parent command/EOF, then enforce a total deadline."""

    for line in stream:
        if line.strip() == SHUTDOWN_STDIN_COMMAND:
            break
    controller.request()
    if not controller.completed.wait(FORCED_EXIT_TIMEOUT_SECONDS):
        # The Rust parent may only be able to kill PyInstaller's one-file
        # bootloader. Enforce the final deadline inside the real Python child.
        os._exit(1)


async def serve(args: argparse.Namespace, bootstrap_token: str, parent_stream: Any = None) -> None:
    configure_desktop_environment(args.data_dir, bootstrap_token)
    listener = reserve_loopback_socket(args.port)
    port = int(listener.getsockname()[1])
    parent_stream = parent_stream or sys.stdin
    shutdown = ShutdownController()
    Thread(target=watch_parent, args=(parent_stream, shutdown), daemon=True).start()

    # Import only after desktop environment variables are set. app.py creates
    # its module-level ASGI application at import time.
    from interview_os.api import app as application_module

    application = application_module.app
    # create_app captured the token in middleware; remove the environment copy
    # once startup configuration is complete.
    os.environ.pop("INTERVIEW_OS_BOOTSTRAP_TOKEN", None)
    config = uvicorn.Config(
        application,
        host=DEFAULT_HOST,
        port=port,
        log_level=args.log_level,
        access_log=False,
        loop="asyncio",
        http="h11",
        ws="none",
        lifespan="on",
        workers=1,
        proxy_headers=False,
        server_header=False,
        timeout_graceful_shutdown=15,
    )
    server = uvicorn.Server(config)
    shutdown.attach(server)
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("desktop service stopped before readiness")
            await asyncio.sleep(0.01)
        emit_handshake(port)
        await server_task
    finally:
        listener.close()
        if not server_task.done():
            server.should_exit = True
            await server_task
        shutdown.finish()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bootstrap_token = read_bootstrap_token()
        asyncio.run(serve(args, bootstrap_token, sys.stdin))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary must return a clean failure
        # Do not print exception text: provider URLs and local paths can be
        # embedded in third-party error messages.
        print(f"InterviewOS desktop service failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
