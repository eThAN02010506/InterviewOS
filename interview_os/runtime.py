"""Runtime paths shared by browser-server and future desktop launchers."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DESKTOP_MODE = "desktop"


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved writable locations without coupling services to the process cwd.

    Server mode deliberately preserves the project's historical locations. A
    desktop launcher opts into OS application storage by setting
    ``INTERVIEW_OS_RUNTIME_MODE=desktop`` (or an explicit data directory).
    """

    mode: str
    data_dir: Path
    database_path: Path
    settings_path: Path
    recordings_dir: Path

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.database_path.as_posix()}"


def platform_data_dir(
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the conventional per-user application directory for the OS."""

    platform = platform or sys.platform
    environ = os.environ if environ is None else environ
    home = home or Path.home()
    if platform == "darwin":
        return home / "Library" / "Application Support" / "InterviewOS"
    if platform.startswith("win"):
        local_app_data = environ.get("LOCALAPPDATA")
        return Path(local_app_data) / "InterviewOS" if local_app_data else home / "InterviewOS"
    xdg_data_home = environ.get("XDG_DATA_HOME")
    return (Path(xdg_data_home) if xdg_data_home else home / ".local" / "share") / "interview-os"


def resolve_runtime_paths(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
) -> RuntimePaths:
    """Resolve storage paths for the current launch mode.

    An explicit ``INTERVIEW_OS_DATA_DIR`` implies desktop-style consolidated
    storage. Without it, normal server launches retain legacy paths so existing
    databases and settings continue to appear unchanged.
    """

    environ = os.environ if environ is None else environ
    home = home or Path.home()
    cwd = cwd or Path.cwd()
    configured_data_dir = environ.get("INTERVIEW_OS_DATA_DIR", "").strip()
    requested_mode = environ.get("INTERVIEW_OS_RUNTIME_MODE", "server").strip().lower()
    desktop = requested_mode == DESKTOP_MODE or bool(configured_data_dir)

    if desktop:
        data_dir = (
            Path(configured_data_dir).expanduser().resolve()
            if configured_data_dir
            else platform_data_dir(platform=platform, environ=environ, home=home)
        )
        database_path = data_dir / "interview_os.db"
        settings_path = data_dir / "settings.json"
        recordings_dir = data_dir / "recordings"
        mode = DESKTOP_MODE
    else:
        data_dir = home / ".interview_os"
        database_path = cwd / "interview_os.db"
        settings_path = data_dir / "settings.json"
        recordings_dir = cwd / "data" / "recordings"
        mode = "server"

    configured_settings = environ.get("INTERVIEW_OS_SETTINGS_PATH", "").strip()
    configured_recordings = environ.get("INTERVIEW_OS_RECORDINGS_DIR", "").strip()
    if configured_settings:
        settings_path = Path(configured_settings).expanduser()
    if configured_recordings:
        recordings_dir = Path(configured_recordings).expanduser()

    return RuntimePaths(
        mode=mode,
        data_dir=data_dir,
        database_path=database_path,
        settings_path=settings_path,
        recordings_dir=recordings_dir,
    )
