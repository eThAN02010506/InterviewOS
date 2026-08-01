"""Permission-restricted, atomic storage for local provider settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class LocalSettingsStore:
    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv("INTERVIEW_OS_SETTINGS_PATH")
        self.path = (
            Path(configured).expanduser()
            if configured
            else Path.home() / ".interview_os" / "settings.json"
        )

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        os.chmod(self.path, 0o600)
