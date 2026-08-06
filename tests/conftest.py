"""Shared test fixtures.

``test_settings_isolation`` points the default settings store at a temp
directory so tests never read or write the user's real
``~/.interview_os/settings.json`` (e.g. a persisted live-audio mode would
otherwise silently flip an ASR test onto the omni path).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERVIEW_OS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    yield
