import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_desktop_sidecar import build_command, validate_python_architecture


def test_pyinstaller_command_names_tauri_target_binary(tmp_path):
    command = build_command(tmp_path, "aarch64-apple-darwin")

    assert command[:3] == [__import__("sys").executable, "-m", "PyInstaller"]
    assert command[command.index("--name") + 1] == (
        "interview-os-sidecar-aarch64-apple-darwin"
    )
    assert "--onefile" in command
    # Project imports are explicit. Avoid a broad package crawl: it executes
    # PyInstaller hook discovery for every module and can hang on build hosts.
    assert "--collect-submodules" not in command
    assert "uvicorn.lifespan.on" in command
    assert "aiosqlite" in command
    assert str(Path("scripts/desktop_sidecar_entry.py")) in command[-1]


def test_pyinstaller_signs_inner_binary_before_tauri_on_macos(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.build_desktop_sidecar.sys.platform", "darwin")
    command = build_command(tmp_path, "aarch64-apple-darwin", "Developer ID Application: Test")

    assert command[command.index("--codesign-identity") + 1] == (
        "Developer ID Application: Test"
    )


def test_sidecar_build_rejects_python_rust_architecture_mismatch():
    validate_python_architecture("aarch64-apple-darwin", "arm64")
    validate_python_architecture("x86_64-pc-windows-msvc", "AMD64")
    with pytest.raises(RuntimeError, match="does not match"):
        validate_python_architecture("aarch64-apple-darwin", "x86_64")


def test_desktop_build_always_rebuilds_and_smokes_the_sidecar():
    package = json.loads(Path("desktop/package.json").read_text(encoding="utf-8"))
    scripts = package["scripts"]

    assert "build_desktop_sidecar.py" in scripts["sidecar:build"]
    assert "smoke_desktop_sidecar.py" in scripts["sidecar:smoke"]
    assert "sidecar:build" in scripts["prepare:release"]
    assert "sidecar:smoke" in scripts["prepare:release"]
    assert "release:guard" in scripts["prepare:release"]
    assert "sidecar:build" in scripts["prepare:local"]
    assert "sidecar:smoke" in scripts["prepare:local"]
    assert "local:guard" in scripts["prepare:local"]
    assert "tauri.local.conf.json" in scripts["build:local"]

    release_config = json.loads(Path("desktop/tauri.conf.json").read_text(encoding="utf-8"))
    local_config = json.loads(
        Path("desktop/tauri.local.conf.json").read_text(encoding="utf-8")
    )
    assert release_config["build"]["beforeBuildCommand"] == "npm run prepare:release"
    assert release_config["build"]["beforeDevCommand"] == "npm run sidecar:build"
    assert local_config["build"]["beforeBuildCommand"] == "npm run prepare:local"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_desktop_python_runner_honors_explicit_build_interpreter(tmp_path):
    fake_python = tmp_path / "controlled-python"
    fake_python.write_text(
        "#!/bin/sh\nprintf 'runner=%s\\narg=%s\\n' \"$0\" \"$1\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["INTERVIEW_OS_BUILD_PYTHON"] = str(fake_python)

    result = subprocess.run(
        ["node", "desktop/scripts/run-python.mjs", "fake-script.py"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert f"runner={fake_python}" in result.stdout
    assert "fake-script.py" in result.stdout


def test_macos_bundle_declares_local_network_and_microphone_usage():
    with Path("desktop/Info.plist").open("rb") as file:
        info = plistlib.load(file)

    transport = info["NSAppTransportSecurity"]
    assert transport["NSAllowsLocalNetworking"] is True
    assert "NSAllowsArbitraryLoads" not in transport
    assert "local AI services" in info["NSLocalNetworkUsageDescription"]
    assert info["NSMicrophoneUsageDescription"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS build guard")
def test_local_adhoc_guard_rejects_release_credentials():
    environment = os.environ.copy()
    environment["APPLE_SIGNING_IDENTITY"] = "Developer ID Application: Example"
    result = subprocess.run(
        ["node", "desktop/scripts/require-local-adhoc.mjs"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Local build refused" in result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS build guard")
def test_release_guard_rejects_non_distribution_identity():
    environment = os.environ.copy()
    environment["APPLE_SIGNING_IDENTITY"] = "bogus"
    result = subprocess.run(
        ["node", "desktop/scripts/require-release-signing.mjs"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Developer ID Application" in result.stderr


def test_macos_hardened_runtime_allows_microphone_capture():
    config = json.loads(Path("desktop/tauri.conf.json").read_text(encoding="utf-8"))
    macos = config["bundle"]["macOS"]
    with Path("desktop/Entitlements.plist").open("rb") as file:
        entitlements = plistlib.load(file)

    assert macos["entitlements"] == "Entitlements.plist"
    assert entitlements["com.apple.security.device.audio-input"] is True
    assert "com.apple.security.cs.disable-library-validation" not in entitlements


def test_macos_adhoc_library_exception_is_confined_to_local_build():
    local_config = json.loads(
        Path("desktop/tauri.local.conf.json").read_text(encoding="utf-8")
    )
    with Path("desktop/Entitlements.local-adhoc.plist").open("rb") as file:
        entitlements = plistlib.load(file)

    assert local_config["bundle"]["macOS"]["signingIdentity"] == "-"
    assert local_config["bundle"]["macOS"]["entitlements"] == (
        "Entitlements.local-adhoc.plist"
    )
    assert entitlements["com.apple.security.cs.disable-library-validation"] is True
