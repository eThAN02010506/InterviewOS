"""Build the current platform's Python service as a Tauri sidecar binary."""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "desktop" / "binaries"


def target_triple() -> str:
    result = subprocess.run(
        ["rustc", "--print", "host-tuple"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError("rustc returned an invalid target triple")
    return value


def validate_python_architecture(triple: str, machine: str | None = None) -> None:
    """Prevent a translated Python binary from being mislabeled for native Tauri."""

    actual = (machine or platform.machine()).lower()
    actual = {"arm64": "aarch64", "amd64": "x86_64"}.get(actual, actual)
    expected = triple.split("-", 1)[0].lower()
    if actual != expected:
        raise RuntimeError(
            f"Python architecture {actual!r} does not match Rust target {expected!r}"
        )


def build_command(output_dir: Path, triple: str, signing_identity: str = "") -> list[str]:
    executable_suffix = ".exe" if sys.platform == "win32" else ""
    binary_name = f"interview-os-sidecar-{triple}{executable_suffix}"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--name",
        binary_name.removesuffix(executable_suffix),
        "--distpath",
        str(output_dir),
        "--workpath",
        str(ROOT / "build" / "desktop" / triple),
        "--specpath",
        str(ROOT / "build" / "desktop" / "spec"),
        "--paths",
        str(ROOT),
        "--add-data",
        f"{ROOT / 'interview_os' / 'web'}{os.pathsep}interview_os/web",
        "--hidden-import",
        "uvicorn.loops.asyncio",
        "--hidden-import",
        "uvicorn.protocols.http.h11_impl",
        "--hidden-import",
        "uvicorn.lifespan.on",
        "--hidden-import",
        "sqlalchemy.dialects.sqlite.aiosqlite",
        "--hidden-import",
        "aiosqlite",
    ]
    if signing_identity and sys.platform == "darwin":
        command.extend(["--codesign-identity", signing_identity])
    command.append(str(ROOT / "scripts" / "desktop_sidecar_entry.py"))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--signing-identity",
        default=os.getenv("APPLE_SIGNING_IDENTITY", ""),
        help="macOS identity passed to PyInstaller before Tauri signs the outer app",
    )
    args = parser.parse_args()
    triple = target_triple()
    validate_python_architecture(triple)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Keep PyInstaller's mutable cache inside the ignored build tree. This
    # makes sandboxed/CI builds independent of the user's home directory.
    environment["PYINSTALLER_CONFIG_DIR"] = str(ROOT / "build" / "desktop" / "pyinstaller")
    subprocess.run(
        build_command(args.output_dir.resolve(), triple, args.signing_identity),
        cwd=ROOT,
        env=environment,
        check=True,
    )
    suffix = ".exe" if sys.platform == "win32" else ""
    binary = args.output_dir / f"interview-os-sidecar-{triple}{suffix}"
    if not binary.is_file():
        raise RuntimeError("PyInstaller completed without the expected sidecar binary")
    print(f"Built desktop sidecar: {binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
