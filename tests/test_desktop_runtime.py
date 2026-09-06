from pathlib import Path

from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.database.storage import Storage
from interview_os.runtime import platform_data_dir, resolve_runtime_paths
from interview_os.services.settings_service import LocalSettingsStore


def test_server_paths_preserve_existing_project_layout():
    paths = resolve_runtime_paths(
        environ={},
        platform="darwin",
        home=Path("/Users/tester"),
        cwd=Path("/workspace/interview-os"),
    )

    assert paths.mode == "server"
    assert paths.database_path == Path("/workspace/interview-os/interview_os.db")
    assert paths.settings_path == Path("/Users/tester/.interview_os/settings.json")
    assert paths.recordings_dir == Path("/workspace/interview-os/data/recordings")


def test_desktop_paths_follow_each_platform_convention():
    assert platform_data_dir(platform="darwin", environ={}, home=Path("/Users/tester")) == (
        Path("/Users/tester/Library/Application Support/InterviewOS")
    )
    assert platform_data_dir(
        platform="win32",
        environ={"LOCALAPPDATA": "/Users/tester/AppData/Local"},
        home=Path("/Users/tester"),
    ) == Path("/Users/tester/AppData/Local/InterviewOS")
    assert platform_data_dir(
        platform="linux",
        environ={"XDG_DATA_HOME": "/home/tester/.data"},
        home=Path("/home/tester"),
    ) == Path("/home/tester/.data/interview-os")


def test_explicit_data_directory_consolidates_desktop_state(tmp_path):
    data_dir = tmp_path / "desktop-data"
    paths = resolve_runtime_paths(
        environ={"INTERVIEW_OS_DATA_DIR": str(data_dir)},
        home=tmp_path,
        cwd=tmp_path / "checkout",
    )

    assert paths.mode == "desktop"
    assert paths.data_dir == data_dir
    assert paths.database_path == data_dir / "interview_os.db"
    assert paths.settings_path == data_dir / "settings.json"
    assert paths.recordings_dir == data_dir / "recordings"
    assert paths.database_url.endswith("/desktop-data/interview_os.db")


def test_optional_bootstrap_token_protects_sidecar_api(monkeypatch, tmp_path):
    monkeypatch.setenv("INTERVIEW_OS_RUNTIME_MODE", "desktop")
    monkeypatch.setenv("INTERVIEW_OS_DATA_DIR", str(tmp_path / "desktop"))
    monkeypatch.setenv("INTERVIEW_OS_BOOTSTRAP_TOKEN", "one-time-secret")
    app = create_app(
        storage=Storage(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"),
        llm_client=object(),
        configure_llm=False,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
        require_auth=False,
    )

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 403
        assert client.get(
            "/health", headers={"X-InterviewOS-Bootstrap": "one-time-secret"}
        ).status_code == 200
        preflight = client.options(
            "/api/interviews/sessions",
            headers={
                "Origin": "http://tauri.localhost",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-InterviewOS-Bootstrap",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://tauri.localhost"
