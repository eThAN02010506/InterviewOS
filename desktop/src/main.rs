use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{webview::PageLoadEvent, Emitter, Manager, RunEvent, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

const HANDSHAKE_PROTOCOL: u8 = 1;
const HANDSHAKE_PREFIX: &str = "INTERVIEW_OS_HANDSHAKE ";
const BOOTSTRAP_STDIN_PREFIX: &str = "INTERVIEW_OS_BOOTSTRAP ";
const SHUTDOWN_STDIN_COMMAND: &str = "INTERVIEW_OS_SHUTDOWN 1\n";
const SIDECAR_NAME: &str = "interview-os-sidecar";

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FrontendRuntime {
    api_base_url: String,
    bootstrap_token: String,
    mode: &'static str,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SidecarHandshake {
    protocol: u8,
    status: String,
    api_base_url: String,
    mode: String,
}

#[derive(Clone, Default)]
enum LaunchStatus {
    #[default]
    Starting,
    Ready(FrontendRuntime),
    Failed(&'static str),
}

#[derive(Default)]
struct DesktopState {
    status: Mutex<LaunchStatus>,
    child: Mutex<Option<CommandChild>>,
    exiting: AtomicBool,
    renderer_document_generation: AtomicU64,
    renderer_shutdown_bridge_generation: AtomicU64,
    renderer_shutdown_requested: AtomicBool,
    renderer_ready_to_exit: AtomicBool,
    shutdown_generation: AtomicU64,
    requested_exit_code: Mutex<i32>,
    shutdown_complete: AtomicBool,
    terminated: AtomicBool,
}

fn new_bootstrap_token() -> Result<String, &'static str> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|_| "could not initialize desktop security")?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn parse_handshake(line: &str, token: &str) -> Result<FrontendRuntime, &'static str> {
    let payload = line
        .strip_prefix(HANDSHAKE_PREFIX)
        .ok_or("desktop service returned an invalid handshake")?;
    let handshake: SidecarHandshake = serde_json::from_str(payload)
        .map_err(|_| "desktop service returned an invalid handshake")?;
    let port = handshake
        .api_base_url
        .strip_prefix("http://127.0.0.1:")
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|value| *value > 0);
    if handshake.protocol != HANDSHAKE_PROTOCOL
        || handshake.status != "ready"
        || handshake.mode != "desktop"
        || port.is_none()
    {
        return Err("desktop service returned an invalid handshake");
    }
    Ok(FrontendRuntime {
        api_base_url: handshake.api_base_url,
        bootstrap_token: token.to_owned(),
        mode: "desktop",
    })
}

fn set_failed(state: &DesktopState, message: &'static str) {
    if let Ok(mut status) = state.status.lock() {
        if !matches!(*status, LaunchStatus::Ready(_)) {
            *status = LaunchStatus::Failed(message);
        }
    }
}

fn publish_ready_if_starting(state: &DesktopState, runtime: FrontendRuntime) -> bool {
    let Ok(mut status) = state.status.lock() else {
        return false;
    };
    if matches!(*status, LaunchStatus::Starting) {
        *status = LaunchStatus::Ready(runtime);
        true
    } else {
        false
    }
}

fn claim_startup_timeout(state: &DesktopState) -> Result<bool, &'static str> {
    let mut status = state
        .status
        .lock()
        .map_err(|_| "desktop service state is unavailable")?;
    match &*status {
        LaunchStatus::Ready(_) | LaunchStatus::Failed(_) => Ok(false),
        LaunchStatus::Starting => {
            *status = LaunchStatus::Failed("desktop service startup timed out");
            Ok(true)
        }
    }
}

fn begin_renderer_document(state: &DesktopState) -> bool {
    // A bridge listener belongs to one JavaScript document, not to the desktop
    // process. Advancing the generation invalidates the previous document's
    // acknowledgement before a reload or navigation can install a new one.
    state
        .renderer_document_generation
        .fetch_add(1, Ordering::AcqRel);
    state
        .renderer_shutdown_bridge_generation
        .store(0, Ordering::Release);
    state.renderer_ready_to_exit.store(false, Ordering::Release);
    state.renderer_shutdown_requested.load(Ordering::Acquire)
}

fn acknowledge_renderer_shutdown_bridge(state: &DesktopState) -> bool {
    let generation = state.renderer_document_generation.load(Ordering::Acquire);
    if generation == 0 {
        return false;
    }
    state
        .renderer_shutdown_bridge_generation
        .store(generation, Ordering::Release);
    true
}

fn renderer_bridge_ready_for_current_document(state: &DesktopState) -> bool {
    let generation = state.renderer_document_generation.load(Ordering::Acquire);
    generation != 0
        && state
            .renderer_shutdown_bridge_generation
            .load(Ordering::Acquire)
            == generation
}

fn resume_exit_after_document_loss<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    state: &DesktopState,
) {
    if !begin_renderer_document(state) {
        return;
    }

    // If navigation or a renderer crash destroys the document while its
    // cleanup handshake is pending, that document can no longer answer. Resume
    // native shutdown immediately instead of waiting for the 150s watchdog.
    state.renderer_ready_to_exit.store(true, Ordering::Release);
    let code = state
        .requested_exit_code
        .lock()
        .map(|value| *value)
        .unwrap_or(0);
    app.exit(code);
}

fn process_failure_requires_exit(state: &DesktopState, message: &'static str) -> bool {
    if state.exiting.load(Ordering::Acquire) {
        return false;
    }
    if let Ok(mut status) = state.status.lock() {
        let renderer_has_credentials = matches!(*status, LaunchStatus::Ready(_));
        *status = LaunchStatus::Failed(message);
        return renderer_has_credentials;
    }
    true
}

#[tauri::command]
async fn desktop_runtime_config(
    state: State<'_, DesktopState>,
) -> Result<FrontendRuntime, &'static str> {
    // A signed one-file Python binary can spend noticeable time unpacking on
    // first launch (especially while endpoint protection scans it).
    for _ in 0..1800 {
        let status = state
            .status
            .lock()
            .map_err(|_| "desktop service state is unavailable")?
            .clone();
        match status {
            LaunchStatus::Ready(runtime) => return Ok(runtime),
            LaunchStatus::Failed(message) => return Err(message),
            LaunchStatus::Starting => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
    // Atomically claim the Starting -> Failed transition. A handshake that won
    // the same status lock first remains Ready; a late handshake cannot revive
    // a process after this timeout claimant starts terminating it.
    if claim_startup_timeout(&state)? {
        if let Ok(mut child) = state.child.lock() {
            if let Some(process) = child.take() {
                let _ = process.kill();
            }
        }
        return Err("desktop service startup timed out");
    }
    match state
        .status
        .lock()
        .map_err(|_| "desktop service state is unavailable")?
        .clone()
    {
        LaunchStatus::Ready(runtime) => Ok(runtime),
        LaunchStatus::Failed(message) => Err(message),
        LaunchStatus::Starting => Err("desktop service state is unavailable"),
    }
}

#[tauri::command]
fn desktop_shutdown_bridge_ready(state: State<'_, DesktopState>) {
    acknowledge_renderer_shutdown_bridge(&state);
}

#[tauri::command]
fn desktop_renderer_ready_to_exit(app: tauri::AppHandle, state: State<'_, DesktopState>) {
    if !state.renderer_shutdown_requested.load(Ordering::Acquire)
        || !renderer_bridge_ready_for_current_document(&state)
    {
        return;
    }
    state.renderer_ready_to_exit.store(true, Ordering::Release);
    let code = state
        .requested_exit_code
        .lock()
        .map(|value| *value)
        .unwrap_or(0);
    app.exit(code);
}

fn cancel_renderer_shutdown(state: &DesktopState) {
    state
        .renderer_shutdown_requested
        .store(false, Ordering::Release);
    state.renderer_ready_to_exit.store(false, Ordering::Release);
    state.shutdown_generation.fetch_add(1, Ordering::AcqRel);
    if let Ok(mut code) = state.requested_exit_code.lock() {
        *code = 0;
    }
}

#[tauri::command]
fn desktop_renderer_cancel_exit(state: State<'_, DesktopState>) {
    cancel_renderer_shutdown(&state);
}

fn main() {
    let application_builder = tauri::Builder::default()
        // This must be the first plugin: two sidecars cannot safely share the
        // SQLite database and in-process workflow locks.
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .manage(DesktopState::default())
        .invoke_handler(tauri::generate_handler![
            desktop_runtime_config,
            desktop_shutdown_bridge_ready,
            desktop_renderer_ready_to_exit,
            desktop_renderer_cancel_exit
        ])
        .on_page_load(|webview, payload| {
            if webview.label() == "main" && payload.event() == PageLoadEvent::Started {
                let state = webview.state::<DesktopState>();
                resume_exit_after_document_loss(webview.app_handle(), &state);
            }
        });

    #[cfg(target_os = "macos")]
    let application_builder = application_builder.on_web_content_process_terminate(|webview| {
        if webview.label() == "main" {
            let state = webview.state::<DesktopState>();
            resume_exit_after_document_loss(webview.app_handle(), &state);
        }
    });

    let application = application_builder
        .setup(|app| {
            let bootstrap_token = match new_bootstrap_token() {
                Ok(token) => token,
                Err(message) => {
                    set_failed(&app.state::<DesktopState>(), message);
                    return Ok(());
                }
            };
            // Keep large/local interview artifacts out of Windows roaming
            // profiles. On macOS/Linux Tauri maps this to the normal app-data
            // location; the bundle identifier keeps desktop data isolated.
            let data_dir = match app.path().app_local_data_dir() {
                Ok(path) => path,
                Err(_) => {
                    set_failed(
                        &app.state::<DesktopState>(),
                        "desktop application data directory is unavailable",
                    );
                    return Ok(());
                }
            };
            let command = match app.shell().sidecar(SIDECAR_NAME) {
                Ok(command) => command.arg("--data-dir").arg(data_dir),
                Err(_) => {
                    set_failed(
                        &app.state::<DesktopState>(),
                        "desktop service executable is unavailable",
                    );
                    return Ok(());
                }
            };
            let (mut events, mut child) = match command.spawn() {
                Ok(process) => process,
                Err(_) => {
                    set_failed(
                        &app.state::<DesktopState>(),
                        "could not start InterviewOS desktop service",
                    );
                    return Ok(());
                }
            };
            if child
                .write(format!("{BOOTSTRAP_STDIN_PREFIX}{bootstrap_token}\n").as_bytes())
                .is_err()
            {
                let _ = child.kill();
                set_failed(
                    &app.state::<DesktopState>(),
                    "could not initialize InterviewOS desktop service",
                );
                return Ok(());
            }
            let state = app.state::<DesktopState>();
            match state.child.lock() {
                Ok(mut stored_child) => {
                    stored_child.replace(child);
                }
                Err(_) => {
                    let _ = child.kill();
                    set_failed(&state, "desktop service state is unavailable");
                    return Ok(());
                }
            }

            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = events.recv().await {
                    match event {
                        CommandEvent::Stdout(bytes) => {
                            let line = String::from_utf8_lossy(&bytes);
                            let line = line.trim();
                            if !line.starts_with(HANDSHAKE_PREFIX) {
                                continue;
                            }
                            match parse_handshake(line, &bootstrap_token) {
                                Ok(runtime) => {
                                    publish_ready_if_starting(
                                        &handle.state::<DesktopState>(),
                                        runtime,
                                    );
                                }
                                Err(message) => {
                                    set_failed(&handle.state::<DesktopState>(), message)
                                }
                            }
                        }
                        CommandEvent::Terminated(_) => {
                            let state = handle.state::<DesktopState>();
                            state.terminated.store(true, Ordering::Release);
                            if process_failure_requires_exit(
                                &state,
                                "desktop service stopped unexpectedly",
                            ) {
                                // Once the renderer has received credentials, keeping it
                                // alive after the port is released could leak those headers
                                // to a local process that rebinds the same port.
                                state.renderer_ready_to_exit.store(true, Ordering::Release);
                                if let Ok(mut code) = state.requested_exit_code.lock() {
                                    *code = 1;
                                }
                                handle.exit(1);
                            }
                        }
                        CommandEvent::Error(_) => {
                            if process_failure_requires_exit(
                                &handle.state::<DesktopState>(),
                                "desktop service communication failed",
                            ) {
                                let state = handle.state::<DesktopState>();
                                state.renderer_ready_to_exit.store(true, Ordering::Release);
                                if let Ok(mut code) = state.requested_exit_code.lock() {
                                    *code = 1;
                                }
                                handle.exit(1);
                            }
                        }
                        CommandEvent::Stderr(_) => {
                            // Intentionally do not forward sidecar output: third-party
                            // exceptions can contain local paths or provider details.
                        }
                        _ => {}
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build InterviewOS desktop application");

    application.run(|handle, event| match event {
        RunEvent::ExitRequested { api, code, .. } => {
            let state = handle.state::<DesktopState>();
            if state.shutdown_complete.load(Ordering::Acquire) {
                return;
            }
            let renderer_can_finalize = state
                .status
                .lock()
                .map(|status| matches!(*status, LaunchStatus::Ready(_)))
                .unwrap_or(false)
                && renderer_bridge_ready_for_current_document(&state);
            if renderer_can_finalize && !state.renderer_ready_to_exit.load(Ordering::Acquire) {
                api.prevent_exit();
                if !state
                    .renderer_shutdown_requested
                    .swap(true, Ordering::AcqRel)
                {
                    let shutdown_generation =
                        state.shutdown_generation.fetch_add(1, Ordering::AcqRel) + 1;
                    let exit_code = code.unwrap_or(0);
                    if let Ok(mut requested_code) = state.requested_exit_code.lock() {
                        *requested_code = exit_code;
                    }
                    if handle.emit("desktop-shutdown-requested", ()).is_err() {
                        state.renderer_ready_to_exit.store(true, Ordering::Release);
                        handle.exit(exit_code);
                    } else {
                        let app = handle.clone();
                        tauri::async_runtime::spawn(async move {
                            // The renderer can spend up to 30s reconciling an
                            // in-flight transition and 90s draining audio. Keep
                            // a safety margin so this watchdog never wins a
                            // legitimate boundary race and forces data loss.
                            tokio::time::sleep(Duration::from_secs(150)).await;
                            let state = app.state::<DesktopState>();
                            let same_request = state.shutdown_generation.load(Ordering::Acquire)
                                == shutdown_generation;
                            if same_request
                                && state.renderer_shutdown_requested.load(Ordering::Acquire)
                                && !state.renderer_ready_to_exit.swap(true, Ordering::AcqRel)
                            {
                                let code = state
                                    .requested_exit_code
                                    .lock()
                                    .map(|value| *value)
                                    .unwrap_or(0);
                                app.exit(code);
                            }
                        });
                    }
                }
                return;
            }
            // Every close request remains blocked until the one cleanup task
            // finishes; repeated clicks must not bypass graceful shutdown.
            api.prevent_exit();
            if !state.exiting.swap(true, Ordering::AcqRel) {
                let exit_code = code.unwrap_or(0);
                let app = handle.clone();
                tauri::async_runtime::spawn(async move {
                    let state = app.state::<DesktopState>();
                    let has_child = if let Ok(mut child) = state.child.lock() {
                        if let Some(process) = child.as_mut() {
                            let _ = process.write(SHUTDOWN_STDIN_COMMAND.as_bytes());
                        }
                        child.is_some()
                    } else {
                        false
                    };
                    // Python enforces a 20-second total shutdown deadline in
                    // the real one-file child. Give it two extra seconds before
                    // killing the PyInstaller bootloader as a final fallback.
                    if has_child {
                        for _ in 0..440 {
                            if state.terminated.load(Ordering::Acquire) {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }
                    }
                    if let Ok(mut child) = state.child.lock() {
                        if let Some(process) = child.take() {
                            if !state.terminated.load(Ordering::Acquire) {
                                let _ = process.kill();
                            }
                        }
                    }
                    state.shutdown_complete.store(true, Ordering::Release);
                    app.exit(exit_code);
                });
            }
        }
        RunEvent::Exit => {
            if let Ok(mut child) = handle.state::<DesktopState>().child.lock() {
                if let Some(process) = child.take() {
                    let _ = process.kill();
                }
            }
        }
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn parses_only_versioned_loopback_handshake() {
        let token = "a".repeat(64);
        let valid = concat!(
            "INTERVIEW_OS_HANDSHAKE ",
            r#"{"protocol":1,"status":"ready","apiBaseUrl":"http://127.0.0.1:43123","mode":"desktop"}"#
        );
        let runtime = parse_handshake(valid, &token).expect("valid handshake");
        assert_eq!(runtime.api_base_url, "http://127.0.0.1:43123");
        assert_eq!(runtime.bootstrap_token, token);
        assert!(parse_handshake("{}", &token).is_err());
        assert!(parse_handshake(
            "INTERVIEW_OS_HANDSHAKE {\"protocol\":2,\"status\":\"ready\",\"apiBaseUrl\":\"http://127.0.0.1:1\",\"mode\":\"desktop\"}",
            &token
        )
        .is_err());
        assert!(parse_handshake(
            "INTERVIEW_OS_HANDSHAKE {\"protocol\":1,\"status\":\"ready\",\"apiBaseUrl\":\"http://example.com:8000\",\"mode\":\"desktop\"}",
            &token
        )
        .is_err());
    }

    #[test]
    fn bootstrap_token_has_256_bits_of_entropy() {
        let first = new_bootstrap_token().expect("token");
        let second = new_bootstrap_token().expect("token");
        assert_eq!(first.len(), 64);
        assert_ne!(first, second);
        assert!(first.bytes().all(|value| value.is_ascii_hexdigit()));
    }

    #[test]
    fn ready_process_failure_invalidates_renderer_credentials() {
        let state = DesktopState::default();
        *state.status.lock().expect("status") = LaunchStatus::Ready(FrontendRuntime {
            api_base_url: "http://127.0.0.1:43123".into(),
            bootstrap_token: "a".repeat(64),
            mode: "desktop",
        });

        assert!(process_failure_requires_exit(
            &state,
            "desktop service stopped unexpectedly"
        ));
        assert!(matches!(
            *state.status.lock().expect("status"),
            LaunchStatus::Failed(_)
        ));
    }

    #[test]
    fn cancelled_renderer_shutdown_invalidates_the_old_watchdog_generation() {
        let state = DesktopState::default();
        state
            .renderer_shutdown_requested
            .store(true, Ordering::Release);
        state.shutdown_generation.store(7, Ordering::Release);
        *state.requested_exit_code.lock().expect("exit code") = 3;

        cancel_renderer_shutdown(&state);

        assert!(!state.renderer_shutdown_requested.load(Ordering::Acquire));
        assert!(!state.renderer_ready_to_exit.load(Ordering::Acquire));
        assert_eq!(state.shutdown_generation.load(Ordering::Acquire), 8);
        assert_eq!(*state.requested_exit_code.lock().expect("exit code"), 0);
    }

    #[test]
    fn renderer_bridge_ack_is_scoped_to_the_current_document() {
        let state = DesktopState::default();

        assert!(!acknowledge_renderer_shutdown_bridge(&state));
        assert!(!renderer_bridge_ready_for_current_document(&state));

        assert!(!begin_renderer_document(&state));
        assert!(acknowledge_renderer_shutdown_bridge(&state));
        assert!(renderer_bridge_ready_for_current_document(&state));

        assert!(!begin_renderer_document(&state));
        assert!(!renderer_bridge_ready_for_current_document(&state));

        assert!(acknowledge_renderer_shutdown_bridge(&state));
        assert!(renderer_bridge_ready_for_current_document(&state));
    }

    #[test]
    fn replacing_a_document_reports_a_pending_shutdown() {
        let state = DesktopState::default();
        begin_renderer_document(&state);
        acknowledge_renderer_shutdown_bridge(&state);
        state
            .renderer_shutdown_requested
            .store(true, Ordering::Release);
        state.renderer_ready_to_exit.store(true, Ordering::Release);

        assert!(begin_renderer_document(&state));
        assert!(!renderer_bridge_ready_for_current_document(&state));
        assert!(!state.renderer_ready_to_exit.load(Ordering::Acquire));
    }

    #[test]
    fn timeout_claim_is_terminal_for_late_handshakes() {
        let state = DesktopState::default();

        assert!(claim_startup_timeout(&state).expect("timeout claim"));
        assert!(!publish_ready_if_starting(
            &state,
            FrontendRuntime {
                api_base_url: "http://127.0.0.1:43123".into(),
                bootstrap_token: "a".repeat(64),
                mode: "desktop",
            }
        ));
        assert!(matches!(
            *state.status.lock().expect("status"),
            LaunchStatus::Failed("desktop service startup timed out")
        ));
        assert!(!claim_startup_timeout(&state).expect("second timeout claim"));
    }

    #[test]
    fn ready_handshake_wins_over_a_later_timeout_claim() {
        let state = DesktopState::default();
        let runtime = FrontendRuntime {
            api_base_url: "http://127.0.0.1:43123".into(),
            bootstrap_token: "a".repeat(64),
            mode: "desktop",
        };

        assert!(publish_ready_if_starting(&state, runtime));
        assert!(!claim_startup_timeout(&state).expect("timeout claim"));
        assert!(matches!(
            *state.status.lock().expect("status"),
            LaunchStatus::Ready(_)
        ));
    }

    #[test]
    fn timeout_and_ready_race_has_exactly_one_winner() {
        for _ in 0..64 {
            let state = Arc::new(DesktopState::default());
            let barrier = Arc::new(Barrier::new(2));

            let ready_state = Arc::clone(&state);
            let ready_barrier = Arc::clone(&barrier);
            let ready = std::thread::spawn(move || {
                ready_barrier.wait();
                publish_ready_if_starting(
                    &ready_state,
                    FrontendRuntime {
                        api_base_url: "http://127.0.0.1:43123".into(),
                        bootstrap_token: "a".repeat(64),
                        mode: "desktop",
                    },
                )
            });

            let timeout_state = Arc::clone(&state);
            let timeout = std::thread::spawn(move || {
                barrier.wait();
                claim_startup_timeout(&timeout_state).expect("timeout claim")
            });

            let ready_won = ready.join().expect("ready thread");
            let timeout_won = timeout.join().expect("timeout thread");
            assert_ne!(ready_won, timeout_won);
            assert!(matches!(
                *state.status.lock().expect("status"),
                LaunchStatus::Ready(_) | LaunchStatus::Failed(_)
            ));
        }
    }
}
