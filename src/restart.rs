//! "Rebuild & restart": close the app, rebuild it from source and start the
//! new build, which reopens the project. A running sequence is saved by its
//! runner first and continues in the new app (see `sequence::checkpoint`).

use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

static REQUESTED: AtomicBool = AtomicBool::new(false);

/// Ask the app to save, rebuild and restart (done on its next frame).
pub fn request() {
    REQUESTED.store(true, Ordering::SeqCst);
    crate::app::request_repaint();
}

pub fn take_requested() -> bool {
    REQUESTED.swap(false, Ordering::SeqCst)
}

/// The source folder this app was built from, if it is still there.
pub fn source_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.join("Cargo.toml").is_file().then_some(dir)
}

/// What the restarted app reopens.
#[derive(Serialize, Deserialize)]
pub struct Handoff {
    pub project: PathBuf,
    /// An untitled project (Save still asks where to put it).
    pub temp: bool,
}

fn handoff_path() -> PathBuf {
    crate::app::CONFIG_DIR.config_local_dir().join("restart_handoff.json")
}

pub fn save_handoff(h: &Handoff) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(h).map_err(std::io::Error::other)?;
    std::fs::write(handoff_path(), json)
}

/// The hand-off left by the app before this one (read once).
pub fn take_handoff() -> Option<Handoff> {
    let path = handoff_path();
    let text = std::fs::read_to_string(&path).ok()?;
    let _ = std::fs::remove_file(&path);
    serde_json::from_str(&text)
        .inspect_err(|e| warn!("ignoring the restart hand-off: {e}"))
        .ok()
}

pub fn clear_handoff() {
    let _ = std::fs::remove_file(handoff_path());
}

/// `cargo build` arguments matching this build (profile and features), so
/// the rebuilt exe replaces this one.
fn build_args() -> Vec<&'static str> {
    let mut args = vec!["build", "--bin", "scan_control"];
    if !cfg!(debug_assertions) {
        args.push("--release");
    }
    if cfg!(feature = "v5e") {
        args.extend(["--features", "v5e"]);
    }
    args
}

/// Start a console that waits for this app to exit, rebuilds it and starts
/// the new exe. A failed build stays open showing the errors; running the
/// script again (path printed) retries.
pub fn spawn_rebuild() -> std::io::Result<()> {
    let dir = source_dir().ok_or_else(|| std::io::Error::other("the source folder was not found"))?;
    let exe = std::env::current_exe()?;
    let cargo = option_env!("CARGO").unwrap_or("cargo");
    let script = write_script(&dir, &exe, cargo, std::process::id())?;
    info!("rebuilding with {}", script.display());
    launch(&script)
}

#[cfg(windows)]
fn write_script(dir: &Path, exe: &Path, cargo: &str, pid: u32) -> std::io::Result<PathBuf> {
    let args = build_args().join(" ");
    let text = format!(
        "@echo off\r\n\
         title Scan Control rebuild\r\n\
         cd /d \"{dir}\"\r\n\
         echo Waiting for Scan Control to close...\r\n\
         :wait\r\n\
         tasklist /FI \"PID eq {pid}\" 2>nul | find \"{pid}\" >nul && (timeout /t 1 /nobreak >nul & goto wait)\r\n\
         echo Building...\r\n\
         \"{cargo}\" {args}\r\n\
         if errorlevel 1 (\r\n\
         echo.\r\n\
         echo BUILD FAILED. Fix the error, then run this again to retry:\r\n\
         echo   %~f0\r\n\
         echo A running sequence was saved and continues when the app starts.\r\n\
         pause\r\n\
         exit /b 1\r\n\
         )\r\n\
         start \"\" \"{exe}\"\r\n",
        dir = dir.display(),
        exe = exe.display(),
    );
    let path = crate::app::CONFIG_DIR.config_local_dir().join("rebuild.cmd");
    std::fs::write(&path, text)?;
    Ok(path)
}

#[cfg(windows)]
fn launch(script: &Path) -> std::io::Result<()> {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
    std::process::Command::new("cmd")
        .arg("/C")
        .arg(script)
        .creation_flags(CREATE_NEW_CONSOLE)
        .spawn()
        .map(|_| ())
}

#[cfg(not(windows))]
fn write_script(dir: &Path, exe: &Path, cargo: &str, pid: u32) -> std::io::Result<PathBuf> {
    let args = build_args().join(" ");
    let text = format!(
        "#!/bin/sh\n\
         cd '{dir}' || exit 1\n\
         while kill -0 {pid} 2>/dev/null; do sleep 1; done\n\
         '{cargo}' {args} || {{ echo 'BUILD FAILED; run '\"$0\"' again to retry'; exit 1; }}\n\
         '{exe}' &\n",
        dir = dir.display(),
        exe = exe.display(),
    );
    let path = crate::app::CONFIG_DIR.config_local_dir().join("rebuild.sh");
    std::fs::write(&path, text)?;
    Ok(path)
}

#[cfg(not(windows))]
fn launch(script: &Path) -> std::io::Result<()> {
    std::process::Command::new("sh").arg(script).spawn().map(|_| ())
}
