//! Recently opened projects and imported files, kept in the config folder so
//! they survive restarts.

use std::{
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
};

use serde::{Deserialize, Serialize};

/// How many entries each list keeps.
const MAX_RECENT: usize = 10;

#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecentFiles {
    pub projects: Vec<PathBuf>,
    pub imports: Vec<PathBuf>,
}

fn recent_path() -> PathBuf {
    crate::app::CONFIG_DIR.config_local_dir().join("recent_files.json")
}

static RECENT: LazyLock<Mutex<RecentFiles>> = LazyLock::new(|| {
    Mutex::new(
        std::fs::read_to_string(recent_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default(),
    )
});

/// Current lists (a copy), most recent first.
pub fn recent() -> RecentFiles {
    RECENT.lock().map(|r| r.clone()).unwrap_or_default()
}

fn update(f: impl FnOnce(&mut RecentFiles)) {
    let Ok(mut r) = RECENT.lock() else { return };
    f(&mut r);
    match serde_json::to_string_pretty(&*r) {
        Ok(json) => {
            if let Err(e) = std::fs::write(recent_path(), json) {
                tracing::warn!("could not save recent files: {e}");
            }
        }
        Err(e) => tracing::warn!("could not encode recent files: {e}"),
    }
}

fn push_front(list: &mut Vec<PathBuf>, path: &Path) {
    list.retain(|p| p != path);
    list.insert(0, path.to_path_buf());
    list.truncate(MAX_RECENT);
}

pub fn add_project(path: &Path) {
    update(|r| push_front(&mut r.projects, path));
}
pub fn add_import(path: &Path) {
    update(|r| push_front(&mut r.imports, path));
}
pub fn remove_project(path: &Path) {
    update(|r| r.projects.retain(|p| p != path));
}
pub fn remove_import(path: &Path) {
    update(|r| r.imports.retain(|p| p != path));
}
pub fn clear_projects() {
    update(|r| r.projects.clear());
}
pub fn clear_imports() {
    update(|r| r.imports.clear());
}

/// Menu label: the file name, then its folder (full path is in the hover text).
pub fn label(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    match path.parent().and_then(|p| p.file_name()) {
        Some(dir) => format!("{name}  \u{2014}  {}", dir.to_string_lossy()),
        None => name,
    }
}

