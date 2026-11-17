//! The running sequence, saved to disk after every step so it survives the
//! app being closed (to rebuild it, or a crash). On the next start the
//! runner picks it up where it was. Nanonis keeps scanning meanwhile.

use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tracing::warn;

use super::model::{Dir, ScanStart, Sequence, Step};

/// The scan the current step started (see the runner's `ScanPhase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum SavedScan {
    NotStarted,
    Running(Dir),
    Paused(Dir),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Checkpoint {
    pub steps: Vec<(usize, Step)>,
    pub group: u32,
    pub scan_start: ScanStart,
    /// Step in progress, or next to run.
    pub index: usize,
    pub paused: bool,
    /// Step `index` had begun (it may be half done).
    pub step_started: bool,
    pub scan: SavedScan,
    /// Scanning time of the current scan before `scan_since`, s.
    pub scan_active_s: f64,
    /// When the current scan (re)started running, Unix s.
    pub scan_since_unix: Option<u64>,
    pub scan_expected_s: f64,
    pub pause_after_scan: bool,
    /// The app was closed on purpose to be rebuilt, with the sequence
    /// running: carry on by itself once Nanonis is connected again.
    pub resume_after_restart: bool,
    /// The editor's plan (entry names, frames for the map).
    pub plan: Sequence,
    /// Layer group of the run's survey tiles, so tiles taken after a
    /// restart join the same survey.
    #[serde(default)]
    pub survey_group: Option<String>,
    /// Stamps of the survey tiles taken so far (uuids), to line up at the end.
    #[serde(default)]
    pub survey_tiles: Vec<String>,
}

fn path() -> PathBuf {
    crate::app::CONFIG_DIR.config_local_dir().join("sequence_checkpoint.json")
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

pub(super) fn load() -> Option<Checkpoint> {
    let text = std::fs::read_to_string(path()).ok()?;
    match serde_json::from_str::<Checkpoint>(&text) {
        Ok(c) if c.index < c.steps.len() => Some(c),
        Ok(_) => None,
        Err(e) => {
            warn!("ignoring the saved sequence (unreadable: {e})");
            None
        }
    }
}

/// Write `json` (a [`Checkpoint`]), replacing the old file in one step.
pub(super) fn write(json: &str) {
    let path = path();
    let tmp = path.with_extension("json.tmp");
    let result = std::fs::write(&tmp, json).and_then(|_| std::fs::rename(&tmp, &path));
    if let Err(e) = result {
        warn!("could not save the sequence checkpoint: {e}");
    }
}

pub(super) fn clear() {
    let _ = std::fs::remove_file(path());
}
