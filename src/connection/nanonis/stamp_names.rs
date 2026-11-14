//! Name stamps after the .sxm file Nanonis saved the same scan to.
//!
//! A new stamp starts with the base name; a background search of the
//! Nanonis session folder finds the file holding the same data (it may only
//! appear when the scan finishes) and the stamp is renamed to the file name,
//! e.g. `20261112_0006`. Older stamps in a project that don't have a file
//! name yet are looked up once in the session folder and its neighbours.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use crossbeam::channel::Receiver;
use uuid::Uuid;

use crate::analysis::sxm_match::{list_sxm, Sxm};

/// How often to look for files while stamps wait for a name.
const INTERVAL: Duration = Duration::from_secs(5);
/// Give up on a new stamp's file after this long.
const GIVE_UP: Duration = Duration::from_secs(3 * 3600);
/// Only files modified this recently are checked for new stamps.
const RECENT: Duration = Duration::from_secs(6 * 3600);

struct Pending {
    uuid: Uuid,
    data: Arc<Box<[f32]>>,
    size: [u32; 2],
    since: Instant,
    /// An existing stamp: search all session folders once.
    wide: bool,
}

#[derive(Default)]
pub struct StampNamer {
    pending: Vec<Pending>,
    /// Stamps already looked up (or queued), so each is tried once.
    tried: HashSet<Uuid>,
    job: Option<Receiver<Vec<(Uuid, String)>>>,
    /// Stamps the running job is checking.
    in_job: Vec<Uuid>,
    last_run: Option<Instant>,
}

/// Looks like a Nanonis file name already (ends in `_` and 4+ digits).
pub fn is_file_name(name: &str) -> bool {
    name.rsplit_once('_')
        .is_some_and(|(_, n)| n.len() >= 4 && n.chars().all(|c| c.is_ascii_digit()))
}

impl StampNamer {
    /// A new stamp: name it after its file once Nanonis has saved it.
    pub fn add_new(&mut self, uuid: Uuid, data: Arc<Box<[f32]>>, size: [u32; 2]) {
        self.tried.insert(uuid);
        self.pending.push(Pending { uuid, data, size, since: Instant::now(), wide: false });
    }

    /// An existing stamp without a file name: look it up once.
    pub fn add_existing(&mut self, uuid: Uuid, data: Arc<Box<[f32]>>, size: [u32; 2]) {
        if self.tried.insert(uuid) {
            self.pending.push(Pending { uuid, data, size, since: Instant::now(), wide: true });
        }
    }

    /// Names found since the last call. Starts a background search when
    /// stamps are waiting.
    pub fn poll(&mut self, session_path: &str) -> Vec<(Uuid, String)> {
        let mut found = Vec::new();
        if let Some(rx) = &self.job {
            match rx.try_recv() {
                Ok(names) => {
                    self.job = None;
                    let done: HashSet<Uuid> = names.iter().map(|n| n.0).collect();
                    let in_job = std::mem::take(&mut self.in_job);
                    // Found, or a one-off search of an existing stamp: done.
                    self.pending.retain(|p| !done.contains(&p.uuid) && !(p.wide && in_job.contains(&p.uuid)));
                    found = names;
                }
                Err(crossbeam::channel::TryRecvError::Empty) => return found,
                Err(_) => self.job = None,
            }
        }
        self.pending.retain(|p| p.wide || p.since.elapsed() < GIVE_UP);
        if self.pending.is_empty()
            || session_path.trim().is_empty()
            || self.last_run.is_some_and(|t| t.elapsed() < INTERVAL)
        {
            return found;
        }
        self.last_run = Some(Instant::now());
        let session = PathBuf::from(session_path.trim());
        let jobs: Vec<(Uuid, Arc<Box<[f32]>>, [u32; 2], bool)> =
            self.pending.iter().map(|p| (p.uuid, p.data.clone(), p.size, p.wide)).collect();
        self.in_job = jobs.iter().map(|j| j.0).collect();
        let (tx, rx) = crossbeam::channel::bounded(1);
        std::thread::Builder::new()
            .name("stamp-names".into())
            .spawn(move || {
                let _ = tx.send(search(&session, &jobs));
                crate::app::request_repaint();
            })
            .ok();
        self.job = Some(rx);
        found
    }
}

/// Modification time, newest first.
fn by_newest(mut files: Vec<PathBuf>) -> Vec<PathBuf> {
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
    files.sort_by_key(|p| std::cmp::Reverse(mtime(p)));
    files
}

/// Find the file of each stamp. New stamps: recent files in the session
/// folder. Existing stamps: every file next to the session folder too.
/// Each file is read once and checked against all stamps still unnamed.
fn search(session: &Path, jobs: &[(Uuid, Arc<Box<[f32]>>, [u32; 2], bool)]) -> Vec<(Uuid, String)> {
    let now = SystemTime::now();
    let is_recent = |p: &Path| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .is_ok_and(|t| now.duration_since(t).map_or(true, |d| d < RECENT))
    };
    let root = if jobs.iter().any(|j| j.3) { session.parent().unwrap_or(session) } else { session };
    let mut todo: Vec<usize> = (0..jobs.len()).collect();
    let mut out = Vec::new();
    for path in by_newest(list_sxm(root)) {
        if todo.is_empty() {
            break;
        }
        let in_session = path.parent() == Some(session) && is_recent(&path);
        let want: Vec<usize> = todo.iter().copied().filter(|&i| jobs[i].3 || in_session).collect();
        if want.is_empty() {
            continue;
        }
        let Some(sxm) = Sxm::load(&path) else { continue };
        for i in want {
            if sxm.holds(&jobs[i].1, jobs[i].2) {
                if let Some(stem) = path.file_stem() {
                    out.push((jobs[i].0, stem.to_string_lossy().to_string()));
                }
                todo.retain(|&j| j != i);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Looks up some labelled frames as existing stamps in a real session folder.
    /// `cargo test --release --bin scan_control real_session_lookup -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_session_lookup() {
        use crate::analysis::ring_training::Dataset;
        let session = std::env::var("SESSION_DIR")
            .unwrap_or(r"F:\Data\LT SPM\2026\November\20261110_STM24_As1aNov9".into());
        let samples = Dataset::load(Dataset::default_dir()).samples();
        let picked: Vec<_> = samples.iter().step_by(6).collect();
        let jobs: Vec<_> = picked
            .iter()
            .map(|s| (Uuid::new_v4(), Arc::new(s.data.clone().into_boxed_slice()), [s.record.w, s.record.h], true))
            .collect();
        let t = Instant::now();
        let found = search(Path::new(&session), &jobs);
        println!("{} of {} found in {:?}", found.len(), jobs.len(), t.elapsed());
        for (j, s) in jobs.iter().zip(&picked) {
            let f = found.iter().find(|f| f.0 == j.0).map(|f| f.1.as_str());
            println!("{} -> {f:?}", s.record.source);
            assert_eq!(f, Some(s.record.source.as_str()));
        }
    }

    #[test]
    fn file_names() {
        assert!(is_file_name("20261112_0006"));
        assert!(is_file_name("scan_0012"));
        assert!(!is_file_name("20261112_"));
        assert!(!is_file_name("20261112_(3)"));
        assert!(!is_file_name("stamp"));
    }
}
