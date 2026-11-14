//! Training data and retraining for Ring Assist.
//!
//! The user labels stamped frames (good / corrected direction / should stop).
//! Labels and frame data go into a dataset shared by all projects. Retraining
//! searches the detector's tuning knobs for the values that agree best with
//! the labels, and checks the result with cross-validation so it is only
//! offered when it generalises better than the current settings.

use std::{
    collections::HashMap,
    io::Write as _,
    path::{Path, PathBuf},
    sync::Mutex,
};

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    ring_center::RingParams,
    ring_match::{blind_match, match_rings, Expected, MatchMode, MatchParams, Polarity, RingMatch},
    ring_stamp::{self, quad_to_px_dir, StampVerdict},
};

/// A predicted direction within this angle of the label counts as correct.
const MAX_ANGLE_DEG: f64 = 30.;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LabelKind {
    /// The model's direction was right.
    Good,
    /// The user corrected the direction.
    Direction,
    /// No move wanted here.
    Stop,
    /// The smallest ring is in the frame (stage 1 reached). `dir` is the
    /// direction of the final step into the middle, or zero if not given.
    SmallestRing,
}

/// One labelled frame, as stored in `labels.jsonl`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub kind: LabelKind,
    /// Unit direction in frame pixel axes (+x right, +y down rows); zero for Stop.
    pub dir: [f64; 2],
    pub w: u32,
    pub h: u32,
    pub nm_per_px: f64,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub time: String,
    /// What the GDS predicted for this frame, if it was on a GDS ring system.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gds: Option<GdsInfo>,
}

/// The GDS prediction stored with a label.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GdsInfo {
    /// Unit direction to the ring centre, frame pixel axes.
    pub dir: [f64; 2],
    /// The innermost ring should be in the frame.
    pub smallest: bool,
    /// Expected ring centre, frame pixels (+x right, +y down rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center_px: Option<[f64; 2]>,
    /// Expected (inner, outer) radius of each band, frame pixels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bands_px: Vec<[f64; 2]>,
}

impl GdsInfo {
    /// The rings the GDS expects in the frame, if stored with the label.
    pub fn expected(&self) -> Option<Expected> {
        let c = self.center_px?;
        (!self.bands_px.is_empty()).then(|| Expected {
            center_px: DVec2::from_array(c),
            bands_px: self.bands_px.iter().map(|b| (b[0], b[1])).collect(),
        })
    }
    /// Different enough to rewrite the stored prediction.
    fn differs(&self, other: &GdsInfo) -> bool {
        let near = |a: Option<[f64; 2]>, b: Option<[f64; 2]>| match (a, b) {
            (Some(a), Some(b)) => (DVec2::from_array(a) - DVec2::from_array(b)).length() < 0.5,
            (None, None) => true,
            _ => false,
        };
        self.smallest != other.smallest
            || DVec2::from_array(self.dir).dot(DVec2::from_array(other.dir)) < 0.9998
            || !near(self.center_px, other.center_px)
            || self.bands_px.len() != other.bands_px.len()
            || self.bands_px.iter().zip(&other.bands_px).any(|(a, b)| (a[1] - b[1]).abs() > 0.5)
    }
}

impl Record {
    pub fn dir(&self) -> DVec2 {
        DVec2::from_array(self.dir)
    }
}

pub struct Sample {
    pub record: Record,
    pub data: Vec<f32>,
}

/// Detector settings that retraining tunes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TuneParams {
    pub params: RingParams,
    pub min_radius_um: f64,
    /// Matching the scan to the GDS rings (refines the GDS prediction).
    pub matching: MatchParams,
}

impl Default for TuneParams {
    fn default() -> Self {
        Self { params: RingParams::default(), min_radius_um: 1.5, matching: MatchParams::default() }
    }
}

impl TuneParams {
    pub fn path() -> PathBuf {
        crate::app::CONFIG_DIR.config_local_dir().join("ring_params.json")
    }
    /// Saved settings, or the defaults if there are none.
    pub fn load() -> Self {
        std::fs::read_to_string(Self::path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
    pub fn save(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(Self::path(), json).map_err(|e| e.to_string())
    }
    /// Human-readable list of knobs that differ from `other`.
    pub fn changes_from(&self, other: &TuneParams) -> Vec<String> {
        let mut out = Vec::new();
        let (a, b) = (&other.params, &self.params);
        if a.target_px != b.target_px {
            out.push(format!("downsample {:?} → {:?} px", a.target_px, b.target_px));
        }
        let mut f = |name: &str, x: f64, y: f64| {
            if (x - y).abs() > 1e-9 {
                out.push(format!("{name} {x} → {y}"));
            }
        };
        f("blur", a.blur_sigma as f64, b.blur_sigma as f64);
        f("edge fraction", a.edge_fraction as f64, b.edge_fraction as f64);
        f("angle scale", a.angle_scale, b.angle_scale);
        f("unsure below inliers", a.min_inlier_fraction, b.min_inlier_fraction);
        f("min ring radius µm", other.min_radius_um, self.min_radius_um);
        out
    }
}

/// Labelled frames on disk: `labels.jsonl` plus one `<id>.bin` (raw f32) per frame.
pub struct Dataset {
    dir: PathBuf,
    /// Latest label per id, in first-seen order.
    records: Vec<Record>,
}

impl Dataset {
    pub fn default_dir() -> PathBuf {
        crate::app::CONFIG_DIR.config_local_dir().join("ring_training")
    }
    pub fn load(dir: PathBuf) -> Self {
        let mut records: Vec<Record> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(dir.join("labels.jsonl")) {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                match serde_json::from_str::<Record>(line) {
                    Ok(r) => match index.get(&r.id) {
                        Some(&i) => records[i] = r,
                        None => {
                            index.insert(r.id.clone(), records.len());
                            records.push(r);
                        }
                    },
                    Err(e) => tracing::warn!("skipping bad ring label line: {e}"),
                }
            }
        }
        Self { dir, records }
    }
    pub fn dir(&self) -> &Path {
        &self.dir
    }
    pub fn len(&self) -> usize {
        self.records.len()
    }
    /// (good, corrected, stop, smallest ring)
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let n = |k| self.records.iter().filter(|r| r.kind == k).count();
        (
            n(LabelKind::Good),
            n(LabelKind::Direction),
            n(LabelKind::Stop),
            n(LabelKind::SmallestRing),
        )
    }
    /// Add or replace the label for frame `id`.
    pub fn add(&mut self, record: Record, data: &[f32]) -> Result<(), String> {
        if data.len() != record.w as usize * record.h as usize {
            return Err("frame size does not match its data".into());
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
        std::fs::write(self.dir.join(format!("{}.bin", record.id)), bytemuck::cast_slice(data))
            .map_err(|e| e.to_string())?;
        let line = serde_json::to_string(&record).map_err(|e| e.to_string())?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("labels.jsonl"))
            .map_err(|e| e.to_string())?;
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
        match self.records.iter_mut().find(|r| r.id == record.id) {
            Some(r) => *r = record,
            None => self.records.push(record),
        }
        Ok(())
    }
    pub fn gds_of(&self, id: &str) -> Option<Option<GdsInfo>> {
        self.records.iter().find(|r| r.id == id).map(|r| r.gds.clone())
    }
    /// Set the GDS prediction of existing labels (e.g. after the GDS was
    /// placed or moved). Rewrites `labels.jsonl` once if anything changed.
    pub fn update_gds(&mut self, updates: &[(String, Option<GdsInfo>)]) -> Result<usize, String> {
        let mut changed = 0;
        for (id, g) in updates {
            if let Some(r) = self.records.iter_mut().find(|r| &r.id == id) {
                let differs = match (&r.gds, g) {
                    (Some(a), Some(b)) => a.differs(b),
                    (None, None) => false,
                    _ => true,
                };
                if differs {
                    r.gds = g.clone();
                    changed += 1;
                }
            }
        }
        if changed > 0 {
            self.rewrite()?;
        }
        Ok(changed)
    }
    /// Set the source name (the Nanonis file name) of existing labels.
    pub fn set_sources(&mut self, updates: &[(String, String)]) -> Result<usize, String> {
        let mut changed = 0;
        for (id, name) in updates {
            if let Some(r) = self.records.iter_mut().find(|r| &r.id == id && &r.source != name) {
                r.source = name.clone();
                changed += 1;
            }
        }
        if changed > 0 {
            self.rewrite()?;
        }
        Ok(changed)
    }
    /// Write the latest label of every frame back to `labels.jsonl`.
    fn rewrite(&self) -> Result<(), String> {
        let mut text = String::new();
        for r in &self.records {
            text += &serde_json::to_string(r).map_err(|e| e.to_string())?;
            text.push('\n');
        }
        let tmp = self.dir.join("labels.jsonl.tmp");
        std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, self.dir.join("labels.jsonl")).map_err(|e| e.to_string())
    }
    /// Read every labelled frame. Frames whose data file is missing are skipped.
    pub fn samples(&self) -> Vec<Sample> {
        self.records
            .iter()
            .filter_map(|r| {
                let bytes = std::fs::read(self.dir.join(format!("{}.bin", r.id))).ok()?;
                if bytes.len() != r.w as usize * r.h as usize * 4 {
                    return None;
                }
                let data = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Some(Sample { record: r.clone(), data })
            })
            .collect()
    }
}

/// Does the detector with `tune` agree with this sample's label?
pub fn score(sample: &Sample, tune: &TuneParams) -> bool {
    let r = &sample.record;
    let size = [r.w, r.h];
    let Some(res) = ring_stamp::analyse_frame(
        &sample.data,
        size,
        ring_stamp::frame_for(size, r.nm_per_px),
        &tune.params,
        tune.min_radius_um,
    ) else {
        return r.kind == LabelKind::Stop
            || (r.kind == LabelKind::SmallestRing && r.dir() == DVec2::ZERO);
    };
    let label = r.dir();
    match (r.kind, res.verdict) {
        (LabelKind::Stop, StampVerdict::Unsure(_)) => true,
        (LabelKind::Stop, _) => false,
        // At the smallest ring the ring centre is the middle: right if the
        // model points at the middle (when the user gave that direction), or
        // stops when no direction was given.
        (LabelKind::SmallestRing, StampVerdict::Unsure(_)) => label == DVec2::ZERO,
        (LabelKind::SmallestRing, StampVerdict::Arc) => {
            label != DVec2::ZERO
                && quad_to_px_dir(res.dir_quad, size).dot(label)
                    >= MAX_ANGLE_DEG.to_radians().cos()
        }
        (LabelKind::SmallestRing, StampVerdict::ArmEnd) => false,
        (_, StampVerdict::Unsure(_)) => false,
        (_, StampVerdict::Arc) => {
            let pred = quad_to_px_dir(res.dir_quad, size);
            pred.dot(label) >= MAX_ANGLE_DEG.to_radians().cos()
        }
        (_, StampVerdict::ArmEnd) => {
            // Sideways move: right if the label points clearly left/right that way.
            let pred = quad_to_px_dir(res.dir_quad, size);
            label.x.abs() > 0.5 && pred.x.signum() == label.x.signum()
        }
    }
}

/// Does a ring-centre prediction (direction in pixel axes, and whether the
/// smallest ring is in the frame) agree with this label?
fn judge(r: &Record, dir: DVec2, smallest: bool) -> bool {
    let cos = MAX_ANGLE_DEG.to_radians().cos();
    match r.kind {
        LabelKind::Good | LabelKind::Direction => dir.dot(r.dir()) >= cos,
        LabelKind::Stop => smallest,
        LabelKind::SmallestRing => smallest && (r.dir() == DVec2::ZERO || dir.dot(r.dir()) >= cos),
    }
}

/// Does the GDS prediction agree with this label? `None` without a GDS.
pub fn score_gds(r: &Record) -> Option<bool> {
    let g = r.gds.as_ref()?;
    Some(judge(r, DVec2::from_array(g.dir), g.smallest))
}

/// Matches computed during one retraining, per sample and match settings.
/// The threshold `min_score` doesn't change the match, so it's not part of
/// the key and trying thresholds is free.
#[derive(Default)]
pub struct MatchCache(Mutex<HashMap<(usize, MatchKey), Option<(usize, RingMatch)>>>);

/// (blind, mode, softness, width slack, polarity)
type MatchKey = (bool, MatchMode, u64, u64, Polarity);

impl MatchCache {
    fn key(s: &Sample, p: &MatchParams, blind: bool) -> (usize, MatchKey) {
        let softness = if p.mode == MatchMode::SoftTwoTone { p.softness.to_bits() } else { 0 };
        (s as *const Sample as usize, (blind, p.mode, softness, p.width_slack.to_bits(), p.polarity))
    }
    fn cached(&self, key: (usize, MatchKey), f: impl FnOnce() -> Option<(usize, RingMatch)>) -> Option<(usize, RingMatch)> {
        if let Some(m) = self.0.lock().unwrap().get(&key) {
            return *m;
        }
        let m = f();
        self.0.lock().unwrap().insert(key, m);
        m
    }
    fn get(&self, s: &Sample, p: &MatchParams) -> Option<RingMatch> {
        self.cached(Self::key(s, p, false), || {
            let e = s.record.gds.as_ref()?.expected()?;
            match_rings(&s.data, [s.record.w, s.record.h], &e, p).map(|m| (0, m))
        })
        .map(|m| m.1)
    }
    /// Blind match against ring kinds given in world nm.
    fn get_blind(&self, s: &Sample, p: &MatchParams, kinds_nm: &[Vec<(f64, f64)>]) -> Option<(usize, RingMatch)> {
        self.cached(Self::key(s, p, true), || {
            let k = 1. / s.record.nm_per_px;
            let kinds: Vec<Vec<(f64, f64)>> =
                kinds_nm.iter().map(|b| b.iter().map(|x| (x.0 * k, x.1 * k)).collect()).collect();
            blind_match(&s.data, [s.record.w, s.record.h], &kinds, p)
        })
    }
}

/// Blind matching (no GDS placement, only the ring kinds in world nm) for
/// this label. An untrusted match falls back to the detector. Returns
/// (right, confident match).
pub fn score_blind(
    sample: &Sample,
    tune: &TuneParams,
    p: &MatchParams,
    kinds_nm: &[Vec<(f64, f64)>],
    cache: &MatchCache,
) -> (bool, bool) {
    if p.mode == MatchMode::Off || kinds_nm.is_empty() {
        return (score(sample, tune), false);
    }
    let r = &sample.record;
    match cache.get_blind(sample, p, kinds_nm).filter(|m| m.1.score >= p.min_score) {
        None => (score(sample, tune), false),
        Some((kind, m)) => {
            let fc = DVec2::new((r.w as f64 - 1.) / 2., (r.h as f64 - 1.) / 2.);
            let v = m.center_px - fc;
            let half_diag = 0.5 * DVec2::new(r.w as f64, r.h as f64).length();
            let inner = kinds_nm[kind][0].1 / r.nm_per_px;
            (judge(r, v.normalize_or_zero(), v.length() - half_diag <= inner), true)
        }
    }
}

/// Does the GDS prediction, refined by matching the scan against the
/// expected rings, agree with this label? An untrusted match (score below
/// `min_score`) falls back to the plain GDS prediction. `None` for labels
/// without the GDS ring geometry.
pub fn score_match(sample: &Sample, p: &MatchParams, cache: &MatchCache) -> Option<bool> {
    let r = &sample.record;
    let exp = r.gds.as_ref()?.expected()?;
    let Some(m) = cache.get(sample, p).filter(|m| m.score >= p.min_score) else {
        return score_gds(r);
    };
    let fc = DVec2::new((r.w as f64 - 1.) / 2., (r.h as f64 - 1.) / 2.);
    let v = m.center_px - fc;
    let half_diag = 0.5 * DVec2::new(r.w as f64, r.h as f64).length();
    let smallest = v.length() - half_diag <= exp.bands_px[0].1;
    Some(judge(r, v.normalize_or_zero(), smallest))
}

/// Number of samples (with GDS geometry) the match settings get right.
pub fn correct_match(samples: &[&Sample], p: &MatchParams, cache: &MatchCache) -> usize {
    let threads = (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1);
    let chunk = samples.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        samples
            .chunks(chunk)
            .map(|c| s.spawn(move || c.iter().filter(|x| score_match(x, p, cache) == Some(true)).count()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap_or(0))
            .sum()
    })
}

/// Coordinate descent over the match settings, like `search`.
pub fn search_match(samples: &[&Sample], start: MatchParams, cache: &MatchCache) -> (MatchParams, usize) {
    let mut best = start;
    let mut best_n = correct_match(samples, &best, cache);
    for _round in 0..4 {
        let mut improved = false;
        for knob in 0..5 {
            for cand in match_candidates(&best, knob) {
                if cand == best {
                    continue;
                }
                let n = correct_match(samples, &cand, cache);
                if n > best_n {
                    best = cand;
                    best_n = n;
                    improved = true;
                }
            }
        }
        if !improved || best_n == samples.len() {
            break;
        }
    }
    (best, best_n)
}

fn match_candidates(base: &MatchParams, knob: usize) -> Vec<MatchParams> {
    let b = *base;
    match knob {
        0 => MatchMode::ALL.iter().map(|&mode| MatchParams { mode, ..b }).collect(),
        1 => [0.1, 0.25, 0.5, 1.].iter().map(|&softness| MatchParams { softness, ..b }).collect(),
        2 => [0., 0.3, 0.6, 1.].iter().map(|&width_slack| MatchParams { width_slack, ..b }).collect(),
        3 => [0.4, 0.5, 0.6, 0.7, 0.8].iter().map(|&min_score| MatchParams { min_score, ..b }).collect(),
        _ => [Polarity::Auto, Polarity::Raised, Polarity::Recessed]
            .iter()
            .map(|&polarity| MatchParams { polarity, ..b })
            .collect(),
    }
}

/// Number of samples the settings get right, scored in parallel.
pub fn correct(samples: &[&Sample], tune: &TuneParams) -> usize {
    // Use at most half the cores so the UI stays responsive while retraining.
    let threads = (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1);
    let chunk = samples.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        samples
            .chunks(chunk)
            .map(|c| s.spawn(move || c.iter().filter(|x| score(x, tune)).count()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap_or(0))
            .sum()
    })
}

/// Coordinate descent over the tuning knobs, starting from `start`. A knob
/// only changes when that strictly improves the count, so ties keep the
/// current settings.
pub fn search(samples: &[&Sample], start: TuneParams) -> (TuneParams, usize) {
    let mut best = start;
    let mut best_n = correct(samples, &best);
    for _round in 0..4 {
        let mut improved = false;
        for knob in 0..6 {
            for cand in candidates(&best, knob) {
                if cand == best {
                    continue;
                }
                let n = correct(samples, &cand);
                if n > best_n {
                    best = cand;
                    best_n = n;
                    improved = true;
                }
            }
        }
        if !improved || best_n == samples.len() {
            break;
        }
    }
    (best, best_n)
}

fn candidates(base: &TuneParams, knob: usize) -> Vec<TuneParams> {
    let with = |f: &dyn Fn(&mut TuneParams)| {
        let mut t = *base;
        f(&mut t);
        t
    };
    match knob {
        0 => [24usize, 32, 48].iter().map(|&v| with(&|t| t.params.target_px = Some(v))).collect(),
        1 => [1.0f32, 1.5, 2., 3.].iter().map(|&v| with(&|t| t.params.blur_sigma = v)).collect(),
        2 => [0.15f32, 0.2, 0.3, 0.4].iter().map(|&v| with(&|t| t.params.edge_fraction = v)).collect(),
        3 => [0.07, 0.1, 0.15].iter().map(|&v| with(&|t| t.params.angle_scale = v)).collect(),
        4 => [0.08, 0.1, 0.12, 0.15, 0.2, 0.25]
            .iter()
            .map(|&v| with(&|t| t.params.min_inlier_fraction = v))
            .collect(),
        _ => [1.0, 1.5, 2., 3.].iter().map(|&v| with(&|t| t.min_radius_um = v)).collect(),
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub n: usize,
    /// Correct with the current settings.
    pub current_correct: usize,
    /// Correct on held-out folds when the search only saw the other folds.
    pub cv_correct: usize,
    /// Settings found on all samples, and how many of them they get right.
    pub new: TuneParams,
    pub new_correct: usize,
    pub changes: Vec<String>,
    /// Labels with a GDS prediction, and how many it gets right.
    pub gds_n: usize,
    pub gds_correct: usize,
    /// GDS where available, the current settings elsewhere.
    pub combined_correct: usize,
    /// Labels with the GDS ring geometry (needed for matching).
    pub match_n: usize,
    /// Right per matching mode (other match settings as now), out of `match_n`.
    pub mode_correct: Vec<(MatchMode, usize)>,
    /// Match settings: right now, on held-out folds, and new settings on all.
    pub match_current_correct: usize,
    pub match_cv_correct: usize,
    pub match_new_correct: usize,
    pub match_changes: Vec<String>,
    /// Blind matching (no GDS placement needed) per mode, on all labels:
    /// (mode, right, confident matches). `Off` is the detector alone.
    pub blind: Vec<(MatchMode, usize, usize)>,
}

impl Report {
    /// Offer new settings only if they generalise better and differ.
    pub fn is_improvement(&self) -> bool {
        self.detector_improves() || self.match_improves()
    }
    pub fn detector_improves(&self) -> bool {
        self.cv_correct > self.current_correct && !self.changes.is_empty()
    }
    pub fn match_improves(&self) -> bool {
        self.match_cv_correct > self.match_current_correct && !self.match_changes.is_empty()
    }
}

/// Score the current settings, cross-validate the search, and find new
/// settings on all samples.
pub fn evaluate(samples: &[Sample], current: TuneParams, kinds_nm: &[Vec<(f64, f64)>]) -> Report {
    let all: Vec<&Sample> = samples.iter().collect();
    let n = all.len();
    let current_correct = correct(&all, &current);
    let folds = n.min(5).max(1);
    let mut cv_correct = 0;
    if folds >= 2 {
        for k in 0..folds {
            let train: Vec<&Sample> =
                all.iter().enumerate().filter(|(i, _)| i % folds != k).map(|(_, s)| *s).collect();
            let test: Vec<&Sample> =
                all.iter().enumerate().filter(|(i, _)| i % folds == k).map(|(_, s)| *s).collect();
            let (fit, _) = search(&train, current);
            cv_correct += correct(&test, &fit);
        }
    } else {
        cv_correct = current_correct;
    }
    let (mut new, new_correct) = search(&all, current);

    // GDS matching, on labels that have the expected ring geometry.
    let cache = MatchCache::default();
    let geo: Vec<&Sample> = all
        .iter()
        .copied()
        .filter(|s| s.record.gds.as_ref().and_then(|g| g.expected()).is_some())
        .collect();
    let cur_m = current.matching;
    let mode_correct = MatchMode::ALL
        .iter()
        .map(|&mode| (mode, correct_match(&geo, &MatchParams { mode, ..cur_m }, &cache)))
        .collect();
    let match_current_correct = correct_match(&geo, &cur_m, &cache);
    let mfolds = geo.len().min(5);
    let mut match_cv_correct = match_current_correct;
    if mfolds >= 2 {
        match_cv_correct = 0;
        for k in 0..mfolds {
            let train: Vec<&Sample> =
                geo.iter().enumerate().filter(|(i, _)| i % mfolds != k).map(|(_, s)| *s).collect();
            let test: Vec<&Sample> =
                geo.iter().enumerate().filter(|(i, _)| i % mfolds == k).map(|(_, s)| *s).collect();
            let (fit, _) = search_match(&train, cur_m, &cache);
            match_cv_correct += correct_match(&test, &fit, &cache);
        }
    }
    let (match_new, match_new_correct) = search_match(&geo, cur_m, &cache);
    new.matching = match_new;

    // Blind matching: every label, only the ring kinds needed.
    let blind = if kinds_nm.is_empty() {
        Vec::new()
    } else {
        MatchMode::ALL
            .iter()
            .map(|&mode| {
                let p = MatchParams { mode, ..cur_m };
                let res = par_map(&all, |s| score_blind(s, &current, &p, kinds_nm, &cache));
                (mode, res.iter().filter(|r| r.0).count(), res.iter().filter(|r| r.1).count())
            })
            .collect()
    };
    let gds: Vec<bool> = all.iter().filter_map(|s| score_gds(&s.record)).collect();
    let no_gds: Vec<&Sample> = all.iter().copied().filter(|s| s.record.gds.is_none()).collect();
    let gds_correct = gds.iter().filter(|&&b| b).count();
    let combined_correct = gds_correct + correct(&no_gds, &current);
    Report {
        n,
        current_correct,
        cv_correct,
        changes: new.changes_from(&current),
        new,
        new_correct,
        gds_n: gds.len(),
        gds_correct,
        combined_correct,
        match_n: geo.len(),
        mode_correct,
        match_current_correct,
        match_cv_correct,
        match_new_correct,
        match_changes: match_new.changes_from(&cur_m),
        blind,
    }
}

/// `f` over the samples on half the cores, results in order.
fn par_map<T: Send>(samples: &[&Sample], f: impl Fn(&Sample) -> T + Sync) -> Vec<T> {
    let threads = (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1);
    let chunk = samples.len().div_ceil(threads).max(1);
    let f = &f;
    std::thread::scope(|sc| {
        samples
            .chunks(chunk)
            .map(|c| sc.spawn(move || c.iter().map(|s| f(s)).collect::<Vec<T>>()))
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rings centred at `centre` (pixels) in a `w`×`w` frame, 12 px bands.
    fn rings(w: usize, centre: DVec2) -> Vec<f32> {
        (0..w * w)
            .map(|i| {
                let p = DVec2::new((i % w) as f64, (i / w) as f64);
                let r = (p - centre).length();
                if (r / 12.).floor() as i64 % 2 == 0 { 1. } else { 0. }
            })
            .collect()
    }

    fn sample(kind: LabelKind, dir: DVec2, data: Vec<f32>, w: u32) -> Sample {
        Sample {
            record: Record {
                id: uuid::Uuid::new_v4().to_string(),
                kind,
                dir: dir.normalize_or_zero().to_array(),
                w,
                h: w,
                nm_per_px: 3000. / w as f64,
                source: String::new(),
                time: String::new(),
                gds: None,
            },
            data,
        }
    }

    fn base() -> TuneParams {
        TuneParams {
            params: RingParams { target_px: None, ..RingParams::default() },
            min_radius_um: 0.5,
            matching: MatchParams::default(),
        }
    }

    #[test]
    fn scoring_rules() {
        let w = 96;
        let data = rings(w, DVec2::new(-60., -40.));
        let toward = DVec2::new(-60., -40.) - DVec2::splat(47.5);
        assert!(score(&sample(LabelKind::Good, toward, data.clone(), w as u32), &base()));
        // Opposite direction is wrong.
        assert!(!score(&sample(LabelKind::Direction, -toward, data.clone(), w as u32), &base()));
        // A clear ring frame is not a stop.
        assert!(!score(&sample(LabelKind::Stop, DVec2::ZERO, data, w as u32), &base()));
        // Flat frame: the detector is unsure, which matches a Stop label.
        let flat = vec![0.5f32; w * w];
        assert!(score(&sample(LabelKind::Stop, DVec2::ZERO, flat.clone(), w as u32), &base()));
        // Smallest ring without a direction: right when the model stops...
        assert!(score(&sample(LabelKind::SmallestRing, DVec2::ZERO, flat, w as u32), &base()));
        // ...and with a direction to the middle: right when it points there.
        let data = rings(w, DVec2::new(-60., -40.));
        assert!(score(&sample(LabelKind::SmallestRing, toward, data.clone(), w as u32), &base()));
        assert!(!score(&sample(LabelKind::SmallestRing, -toward, data, w as u32), &base()));
    }

    #[test]
    fn dataset_round_trip_last_label_wins() {
        let dir = std::env::temp_dir().join(format!("ring_ds_{}", uuid::Uuid::new_v4()));
        let mut ds = Dataset::load(dir.clone());
        let data = vec![1f32, 2., 3., 4.];
        let mut r = sample(LabelKind::Good, DVec2::X, data.clone(), 2).record;
        ds.add(r.clone(), &data).unwrap();
        r.kind = LabelKind::Stop;
        ds.add(r.clone(), &data).unwrap();
        let other = sample(LabelKind::Direction, DVec2::Y, data.clone(), 2).record;
        ds.add(other, &data).unwrap();

        let ds = Dataset::load(dir.clone());
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.counts(), (0, 1, 1, 0));
        let samples = ds.samples();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].record.kind, LabelKind::Stop);
        assert_eq!(samples[0].data, data);

        // GDS predictions can be added later and survive a reload.
        let mut ds = ds;
        let id = samples[1].record.id.clone();
        let g = GdsInfo { dir: [0., 1.], smallest: false, ..Default::default() };
        assert_eq!(ds.update_gds(&[(id.clone(), Some(g.clone()))]).unwrap(), 1);
        assert_eq!(ds.update_gds(&[(id.clone(), Some(g.clone()))]).unwrap(), 0);
        // Adding the ring geometry later counts as a change.
        let geo = GdsInfo { center_px: Some([10., 20.]), bands_px: vec![[5., 9.]], ..g.clone() };
        assert_eq!(ds.update_gds(&[(id.clone(), Some(geo))]).unwrap(), 1);
        assert_eq!(ds.update_gds(&[(id.clone(), Some(g.clone()))]).unwrap(), 1);
        let ds = Dataset::load(dir.clone());
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.gds_of(&id), Some(Some(g)));
        // That label points along +y, as does the GDS: right.
        let rec = ds.samples().into_iter().find(|s| s.record.id == id).unwrap().record;
        assert_eq!(score_gds(&rec), Some(true));
        let mut wrong = rec.clone();
        wrong.gds = Some(GdsInfo { dir: [0., -1.], smallest: false, ..Default::default() });
        assert_eq!(score_gds(&wrong), Some(false));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn search_fixes_bad_settings_and_cv_is_honest() {
        let w = 96;
        let mut samples = Vec::new();
        for (cx, cy) in [(-60., -40.), (160., -50.), (-50., 150.), (150., 140.), (48., -90.), (-90., 48.)] {
            let c = DVec2::new(cx, cy);
            samples.push(sample(LabelKind::Direction, c - DVec2::splat(47.5), rings(w, c), w as u32));
        }
        samples.push(sample(LabelKind::Stop, DVec2::ZERO, vec![0.5; w * w], w as u32));
        // A start that calls everything unsure.
        let mut bad = base();
        bad.min_radius_um = 50.;
        let report = evaluate(&samples, bad, &[]);
        assert_eq!(report.current_correct, 1);
        assert!(report.new_correct > report.current_correct, "{report:?}");
        assert!(report.is_improvement(), "{report:?}");

        // Already-good settings: nothing better to offer.
        let good = evaluate(&samples, report.new, &[]);
        assert!(!good.is_improvement(), "{good:?}");
    }

    /// Frames on device-like rings whose GDS placement is off by enough that
    /// the plain GDS direction is wrong; matching the scan fixes it. Blobs
    /// and noise are in every frame.
    #[test]
    fn matching_beats_plain_gds_when_the_gds_is_off() {
        use crate::analysis::ring_match::tests::{device_bands, scan};
        let w = 192u32;
        let fc = DVec2::splat(95.5);
        let mut samples = Vec::new();
        for (k, (truth, gds)) in [
            // True centre and where the GDS puts it (px).
            // The first four are ~70 px off sideways: more than 30° wrong.
            (DVec2::new(95., 0.), DVec2::new(165., 0.)),
            (DVec2::new(0., 95.), DVec2::new(0., 25.)),
            (DVec2::new(191., 95.), DVec2::new(191., 165.)),
            (DVec2::new(95., 191.), DVec2::new(25., 191.)),
            (DVec2::new(-50., -50.), DVec2::new(-50., -50.)),
            (DVec2::new(240., 240.), DVec2::new(240., 240.)),
        ]
        .into_iter()
        .enumerate()
        {
            let data = scan(w as usize, truth, &device_bands(), 0., 1., k as u64 + 1);
            let mut s = sample(LabelKind::Direction, truth - fc, data, w);
            s.record.gds = Some(GdsInfo {
                dir: (gds - fc).normalize().to_array(),
                smallest: false,
                center_px: Some(gds.to_array()),
                bands_px: device_bands().iter().map(|b| [b.0, b.1]).collect(),
            });
            samples.push(s);
        }
        let mut tune = base();
        tune.matching.mode = MatchMode::Off;
        let r = evaluate(&samples, tune, &[]);
        assert_eq!(r.match_n, 6);
        let by = |m: MatchMode| r.mode_correct.iter().find(|x| x.0 == m).unwrap().1;
        assert_eq!(by(MatchMode::Off), r.gds_correct, "{r:?}");
        assert_eq!(r.gds_correct, 2, "{r:?}");
        for m in [MatchMode::Smooth, MatchMode::SoftTwoTone, MatchMode::HardTwoTone] {
            assert_eq!(by(m), 6, "{m:?}: {r:?}");
        }
        assert!(r.match_improves(), "{r:?}");
        assert_ne!(r.new.matching.mode, MatchMode::Off);
    }

    /// Prints how the saved settings, retraining and the GDS do on the shared
    /// dataset (read-only):
    /// `cargo test --release --bin scan_control dataset_report -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dataset_report() {
        let samples = Dataset::load(Dataset::default_dir()).samples();
        let tune = TuneParams::load();
        // Ring kinds from the device GDS, shrunk to 0.75 as on the map.
        let scale: f64 = std::env::var("GDS_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.75);
        let polys = crate::scan_view::gds_image::read_polys("As_Implanted_MLA150.GDS");
        let kinds = crate::analysis::gds_prior::ring_kinds(
            &crate::analysis::gds_prior::find_ring_systems(&polys),
            scale,
        );
        println!("ring kinds (nm, scale {scale}): {kinds:?}");
        let cache = MatchCache::default();
        let p = tune.matching;
        println!("{:24} {:10} {:9} {:6} {}", "source", "kind", "detector", "gds", "blind (score, verdict)");
        // (direction gap, right) of every trusted blind match.
        let mut gaps: Vec<(f64, bool)> = Vec::new();
        for s in &samples {
            if let Some((_, m)) = cache.get_blind(s, &p, &kinds).filter(|m| m.1.score >= p.min_score) {
                let ok = score_blind(s, &tune, &p, &kinds, &cache).0;
                gaps.push((m.dir_gap(), ok));
                println!("  {:24} score {:.2} opposite {:.2} gap {:.3} {}", s.record.source, m.score, m.opposite_score, m.dir_gap(), if ok { "right" } else { "WRONG" });
            }
        }
        println!("direction confidence (gap) threshold: right/sure of {} trusted matches", gaps.len());
        for t in [0., 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2] {
            let sure: Vec<_> = gaps.iter().filter(|g| g.0 >= t).collect();
            let right = sure.iter().filter(|g| g.1).count();
            println!("  gap >= {t:.3}: {right}/{} right, covers {:.0}% of frames", sure.len(), 100. * sure.len() as f64 / samples.len() as f64);
        }
        for s in &samples {
            let gds = match score_gds(&s.record) {
                Some(true) => "right",
                Some(false) => "wrong",
                None => "-",
            };
            let det = if score(s, &tune) { "right" } else { "wrong" };
            let blind = match cache.get_blind(s, &p, &kinds) {
                Some((_, m)) => {
                    let (ok, conf) = score_blind(s, &tune, &p, &kinds, &cache);
                    let used = if conf { "match" } else { "detector" };
                    format!("{:.2} {} ({used})", m.score, if ok { "right" } else { "wrong" })
                }
                None => "-".into(),
            };
            println!("{:24} {:10} {:9} {:6} {blind}", s.record.source, format!("{:?}", s.record.kind), det, gds);
            if let (Ok(dir), Some(m)) = (std::env::var("DUMP_DIR"), cache.get_blind(s, &p, &kinds)) {
                let ok = score_blind(s, &tune, &p, &kinds, &cache).0;
                let name = format!(
                    "{:02}_{}_{}.png",
                    samples.iter().position(|x| std::ptr::eq(x, s)).unwrap_or(0),
                    s.record.source.trim_end_matches(".sxm").replace(['(', ')', ' '], ""),
                    if ok { "right" } else { "wrong" }
                );
                std::fs::create_dir_all(&dir).unwrap();
                dump_blind(&std::path::Path::new(&dir).join(name), s, &p, &kinds, m);
            }
        }
        let r = evaluate(&samples, tune, &kinds);
        let pct = |a: usize, n: usize| if n > 0 { 100. * a as f64 / n as f64 } else { 0. };
        println!("frames: {}", r.n);
        println!("current settings: {}/{} ({:.0}%)", r.current_correct, r.n, pct(r.current_correct, r.n));
        println!("new settings, held out: {}/{} ({:.0}%)", r.cv_correct, r.n, pct(r.cv_correct, r.n));
        println!("new settings, all: {}/{} ({:.0}%)", r.new_correct, r.n, pct(r.new_correct, r.n));
        println!("changes: {:?}", r.changes);
        println!("GDS only: {}/{} ({:.0}%)", r.gds_correct, r.gds_n, pct(r.gds_correct, r.gds_n));
        println!("GDS else detector: {}/{} ({:.0}%)", r.combined_correct, r.n, pct(r.combined_correct, r.n));
        println!("labels with ring geometry: {}", r.match_n);
        for (m, c) in &r.mode_correct {
            println!("  {}: {}/{}", m.name(), c, r.match_n);
        }
        println!("blind matching (else detector), all {} labels:", r.n);
        for (m, c, conf) in &r.blind {
            let name = if *m == MatchMode::Off { "detector only" } else { m.name() };
            println!("  {name}: {c}/{} ({:.0}%), {conf} confident", r.n, pct(*c, r.n));
        }
    }

    /// Blind matching on the forward scan, the backward scan (from the
    /// label's .sxm file) and the better fitting of the two, with timings.
    /// `SXM_DIR=dir cargo test --release --bin scan_control forward_and_backward -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn forward_and_backward() {
        use crate::analysis::sxm_match::{backward_for, list_sxm};
        let dir = std::env::var("SXM_DIR").unwrap_or(r"F:\Data\LT SPM\2026\November".into());
        let files = list_sxm(Path::new(&dir));
        let samples = Dataset::load(Dataset::default_dir()).samples();
        let tune = TuneParams::load();
        let polys = crate::scan_view::gds_image::read_polys("As_Implanted_MLA150.GDS");
        let kinds = crate::analysis::gds_prior::ring_kinds(&crate::analysis::gds_prior::find_ring_systems(&polys), 0.75);
        let p = tune.matching;
        let cache = MatchCache::default();
        let (mut n, mut fwd_ok, mut bwd_ok, mut best_ok, mut bwd_better) = (0, 0, 0, 0, 0);
        let mut time = std::time::Duration::ZERO;
        // The cache is keyed by sample address: keep every backward sample alive.
        let pairs: Vec<(&Sample, Sample)> = samples
            .iter()
            .filter_map(|s| {
                let Some(path) = files.iter().find(|f| f.file_stem().is_some_and(|x| x == s.record.source.as_str())) else {
                    println!("{}: no file", s.record.source);
                    return None;
                };
                let Some(b) = backward_for(path, &s.data, [s.record.w, s.record.h]) else {
                    println!("{}: no backward scan", s.record.source);
                    return None;
                };
                Some((s, Sample { record: s.record.clone(), data: b }))
            })
            .collect();
        for (s, sb) in &pairs {
            let (s, sb) = (*s, sb);
            let t = std::time::Instant::now();
            let mf = cache.get_blind(s, &p, &kinds);
            let mb = cache.get_blind(sb, &p, &kinds);
            time += t.elapsed();
            let f_ok = score_blind(s, &tune, &p, &kinds, &cache).0;
            let b_ok = score_blind(sb, &tune, &p, &kinds, &cache).0;
            let use_b = mb.is_some_and(|b| mf.is_none_or(|f| b.1.score > f.1.score));
            let ok = if use_b { b_ok } else { f_ok };
            n += 1;
            fwd_ok += f_ok as usize;
            bwd_ok += b_ok as usize;
            best_ok += ok as usize;
            bwd_better += use_b as usize;
            println!(
                "{:16} fwd {:.2} {:5} bwd {:.2} {:5} -> {}",
                s.record.source,
                mf.map_or(0., |m| m.1.score),
                if f_ok { "right" } else { "wrong" },
                mb.map_or(0., |m| m.1.score),
                if b_ok { "right" } else { "wrong" },
                if ok { "right" } else { "WRONG" }
            );
        }
        println!("{n} frames with both scans: forward {fwd_ok}, backward {bwd_ok}, better fit {best_ok} (backward chosen {bwd_better})");
        println!("{:?} per frame for both scans", time / n.max(1) as u32);
    }

    /// Three panels: flattened scan, its two tones, and the two tones with
    /// the matched ring edges (green), the label direction (white) and the
    /// matched direction (red).
    fn dump_blind(path: &Path, s: &Sample, p: &MatchParams, kinds_nm: &[Vec<(f64, f64)>], m: (usize, RingMatch)) {
        use crate::analysis::ring_match::blind_prepared;
        const UP: usize = 4;
        let (kind, m) = m;
        let r = &s.record;
        let bands: Vec<(f64, f64)> = kinds_nm[kind].iter().map(|b| (b.0 / r.nm_per_px, b.1 / r.nm_per_px)).collect();
        let size = [r.w, r.h];
        let smooth = MatchParams { mode: MatchMode::Smooth, ..*p };
        let tone = MatchParams { mode: MatchMode::HardTwoTone, ..*p };
        let (Some(a), Some(b)) = (
            blind_prepared(&s.data, size, &bands, &smooth, m.raised),
            blind_prepared(&s.data, size, &bands, &tone, m.raised),
        ) else {
            return;
        };
        let (w, h, k) = (a.w, a.h, a.k as f64);
        let mut im = image::RgbImage::new((3 * w * UP + 2 * UP) as u32, (h * UP) as u32);
        let gray = |img: &crate::analysis::ring_match::Prepared| {
            let mut v: Vec<f64> = img.v.iter().copied().filter(|x| x.is_finite()).collect();
            v.sort_by(f64::total_cmp);
            let (lo, hi) = (v[v.len() / 100], v[v.len() * 99 / 100].max(v[v.len() / 100] + 1e-12));
            move |x: f64| (((x - lo) / (hi - lo)).clamp(0., 1.) * 255.) as u8
        };
        for (panel, img) in [(0usize, &a), (1, &b), (2, &b)] {
            let g = gray(img);
            for y in 0..h * UP {
                for x in 0..w * UP {
                    let v = img.v[(y / UP) * w + x / UP];
                    let c = if v.is_finite() { [g(v); 3] } else { [60, 0, 60] };
                    im.put_pixel((panel * (w * UP + UP) + x) as u32, y as u32, image::Rgb(c));
                }
            }
        }
        let x0 = 2 * (w * UP + UP);
        let mut put = |x: f64, y: f64, c: [u8; 3]| {
            // Frame pixels → panel 3 image pixels.
            let (wx, wy) = ((x - (k - 1.) / 2.) / k, (y - (k - 1.) / 2.) / k);
            let (px, py) = ((wx + 0.5) * UP as f64, (wy + 0.5) * UP as f64);
            if px >= 0. && py >= 0. && px < (w * UP) as f64 && py < (h * UP) as f64 {
                im.put_pixel((x0 + px as usize) as u32, py as u32, image::Rgb(c));
            }
        };
        let c = m.center_px;
        for &(r0, r1) in &bands {
            for rad in [r0 - m.width_bias_px / 2., r1 + m.width_bias_px / 2.] {
                let n = (rad * 8.).max(64.) as usize;
                for i in 0..n {
                    let t = i as f64 / n as f64 * std::f64::consts::TAU;
                    put(c.x + rad * t.cos(), c.y + rad * t.sin(), [0, 220, 0]);
                }
            }
        }
        let fc = DVec2::new((r.w as f64 - 1.) / 2., (r.h as f64 - 1.) / 2.);
        let len = 0.4 * r.w.min(r.h) as f64;
        for (d, col) in [(r.dir(), [255, 255, 255]), ((c - fc).normalize_or_zero(), [255, 40, 40])] {
            for i in 0..400 {
                let q = fc + d * len * i as f64 / 400.;
                for o in [-1., 0., 1.] {
                    put(q.x + o, q.y, col);
                    put(q.x, q.y + o, col);
                }
            }
        }
        im.save(path).unwrap();
    }

    /// Finds the Nanonis .sxm file each label's frame came from (by its Z
    /// data) and prints the mapping; with `WRITE=1` it also stores the file
    /// name as the label's source.
    /// `SXM_DIR=dir cargo test --release --bin scan_control name_labels_from_sxm -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn name_labels_from_sxm() {
        use crate::analysis::sxm_match::find_matching_sxm;
        let dir = std::env::var("SXM_DIR").unwrap_or(r"F:\Data\LT SPM\2026\November".into());
        let mut ds = Dataset::load(Dataset::default_dir());
        let samples = ds.samples();
        let files = crate::analysis::sxm_match::list_sxm(Path::new(&dir));
        println!("{} .sxm files under {dir}", files.len());
        let mut updates = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            let found = find_matching_sxm(&s.data, [s.record.w, s.record.h], &files);
            let name = found.as_ref().map(|p| p.file_stem().unwrap().to_string_lossy().to_string());
            println!(
                "{i:02} {:24} {:?} {} -> {}",
                s.record.source,
                s.record.kind,
                &s.record.time.get(..19).unwrap_or(""),
                found.as_ref().map_or("NOT FOUND".into(), |p| p.display().to_string())
            );
            if let Some(n) = name {
                updates.push((s.record.id.clone(), n));
            }
        }
        if std::env::var("WRITE").is_ok_and(|v| v == "1") {
            println!("updated {} labels", ds.set_sources(&updates).unwrap());
        }
    }

    /// Times one analysis per labelled frame in the current build profile:
    /// `cargo test --bin scan_control time_analysis -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn time_analysis() {
        let samples = Dataset::load(Dataset::default_dir()).samples();
        let tune = TuneParams::load();
        for s in &samples {
            let t = std::time::Instant::now();
            let _ = score(s, &tune);
            println!("{}x{}	{:.1} ms	{}", s.record.w, s.record.h, t.elapsed().as_secs_f64() * 1e3, s.record.source);
        }
    }

    /// Seeds the shared dataset with the Nov 2026 training scans, labelled
    /// from the arrows drawn on their jpgs. Run once:
    /// `cargo test --bin scan_control seed_ring_dataset -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn seed_ring_dataset() {
        use crate::analysis::ring_center::sxm_harness::load_z;
        let dir = std::env::var("RING_SEED_DIR")
            .unwrap_or(r"F:\Data\LT SPM\2026\November\ring training data".into());
        // Compass degrees (0 = up, 90 = right); None = should stop.
        let labels: [(&str, Option<f64>); 10] = [
            ("20261102_0007", Some(300.)),
            ("20261102_0011", Some(270.)),
            ("20261103_0001", Some(270.)),
            ("20261103_0003", Some(0.)),
            ("20261105_0001", Some(300.)),
            ("20261105_0003", Some(270.)),
            ("20261109_0002", Some(270.)),
            ("20261109_0011", Some(75.)),
            ("20261110_0001", Some(340.)),
            ("20261105_0004", None),
        ];
        let mut ds = Dataset::load(Dataset::default_dir());
        for (stem, compass) in labels {
            let path = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .find(|p| {
                    p.extension().is_some_and(|e| e == "sxm")
                        && p.file_stem().is_some_and(|s| s.to_string_lossy().contains(stem))
                });
            let Some(path) = path else {
                println!("{stem}: not found");
                continue;
            };
            let (z, w, h, range) = load_z(&path).unwrap();
            let (kind, d) = match compass {
                // Pixel axes: +x right, +y down.
                Some(c) => {
                    let c = f64::to_radians(c);
                    (LabelKind::Direction, DVec2::new(c.sin(), -c.cos()))
                }
                None => (LabelKind::Stop, DVec2::ZERO),
            };
            let rec = Record {
                id: format!("seed-{stem}"),
                kind,
                dir: d.to_array(),
                w: w as u32,
                h: h as u32,
                nm_per_px: range[0] as f64 * 1e9 / w as f64,
                source: path.file_name().unwrap().to_string_lossy().into(),
                time: "seed".into(),
                gds: None,
            };
            ds.add(rec, &z).unwrap();
            println!("{stem}: added");
        }
        let samples = ds.samples();
        let report = evaluate(&samples, TuneParams::default(), &[]);
        println!("{report:#?}");
    }
}
