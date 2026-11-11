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
};

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    ring_center::RingParams,
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
}

impl Default for TuneParams {
    fn default() -> Self {
        Self { params: RingParams::default(), min_radius_um: 1.5 }
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
    pub fn counts(&self) -> (usize, usize, usize) {
        let n = |k| self.records.iter().filter(|r| r.kind == k).count();
        (n(LabelKind::Good), n(LabelKind::Direction), n(LabelKind::Stop))
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
        return r.kind == LabelKind::Stop;
    };
    let label = r.dir();
    match (r.kind, res.verdict) {
        (LabelKind::Stop, StampVerdict::Unsure(_)) => true,
        (LabelKind::Stop, _) => false,
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
}

impl Report {
    /// Offer the new settings only if they generalise better and differ.
    pub fn is_improvement(&self) -> bool {
        self.cv_correct > self.current_correct && !self.changes.is_empty()
    }
}

/// Score the current settings, cross-validate the search, and find new
/// settings on all samples.
pub fn evaluate(samples: &[Sample], current: TuneParams) -> Report {
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
    let (new, new_correct) = search(&all, current);
    Report { n, current_correct, cv_correct, changes: new.changes_from(&current), new, new_correct }
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
            },
            data,
        }
    }

    fn base() -> TuneParams {
        TuneParams {
            params: RingParams { target_px: None, ..RingParams::default() },
            min_radius_um: 0.5,
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
        assert!(score(&sample(LabelKind::Stop, DVec2::ZERO, flat, w as u32), &base()));
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
        assert_eq!(ds.counts(), (0, 1, 1));
        let samples = ds.samples();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].record.kind, LabelKind::Stop);
        assert_eq!(samples[0].data, data);
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
        let report = evaluate(&samples, bad);
        assert_eq!(report.current_correct, 1);
        assert!(report.new_correct > report.current_correct, "{report:?}");
        assert!(report.is_improvement(), "{report:?}");

        // Already-good settings: nothing better to offer.
        let good = evaluate(&samples, report.new);
        assert!(!good.is_improvement(), "{good:?}");
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
            };
            ds.add(rec, &z).unwrap();
            println!("{stem}: added");
        }
        let samples = ds.samples();
        let report = evaluate(&samples, TuneParams::default());
        println!("{report:#?}");
    }
}
