//! Labelled tip-quality examples: the user's judgement of archive scans,
//! with the detector's scores, for checking and tuning the detector and,
//! later, training a learned model.
//!
//! Kept under `CONFIG_DIR/tip_training`: `labels.jsonl` (one record per
//! line) and, when asked, `<id>.bin` with the image (f32 little-endian,
//! row-major) so the set doesn't depend on the archive staying put.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::analysis::tip_quality::{analyse, TipParams, TipReport, Verdict, FEATURE_NAMES};

/// The user's judgement of a scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TipLabel {
    Good,
    /// Two apexes: one ghost copy.
    Double,
    /// Three or more apexes.
    Multi,
    Blunt,
    /// Tip changes, streaks.
    Unstable,
}
impl TipLabel {
    pub const ALL: [TipLabel; 5] = [TipLabel::Good, TipLabel::Double, TipLabel::Multi, TipLabel::Blunt, TipLabel::Unstable];
    pub fn name(self) -> &'static str {
        match self {
            TipLabel::Good => "Good",
            TipLabel::Double => "Double",
            TipLabel::Multi => "Multi",
            TipLabel::Blunt => "Blunt",
            TipLabel::Unstable => "Unstable",
        }
    }
    /// Keyboard shortcut in the labelling view.
    pub fn key(self) -> egui::Key {
        match self {
            TipLabel::Good => egui::Key::G,
            TipLabel::Double => egui::Key::D,
            TipLabel::Multi => egui::Key::M,
            TipLabel::Blunt => egui::Key::B,
            TipLabel::Unstable => egui::Key::U,
        }
    }
    pub fn is_bad(self) -> bool {
        self != TipLabel::Good
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    /// The .sxm file it came from.
    pub path: String,
    pub channel: String,
    pub label: TipLabel,
    /// Unix time, s.
    pub unix: u64,
    pub size: [u32; 2],
    /// Frame (unit quad to nm), for the pixel size.
    pub frame: [f64; 6],
    /// The image was saved as `<id>.bin`.
    #[serde(default)]
    pub has_data: bool,
    /// The detector's result when labelled (or last re-analysed).
    pub report: TipReport,
}

pub fn dir() -> PathBuf {
    crate::app::CONFIG_DIR.config_local_dir().join("tip_training")
}

/// All records (later lines replace earlier ones for the same file).
pub fn load() -> Vec<Record> {
    let text = std::fs::read_to_string(dir().join("labels.jsonl")).unwrap_or_default();
    let mut out: Vec<Record> = Vec::new();
    for r in text.lines().filter_map(|l| serde_json::from_str::<Record>(l).ok()) {
        match out.iter_mut().find(|o| o.path == r.path) {
            Some(o) => *o = r,
            None => out.push(r),
        }
    }
    out
}

pub fn save_all(records: &[Record]) -> Result<(), String> {
    std::fs::create_dir_all(dir()).map_err(|e| e.to_string())?;
    let text: String = records.iter().filter_map(|r| serde_json::to_string(r).ok()).map(|l| l + "\n").collect();
    std::fs::write(dir().join("labels.jsonl"), text).map_err(|e| e.to_string())
}

pub fn save_data(id: &str, data: &[f32]) -> Result<(), String> {
    std::fs::create_dir_all(dir()).map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir().join(format!("{id}.bin")), bytes).map_err(|e| e.to_string())
}

fn load_data(id: &str, size: [u32; 2]) -> Option<Vec<f32>> {
    let bytes = std::fs::read(dir().join(format!("{id}.bin"))).ok()?;
    let d: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (d.len() == size[0] as usize * size[1] as usize).then_some(d)
}

/// Every .sxm file under `root`, sorted.
pub fn find_sxm(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("sxm")) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Run the detector again on a record (from its saved image, else its
/// file), e.g. after the detector changed.
pub fn reanalyse(r: &Record, params: &TipParams) -> Option<TipReport> {
    let frame = glam::DAffine2::from_cols_array(&r.frame);
    if r.has_data {
        if let Some(d) = load_data(&r.id, r.size) {
            return Some(analyse(&d, None, r.size, frame, params));
        }
    }
    let f = crate::analysis::sxm_match::load_frame(Path::new(&r.path), Some(&r.channel)).ok()?;
    Some(analyse(&f.forward, Some(&f.backward), f.size, f.frame, params))
}

/// The detector against the labels.
#[derive(Debug, Clone, Default)]
pub struct Evaluation {
    /// counts[label][verdict]: verdict 0 Good, 1 Suspect, 2 Poor, 3 Unsure.
    pub counts: [[usize; 4]; 5],
    /// Bad (any non-Good label) caught as Suspect or Poor, and Good passed
    /// as Good, over the judged (not Unsure) records.
    pub caught_bad: (usize, usize),
    pub passed_good: (usize, usize),
    /// The same split by pixel size: (upper bound nm, caught, bad, passed, good).
    pub by_scale: Vec<(f64, usize, usize, usize, usize)>,
}
impl Evaluation {
    pub fn balanced_accuracy(&self) -> f64 {
        let r = |(a, b): (usize, usize)| if b == 0 { 0.5 } else { a as f64 / b as f64 };
        0.5 * (r(self.caught_bad) + r(self.passed_good))
    }
}

fn verdict_index(v: &Verdict) -> usize {
    match v {
        Verdict::Good => 0,
        Verdict::Suspect => 1,
        Verdict::Poor => 2,
        Verdict::Unsure(_) => 3,
    }
}

/// Judge every record's stored scores with `params` and compare.
pub fn evaluate(records: &[Record], params: &TipParams) -> Evaluation {
    let scales = [0.05, 0.2, 1., 5., f64::INFINITY];
    let mut e = Evaluation { by_scale: scales.iter().map(|s| (*s, 0, 0, 0, 0)).collect(), ..Default::default() };
    for r in records {
        let (v, _) = params.judge(&r.report);
        let vi = verdict_index(&v);
        let li = TipLabel::ALL.iter().position(|l| *l == r.label).unwrap_or(0);
        e.counts[li][vi] += 1;
        if vi == 3 {
            continue;
        }
        let si = scales.iter().position(|s| r.report.px_nm <= *s).unwrap_or(scales.len() - 1);
        if r.label.is_bad() {
            e.caught_bad.1 += 1;
            e.by_scale[si].2 += 1;
            if vi > 0 {
                e.caught_bad.0 += 1;
                e.by_scale[si].1 += 1;
            }
        } else {
            e.passed_good.1 += 1;
            e.by_scale[si].4 += 1;
            if vi == 0 {
                e.passed_good.0 += 1;
                e.by_scale[si].3 += 1;
            }
        }
    }
    e
}

/// Search the multi-tip and instability thresholds for the best balanced
/// accuracy on the labels (Suspect counts as caught). Returns the tuned
/// params and their evaluation.
pub fn tune(records: &[Record], start: &TipParams) -> (TipParams, Evaluation) {
    let mut best = (*start, evaluate(records, start));
    let grid: Vec<f32> = (1..=30).map(|i| i as f32 * 0.025).collect();
    for &sm in &grid {
        for &si in &grid {
            let mut p = *start;
            p.suspect_multi = sm;
            p.poor_multi = (sm * 1.6).min(1.);
            p.suspect_instability = si;
            p.poor_instability = (si * 2.).min(1.);
            let e = evaluate(records, &p);
            if e.balanced_accuracy() > best.1.balanced_accuracy() + 1e-9 {
                best = (p, e);
            }
        }
    }
    best
}

/// Write `features.csv` (one row per record: label, file, the detector's
/// feature vector) next to the labels, for training a model.
pub fn export_csv(records: &[Record]) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir()).map_err(|e| e.to_string())?;
    let mut text = format!("id,label,path,width,height,has_data,{}
", FEATURE_NAMES.join(","));
    for r in records {
        let f: Vec<String> = r.report.features().iter().map(|v| v.to_string()).collect();
        text += &format!(
            "{},{},\"{}\",{},{},{},{}
",
            r.id,
            r.label.name(),
            r.path.replace('"', "'"),
            r.size[0],
            r.size[1],
            r.has_data,
            f.join(",")
        );
    }
    let path = dir().join("features.csv");
    std::fs::write(&path, text).map_err(|e| e.to_string())?;
    Ok(path)
}
