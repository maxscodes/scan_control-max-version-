//! Automated coarse-motor calibration: scan, step one way, scan, …, step back
//! the same number of times, then the same for Y, and measure each move from
//! how far the image moved between consecutive scans.
//!
//! With `repeats` = n: S0 · n × (+X · scan) · n × (−X · scan) · n × (+Y ·
//! scan) · n × (−Y · scan), ending where it started. Each direction's result
//! is the average of its n moves; their spread shows how repeatable it is.
//!
//! With several step `sizes` the moves go from small to large step counts
//! (each size +X, −X, +Y, −Y, so the tip keeps coming back), every move is
//! searched for near where the smaller ones say it went, and each direction
//! is fitted with a line, move = per-move shift + steps × step: the slope is
//! the step (a 2-D vector, so cross-coupling shows), separate from what every
//! move adds whatever its length (approach, drift, short first steps).
//! Every scan goes down. Alternating would save the tip's trip back to the
//! top, but an up scan and a down scan of the same spot don't line up (creep
//! and drift during the scan shift them by hundreds of nm along y), and that
//! offset went straight into the measured moves.

use glam::{DAffine2, DVec2};

use crate::{
    analysis::register::{image_shift, shift_near, GOOD_SCORE},
    connection::{
        nanonis::{
            course_motion::CoarseCalibration,
            sequence::{CoarseSpec, Dir, FrameSpec, Position, ScanEntry, ScanStart, Sequence, Speed},
        },
        FrameData,
    },
};

/// What the user asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalPlan {
    /// Steps per move along X and Y (positive).
    pub steps: [i32; 2],
    /// Moves per direction (≥ 1); more average out step-to-step scatter.
    /// With several step sizes: moves per direction and size.
    pub repeats: u32,
    /// Different step counts per direction, evenly spaced from `min_steps`
    /// up to `steps` (most moves large, as in grids); 1 = every move `steps`.
    pub sizes: u32,
    pub min_steps: [i32; 2],
    /// Which motor axes to calibrate (X, Y): one can be redone alone.
    pub axes: [bool; 2],
    /// Wait after each approach before scanning, s.
    pub settle_s: f64,
    pub withdraw: bool,
    pub approach: bool,
    /// Driver amplitudes (V) when the run started.
    pub volts: DVec2,
    /// Square frame (nm) and lines (= pixels per line) for the scans.
    pub frame_nm: f64,
    pub lines: u32,
    /// How each calibration scan is started.
    pub scan_start: ScanStart,
    /// Forward time per line (s) for the calibration scans; `None` keeps
    /// Nanonis' speed.
    pub line_time_s: Option<f64>,
    /// Forward time per line to put back after.
    pub restore_line_time_s: Option<f64>,
    /// Frame size (nm) and resolution [pixels, lines] to put back after.
    pub restore: Option<([f64; 2], [u32; 2])>,
}

impl CalPlan {
    /// Step counts used along `axis`, small to large.
    pub fn counts(&self, axis: usize) -> Vec<i32> {
        let max = self.steps[axis].max(1);
        let k = self.sizes.max(1) as usize;
        if k == 1 {
            return vec![max];
        }
        let min = self.min_steps[axis].clamp(1, max);
        let mut v: Vec<i32> = (0..k)
            .map(|i| {
                let t = i as f64 / (k - 1) as f64;
                (min as f64 + (max - min) as f64 * t).round() as i32
            })
            .collect();
        v.dedup();
        v
    }
    /// The moves in order: (motor axis, positive direction, steps > 0).
    pub fn moves_with_steps(&self) -> Vec<(usize, bool, i32)> {
        let n = self.repeats.max(1) as usize;
        if self.sizes <= 1 {
            return [(0, true), (0, false), (1, true), (1, false)]
                .iter()
                .filter(|&&(axis, _)| self.axes[axis])
                .flat_map(|&(axis, pos)| std::iter::repeat_n((axis, pos, self.steps[axis]), n))
                .collect();
        }
        // Small to large; each size there and back, so the tip keeps
        // returning to where it started.
        let counts = [self.counts(0), self.counts(1)];
        let sizes = counts[0].len().max(counts[1].len());
        let mut out = Vec::new();
        for i in 0..sizes {
            for _ in 0..n {
                for axis in 0..2 {
                    if !self.axes[axis] {
                        continue;
                    }
                    if let Some(&c) = counts[axis].get(i) {
                        out.push((axis, true, c));
                        out.push((axis, false, c));
                    }
                }
            }
        }
        out
    }
    /// The moves in order: (motor axis, positive direction).
    pub fn moves(&self) -> Vec<(usize, bool)> {
        self.moves_with_steps().into_iter().map(|(a, p, _)| (a, p)).collect()
    }
    /// Number of scans the run takes.
    pub fn scans(&self) -> usize {
        self.moves().len() + 1
    }
    /// Rough length of the run, s: the scans at `line_s` (forward +
    /// backward) per line, plus `move_s` per withdraw/move/approach and the
    /// settle time after each.
    pub fn estimate_s(&self, line_s: f64, move_s: f64) -> f64 {
        let scan = self.lines as f64 * line_s + self.scan_start.extra_s();
        self.scans() as f64 * scan + self.moves().len() as f64 * (move_s + self.settle_s)
    }
}

fn dir_name(axis: usize, positive: bool) -> &'static str {
    match (axis, positive) {
        (0, true) => "+X",
        (0, false) => "\u{2212}X",
        (_, true) => "+Y",
        (_, false) => "\u{2212}Y",
    }
}

/// The scans and moves as sequence entries.
pub fn build_sequence(plan: &CalPlan) -> Sequence {
    let moves = plan.moves_with_steps();
    let mut entries = Vec::with_capacity(moves.len() + 2);
    entries.push(ScanEntry {
        name: "cal start".into(),
        // The calibration frame size and resolution (the frame stays
        // centred where it is); the later scans keep them.
        frame: FrameSpec { position: Position::Keep, size: Some([plan.frame_nm; 2]), angle: None },
        resolution: Some([plan.lines; 2]),
        speed: plan.line_time_s.map(Speed::LineTime),
        dir: Dir::Down,
        ..Default::default()
    });
    for (k, &(axis, pos, n)) in moves.iter().enumerate() {
        let mut steps = [0, 0];
        steps[axis] = if pos { n } else { -n };
        let name = if plan.sizes > 1 {
            format!("cal {} {n} steps", dir_name(axis, pos))
        } else {
            format!("cal {} {}", dir_name(axis, pos), k % plan.repeats.max(1) as usize + 1)
        };
        entries.push(ScanEntry {
            name,
            coarse: Some(CoarseSpec { steps, withdraw: plan.withdraw, approach: plan.approach }),
            settle_s: plan.settle_s,
            dir: Dir::Down,
            ..Default::default()
        });
    }
    let mut seq = Sequence { entries, scan_start: plan.scan_start };
    seq.entries.extend(restore_sequence(plan).entries);
    seq
}

/// Just putting the frame size and resolution back (also used after an
/// aborted run).
pub fn restore_sequence(plan: &CalPlan) -> Sequence {
    let entries = plan
        .restore
        .map(|(size, res)| ScanEntry {
            name: "restore frame".into(),
            frame: FrameSpec { position: Position::Keep, size: Some(size), angle: None },
            resolution: Some(res),
            speed: plan.line_time_s.and(plan.restore_line_time_s).map(Speed::LineTime),
            scan: false,
            stamp: false,
            ..Default::default()
        })
        .into_iter()
        .collect();
    Sequence { entries, scan_start: plan.scan_start }
}

/// One scan of the run: forward image and the scan frame (piezo nm).
#[derive(Clone)]
pub struct CalFrame {
    pub data: FrameData,
    pub frame: DAffine2,
    /// Where the image came from (the .sxm name, or "live view").
    pub source: String,
}
impl std::fmt::Debug for CalFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CalFrame({}\u{d7}{})", self.data.size[0], self.data.size[1])
    }
}

/// One coarse move of the run and how far its scans say it went.
#[derive(Debug, Clone, Copy)]
pub struct MoveMeasure {
    pub axis: usize,
    pub positive: bool,
    /// Signed steps of this move.
    pub steps: i32,
    /// Content shift of the scan after the move relative to the one before,
    /// pixels (`image_shift` convention), if known: the automatic match, or
    /// the user's alignment when `manual`.
    pub shift_px: Option<DVec2>,
    /// The automatic match, kept to go back to.
    pub auto_px: Option<DVec2>,
    /// Score of the automatic match.
    pub score: f64,
    /// Aligned by eye.
    pub manual: bool,
}
impl MoveMeasure {
    pub fn name(&self) -> &'static str {
        dir_name(self.axis, self.positive)
    }
}

/// Measured result for one motor direction (average of its moves).
#[derive(Debug, Clone, Copy)]
pub struct DirResult {
    pub axis: usize,
    pub positive: bool,
    /// Signed steps per move (the largest, with several sizes) and the
    /// smallest.
    pub steps: i32,
    pub steps_min: i32,
    /// Tip move (nm, scan-area axes) for `steps` steps: the average, or
    /// from the fitted line.
    #[allow(dead_code)] // kept with the result for inspection
    pub moved_nm: DVec2,
    /// Tip move per signed step, nm (scan-area axes).
    pub per_step_nm: DVec2,
    /// With several step sizes: what every move adds whatever its length,
    /// nm (the fitted line's intercept).
    pub offset_nm: Option<DVec2>,
    /// New calibration column (m per step per V).
    pub column: DVec2,
    pub old_column: DVec2,
    /// Worst automatic match among the moves used (1 = identical; 1 if
    /// all were aligned by eye).
    pub score: f64,
    /// Moves that matched well or were aligned by eye, of all moves in
    /// this direction, and how many of them by eye.
    pub good: usize,
    pub total: usize,
    pub manual: usize,
    /// Spread of nm/step between the moves used, % of the mean (0 with
    /// one); with several sizes, the scatter about the fitted line, % of
    /// the average move.
    pub spread_pct: f64,
    /// Apply this row (default: when a move matched well).
    pub use_it: bool,
}
impl DirResult {
    pub fn name(&self) -> &'static str {
        dir_name(self.axis, self.positive)
    }
    /// Distance per step, nm.
    pub fn nm_per_step(&self) -> f64 {
        self.per_step_nm.length()
    }
    /// Tip move per step taken in this direction (sign included), nm.
    pub fn step_vector_nm(&self) -> DVec2 {
        self.per_step_nm * self.steps.signum() as f64
    }
}

#[derive(Debug, Clone)]
pub struct CalResult {
    pub rows: Vec<DirResult>,
    /// Why there are no (or fewer) rows.
    pub problem: Option<String>,
    pub volts: DVec2,
    /// The scans (move k goes from scan k to scan k + 1) and the moves.
    pub frames: Vec<CalFrame>,
    pub moves: Vec<MoveMeasure>,
    old: CoarseCalibration,
}

/// Tip move (nm, scan-area axes) for a content shift (pixels) between two
/// scans of `frame`: the image content moves the opposite way to the tip.
pub fn shift_to_tip(frame: &CalFrame, shift_px: DVec2) -> DVec2 {
    let [w, h] = frame.data.size;
    // Pixel (column, row) steps → frame-local units (row 0 = local y −0.5)
    // → piezo nm through the frame's axes.
    -(frame.frame.matrix2 * DVec2::new(shift_px.x / w.max(1) as f64, shift_px.y / h.max(1) as f64))
}

/// Content shift (pixels) between two scans of `frame` for a tip move (nm):
/// the inverse of [`shift_to_tip`].
pub fn tip_to_shift(frame: &CalFrame, tip: DVec2) -> DVec2 {
    let [w, h] = frame.data.size;
    let m = frame.frame.matrix2;
    if m.determinant().abs() < 1e-12 {
        return DVec2::ZERO;
    }
    let local = m.inverse() * tip;
    -DVec2::new(local.x * w as f64, local.y * h as f64)
}

/// Content shift (pixels) of `b` relative to `a` near the shift a tip move
/// of `predicted` nm would give, and the match score.
fn match_near(a: &CalFrame, b: &CalFrame, predicted: DVec2) -> Option<(DVec2, f64)> {
    let [w, h] = a.data.size;
    if b.data.size != [w, h] {
        return None;
    }
    let guess = tip_to_shift(a, predicted);
    let radius = (0.3 * guess.length()).max(0.08 * w.max(h) as f64);
    let s = shift_near(&a.data.data, &b.data.data, w as usize, h as usize, (guess.x, guess.y), radius, 0.15)?;
    Some((DVec2::new(s.dx, s.dy), s.score))
}

/// Content shift (pixels) of `b` relative to `a` and the match score.
fn match_scans(a: &CalFrame, b: &CalFrame) -> Option<(DVec2, f64)> {
    let [w, h] = a.data.size;
    if b.data.size != [w, h] {
        return None;
    }
    let s = image_shift(&a.data.data, &b.data.data, w as usize, h as usize)?;
    Some((DVec2::new(s.dx, s.dy), s.score))
}

/// Tip move (nm, scan-area axes) between two scans of the same frame, and
/// the match score.
#[cfg(test)]
pub fn tip_move(a: &CalFrame, b: &CalFrame) -> Option<(DVec2, f64)> {
    match_scans(a, b).map(|(s, score)| (shift_to_tip(a, s), score))
}

/// Measure every move and turn each direction into a calibration column.
pub fn compute(frames: &[CalFrame], plan: &CalPlan, old: &CoarseCalibration) -> CalResult {
    let mut problem = None;
    if frames.len() < plan.scans() {
        problem = Some(format!(
            "Stopped after {} of {} scans: the {} moves done are used.",
            frames.len(),
            plan.scans(),
            frames.len().saturating_sub(1)
        ));
    } else if frames.len() > plan.scans() {
        problem = Some(format!("{} scans captured for a {}-scan run", frames.len(), plan.scans()));
    }
    // In order, so each move is searched for near where the earlier (smaller)
    // moves the same way say it went: a look-alike far from there can't win.
    let mut known: Vec<((usize, bool), i32, DVec2)> = Vec::new();
    let mut moves = Vec::new();
    for (k, &(axis, positive, n)) in plan.moves_with_steps().iter().enumerate() {
        let steps = if positive { n } else { -n };
        let m = match (frames.get(k), frames.get(k + 1)) {
            (Some(a), Some(b)) => {
                let mine: Vec<_> = known.iter().filter(|kk| kk.0 == (axis, positive)).collect();
                let guided = (!mine.is_empty()).then(|| {
                    let num: DVec2 = mine.iter().map(|kk| kk.2 * kk.1 as f64).sum();
                    let den: f64 = mine.iter().map(|kk| (kk.1 as f64).powi(2)).sum();
                    match_near(a, b, num / den * steps as f64)
                });
                guided.flatten().filter(|g| g.1 >= GOOD_SCORE).or_else(|| match_scans(a, b))
            }
            _ => None,
        };
        if let (Some((shift, score)), Some(a)) = (m, frames.get(k)) {
            if score >= GOOD_SCORE {
                known.push(((axis, positive), steps, shift_to_tip(a, shift)));
            }
        }
        moves.push(MoveMeasure {
            axis,
            positive,
            steps,
            shift_px: m.map(|m| m.0),
            auto_px: m.map(|m| m.0),
            score: m.map_or(0., |m| m.1),
            manual: false,
        });
    }
    let mut r = CalResult {
        rows: Vec::new(),
        problem,
        volts: plan.volts,
        frames: frames.to_vec(),
        moves,
        old: *old,
    };
    r.update_rows(None);
    if r.rows.is_empty() && r.problem.is_none() {
        r.problem = Some("no moves could be measured".into());
    }
    r
}

impl CalResult {
    /// The scans before and after move `k`.
    pub fn pair(&self, k: usize) -> Option<(&CalFrame, &CalFrame)> {
        Some((self.frames.get(k)?, self.frames.get(k + 1)?))
    }
    /// Tip move (nm) of move `k` for a content shift (pixels).
    pub fn tip_for(&self, k: usize, shift_px: DVec2) -> Option<DVec2> {
        Some(shift_to_tip(self.frames.get(k)?, shift_px))
    }
    /// Use the user's alignment for move `k`.
    pub fn set_manual(&mut self, k: usize, shift_px: DVec2) {
        let Some(m) = self.moves.get_mut(k) else { return };
        m.shift_px = Some(shift_px);
        m.manual = true;
        let dir = (m.axis, m.positive);
        self.update_rows(Some(dir));
    }
    /// Go back to the automatic match for move `k`.
    pub fn set_automatic(&mut self, k: usize) {
        let Some(m) = self.moves.get_mut(k) else { return };
        m.shift_px = m.auto_px;
        m.manual = false;
        let dir = (m.axis, m.positive);
        self.update_rows(Some(dir));
    }

    /// Rebuild the rows from the moves, keeping the user's Use ticks except
    /// for direction `changed`.
    fn update_rows(&mut self, changed: Option<(usize, bool)>) {
        let prev: Vec<((usize, bool), bool)> = self.rows.iter().map(|r| ((r.axis, r.positive), r.use_it)).collect();
        let mut rows = Vec::new();
        for (axis, positive) in [(0, true), (0, false), (1, true), (1, false)] {
            let volts = if axis == 0 { self.volts.x } else { self.volts.y };
            // (tip move, score, by eye, signed steps) of each measured move this way.
            let mine: Vec<(DVec2, f64, bool, i32)> = self
                .moves
                .iter()
                .enumerate()
                .filter(|(_, m)| (m.axis, m.positive) == (axis, positive))
                .filter_map(|(k, m)| Some((self.tip_for(k, m.shift_px?)?, m.score, m.manual, m.steps)))
                .collect();
            if mine.is_empty() {
                continue;
            }
            let total = self.moves.iter().filter(|m| (m.axis, m.positive) == (axis, positive)).count();
            let good: Vec<(DVec2, f64, bool, i32)> =
                mine.iter().copied().filter(|(_, s, manual, _)| *manual || *s >= GOOD_SCORE).collect();
            // Use the good moves; with none, show the best one (unticked).
            let used: Vec<(DVec2, f64, bool, i32)> = if good.is_empty() {
                mine.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)).into_iter().collect()
            } else {
                good.clone()
            };
            let n = used.len() as f64;
            let score = used.iter().filter(|u| !u.2).map(|u| u.1).fold(1f64, f64::min);
            let steps = used.iter().map(|u| u.3).max_by_key(|s| s.abs()).unwrap_or(1);
            let steps_min = used.iter().map(|u| u.3).min_by_key(|s| s.abs()).unwrap_or(steps);
            let mut sizes: Vec<i32> = used.iter().map(|u| u.3.abs()).collect();
            sizes.sort_unstable();
            sizes.dedup();
            let (per_step_nm, offset_nm, spread_pct) = if sizes.len() >= 2 {
                // Least squares: move = offset + steps × per_step.
                let nbar = used.iter().map(|u| u.3 as f64).sum::<f64>() / n;
                let mbar = used.iter().map(|u| u.0).sum::<DVec2>() / n;
                let sxx: f64 = used.iter().map(|u| (u.3 as f64 - nbar).powi(2)).sum();
                let sxm: DVec2 = used.iter().map(|u| (u.0 - mbar) * (u.3 as f64 - nbar)).sum();
                let per = sxm / sxx;
                let offset = mbar - per * nbar;
                let rss: f64 = used.iter().map(|u| (u.0 - offset - per * u.3 as f64).length_squared()).sum();
                let rms = (rss / (n - 2.).max(1.)).sqrt();
                let mean_len = used.iter().map(|u| u.0.length()).sum::<f64>() / n;
                (per, Some(offset), if mean_len > 0. { rms / mean_len * 100. } else { 0. })
            } else {
                // One size: average per step (through the origin).
                let num: DVec2 = used.iter().map(|u| u.0 * u.3 as f64).sum();
                let den: f64 = used.iter().map(|u| (u.3 as f64).powi(2)).sum();
                let per = num / den.max(f64::MIN_POSITIVE);
                let lengths: Vec<f64> = used.iter().map(|u| u.0.length() / u.3.unsigned_abs().max(1) as f64).collect();
                let mean = lengths.iter().sum::<f64>() / n;
                let spread = if used.len() > 1 && mean > 0. {
                    (lengths.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (n - 1.)).sqrt() / mean * 100.
                } else {
                    0.
                };
                (per, None, spread)
            };
            let moved_nm = per_step_nm * steps as f64 + offset_nm.unwrap_or(DVec2::ZERO);
            // Per signed step, as in CoarseCalibration.
            let column = per_step_nm / volts * 1e-9;
            let default_use = !good.is_empty() && column.is_finite();
            let use_it = match prev.iter().find(|p| p.0 == (axis, positive)) {
                Some(&(_, u)) if changed != Some((axis, positive)) => u && column.is_finite(),
                _ => default_use,
            };
            rows.push(DirResult {
                axis,
                positive,
                steps,
                steps_min,
                moved_nm,
                per_step_nm,
                offset_nm,
                column,
                old_column: self.old.column(axis, positive),
                score,
                good: good.len(),
                total,
                manual: good.iter().filter(|g| g.2).count(),
                spread_pct,
                use_it,
            });
        }
        self.rows = rows;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn surface(x: f64, y: f64) -> f32 {
        let mut v = 0.;
        for k in 1..6 {
            let k = k as f64;
            v += (x * 0.011 * k + (y * 0.007 * k).sin() * 3.).sin() / k
                + (y * 0.013 * k + (x * 0.005 * k).cos() * 2.).cos() / k;
        }
        v as f32
    }

    /// Scan of a 1000 nm square frame (128 px) with the tip at `tip` (nm):
    /// pixel (i, j) sees the sample at frame-local position plus the tip.
    fn scan(frame: DAffine2, tip: DVec2) -> CalFrame {
        let n = 128;
        let mut d = vec![0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let local = DVec2::new((i as f64 + 0.5) / n as f64 - 0.5, (j as f64 + 0.5) / n as f64 - 0.5);
                let p = frame.transform_point2(local) + tip;
                d[j * n + i] = surface(p.x, p.y);
            }
        }
        CalFrame {
            data: FrameData { size: [n as u32, n as u32], data: Arc::new(d.into_boxed_slice()) },
            frame,
            source: String::new(),
        }
    }

    #[test]
    fn measures_the_tip_move_with_the_right_sign() {
        // Same convention as the live frame: scale (w, −h), angle, centre.
        for angle in [0f64, 20.] {
            let frame = DAffine2::from_scale_angle_translation(
                DVec2::new(1000., -1000.),
                -angle.to_radians(),
                DVec2::new(50., -30.),
            );
            let before = scan(frame, DVec2::ZERO);
            let after = scan(frame, DVec2::new(300., -120.));
            let (moved, score) = tip_move(&before, &after).unwrap();
            assert!((moved - DVec2::new(300., -120.)).length() < 10., "angle {angle}: {moved:?}");
            assert!(score > 0.9);
        }
    }

    #[test]
    fn plan_sets_the_frame_and_puts_it_back() {
        use crate::connection::nanonis::sequence::{compile, Step};
        let plan = CalPlan {
            steps: [12, 9],
            settle_s: 20.,
            withdraw: true,
            approach: true,
            volts: DVec2::ONE,
            frame_nm: 1500.,
            lines: 128,
            repeats: 1,
            sizes: 1,
            min_steps: [1, 1],
            axes: [true, true],
            scan_start: ScanStart::default(),
            line_time_s: None,
            restore_line_time_s: None,
            restore: Some(([3500., 3500.], [512, 512])),
        };
        let steps: Vec<Step> = compile(&build_sequence(&plan), 0, Dir::Down)
            .into_iter()
            .map(|(_, s)| s)
            .collect();
        let scans = steps.iter().filter(|s| matches!(s, Step::Scan(_))).count();
        assert_eq!(scans, plan.scans());
        // Calibration frame and resolution before the first scan.
        let first_scan = steps.iter().position(|s| matches!(s, Step::Scan(_))).unwrap();
        assert!(steps[..first_scan].contains(&Step::SetResolution([128, 128])));
        assert!(steps[..first_scan].iter().any(|s| matches!(s, Step::SetFrame(f) if f.size == Some([1500., 1500.]))));
        // Back to the user's frame and resolution after the last scan.
        let last_scan = steps.iter().rposition(|s| matches!(s, Step::Scan(_))).unwrap();
        assert!(steps[last_scan..].contains(&Step::SetResolution([512, 512])));
        assert!(steps[last_scan..].iter().any(|s| matches!(s, Step::SetFrame(f) if f.size == Some([3500., 3500.]))));
        // Every scan goes the same way (up/down pairs are offset).
        let dirs: Vec<Dir> = steps
            .iter()
            .filter_map(|s| match s {
                Step::Scan(d) => Some(*d),
                _ => None,
            })
            .collect();
        assert!(dirs.iter().all(|d| *d == Dir::Down), "{dirs:?}");
        // Coarse moves: +X, -X, +Y, -Y.
        let moves: Vec<[i32; 2]> = steps
            .iter()
            .filter_map(|s| match s {
                Step::CoarseMove { steps } => Some(*steps),
                _ => None,
            })
            .collect();
        assert_eq!(moves, vec![[12, 0], [-12, 0], [0, 9], [0, -9]]);
    }

    /// A run stopped part-way still measures the moves it finished.
    #[test]
    fn stopped_run_uses_the_moves_done() {
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(1000., -1000.), 0., DVec2::ZERO);
        let plan = CalPlan {
            steps: [16, 16],
            sizes: 4,
            min_steps: [2, 2],
            repeats: 1,
            settle_s: 0.,
            withdraw: true,
            approach: true,
            volts: DVec2::ONE,
            frame_nm: 1000.,
            lines: 128,
            axes: [true, true],
            scan_start: ScanStart::default(),
            line_time_s: None,
            restore_line_time_s: None,
            restore: None,
        };
        // Stopped after 2 sizes: 8 moves, 9 scans of 17.
        let step = [DVec2::new(0., -30.), DVec2::new(25., 0.)];
        let mut tip = DVec2::ZERO;
        let mut tips = vec![tip];
        for &(axis, pos, n) in plan.moves_with_steps().iter().take(8) {
            tip += step[axis] * if pos { n as f64 } else { -(n as f64) };
            tips.push(tip);
        }
        let frames: Vec<CalFrame> = tips.iter().map(|&t| scan(frame, t)).collect();
        let r = compute(&frames, &plan, &CoarseCalibration::default());
        assert!(r.problem.as_deref().is_some_and(|p| p.contains("Stopped after 9 of 17")), "{:?}", r.problem);
        assert_eq!(r.rows.len(), 4);
        for row in &r.rows {
            assert_eq!((row.good, row.total), (2, 4), "{row:?}");
            assert!((row.nm_per_step() - if row.axis == 0 { 30. } else { 25. }).abs() < 1., "{row:?}");
            assert!(row.use_it);
        }
    }

    /// Step counts small → large: the fit separates the step (with its
    /// cross-coupling) from a fixed shift every move adds.
    #[test]
    fn ramp_fits_step_and_per_move_shift() {
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(1000., -1000.), 0., DVec2::ZERO);
        // X motor: 28 nm/step mostly along −y, 3 nm/step sideways, plus 40 nm
        // along +x on every move (either way). Y motor: 25 nm/step along +x.
        let x_step = DVec2::new(3., -28.);
        let x_shift = DVec2::new(40., 0.);
        let y_step = DVec2::new(25., 0.);
        let plan = CalPlan {
            steps: [16, 16],
            sizes: 4,
            min_steps: [2, 2],
            repeats: 1,
            settle_s: 0.,
            withdraw: true,
            approach: true,
            volts: DVec2::ONE,
            frame_nm: 1000.,
            lines: 128,
            axes: [true, true],
            scan_start: ScanStart::default(),
            line_time_s: None,
            restore_line_time_s: None,
            restore: None,
        };
        assert_eq!(plan.counts(0), vec![2, 7, 11, 16]);
        let moves = plan.moves_with_steps();
        assert_eq!(&moves[..4], &[(0, true, 2), (0, false, 2), (1, true, 2), (1, false, 2)]);
        assert_eq!(moves.len(), 16);
        let mut tip = DVec2::ZERO;
        let mut tips = vec![tip];
        for &(axis, pos, n) in &moves {
            let sign = if pos { 1. } else { -1. };
            tip += if axis == 0 { x_step * sign * n as f64 + x_shift } else { y_step * sign * n as f64 };
            tips.push(tip);
        }
        let frames: Vec<CalFrame> = tips.iter().map(|&t| scan(frame, t)).collect();
        let r = compute(&frames, &plan, &CoarseCalibration::default());
        assert!(r.problem.is_none(), "{:?}", r.problem);
        let row = |axis, pos| r.rows.iter().find(|r| r.axis == axis && r.positive == pos).unwrap();
        for pos in [true, false] {
            let x = row(0, pos);
            assert!((x.step_vector_nm() - x_step * if pos { 1. } else { -1. }).length() < 1., "{x:?}");
            assert!((x.offset_nm.unwrap() - x_shift).length() < 8., "{x:?}");
            assert_eq!((x.steps_min.abs(), x.steps.abs()), (2, 16));
            let y = row(1, pos);
            assert!((y.step_vector_nm() - y_step * if pos { 1. } else { -1. }).length() < 1., "{y:?}");
            assert!(y.offset_nm.unwrap().length() < 8., "{y:?}");
        }
        // The column is the slope, per V: the sideways part is kept.
        let c = row(0, true).column;
        assert!((c - x_step * 1e-9).length() < 1e-9, "{c:?}");
    }

    #[test]
    fn columns_follow_the_steps_sign() {
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(1000., -1000.), 0., DVec2::ZERO);
        // +X: 10 steps moved 300 nm; −X: 10 steps moved back 250 nm.
        let tips = [
            DVec2::ZERO,
            DVec2::new(300., 0.),
            DVec2::new(50., 0.),
            DVec2::new(50., 280.),
            DVec2::new(50., 0.),
        ];
        let frames: Vec<CalFrame> = tips.iter().map(|&t| scan(frame, t)).collect();
        let plan = CalPlan {
            steps: [10, 10],
            settle_s: 0.,
            withdraw: true,
            approach: true,
            volts: DVec2::new(2., 2.),
            frame_nm: 1000.,
            lines: 128,
            repeats: 1,
            sizes: 1,
            min_steps: [1, 1],
            axes: [true, true],
            scan_start: ScanStart::default(),
            line_time_s: None,
            restore_line_time_s: None,
            restore: None,
        };
        let r = compute(&frames, &plan, &CoarseCalibration::default());
        assert!(r.problem.is_none(), "{:?}", r.problem);
        let nm = |k: usize| r.rows[k].nm_per_step();
        assert!((nm(0) - 30.).abs() < 1. && (nm(1) - 25.).abs() < 1., "{:?}", r.rows);
        assert!((nm(2) - 28.).abs() < 1. && (nm(3) - 28.).abs() < 1.);
        // Columns point along the motor axis for both directions: a negative
        // step count times the −X column moves the tip towards −x.
        let c = r.rows[1].column;
        assert!(c.x > 0. && c.y.abs() < c.x * 0.05, "{c:?}");
        // 25 nm per step at 2 V.
        assert!((c.x - 25e-9 / 2.).abs() < 1e-9);
        assert!(r.rows.iter().all(|row| row.use_it));
    }

    #[test]
    fn repeats_are_averaged() {
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(1000., -1000.), 0., DVec2::ZERO);
        // Two moves each way: +X 300 then 320 nm, −X 260 then 260, +Y 280 twice, −Y 280 twice.
        let path = [
            DVec2::ZERO,
            DVec2::new(300., 0.),
            DVec2::new(620., 0.),
            DVec2::new(360., 0.),
            DVec2::new(100., 0.),
            DVec2::new(100., 280.),
            DVec2::new(100., 560.),
            DVec2::new(100., 280.),
            DVec2::new(100., 0.),
        ];
        let frames: Vec<CalFrame> = path.iter().map(|&t| scan(frame, t)).collect();
        let plan = CalPlan {
            steps: [10, 10],
            repeats: 2,
            sizes: 1,
            min_steps: [1, 1],
            axes: [true, true],
            settle_s: 0.,
            withdraw: true,
            approach: true,
            volts: DVec2::ONE,
            frame_nm: 1000.,
            lines: 128,
            scan_start: ScanStart::default(),
            line_time_s: None,
            restore_line_time_s: None,
            restore: None,
        };
        assert_eq!(plan.scans(), 9);
        let r = compute(&frames, &plan, &CoarseCalibration::default());
        assert!(r.problem.is_none(), "{:?}", r.problem);
        let px = &r.rows[0];
        assert_eq!((px.good, px.total), (2, 2));
        assert!((px.nm_per_step() - 31.).abs() < 1., "{px:?}");
        // 300 and 320 nm: spread ≈ 4.6 %.
        assert!((px.spread_pct - 4.6).abs() < 1.5, "{px:?}");
        assert!((r.rows[1].nm_per_step() - 26.).abs() < 1.);
        assert!(r.rows[1].spread_pct < 1.);
    }

    #[test]
    fn alignment_by_eye_replaces_the_match() {
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(1000., -1000.), 0., DVec2::ZERO);
        let tips = [DVec2::ZERO, DVec2::new(300., 0.), DVec2::ZERO, DVec2::new(0., 280.), DVec2::ZERO];
        let frames: Vec<CalFrame> = tips.iter().map(|&t| scan(frame, t)).collect();
        let plan = CalPlan {
            steps: [10, 10],
            repeats: 1,
            sizes: 1,
            min_steps: [1, 1],
            axes: [true, true],
            settle_s: 0.,
            withdraw: true,
            approach: true,
            volts: DVec2::ONE,
            frame_nm: 1000.,
            lines: 128,
            scan_start: ScanStart::default(),
            line_time_s: None,
            restore_line_time_s: None,
            restore: None,
        };
        let mut r = compute(&frames, &plan, &CoarseCalibration::default());
        assert!((r.rows[0].nm_per_step() - 30.).abs() < 1.);
        r.rows[2].use_it = false;
        // By eye: +X moved the content 51.2 px left (128 px = 1000 nm), a 400 nm tip move.
        r.set_manual(0, DVec2::new(-51.2, 0.));
        assert!((r.rows[0].nm_per_step() - 40.).abs() < 0.1, "{:?}", r.rows[0]);
        assert_eq!(r.rows[0].manual, 1);
        assert!(r.rows[0].use_it);
        // Other rows keep their ticks.
        assert!(!r.rows[2].use_it);
        r.set_automatic(0);
        assert!((r.rows[0].nm_per_step() - 30.).abs() < 1.);
        assert_eq!(r.rows[0].manual, 0);
    }

    #[test]
    fn one_axis_only() {
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(1000., -1000.), 0., DVec2::ZERO);
        let tips = [DVec2::ZERO, DVec2::new(0., 280.), DVec2::ZERO];
        let frames: Vec<CalFrame> = tips.iter().map(|&t| scan(frame, t)).collect();
        let plan = CalPlan {
            steps: [10, 10],
            repeats: 1,
            sizes: 1,
            min_steps: [1, 1],
            axes: [false, true],
            settle_s: 0.,
            withdraw: true,
            approach: true,
            volts: DVec2::ONE,
            frame_nm: 1000.,
            lines: 128,
            scan_start: ScanStart::default(),
            line_time_s: None,
            restore_line_time_s: None,
            restore: None,
        };
        assert_eq!(plan.moves(), vec![(1, true), (1, false)]);
        assert_eq!(plan.scans(), 3);
        let r = compute(&frames, &plan, &CoarseCalibration::default());
        assert!(r.problem.is_none(), "{:?}", r.problem);
        assert_eq!(r.rows.len(), 2);
        assert!(r.rows.iter().all(|row| row.axis == 1));
        assert!((r.rows[0].nm_per_step() - 28.).abs() < 1.);
    }

    #[test]
    fn sets_the_scan_speed_and_puts_it_back() {
        use crate::connection::nanonis::sequence::{compile, Step};
        let plan = CalPlan {
            steps: [10, 10],
            repeats: 1,
            sizes: 1,
            min_steps: [1, 1],
            axes: [true, true],
            settle_s: 0.,
            withdraw: true,
            approach: true,
            volts: DVec2::ONE,
            frame_nm: 2000.,
            lines: 64,
            scan_start: ScanStart::default(),
            line_time_s: Some(0.5),
            restore: Some(([3500., 3500.], [512, 512])),
            restore_line_time_s: Some(1.2),
        };
        let steps: Vec<Step> = compile(&build_sequence(&plan), 0, Dir::Down).into_iter().map(|(_, s)| s).collect();
        let first_scan = steps.iter().position(|s| matches!(s, Step::Scan(_))).unwrap();
        let last_scan = steps.iter().rposition(|s| matches!(s, Step::Scan(_))).unwrap();
        assert!(steps[..first_scan].contains(&Step::SetSpeed(Speed::LineTime(0.5))), "{steps:?}");
        assert!(steps[last_scan..].contains(&Step::SetSpeed(Speed::LineTime(1.2))), "{steps:?}");
    }

    #[test]
    fn estimate_counts_scans_moves_and_restarts() {
        let plan = CalPlan {
            steps: [10, 10],
            repeats: 1,
            sizes: 1,
            min_steps: [1, 1],
            axes: [true, true],
            settle_s: 10.,
            withdraw: true,
            approach: true,
            volts: DVec2::ONE,
            frame_nm: 2000.,
            lines: 64,
            scan_start: ScanStart { starts: 2, gap_s: 3., park_s: 0. },
            line_time_s: None,
            restore: None,
            restore_line_time_s: None,
        };
        // 5 scans × (64 × 1 s + 3 s) + 4 moves × (30 + 10) s
        assert!((plan.estimate_s(1., 30.) - (5. * 67. + 160.)).abs() < 1e-9);
    }
}

// ---- Keeping the scans of the last applied calibration ----

#[derive(serde::Serialize, serde::Deserialize)]
struct SavedMove {
    axis: usize,
    positive: bool,
    steps: i32,
    shift_px: Option<[f64; 2]>,
    auto_px: Option<[f64; 2]>,
    score: f64,
    manual: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SavedFrame {
    size: [u32; 2],
    frame: [f64; 6],
    source: String,
    /// Raw f32 (little-endian) forward image, row-major.
    file: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SavedResult {
    unix: u64,
    how: String,
    volts: [f64; 2],
    old: CoarseCalibration,
    applied: CoarseCalibration,
    moves: Vec<SavedMove>,
    frames: Vec<SavedFrame>,
}

/// Where the last applied calibration's scans are kept.
pub fn saved_dir() -> std::path::PathBuf {
    crate::app::CONFIG_DIR.config_local_dir().join("last_calibration")
}

impl CalResult {
    /// Keep this run's scans and alignments (the "check by eye" pairs) in
    /// [`saved_dir`], replacing the last ones, with a PNG per move (before |
    /// after). `applied` is the calibration that was applied from it.
    pub fn save(&self, applied: &CoarseCalibration, how: &str) -> Result<std::path::PathBuf, String> {
        let dir = saved_dir();
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
        }
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut frames = Vec::new();
        for (i, f) in self.frames.iter().enumerate() {
            let file = format!("scan_{i:02}.bin");
            let bytes: Vec<u8> = f.data.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join(&file), bytes).map_err(|e| e.to_string())?;
            frames.push(SavedFrame { size: f.data.size, frame: f.frame.to_cols_array(), source: f.source.clone(), file });
        }
        for (k, m) in self.moves.iter().enumerate() {
            if let Some((a, b)) = self.pair(k) {
                let name = format!("move_{k:02}_{}_{}steps.png", m.name().replace('\u{2212}', "-"), m.steps);
                if let Err(e) = pair_png(a, b, &dir.join(name)) {
                    tracing::warn!("could not write the calibration image: {e}");
                }
            }
        }
        let saved = SavedResult {
            unix: crate::analysis::coarse_fit::unix_now(),
            how: how.to_string(),
            volts: self.volts.to_array(),
            old: self.old,
            applied: *applied,
            moves: self
                .moves
                .iter()
                .map(|m| SavedMove {
                    axis: m.axis,
                    positive: m.positive,
                    steps: m.steps,
                    shift_px: m.shift_px.map(|v| v.to_array()),
                    auto_px: m.auto_px.map(|v| v.to_array()),
                    score: m.score,
                    manual: m.manual,
                })
                .collect(),
            frames,
        };
        let json = serde_json::to_string_pretty(&saved).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("calibration.json"), json).map_err(|e| e.to_string())?;
        Ok(dir)
    }

    /// The last applied calibration's run, as saved by [`Self::save`], and
    /// a line saying when and how it was applied.
    pub fn load_saved() -> Result<(Self, String), String> {
        let dir = saved_dir();
        let text = std::fs::read_to_string(dir.join("calibration.json")).map_err(|e| e.to_string())?;
        let s: SavedResult = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let mut frames = Vec::new();
        for f in &s.frames {
            let bytes = std::fs::read(dir.join(&f.file)).map_err(|e| e.to_string())?;
            let data: Vec<f32> =
                bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            if data.len() != f.size[0] as usize * f.size[1] as usize {
                return Err(format!("{} has the wrong size", f.file));
            }
            frames.push(CalFrame {
                data: FrameData { size: f.size, data: std::sync::Arc::new(data.into_boxed_slice()) },
                frame: DAffine2::from_cols_array(&f.frame),
                source: f.source.clone(),
            });
        }
        let moves = s
            .moves
            .iter()
            .map(|m| MoveMeasure {
                axis: m.axis,
                positive: m.positive,
                steps: m.steps,
                shift_px: m.shift_px.map(DVec2::from_array),
                auto_px: m.auto_px.map(DVec2::from_array),
                score: m.score,
                manual: m.manual,
            })
            .collect();
        let mut r = CalResult { rows: Vec::new(), problem: None, volts: DVec2::from_array(s.volts), frames, moves, old: s.old };
        r.update_rows(None);
        let when = crate::connection::nanonis::course_motion::CalibrationStamp { unix: Some(s.unix), how: s.how };
        Ok((r, when.describe()))
    }
}

/// Before | after, grey, each levelled line by line and scaled to its 1–99 %
/// range.
fn pair_png(a: &CalFrame, b: &CalFrame, path: &std::path::Path) -> Result<(), String> {
    let gray = |f: &CalFrame| -> (u32, u32, Vec<u8>) {
        let [w, h] = f.data.size;
        let (wu, hu) = (w as usize, h as usize);
        let mut d: Vec<f32> = f.data.data.to_vec();
        for y in 0..hu {
            let row = &mut d[y * wu..(y + 1) * wu];
            let fin: Vec<f32> = row.iter().copied().filter(|v| v.is_finite()).collect();
            if fin.is_empty() {
                continue;
            }
            let m = fin.iter().sum::<f32>() / fin.len() as f32;
            row.iter_mut().for_each(|v| *v -= m);
        }
        let mut fin: Vec<f32> = d.iter().copied().filter(|v| v.is_finite()).collect();
        fin.sort_by(f32::total_cmp);
        let (lo, hi) = if fin.is_empty() {
            (0., 1.)
        } else {
            (fin[fin.len() / 100], fin[(fin.len() * 99 / 100).min(fin.len() - 1)])
        };
        let span = (hi - lo).max(1e-30);
        let px = d.iter().map(|v| if v.is_finite() { (((v - lo) / span).clamp(0., 1.) * 255.) as u8 } else { 0 }).collect();
        (w, h, px)
    };
    let (wa, ha, pa) = gray(a);
    let (wb, hb, pb) = gray(b);
    let gap = 8;
    let (w, h) = (wa + gap + wb, ha.max(hb));
    let mut img = image::GrayImage::from_pixel(w, h, image::Luma([255]));
    for y in 0..ha {
        for x in 0..wa {
            img.put_pixel(x, y, image::Luma([pa[(y * wa + x) as usize]]));
        }
    }
    for y in 0..hb {
        for x in 0..wb {
            img.put_pixel(wa + gap + x, y, image::Luma([pb[(y * wb + x) as usize]]));
        }
    }
    img.save(path).map_err(|e| e.to_string())
}
