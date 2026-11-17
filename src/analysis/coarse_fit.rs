//! Coarse calibration learned from the user's corrections.
//!
//! After a coarse move the map places the next frame where the calibration
//! predicts it went. When the user drags a stamp (or the scan region) to
//! where it really is, the drag measures how far the moves since the last
//! known position really went. Each such measurement is an [`Observation`];
//! [`fit`] turns many of them into a calibration, pulled toward the last
//! calibration test so a few drags can't throw it off.
//!
//! Everything here is pure: no UI, files or instrument.

use glam::{DMat2, DVec2};
#[cfg(test)]
use glam::IVec2;
use serde::{Deserialize, Serialize};

use crate::connection::nanonis::course_motion::CoarseCalibration;

/// One finished coarse move, as logged when it completed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoveRecord {
    /// Increasing within one scan area.
    pub id: u64,
    /// Unix time, s.
    pub unix: u64,
    pub steps: [i32; 2],
    /// Drive amplitudes (V) and frequencies (Hz) during the move.
    pub volts: [f64; 2],
    #[serde(default)]
    pub freq: [f64; 2],
    /// Where the calibration in use put the move (nm, scan-area axes).
    pub predicted_nm: [f64; 2],
    /// The calibration used for the prediction was the learned one.
    #[serde(default)]
    pub learned: bool,
}

/// A place on the map where the tip's coarse position is known: a stamp
/// (with any drag the user gave it) or a drag of the scan region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    pub key: uuid::Uuid,
    /// The last coarse move before it (`None`: before any logged move).
    pub last_move: Option<u64>,
    pub unix: u64,
    /// The scan area's true world position (translation) there.
    pub true_world: DVec2,
    /// Scan-area axes to world (rotation and scale of the world transform).
    pub world_m2: DMat2,
    /// The user corrected this position (dragged it).
    pub corrected: bool,
}

/// A scan-region drag, kept with the move log (stamps carry their own).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DragAnchor {
    pub key: String,
    pub last_move: Option<u64>,
    pub unix: u64,
    pub true_world: [f64; 2],
    pub world_m2: [f64; 4],
}
impl DragAnchor {
    pub fn anchor(&self) -> Option<Anchor> {
        Some(Anchor {
            key: uuid::Uuid::parse_str(&self.key).ok()?,
            last_move: self.last_move,
            unix: self.unix,
            true_world: DVec2::from_array(self.true_world),
            world_m2: DMat2::from_cols_array(&self.world_m2),
            corrected: true,
        })
    }
}

/// The coarse moves and scan-region drags of one scan area (saved with
/// the project).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoarseLog {
    pub moves: Vec<MoveRecord>,
    pub drags: Vec<DragAnchor>,
}
impl CoarseLog {
    pub fn last_move(&self) -> Option<u64> {
        self.moves.last().map(|m| m.id)
    }
    pub fn next_id(&self) -> u64 {
        self.moves.last().map_or(0, |m| m.id + 1)
    }
}

/// Where a stamp was put when it was taken, to see later how far the user
/// dragged it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoarseRef {
    pub last_move: Option<u64>,
    pub unix: u64,
    /// Scan area world transform, and the stamp's own transform, then.
    pub world_at: [f64; 6],
    pub stamp_at: [f64; 6],
}
impl CoarseRef {
    pub fn new(last_move: Option<u64>, world: glam::DAffine2, stamp: glam::DAffine2) -> Self {
        Self { last_move, unix: unix_now(), world_at: world.to_cols_array(), stamp_at: stamp.to_cols_array() }
    }
    /// The anchor for a stamp now at `current` (its transform). Only a
    /// translation counts as a correction: a rotated or scaled stamp
    /// isn't used.
    pub fn anchor(&self, key: uuid::Uuid, current: glam::DAffine2) -> Option<Anchor> {
        let world = glam::DAffine2::from_cols_array(&self.world_at);
        let at = glam::DAffine2::from_cols_array(&self.stamp_at);
        if !current.matrix2.abs_diff_eq(at.matrix2, 1e-9 * at.matrix2.x_axis.length().max(1e-12)) {
            return None;
        }
        let drag = current.translation - at.translation;
        // Below a thousandth of the frame: not dragged.
        let corrected = drag.length() > 1e-3 * at.matrix2.x_axis.length();
        Some(Anchor {
            key,
            last_move: self.last_move,
            unix: self.unix,
            true_world: world.translation + if corrected { drag } else { DVec2::ZERO },
            world_m2: world.matrix2,
            corrected,
        })
    }
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A move inside an observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObsMove {
    pub steps: [i32; 2],
    pub volts: [f64; 2],
}

/// The moves between two known positions and how far they really went.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// The corrected stamp or drag this was measured at.
    pub key: String,
    pub unix: u64,
    pub moves: Vec<ObsMove>,
    /// Real displacement (nm, scan-area axes).
    pub actual_nm: [f64; 2],
    /// What the calibration in use at the time predicted (nm).
    pub predicted_nm: [f64; 2],
    /// Left out of the fit by the user.
    #[serde(default)]
    pub ignored: bool,
}
impl Observation {
    pub fn actual(&self) -> DVec2 {
        DVec2::from_array(self.actual_nm)
    }
    pub fn predicted(&self) -> DVec2 {
        DVec2::from_array(self.predicted_nm)
    }
    /// Signed steps × volts (×1e9, so columns are in nm/step/V) for each
    /// direction column, in the order +X, −X, +Y, −Y.
    fn design(&self) -> [f64; 4] {
        let mut a = [0.; 4];
        for m in &self.moves {
            let (sx, sy) = (m.steps[0] as f64 * m.volts[0], m.steps[1] as f64 * m.volts[1]);
            if m.steps[0] >= 0 { a[0] += sx } else { a[1] += sx }
            if m.steps[1] >= 0 { a[2] += sy } else { a[3] += sy }
        }
        a
    }
    /// "+40 X, −12 Y" style summary.
    pub fn moves_text(&self) -> String {
        let (mut x, mut y) = (0i64, 0i64);
        for m in &self.moves {
            x += m.steps[0] as i64;
            y += m.steps[1] as i64;
        }
        let mut s = Vec::new();
        if x != 0 {
            s.push(format!("{x:+} X"));
        }
        if y != 0 {
            s.push(format!("{y:+} Y"));
        }
        let n = self.moves.len();
        format!("{}{}", s.join(", "), if n > 1 { format!(" ({n} moves)") } else { String::new() })
    }
}

/// The (+X, −X, +Y, −Y) order used by the fit.
pub const DIRS: [(usize, bool, &str); 4] =
    [(0, true, "+X"), (0, false, "\u{2212}X"), (1, true, "+Y"), (1, false, "\u{2212}Y")];

/// Turn the anchors (any order) and the move log into observations: for
/// each corrected position, the moves since the previous known position.
pub fn observations(moves: &[MoveRecord], anchors: &[Anchor]) -> Vec<Observation> {
    let mut anchors = anchors.to_vec();
    anchors.sort_by_key(|a| (a.last_move.map_or(0, |m| m + 1), a.unix));
    // Groups of anchors after the same move.
    let mut groups: Vec<Vec<Anchor>> = Vec::new();
    for a in anchors {
        match groups.last_mut() {
            Some(g) if g[0].last_move == a.last_move => g.push(a),
            _ => groups.push(vec![a]),
        }
    }
    let mut out = Vec::new();
    for w in groups.windows(2) {
        let (prev, cur) = (&w[0], &w[1]);
        // The latest known position before the moves, and the most
        // refined (latest) correction after them.
        let p = prev.last().unwrap();
        let Some(s) = cur.iter().rev().find(|a| a.corrected) else { continue };
        let lo = p.last_move.map_or(0, |m| m + 1);
        let hi = s.last_move.map_or(0, |m| m + 1);
        let seg: Vec<&MoveRecord> = moves.iter().filter(|m| m.id + 1 > lo && m.id + 1 <= hi).collect();
        if seg.is_empty() {
            continue;
        }
        let inv = s.world_m2;
        if inv.determinant().abs() < 1e-18 {
            continue;
        }
        let actual = inv.inverse() * (s.true_world - p.true_world);
        let predicted: DVec2 = seg.iter().map(|m| DVec2::from_array(m.predicted_nm)).sum();
        if !actual.is_finite() {
            continue;
        }
        out.push(Observation {
            key: s.key.to_string(),
            unix: s.unix,
            moves: seg.iter().map(|m| ObsMove { steps: m.steps, volts: m.volts }).collect(),
            actual_nm: actual.to_array(),
            predicted_nm: predicted.to_array(),
            ignored: false,
        });
    }
    out
}

/// Fit settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitParams {
    /// How many observations' worth of weight the last test gets.
    pub prior_weight: f64,
    /// Newer corrections count more: weight halves every this many.
    pub half_life: f64,
    /// Leave out moves whose length is off from the test's prediction by
    /// more than this factor (either way).
    pub max_ratio: f64,
    /// Leave out residuals beyond this many robust standard deviations.
    pub outlier_sigma: f64,
}
impl Default for FitParams {
    fn default() -> Self {
        Self { prior_weight: 1., half_life: 50., max_ratio: 2.5, outlier_sigma: 3. }
    }
}

/// Fit result for one direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirFit {
    /// Observations that moved in this direction (and were used).
    pub n: usize,
    /// ± one standard deviation of the column, as % of its length
    /// (`None`: not enough data to say).
    pub sigma_pct: Option<f64>,
    /// Change of the column length against the last test, %.
    pub change_pct: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    pub calib: CoarseCalibration,
    pub dirs: [DirFit; 4],
    /// Per observation (same order as given): used, and the residual (nm)
    /// against the fitted calibration.
    pub used: Vec<bool>,
    pub residual_nm: Vec<f64>,
    pub n_used: usize,
}
impl FitResult {
    /// Largest uncertainty over the directions that have data.
    pub fn worst_sigma_pct(&self) -> Option<f64> {
        let with: Vec<f64> = self.dirs.iter().filter(|d| d.n > 0).filter_map(|d| d.sigma_pct).collect();
        if with.is_empty() || self.dirs.iter().any(|d| d.n > 0 && d.sigma_pct.is_none()) {
            return None;
        }
        with.into_iter().reduce(f64::max)
    }
}

/// Columns in nm/step/V, (+X, −X, +Y, −Y).
fn columns(c: &CoarseCalibration) -> [DVec2; 4] {
    DIRS.map(|(axis, pos, _)| c.column(axis, pos) * 1e9)
}

/// Prediction (nm) of an observation's moves with columns `cols`.
fn predict(a: &[f64; 4], cols: &[DVec2; 4]) -> DVec2 {
    (0..4).map(|d| cols[d] * a[d]).sum()
}

/// Fit the calibration to the observations (ignored ones are skipped),
/// pulled toward `baseline`. Observations are weighted by recency: their
/// order in `obs` doesn't matter, `unix` does.
pub fn fit(baseline: &CoarseCalibration, obs: &[Observation], p: &FitParams) -> FitResult {
    let base = columns(baseline);
    let design: Vec<[f64; 4]> = obs.iter().map(|o| o.design()).collect();
    // Recency weights: newest = 1.
    let mut order: Vec<usize> = (0..obs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(obs[i].unix));
    let mut recency = vec![0.; obs.len()];
    for (rank, &i) in order.iter().enumerate() {
        recency[i] = 0.5f64.powf(rank as f64 / p.half_life.max(1.));
    }
    // First cut: user-ignored, empty, and moves far off the test's
    // prediction (a mis-drag onto the wrong feature).
    let mut used: Vec<bool> = obs
        .iter()
        .zip(&design)
        .map(|(o, a)| {
            if o.ignored || a.iter().all(|v| *v == 0.) {
                return false;
            }
            let pred = predict(a, &base).length();
            let act = o.actual().length();
            if pred < 1e-9 || act < 1e-9 {
                return false;
            }
            let r = act / pred;
            r <= p.max_ratio && r >= 1. / p.max_ratio
        })
        .collect();

    let mut cols = base;
    let mut cov_diag = [[f64::NAN; 2]; 4];
    let mut sigma2 = f64::NAN;
    for pass in 0..3 {
        // Normal equations, shared by the x and y components.
        let mut ata = nalgebra::Matrix4::<f64>::zeros();
        let mut atb = [nalgebra::Vector4::<f64>::zeros(), nalgebra::Vector4::<f64>::zeros()];
        // Prior strength per direction: `prior_weight` typical observations.
        let mut typical = [0f64; 4];
        let mut count = [0usize; 4];
        for (i, a) in design.iter().enumerate() {
            if !used[i] {
                continue;
            }
            let w = recency[i];
            for r in 0..4 {
                if a[r] != 0. {
                    typical[r] += a[r].abs();
                    count[r] += 1;
                }
                for c in 0..4 {
                    ata[(r, c)] += w * a[r] * a[c];
                }
                let b = obs[i].actual();
                atb[0][r] += w * a[r] * b.x;
                atb[1][r] += w * a[r] * b.y;
            }
        }
        for d in 0..4 {
            let t = if count[d] > 0 { typical[d] / count[d] as f64 } else { 1. };
            let lam = p.prior_weight.max(1e-6) * t * t;
            ata[(d, d)] += lam;
            atb[0][d] += lam * base[d].x;
            atb[1][d] += lam * base[d].y;
        }
        let Some(inv) = ata.try_inverse() else { break };
        let sx = inv * atb[0];
        let sy = inv * atb[1];
        cols = [0, 1, 2, 3].map(|d| DVec2::new(sx[d], sy[d]));

        // Residuals and their spread (per component).
        let res: Vec<f64> =
            design.iter().zip(obs).map(|(a, o)| (o.actual() - predict(a, &cols)).length()).collect();
        let (mut sw, mut swr2, mut n) = (0., 0., 0usize);
        for i in 0..obs.len() {
            if used[i] {
                sw += recency[i];
                swr2 += recency[i] * res[i] * res[i];
                n += 1;
            }
        }
        // Each observation gives 2 numbers; 8 unknowns, softened by the prior.
        let dof = (2 * n) as f64 - 8. * (n as f64 / (n as f64 + p.prior_weight)).min(1.);
        sigma2 = if n >= 2 && dof > 0.5 && sw > 0. {
            (swr2 / sw) * (2 * n) as f64 / dof / 2.
        } else {
            f64::NAN
        };
        for d in 0..4 {
            cov_diag[d] = [sigma2 * inv[(d, d)], sigma2 * inv[(d, d)]];
        }
        if pass == 2 || !sigma2.is_finite() {
            break;
        }
        // Drop residual outliers (robust spread), then refit.
        let mut r_used: Vec<f64> = (0..obs.len()).filter(|&i| used[i]).map(|i| res[i]).collect();
        if r_used.len() < 4 {
            break;
        }
        r_used.sort_by(f64::total_cmp);
        let med = r_used[r_used.len() / 2];
        let robust = (1.4826 * med).max(1e-6);
        let mut changed = false;
        for i in 0..obs.len() {
            let floor = 0.05 * obs[i].actual().length();
            if used[i] && res[i] > p.outlier_sigma * robust && res[i] > floor {
                used[i] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut calib = *baseline;
    let mut dirs = [DirFit { n: 0, sigma_pct: None, change_pct: 0. }; 4];
    for (d, (axis, pos, _)) in DIRS.iter().enumerate() {
        let n = (0..obs.len()).filter(|&i| used[i] && design[i][d] != 0.).count();
        let col = if n > 0 { cols[d] } else { base[d] };
        calib.set_column(*axis, *pos, col * 1e-9);
        let len = col.length();
        let sigma = (cov_diag[d][0] + cov_diag[d][1]).sqrt();
        dirs[d] = DirFit {
            n,
            sigma_pct: (n > 0 && sigma.is_finite() && len > 0.).then(|| 100. * sigma / len),
            change_pct: if base[d].length() > 0. { 100. * (len / base[d].length() - 1.) } else { 0. },
        };
    }
    let residual_nm =
        design.iter().zip(obs).map(|(a, o)| (o.actual() - predict(a, &columns(&calib))).length()).collect();
    let n_used = used.iter().filter(|u| **u).count();
    let _ = sigma2;
    FitResult { calib, dirs, used, residual_nm, n_used }
}

/// Real displacement (nm) of `steps` at `volts` with `c` (same as
/// `CoarseCalibration::steps_to_real`, here for tests).
#[cfg(test)]
fn real(c: &CoarseCalibration, steps: IVec2, volts: DVec2) -> DVec2 {
    c.steps_to_real(steps, volts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth() -> CoarseCalibration {
        CoarseCalibration {
            x_pos: [12e-9, 1e-9],
            x_neg: [10e-9, 0.5e-9],
            y_pos: [-0.8e-9, 9e-9],
            y_neg: [-0.5e-9, 11e-9],
        }
    }
    fn baseline() -> CoarseCalibration {
        CoarseCalibration::from_matrix(DMat2::IDENTITY * 10e-9)
    }
    /// Deterministic noise in [-1, 1].
    fn noise(i: u64) -> f64 {
        let x = (i.wrapping_mul(2654435761) % 10007) as f64 / 10007.;
        2. * x - 1.
    }
    fn obs_for(steps: &[IVec2], volts: DVec2, unix: u64, noise_nm: f64) -> Observation {
        let t = truth();
        let actual: DVec2 = steps.iter().map(|s| real(&t, *s, volts)).sum::<DVec2>()
            + DVec2::new(noise(unix), noise(unix + 7)) * noise_nm;
        let predicted: DVec2 = steps.iter().map(|s| real(&baseline(), *s, volts)).sum();
        Observation {
            key: format!("k{unix}"),
            unix,
            moves: steps.iter().map(|s| ObsMove { steps: s.to_array(), volts: volts.to_array() }).collect(),
            actual_nm: actual.to_array(),
            predicted_nm: predicted.to_array(),
            ignored: false,
        }
    }

    #[test]
    fn converges_to_truth_with_mixed_moves() {
        let v = DVec2::new(100., 120.);
        let mut obs = Vec::new();
        let pattern = [
            vec![IVec2::new(20, 0)],
            vec![IVec2::new(-25, 0)],
            vec![IVec2::new(0, 30)],
            vec![IVec2::new(0, -15)],
            vec![IVec2::new(10, 0), IVec2::new(0, 12)],
            vec![IVec2::new(-8, -9)],
        ];
        for k in 0..60u64 {
            obs.push(obs_for(&pattern[k as usize % pattern.len()], v, 1000 + k, 50.));
        }
        let r = fit(&baseline(), &obs, &FitParams::default());
        for (axis, pos, name) in DIRS {
            let got = r.calib.column(axis, pos);
            let want = truth().column(axis, pos);
            let err = (got - want).length() / want.length();
            // The pull toward the last test leaves a small bias.
            assert!(err < 0.025, "{name}: {got:?} vs {want:?} ({err})");
        }
        assert!(r.worst_sigma_pct().unwrap() < 5.);
    }

    #[test]
    fn rejects_outliers() {
        let v = DVec2::new(100., 100.);
        let mut obs: Vec<Observation> =
            (0..30u64).map(|k| obs_for(&[IVec2::new(if k % 2 == 0 { 20 } else { -20 }, 0)], v, 10 + k, 20.)).collect();
        // A drag onto the wrong feature: 3x too far.
        let mut bad = obs_for(&[IVec2::new(20, 0)], v, 100, 0.);
        bad.actual_nm = (bad.actual() * 3.).to_array();
        obs.push(bad);
        // A plausible-length but wrong-direction one.
        let mut bad2 = obs_for(&[IVec2::new(20, 0)], v, 101, 0.);
        bad2.actual_nm = [bad2.actual_nm[0] * 0.6, bad2.actual_nm[1] + 6000.];
        obs.push(bad2);
        let r = fit(&baseline(), &obs, &FitParams::default());
        assert!(!r.used[30] && !r.used[31], "{:?}", &r.used[28..]);
        let err = (r.calib.column(0, true) - truth().column(0, true)).length() / truth().column(0, true).length();
        assert!(err < 0.025, "{err}");
    }

    #[test]
    fn no_data_keeps_baseline() {
        let r = fit(&baseline(), &[], &FitParams::default());
        assert_eq!(r.calib, baseline());
        assert!(r.dirs.iter().all(|d| d.n == 0));
    }

    #[test]
    fn few_observations_stay_near_baseline() {
        let v = DVec2::new(100., 100.);
        let obs = vec![obs_for(&[IVec2::new(20, 0)], v, 1, 0.)];
        let r = fit(&baseline(), &obs, &FitParams::default());
        let got = r.calib.column(0, true).x;
        // One observation against a prior worth one: about half way.
        assert!(got > 10.5e-9 && got < 11.5e-9, "{got}");
        // Untouched directions keep the baseline.
        assert_eq!(r.calib.column(1, false), baseline().column(1, false));
    }

    #[test]
    fn observations_from_anchors() {
        let m2 = DMat2::from_angle(0.3) * 2.;
        let moves = vec![
            MoveRecord { id: 0, unix: 1, steps: [10, 0], volts: [100., 100.], freq: [0.; 2], predicted_nm: [10000., 0.], learned: false },
            MoveRecord { id: 1, unix: 2, steps: [0, 5], volts: [100., 100.], freq: [0.; 2], predicted_nm: [0., 5000.], learned: false },
            MoveRecord { id: 2, unix: 3, steps: [-4, 0], volts: [100., 100.], freq: [0.; 2], predicted_nm: [-4000., 0.], learned: false },
        ];
        let key = |n: u128| uuid::Uuid::from_u128(n);
        let true_local = DVec2::new(11000., 4500.);
        let anchors = vec![
            // Before any move.
            Anchor { key: key(1), last_move: None, unix: 0, true_world: DVec2::new(5., 5.), world_m2: m2, corrected: false },
            // After two moves, dragged.
            Anchor {
                key: key(2),
                last_move: Some(1),
                unix: 3,
                true_world: DVec2::new(5., 5.) + m2 * true_local,
                world_m2: m2,
                corrected: true,
            },
            // After the third, not dragged: no observation.
            Anchor { key: key(3), last_move: Some(2), unix: 4, true_world: DVec2::ZERO, world_m2: m2, corrected: false },
        ];
        let obs = observations(&moves, &anchors);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].moves.len(), 2);
        assert!((obs[0].actual() - true_local).length() < 1e-6);
        assert_eq!(obs[0].predicted(), DVec2::new(10000., 5000.));
        assert_eq!(obs[0].key, key(2).to_string());
    }
}
