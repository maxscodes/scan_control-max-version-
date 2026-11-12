//! GDS prior for Ring Assist.
//!
//! The GDS on the map shows where the rings should be. For a scan frame this
//! module:
//! - finds the ring systems in the GDS (concentric annular arcs),
//! - predicts the direction to the ring centre from where the frame sits on
//!   the GDS (an estimate: the GDS placement is never exact),
//! - measures the direction of the broad bands in the scan itself and checks
//!   it against the prediction,
//! - from many stamps, estimates how far the GDS is off on the map.

use std::{collections::BTreeMap, sync::Arc};

use glam::{DAffine2, DMat2, DVec2, Vec2};
use uuid::Uuid;

use crate::{components::selectable_list::SelectableList, connection::LiveImage, view_object::Object};

/// Angle bins (10° each) recording which side of a ring system has arcs.
const BINS: usize = 36;
/// Band normal within this angle of the GDS radial direction counts as agreeing.
pub const AGREE_DEG: f64 = 30.;
/// Below this coherence the scan has no clear band direction.
pub const MIN_COHERENCE: f64 = 0.3;
/// GDS predictions on the map.
pub const GDS_COLOR: egui::Color32 = egui::Color32::from_rgb(0xc3, 0x8c, 0xff);
/// Physical blur for the band measurement (bands are about 1 µm wide).
const BAND_BLUR_NM: f64 = 200.;

// ---------------------------------------------------------------------------
// Ring systems in the GDS
// ---------------------------------------------------------------------------

/// Concentric ring bands sharing one centre.
#[derive(Clone, Debug)]
pub struct RingSystem {
    /// Centre in GDS local coordinates (nm, before the GDS transform).
    pub center: DVec2,
    /// (inner, outer) radius of each band, local nm, innermost first.
    pub bands: Vec<(f64, f64)>,
    /// Angle bins (from +x, counter-clockwise) that have arcs.
    covered: u64,
}

impl RingSystem {
    pub fn outer_radius(&self) -> f64 {
        self.bands.last().map_or(0., |b| b.1)
    }
    fn covers(&self, angle: f64) -> bool {
        let b = angle_bin(angle);
        (0..3).any(|k| self.covered & (1 << ((b + BINS - 1 + k) % BINS)) != 0)
    }
}

fn angle_bin(a: f64) -> usize {
    let t = a.rem_euclid(std::f64::consts::TAU);
    ((t / std::f64::consts::TAU * BINS as f64) as usize).min(BINS - 1)
}

/// One annular arc polygon.
struct Band {
    center: DVec2,
    r_in: f64,
    r_out: f64,
    covered: u64,
}

/// Find ring systems: groups of at least three concentric annular bands.
pub fn find_ring_systems(polys: &BTreeMap<u16, Vec<Vec<Vec2>>>) -> Vec<RingSystem> {
    let bands: Vec<Band> = polys.values().flatten().filter_map(|p| fit_band(p)).collect();
    let mut groups: Vec<Vec<Band>> = Vec::new();
    for b in bands {
        let tol = |r: f64| (0.01 * r).max(100.);
        match groups
            .iter_mut()
            .find(|g| (g[0].center - b.center).length() < tol(b.r_out.max(g[0].r_out)))
        {
            Some(g) => g.push(b),
            None => groups.push(vec![b]),
        }
    }
    groups
        .into_iter()
        .filter_map(|g| {
            let center = g.iter().map(|b| b.center).sum::<DVec2>() / g.len() as f64;
            let covered = g.iter().fold(0, |m, b| m | b.covered);
            let mut radii: Vec<(f64, f64)> = g.iter().map(|b| (b.r_in, b.r_out)).collect();
            radii.sort_by(|a, b| a.0.total_cmp(&b.0));
            // Pieces of the same band (e.g. left and right halves) count once.
            radii.dedup_by(|a, b| (a.0 - b.0).abs() < (0.02 * b.1).max(20.));
            (radii.len() >= 3).then_some(RingSystem { center, bands: radii, covered })
        })
        .collect()
}

/// Fit an annular arc: every vertex on one of two concentric circles.
fn fit_band(poly: &[Vec2]) -> Option<Band> {
    if poly.len() < 12 {
        return None;
    }
    let pts: Vec<DVec2> = poly.iter().map(|p| p.as_dvec2()).collect();
    let mean = pts.iter().copied().sum::<DVec2>() / pts.len() as f64;
    // Kåsa circle fit (relative to the mean for conditioning).
    let mut ata = [[0.; 3]; 3];
    let mut atb = [0.; 3];
    for p in &pts {
        let q = *p - mean;
        let row = [2. * q.x, 2. * q.y, 1.];
        let rhs = q.length_squared();
        for i in 0..3 {
            atb[i] += row[i] * rhs;
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    let s = solve(ata, atb)?;
    let mut c = mean + DVec2::new(s[0], s[1]);
    // Refine centre and both radii together (Gauss-Newton). Each vertex
    // belongs to the inner or outer circle, split half way between them.
    let d: Vec<f64> = pts.iter().map(|p| (*p - c).length()).collect();
    let lo = d.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = d.iter().copied().fold(0., f64::max);
    let (mut r_in, mut r_out) = (lo, hi);
    for _ in 0..30 {
        let mid = (r_in + r_out) / 2.;
        let mut jtj = [[0.; 4]; 4];
        let mut jte = [0.; 4];
        for p in &pts {
            let v = *p - c;
            let di = v.length();
            if di < 1e-9 {
                continue;
            }
            let u = v / di;
            let inner = di < mid;
            let e = di - if inner { r_in } else { r_out };
            let j = [-u.x, -u.y, if inner { -1. } else { 0. }, if inner { 0. } else { -1. }];
            for a in 0..4 {
                jte[a] += j[a] * e;
                for b in 0..4 {
                    jtj[a][b] += j[a] * j[b];
                }
            }
        }
        let Some(step) = solve(jtj, jte.map(|v| -v)) else { break };
        c += DVec2::new(step[0], step[1]);
        r_in += step[2];
        r_out += step[3];
        if step.iter().all(|v| v.abs() < 1e-4) {
            break;
        }
    }
    let tol = (0.02 * r_out).max(20.);
    if r_out - r_in < 2. * tol || r_in < tol {
        return None;
    }
    let on = |p: DVec2| {
        let d = (p - c).length();
        if (d - r_in).abs() <= tol {
            Some(0)
        } else if (d - r_out).abs() <= tol {
            Some(1)
        } else {
            None
        }
    };
    if pts.iter().any(|&p| on(p).is_none()) {
        return None;
    }
    // Angular coverage, sweeping along edges that follow an arc.
    let mut covered = 0u64;
    let ang = |p: DVec2| (p - c).y.atan2((p - c).x);
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        covered |= 1 << angle_bin(ang(a));
        if on(a) == on(b) {
            let (ta, tb) = (ang(a), ang(b));
            let mut dt = (tb - ta).rem_euclid(std::f64::consts::TAU);
            if dt > std::f64::consts::PI {
                dt -= std::f64::consts::TAU;
            }
            let n = (dt.abs() / (std::f64::consts::TAU / BINS as f64 / 2.)).ceil() as usize;
            for k in 1..=n {
                covered |= 1 << angle_bin(ta + dt * k as f64 / n as f64);
            }
        }
    }
    Some(Band { center: c, r_in, r_out, covered })
}

/// Solve a small linear system by Gaussian elimination with partial pivoting.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let piv = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..N {
            let f = a[row][col] / a[col][col];
            for k in col..N {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.; N];
    for row in (0..N).rev() {
        let s: f64 = (row + 1..N).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

// ---------------------------------------------------------------------------
// Prediction for a frame
// ---------------------------------------------------------------------------

/// A GDS on the map with its ring systems.
#[derive(Clone)]
pub struct GdsRings {
    pub uuid: Uuid,
    pub transform: DAffine2,
    pub systems: Arc<Vec<RingSystem>>,
}

/// Visible GDS layers that contain ring systems.
pub fn collect(objects: &SelectableList<Object>) -> Vec<GdsRings> {
    objects
        .iter()
        .filter_map(|o| match &**o {
            Object::Gds { image, hidden: false, .. } => {
                let systems = image.ring_systems();
                (!systems.is_empty()).then(|| GdsRings {
                    uuid: image.uuid(),
                    transform: image.transform,
                    systems,
                })
            }
            _ => None,
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdsPrediction {
    /// Ring centre on the map (world nm).
    pub center_world: DVec2,
    /// Unit direction from the frame centre to the ring centre (world).
    pub dir_world: DVec2,
    pub dist_nm: f64,
    /// The innermost ring should be in (or touching) the frame.
    pub smallest_in_frame: bool,
}

impl GdsPrediction {
    /// The direction in the frame's quad space (as stored for stamps).
    pub fn dir_quad(&self, frame: DAffine2) -> DVec2 {
        (frame.matrix2.inverse() * self.dir_world).normalize_or_zero()
    }
}

/// Predict the move for a frame (quad → world transform) from one GDS.
pub fn predict(systems: &[RingSystem], gds_transform: DAffine2, frame: DAffine2) -> Option<GdsPrediction> {
    let inv = gds_transform.inverse();
    let p = inv.transform_point2(frame.translation);
    let m = inv.matrix2 * frame.matrix2;
    let half_diag = (m * DVec2::new(0.5, 0.5)).length().max((m * DVec2::new(0.5, -0.5)).length());
    let mut best: Option<(f64, usize)> = None;
    for (i, s) in systems.iter().enumerate() {
        let v = p - s.center;
        let d = v.length();
        let r = s.outer_radius();
        if d > r + half_diag + 0.05 * r {
            continue;
        }
        let covered = d < s.bands[0].0 || s.covers(v.y.atan2(v.x));
        let score = d + if covered { 0. } else { r };
        if best.is_none_or(|(b, _)| score < b) {
            best = Some((score, i));
        }
    }
    let s = &systems[best?.1];
    let center_world = gds_transform.transform_point2(s.center);
    let v = center_world - frame.translation;
    Some(GdsPrediction {
        center_world,
        dir_world: v.normalize_or_zero(),
        dist_nm: v.length(),
        smallest_in_frame: (p - s.center).length() - half_diag <= s.bands[0].1,
    })
}

/// Prediction from whichever GDS covers the frame (the nearest centre wins).
pub fn predict_any(gds: &[GdsRings], frame: DAffine2) -> Option<(usize, GdsPrediction)> {
    gds.iter()
        .enumerate()
        .filter_map(|(i, g)| predict(&g.systems, g.transform, frame).map(|p| (i, p)))
        .min_by(|a, b| a.1.dist_nm.total_cmp(&b.1.dist_nm))
}

// ---------------------------------------------------------------------------
// Band direction measured in a scan
// ---------------------------------------------------------------------------

type Job = Box<dyn FnOnce() + Send>;

/// Run band measurements for stamps one at a time on a background thread,
/// and repaint when each is done (opening a project with many stamps must
/// not stall the UI).
pub fn queue_bands(job: Job) {
    static QUEUE: std::sync::LazyLock<crossbeam::channel::Sender<Job>> = std::sync::LazyLock::new(|| {
        let (tx, rx) = crossbeam::channel::unbounded::<Job>();
        std::thread::Builder::new()
            .name("stamp-bands".into())
            .spawn(move || {
                for job in rx.iter() {
                    job();
                    if rx.is_empty() {
                        crate::app::request_repaint();
                    }
                }
            })
            .expect("spawn stamp-bands thread");
        tx
    });
    let _ = QUEUE.send(job);
}

/// Dominant band direction in a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bands {
    /// Unit normal to the bands in pixel axes (+x right, +y down rows). The
    /// sign is arbitrary: bands give an axis, not a direction.
    pub normal_px: DVec2,
    /// 0 = no preferred direction, 1 = perfectly straight parallel bands.
    pub coherence: f64,
}

impl Bands {
    pub fn clear(&self) -> bool {
        self.coherence >= MIN_COHERENCE
    }
    /// The band normal on the map. Normals map with the inverse transpose.
    pub fn normal_world(&self, size: [u32; 2], frame: DAffine2) -> DVec2 {
        let nq = DVec2::new(self.normal_px.x * size[0] as f64, self.normal_px.y * size[1] as f64);
        (frame.matrix2.inverse().transpose() * nq).normalize_or_zero()
    }
}

/// Measure the broad band direction: remove a quadratic background, clip
/// outliers (blobs), downsample, blur at the band scale, and take the
/// structure tensor. Gaps (unscanned rows) are left out.
pub fn band_orientation(data: &[f32], size: [u32; 2], nm_per_px: f64) -> Option<Bands> {
    let (w, h) = (size[0] as usize, size[1] as usize);
    if w < 16 || h < 16 || data.len() != w * h || !(nm_per_px > 0.) {
        return None;
    }
    let valid = data.iter().filter(|v| v.is_finite()).count();
    if valid * 2 < w * h {
        return None;
    }
    let uv = |i: usize| ((i % w) as f64 / w as f64 - 0.5, (i / w) as f64 / h as f64 - 0.5);
    let basis = |(u, t): (f64, f64)| [1., u, t, u * u, t * t, u * t];
    let mut ata = [[0.; 6]; 6];
    let mut atb = [0.; 6];
    let step = (w * h / 20_000).max(1);
    for i in (0..w * h).step_by(step) {
        let v = data[i];
        if !v.is_finite() {
            continue;
        }
        let row = basis(uv(i));
        for a in 0..6 {
            atb[a] += row[a] * v as f64;
            for b in 0..6 {
                ata[a][b] += row[a] * row[b];
            }
        }
    }
    let cf = solve(ata, atb)?;
    let mut res: Vec<f64> = (0..w * h)
        .map(|i| {
            let v = data[i];
            if v.is_finite() {
                let row = basis(uv(i));
                v as f64 - (0..6).map(|k| cf[k] * row[k]).sum::<f64>()
            } else {
                f64::NAN
            }
        })
        .collect();
    let mut vals: Vec<f64> = res.iter().copied().filter(|v| v.is_finite()).collect();
    let n = vals.len();
    let lo = *vals.select_nth_unstable_by(n / 20, f64::total_cmp).1;
    let hi = *vals.select_nth_unstable_by(n * 19 / 20, f64::total_cmp).1;
    if !(hi > lo) {
        return None;
    }
    for v in res.iter_mut().filter(|v| v.is_finite()) {
        *v = v.clamp(lo, hi);
    }

    // Block-average to about 64 px.
    let k = (w.max(h) / 64).max(1);
    let (dw, dh) = (w / k, h / k);
    let mut img = vec![0.; dw * dh];
    let mut mask = vec![0.; dw * dh];
    for by in 0..dh {
        for bx in 0..dw {
            let (mut s, mut c) = (0., 0usize);
            for y in by * k..(by + 1) * k {
                for x in bx * k..(bx + 1) * k {
                    let v = res[y * w + x];
                    if v.is_finite() {
                        s += v;
                        c += 1;
                    }
                }
            }
            if c * 2 >= k * k {
                img[by * dw + bx] = s / c as f64;
                mask[by * dw + bx] = 1.;
            }
        }
    }
    // Normalised blur, so gaps don't create edges.
    let sigma = (BAND_BLUR_NM / (nm_per_px * k as f64)).clamp(0.7, 6.);
    let weighted: Vec<f64> = img.iter().zip(&mask).map(|(v, m)| v * m).collect();
    let bi = blur(&weighted, dw, dh, sigma);
    let bm = blur(&mask, dw, dh, sigma);
    let ok: Vec<bool> = mask.iter().zip(&bm).map(|(&m, &b)| m > 0. && b > 0.5).collect();
    let img: Vec<f64> = bi.iter().zip(&bm).map(|(v, m)| if *m > 1e-9 { v / m } else { 0. }).collect();

    let (mut jxx, mut jyy, mut jxy) = (0., 0., 0.);
    for y in 1..dh - 1 {
        for x in 1..dw - 1 {
            let i = y * dw + x;
            if !(ok[i] && ok[i - 1] && ok[i + 1] && ok[i - dw] && ok[i + dw]) {
                continue;
            }
            let gx = (img[i + 1] - img[i - 1]) / 2.;
            let gy = (img[i + dw] - img[i - dw]) / 2.;
            jxx += gx * gx;
            jyy += gy * gy;
            jxy += gx * gy;
        }
    }
    let tr = jxx + jyy;
    if !(tr > 0.) {
        return None;
    }
    let angle = 0.5 * (2. * jxy).atan2(jxx - jyy);
    let coherence = ((jxx - jyy).powi(2) + 4. * jxy * jxy).sqrt() / tr;
    Some(Bands { normal_px: DVec2::new(angle.cos(), angle.sin()), coherence })
}

fn blur(img: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let r = (3. * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-r..=r).map(|x| (-(x * x) as f64 / (2. * sigma * sigma)).exp()).collect();
    let mut tmp = vec![0.; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.;
            for (j, kv) in kernel.iter().enumerate() {
                let xx = x as isize + j as isize - r;
                if xx >= 0 && (xx as usize) < w {
                    s += kv * img[y * w + xx as usize];
                }
            }
            tmp[y * w + x] = s;
        }
    }
    let ksum: f64 = kernel.iter().sum();
    let mut out = vec![0.; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.;
            for (j, kv) in kernel.iter().enumerate() {
                let yy = y as isize + j as isize - r;
                if yy >= 0 && (yy as usize) < h {
                    s += kv * tmp[yy as usize * w + x];
                }
            }
            out[y * w + x] = s / (ksum * ksum);
        }
    }
    out
}

/// How the scan's bands compare with the GDS prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BandCheck {
    /// Bands run around the predicted centre (angle in degrees).
    Agree(f64),
    /// Clear bands, but at this angle to the predicted rings.
    Disagree(f64),
    /// No clear band direction in the scan.
    NoBands,
}

impl BandCheck {
    pub fn describe(&self) -> String {
        match self {
            BandCheck::Agree(a) => format!("scan bands agree ({a:.0}\u{b0})"),
            BandCheck::Disagree(a) => format!("scan bands disagree ({a:.0}\u{b0})"),
            BandCheck::NoBands => "no clear bands in the scan".into(),
        }
    }
}

/// Angle (degrees, 0..90) between the band normal and the GDS radial direction.
pub fn band_angle_deg(normal_world: DVec2, pred: &GdsPrediction) -> f64 {
    normal_world.dot(pred.dir_world).abs().clamp(0., 1.).acos().to_degrees()
}

pub fn check(bands: Option<Bands>, size: [u32; 2], frame: DAffine2, pred: &GdsPrediction) -> BandCheck {
    match bands.filter(|b| b.clear()) {
        None => BandCheck::NoBands,
        Some(b) => {
            let a = band_angle_deg(b.normal_world(size, frame), pred);
            if a <= AGREE_DEG { BandCheck::Agree(a) } else { BandCheck::Disagree(a) }
        }
    }
}

/// GDS prediction and band check for a stamped image.
#[derive(Clone, Copy, Debug)]
pub struct StampPrior {
    /// Index into the `GdsRings` list.
    pub gds: usize,
    pub pred: GdsPrediction,
    pub check: BandCheck,
}

pub fn stamp_prior(gds: &[GdsRings], image: &LiveImage) -> Option<StampPrior> {
    let (gi, pred) = predict_any(gds, image.transform)?;
    let check = check(image.bands(), image.forward_data.size, image.transform, &pred);
    Some(StampPrior { gds: gi, pred, check })
}

/// Draw a stamp's GDS prediction: a dashed violet arrow toward the ring
/// centre (dimmed with a "?" when the scan's bands disagree).
pub fn draw_stamp(ui: &egui::Ui, stamp_transform: DAffine2, p: &StampPrior) {
    use crate::utils::vec_interop::IntoEgui as _;
    let Some(ctx) = ui.data(|m| m.get_temp::<crate::scan_view::ScanViewCtx>(egui::Id::new(()))) else {
        return;
    };
    let w2s = ctx.world2egui();
    let c = stamp_transform.translation;
    let width = stamp_transform.matrix2.x_axis.length();
    let a = w2s.transform_point2(c).to_egui_pos2();
    let b = w2s.transform_point2(c + p.pred.dir_world * width * 0.35).to_egui_pos2();
    let disagree = matches!(p.check, BandCheck::Disagree(_));
    let color = if disagree { GDS_COLOR.gamma_multiply(0.55) } else { GDS_COLOR };
    let stroke = egui::Stroke::new(1.5, color);
    let painter = ui.painter();
    painter.extend(egui::Shape::dashed_line(&[a, b], stroke, 5., 3.));
    let tip = (b - a).normalized() * 8.;
    painter.arrow(b - tip, tip, stroke);
    if disagree {
        painter.text(
            b + egui::vec2(4., 0.),
            egui::Align2::LEFT_CENTER,
            "?",
            egui::FontId::proportional(12.),
            color,
        );
    }
}

// ---------------------------------------------------------------------------
// Alignment: how far the GDS is off, from many stamps
// ---------------------------------------------------------------------------

/// One stamp's evidence: its bands' normal line should pass through the
/// true ring centre.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub frame_center: DVec2,
    pub normal_world: DVec2,
    pub coherence: f64,
    /// The ring centre the GDS predicts for this stamp (world).
    pub ring_center: DVec2,
}

#[derive(Clone, Copy, Debug)]
pub struct Alignment {
    /// Move the GDS by this (world nm) to best match the scans.
    pub offset: DVec2,
    /// One-sigma uncertainty along x and y (nm).
    pub sigma: DVec2,
    pub used: usize,
}

impl Alignment {
    /// Worth suggesting: clearly larger than its uncertainty and than noise.
    pub fn significant(&self) -> bool {
        self.used >= 3
            && self.offset.length() >= 300.
            && self.offset.length() > 2. * self.sigma.max_element()
    }
}

/// Angular error assumed for a band normal (radians).
const NORMAL_SIGMA_RAD: f64 = 0.2;
/// The GDS is assumed to be within about this of the truth (nm).
const PRIOR_SIGMA_NM: f64 = 5000.;

/// Least-squares GDS offset: each normal line should pass through its ring
/// centre after the shift. A weak prior keeps the shift small when the lines
/// are nearly parallel.
pub fn estimate_offset(obs: &[Observation]) -> Option<Alignment> {
    if obs.len() < 3 {
        return None;
    }
    let mut a = DMat2::IDENTITY * (1. / PRIOR_SIGMA_NM.powi(2));
    let mut b = DVec2::ZERO;
    for o in obs {
        let n = o.normal_world.normalize_or_zero();
        let q = DMat2::IDENTITY - DMat2::from_cols(n * n.x, n * n.y);
        let dist = (o.ring_center - o.frame_center).length().max(1000.);
        let w = o.coherence / (NORMAL_SIGMA_RAD * dist).powi(2);
        a += q * w;
        b += q * (o.frame_center - o.ring_center) * w;
    }
    let cov = a.inverse();
    let offset = cov * b;
    // Scale the uncertainty by how well the lines actually meet (angular
    // scatter), never below a few degrees.
    let rms = (obs
        .iter()
        .map(|o| {
            let n = o.normal_world.normalize_or_zero();
            let v = o.ring_center + offset - o.frame_center;
            let perp = v - n * v.dot(n);
            (perp.length() / v.length().max(1000.)).powi(2)
        })
        .sum::<f64>()
        / obs.len() as f64)
        .sqrt();
    let k = (rms / NORMAL_SIGMA_RAD).max(0.25);
    let sigma = DVec2::new(cov.x_axis.x.sqrt(), cov.y_axis.y.sqrt()) * k;
    (offset.is_finite() && sigma.is_finite()).then_some(Alignment { offset, sigma, used: obs.len() })
}

/// Evidence from the stamps on one GDS, and its alignment estimate.
pub struct AlignmentCheck {
    /// Z stamps on this GDS's rings.
    pub stamps: usize,
    /// Of those with clear bands: (agreeing, total), and the median angle.
    pub agree: usize,
    pub with_bands: usize,
    pub median_deg: f64,
    pub estimate: Option<Alignment>,
}

/// Bands this far from the GDS prediction are treated as junk, not as a
/// sign the GDS is off.
const OUTLIER_DEG: f64 = 40.;

pub fn alignment_check(objects: &SelectableList<Object>, gds: &[GdsRings], gds_uuid: Uuid) -> AlignmentCheck {
    let mut out = AlignmentCheck { stamps: 0, agree: 0, with_bands: 0, median_deg: 0., estimate: None };
    let Some(gi) = gds.iter().position(|g| g.uuid == gds_uuid) else { return out };
    let mut angles = Vec::new();
    let mut obs = Vec::new();
    for o in objects.iter() {
        let Object::ScanImage { image, .. } = &**o else { continue };
        if image.unit != "m" {
            continue;
        }
        let Some(p) = stamp_prior(gds, image).filter(|p| p.gds == gi) else { continue };
        out.stamps += 1;
        let Some(b) = image.bands().filter(|b| b.clear()) else { continue };
        let n = b.normal_world(image.forward_data.size, image.transform);
        let a = band_angle_deg(n, &p.pred);
        angles.push(a);
        if a <= OUTLIER_DEG {
            obs.push(Observation {
                frame_center: image.transform.translation,
                normal_world: n,
                coherence: b.coherence,
                ring_center: p.pred.center_world,
            });
        }
    }
    out.with_bands = angles.len();
    out.agree = angles.iter().filter(|&&a| a <= AGREE_DEG).count();
    if !angles.is_empty() {
        angles.sort_by(f64::total_cmp);
        out.median_deg = angles[angles.len() / 2];
    }
    out.estimate = estimate_offset(&obs);
    out
}

// ---------------------------------------------------------------------------
// Has the GDS changed since stamps were taken?
// ---------------------------------------------------------------------------

/// How much the GDS moved at a point of interest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdsChange {
    pub shift_nm: f64,
    pub rot_deg: f64,
    pub scale: f64,
}

/// Compare an old and current GDS transform where it matters (at `at`,
/// world). `None` when the change is negligible.
pub fn compare(old: DAffine2, new: DAffine2, at: DVec2) -> Option<GdsChange> {
    let local = old.inverse().transform_point2(at);
    let shift_nm = (new.transform_point2(local) - at).length();
    let (s0, a0, _) = old.to_scale_angle_translation();
    let (s1, a1, _) = new.to_scale_angle_translation();
    let mut rot = (a1 - a0).to_degrees();
    rot = (rot + 180.).rem_euclid(360.) - 180.;
    let scale = s1.x / s0.x;
    (shift_nm > 50. || rot.abs() > 0.05 || (scale - 1.).abs() > 1e-3).then_some(GdsChange {
        shift_nm,
        rot_deg: rot,
        scale,
    })
}

/// Stamps whose GDS reference no longer matches the map.
#[derive(Default)]
pub struct RefStatus {
    /// Object indexes of stamps whose GDS has moved, and the largest change.
    pub moved: Vec<usize>,
    pub worst: Option<GdsChange>,
    /// The GDS (uuid) and the transform the first moved stamp was taken with.
    pub restore: Option<(Uuid, DAffine2)>,
    /// Stamps taken with a GDS that is no longer in the project.
    pub replaced: Vec<usize>,
}

pub fn ref_status(objects: &SelectableList<Object>) -> RefStatus {
    let current: Vec<(Uuid, DAffine2)> = objects
        .iter()
        .filter_map(|o| match &**o {
            Object::Gds { image, .. } => Some((image.uuid(), image.transform)),
            _ => None,
        })
        .collect();
    let mut st = RefStatus::default();
    for (i, o) in objects.iter().enumerate() {
        let Object::ScanImage { image, hidden: false, .. } = &**o else { continue };
        let Some((id, old)) = image.gds_ref else { continue };
        match current.iter().find(|(u, _)| *u == id) {
            Some(&(_, now)) => {
                if let Some(c) = compare(old, now, image.transform.translation) {
                    st.moved.push(i);
                    if st.worst.is_none_or(|w| c.shift_nm > w.shift_nm) {
                        st.worst = Some(c);
                    }
                    st.restore.get_or_insert((id, old));
                }
            }
            None if !current.is_empty() => st.replaced.push(i),
            None => {}
        }
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Annular arc polygon (local nm): inner arc out, outer arc back.
    fn arc(c: DVec2, r_in: f64, r_out: f64, a0: f64, a1: f64) -> Vec<Vec2> {
        let n = 40;
        let mut p = Vec::new();
        for k in 0..=n {
            let a = a0 + (a1 - a0) * k as f64 / n as f64;
            p.push((c + r_in * DVec2::new(a.cos(), a.sin())).as_vec2());
        }
        for k in (0..=n).rev() {
            let a = a0 + (a1 - a0) * k as f64 / n as f64;
            p.push((c + r_out * DVec2::new(a.cos(), a.sin())).as_vec2());
        }
        p
    }

    /// Upper and lower half-ring sets with separate centres, like the device.
    fn device() -> BTreeMap<u16, Vec<Vec<Vec2>>> {
        let up = DVec2::new(0., 3000.);
        let down = DVec2::new(0., -3000.);
        let mut polys = Vec::new();
        for k in 0..6 {
            let r = 1000. + 2500. * k as f64;
            polys.push(arc(up, r, r + 1000., 0., std::f64::consts::PI));
            polys.push(arc(down, r, r + 1000., std::f64::consts::PI, std::f64::consts::TAU));
        }
        // A bar and a disc that must not count as rings.
        polys.push(vec![Vec2::new(-20000., -100.), Vec2::new(20000., -100.), Vec2::new(20000., 100.), Vec2::new(-20000., 100.)]);
        polys.push((0..24).map(|k| (DVec2::new(30000., 0.) + 500. * DVec2::from_angle(k as f64 / 24. * std::f64::consts::TAU)).as_vec2()).collect());
        BTreeMap::from([(3u16, polys)])
    }

    #[test]
    fn finds_both_half_ring_systems() {
        let sys = find_ring_systems(&device());
        assert_eq!(sys.len(), 2, "{sys:?}");
        for s in &sys {
            assert_eq!(s.bands.len(), 6);
            assert!((s.center.y.abs() - 3000.).abs() < 1., "{:?}", s.center);
            assert!((s.bands[0].0 - 1000.).abs() < 1.);
        }
    }

    #[test]
    fn predicts_the_right_half_and_direction() {
        let sys = find_ring_systems(&device());
        let gds = DAffine2::from_scale_angle_translation(DVec2::splat(0.75), 0., DVec2::new(5000., 1000.));
        // A 3 µm frame on the upper rings, up-left of their centre.
        let at = gds.transform_point2(DVec2::new(-6000., 3000. + 6000.));
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(3000., -3000.), 0., at);
        let p = predict(&sys, gds, frame).unwrap();
        let want = gds.transform_point2(DVec2::new(0., 3000.));
        assert!((p.center_world - want).length() < 1., "{p:?}");
        assert!(p.dir_world.x > 0.6 && p.dir_world.y < -0.6);
        assert!(!p.smallest_in_frame);
        // Quad space has +y down, so "down-right" on the map is +x +y in quad.
        let q = p.dir_quad(frame);
        assert!(q.x > 0. && q.y > 0.);
        // Lower rings pick the lower centre; near the middle the smallest ring is in view.
        let at = gds.transform_point2(DVec2::new(500., -3000. - 1200.));
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(3000., -3000.), 0., at);
        let p = predict(&sys, gds, frame).unwrap();
        assert!((p.center_world - gds.transform_point2(DVec2::new(0., -3000.))).length() < 1.);
        assert!(p.smallest_in_frame);
        // Far away: no prediction.
        let far = DAffine2::from_scale_angle_translation(DVec2::new(3000., -3000.), 0., DVec2::new(1e6, 0.));
        assert!(predict(&sys, gds, far).is_none());
    }

    #[test]
    fn band_orientation_finds_stripe_normal() {
        // Stripes whose normal is at 30° (pixel axes), on a tilted background
        // with a few bright blobs.
        let (w, h) = (128usize, 128usize);
        let n = DVec2::new(30f64.to_radians().cos(), 30f64.to_radians().sin());
        let data: Vec<f32> = (0..w * h)
            .map(|i| {
                let p = DVec2::new((i % w) as f64, (i / w) as f64);
                let mut v = (p.dot(n) / 40. * std::f64::consts::TAU).sin() + 0.02 * p.x + 0.01 * p.y;
                if (p - DVec2::new(40., 40.)).length() < 5. {
                    v += 10.;
                }
                v as f32
            })
            .collect();
        let b = band_orientation(&data, [w as u32, h as u32], 30.).unwrap();
        assert!(b.normal_px.dot(n).abs() > 0.97, "{b:?}");
        assert!(b.clear());
        // Half-scanned frame (bottom rows missing) still works.
        let mut half = data.clone();
        for v in &mut half[w * 72..] {
            *v = f32::NAN;
        }
        let b = band_orientation(&half, [w as u32, h as u32], 30.).unwrap();
        assert!(b.normal_px.dot(n).abs() > 0.95, "{b:?}");
        // Noise only: no clear direction.
        let noise: Vec<f32> = (0..w * h).map(|i| ((i * 7919 % 1013) as f32 / 1013.) - 0.5).collect();
        let b = band_orientation(&noise, [w as u32, h as u32], 30.).unwrap();
        assert!(!b.clear(), "{b:?}");
    }

    #[test]
    fn normal_maps_to_world_with_rotation_and_flip() {
        // Frame rotated 90° with y flipped (as stamps are): a horizontal
        // band normal in pixels must come out along the world axis it maps to.
        let frame = DAffine2::from_cols(DVec2::new(0., 3000.), DVec2::new(3000., 0.), DVec2::ZERO);
        let b = Bands { normal_px: DVec2::X, coherence: 1. };
        let n = b.normal_world([100, 100], frame);
        assert!((n.abs() - DVec2::Y).length() < 1e-9, "{n:?}");
    }

    #[test]
    fn offset_estimate_recovers_a_shift() {
        // True centre is 1.5 µm right and 1 µm down of where the GDS says.
        let gds_c = DVec2::new(0., 0.);
        let truth = DVec2::new(1500., -1000.);
        let obs: Vec<Observation> = (0..8)
            .map(|k| {
                let a = k as f64 / 8. * std::f64::consts::PI + 0.2;
                let fc = truth + 12000. * DVec2::from_angle(a);
                Observation {
                    frame_center: fc,
                    normal_world: (fc - truth).normalize(),
                    coherence: 0.8,
                    ring_center: gds_c,
                }
            })
            .collect();
        let e = estimate_offset(&obs).unwrap();
        assert!((e.offset - truth).length() < 150., "{e:?}");
        assert!(e.significant());
        // Too few stamps: nothing.
        assert!(estimate_offset(&obs[..2]).is_none());
    }

    #[test]
    fn compare_ignores_tiny_changes() {
        let a = DAffine2::from_scale_angle_translation(DVec2::splat(0.75), 0., DVec2::new(100., 200.));
        assert!(compare(a, a, DVec2::new(-20000., 0.)).is_none());
        let b = DAffine2::from_scale_angle_translation(DVec2::splat(0.75), 0., DVec2::new(2100., 200.));
        let c = compare(a, b, DVec2::new(-20000., 0.)).unwrap();
        assert!((c.shift_nm - 2000.).abs() < 1e-6);
        // A small rotation moves far-away stamps a lot.
        let r = DAffine2::from_scale_angle_translation(DVec2::splat(0.75), 0.01, DVec2::new(100., 200.));
        let c = compare(a, r, DVec2::new(-20000., 0.)).unwrap();
        assert!(c.shift_nm > 150. && (c.rot_deg - 0.573).abs() < 0.01);
    }
}

#[cfg(test)]
mod real {
    use std::io::Write as _;

    use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
    use uuid::Uuid;

    use super::*;
    use crate::connection::FrameData;

    /// Export GDS polygons and stamps from a project for offline inspection,
    /// and print what the GDS prior makes of each labelled stamp.
    /// `PROJECT_FILE=x.scp DUMP_DIR=dir cargo test --bin scan_control real_project -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_project() {
        const GT: TableDefinition<Uuid, [f64; 6]> = TableDefinition::new("gds_transform_table_v1");
        const GL: TableDefinition<Uuid, Vec<u16>> = TableDefinition::new("gds_layer_table_v1");
        const GD: TableDefinition<(Uuid, u16), Vec<Vec<[f32; 2]>>> =
            TableDefinition::new("gds_data_table_v1");
        const LT: TableDefinition<Uuid, [f64; 6]> =
            TableDefinition::new("liveimage_transform_table_v1");
        const LD: TableDefinition<Uuid, [FrameData; 2]> =
            TableDefinition::new("liveimage_data_table_v1");
        const LL: TableDefinition<Uuid, [f64; 3]> =
            TableDefinition::new("liveimage_ringlabel_table_v1");
        const NAMES: TableDefinition<Uuid, (&[u8], bool)> =
            TableDefinition::new("object_data_table_v1");

        let f = std::env::var("PROJECT_FILE").unwrap();
        let out = std::env::var("DUMP_DIR").ok().map(std::path::PathBuf::from);
        let db = Database::open(f).unwrap();
        let txn = db.begin_read().unwrap();
        let gl = txn.open_table(GL).unwrap();
        let gd = txn.open_table(GD).unwrap();
        let gt = txn.open_table(GT).unwrap();
        let mut gds = Vec::new();
        for e in gt.iter().unwrap() {
            let (id, t) = e.unwrap();
            let id = id.value();
            let mut polys = BTreeMap::new();
            for layer in gl.get(id).unwrap().unwrap().value() {
                let p = gd.get((id, layer)).unwrap().unwrap().value();
                polys.insert(
                    layer,
                    p.into_iter().map(|q| q.into_iter().map(Vec2::from_array).collect()).collect(),
                );
            }
            let systems = find_ring_systems(&polys);
            let transform = DAffine2::from_cols_array(&t.value());
            for s in &systems {
                println!(
                    "system at {:?} (world), {} bands, innermost {:?}",
                    transform.transform_point2(s.center).round(),
                    s.bands.len(),
                    s.bands[0]
                );
            }
            gds.push(GdsRings { uuid: id, transform, systems: Arc::new(systems) });
        }
        let lt = txn.open_table(LT).unwrap();
        let ld = txn.open_table(LD).unwrap();
        let ll = txn.open_table(LL).ok();
        let names = txn.open_table(NAMES).unwrap();
        let mut obs = Vec::new();
        let (mut right, mut total) = (0, 0);
        for e in lt.iter().unwrap() {
            let (id, t) = e.unwrap();
            let id = id.value();
            let frame = DAffine2::from_cols_array(&t.value());
            let [fwd, _] = ld.get(id).unwrap().unwrap().value();
            if let Some(dir) = &out {
                std::fs::create_dir_all(dir).unwrap();
                let mut file = std::fs::File::create(dir.join(format!("{id}.bin"))).unwrap();
                for v in fwd.data.iter() {
                    file.write_all(&v.to_le_bytes()).unwrap();
                }
            }
            let name = names
                .get(id)
                .unwrap()
                .map(|v| String::from_utf8_lossy(v.value().0).to_string())
                .unwrap_or_default();
            let label = ll.as_ref().and_then(|t| t.get(id).unwrap()).map(|v| v.value());
            let Some((_, p)) = predict_any(&gds, frame) else {
                println!("{name}: not on a GDS ring system");
                continue;
            };
            let nm_per_px = frame.matrix2.x_axis.length() / fwd.size[0] as f64;
            let bands = band_orientation(&fwd.data, fwd.size, nm_per_px);
            let c = check(bands, fwd.size, frame, &p);
            let mut agree = String::new();
            if let Some(l) = label.filter(|l| l[1] != 0. || l[2] != 0.) {
                let lw = (frame.matrix2 * DVec2::new(l[1], l[2])).normalize();
                let cos = lw.dot(p.dir_world);
                total += 1;
                if cos >= 30f64.to_radians().cos() {
                    right += 1;
                }
                agree = format!("label\u{b7}gds {cos:+.2}");
            }
            println!(
                "{name:15} {:.1} µm, smallest {}, {} {agree}",
                p.dist_nm / 1e3,
                p.smallest_in_frame,
                c.describe()
            );
            if let Some(b) = bands.filter(|b| b.clear()) {
                let n = b.normal_world(fwd.size, frame);
                if band_angle_deg(n, &p) <= 40. {
                    obs.push(Observation {
                        frame_center: frame.translation,
                        normal_world: n,
                        coherence: b.coherence,
                        ring_center: p.center_world,
                    });
                }
            }
        }
        println!("GDS direction right on {right}/{total} labels");
        println!("alignment: {:?}", estimate_offset(&obs));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    /// `cargo test --release --bin scan_control gds_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gds_bench() {
        for n in [240usize, 512, 1024] {
            let data: Vec<f32> = (0..n * n)
                .map(|i| {
                    let p = DVec2::new((i % n) as f64, (i / n) as f64);
                    ((p.length() / 40.).sin() + 0.001 * p.x) as f32
                })
                .collect();
            let t = Instant::now();
            for _ in 0..5 {
                band_orientation(&data, [n as u32, n as u32], 3500. / n as f64);
            }
            let bands = t.elapsed() / 5;
            let t = Instant::now();
            let params = crate::analysis::ring_center::RingParams::default();
            let frame = crate::analysis::ring_stamp::frame_for([n as u32, n as u32], 3500. / n as f64);
            for _ in 0..5 {
                crate::analysis::ring_stamp::analyse_frame(&data, [n as u32, n as u32], frame, &params, 1.5);
            }
            let ring = t.elapsed() / 5;
            println!("{n}x{n}: bands {bands:?}, ring estimator {ring:?}");
        }
        let polys = BTreeMap::from([(3u16, (0..300).map(|k| {
            let c = DVec2::new((k / 12) as f64 * 50000., 0.);
            let r = 1000. + 2500. * (k % 12) as f64;
            (0..=60).map(|j| (c + r * DVec2::from_angle(j as f64 / 60. * 3.14)).as_vec2())
                .chain((0..=60).rev().map(|j| (c + (r + 1000.) * DVec2::from_angle(j as f64 / 60. * 3.14)).as_vec2()))
                .collect()
        }).collect())]);
        let t = Instant::now();
        let sys = find_ring_systems(&polys);
        println!("find_ring_systems (300 arcs): {:?}, {} systems", t.elapsed(), sys.len());
        let g = vec![GdsRings { uuid: Uuid::nil(), transform: DAffine2::IDENTITY, systems: Arc::new(sys) }];
        let t = Instant::now();
        for k in 0..10000 {
            let f = DAffine2::from_scale_angle_translation(DVec2::splat(3000.), 0., DVec2::new(k as f64, 5000.));
            std::hint::black_box(predict_any(&g, f));
        }
        println!("predict_any: {:?} each", t.elapsed() / 10000);
    }
}
