//! Estimate the centre of a concentric ring pattern from a single scan frame.
//!
//! Ring edges have height gradients that point radially, toward or away from
//! the ring centre. Each strong edge pixel therefore defines a line, through the
//! pixel and along its gradient, that passes (approximately) through the centre.
//! The centre is the least-squares intersection of all those lines, refined
//! with IRLS so that non-radial features (dust, tip changes, straight terraces)
//! are down-weighted.
//!
//! Everything here is in pixel coordinates: `x` = column, `y` = row index into
//! the frame data. Converting to world coordinates is up to the caller.

use glam::{DMat2, DVec2};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RingParams {
    /// Subtract each scan line's mean before the plane fit. This removes
    /// line-to-line offsets (tip changes, z drift), but it also erases any ring
    /// edge running parallel to the fast scan axis, because such an edge looks
    /// exactly like a line offset. Off by default.
    pub line_level: bool,
    /// Block-average the frame so its longest side is at most this many
    /// pixels before analysis. This suppresses fine features (atomic steps)
    /// so that only the broad implant bands produce edges. None keeps full res.
    pub target_px: Option<usize>,
    /// Blank out scan lines whose level jumps relative to their neighbours by
    /// more than this many robust standard deviations (tip-change glitches).
    /// None disables.
    pub glitch_sigma: Option<f32>,
    /// Gaussian blur applied before taking gradients (pixels). 0 disables.
    pub blur_sigma: f32,
    /// If set, subtract a heavily blurred copy (sigma in pixels) to remove
    /// broad background such as terraces and tilt bands.
    pub highpass_sigma: Option<f32>,
    /// Fraction of valid pixels (by gradient magnitude) kept as edge pixels.
    pub edge_fraction: f32,
    /// Number of IRLS reweighting passes.
    pub irls_iters: usize,
    /// Angular residual scale (radians) for the Cauchy weights.
    pub angle_scale: f64,
    /// Minimum edge pixels needed to produce an estimate.
    pub min_edges: usize,
    /// Fits with a radius below this (input pixels) are treated as local
    /// features (arm end caps, blobs) rather than rings: their edges are
    /// removed and the fit is repeated. None disables peeling.
    pub min_ring_radius_px: Option<f64>,
    /// Maximum number of local features to peel off.
    pub max_peel: usize,
    /// [`decide`] reports `Unsure` below this inlier fraction. On the training
    /// set, ring scans scored 0.20-0.88 and the middle electrodes 0.08.
    pub min_inlier_fraction: f64,
}

impl Default for RingParams {
    fn default() -> Self {
        // Tuned on the Nov 2026 ring training set (scans of 3.5-3.8 um, ~200 px):
        // see the `ring_on_sxm` harness. Callers should also set
        // `min_ring_radius_px` to about 1.5 um in pixels.
        Self {
            line_level: false,
            target_px: Some(32),
            glitch_sigma: None,
            blur_sigma: 2.0,
            highpass_sigma: None,
            edge_fraction: 0.3,
            irls_iters: 8,
            angle_scale: 0.1,
            min_edges: 50,
            min_ring_radius_px: None,
            max_peel: 3,
            min_inlier_fraction: 0.15,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RingEstimate {
    /// Least-squares centre of the rings (pixels). It may lie far outside the
    /// frame. When `conditioning` is small, only its side (sign) is meaningful.
    pub center_px: DVec2,
    /// Unit vector from the frame centre toward `center_px`.
    pub direction: DVec2,
    /// Dominant edge-normal axis (unit, sign arbitrary). It is reliable even
    /// when the curvature is too weak to locate the centre.
    pub normal_axis: DVec2,
    /// Weighted median distance from edge pixels to the centre (pixels).
    pub radius_px: f64,
    /// λmin / λmax of the normal-equation matrix, in [0, 1]. Near 0 means the
    /// edges are almost parallel, so the distance to the centre is poorly determined.
    pub conditioning: f64,
    /// Fraction of edge pixels whose gradient line passes within
    /// `3 * angle_scale` of the centre. Low values mean the edges are not
    /// radial (noise, no rings in view), so the estimate should not be trusted.
    pub inlier_fraction: f64,
    /// Number of edge pixels used.
    pub n_edges: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RingError {
    TooFewEdges(usize),
    Degenerate,
}

/// Estimate the ring centre from a row-major `width × height` frame. NaN
/// pixels (unscanned lines) are ignored.
pub fn estimate_ring_center(
    data: &[f32],
    width: usize,
    height: usize,
    params: &RingParams,
) -> Result<RingEstimate, RingError> {
    let pre = preprocess(data, width, height, params);
    let mut est = estimate_preprocessed(&pre.img, pre.width, pre.height, pre.scale, params)?.est;
    // Back to input pixel coordinates.
    let f = pre.scale;
    let to_in = |p: DVec2| (p + 0.5) * f - 0.5;
    est.center_px = to_in(est.center_px);
    est.radius_px *= f;
    Ok(est)
}

/// Result of [`preprocess`]: the analysed image, and the factor from its
/// pixels back to input pixels.
pub struct Preprocessed {
    pub img: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub scale: f64,
}

pub fn preprocess(data: &[f32], width: usize, height: usize, params: &RingParams) -> Preprocessed {
    assert_eq!(data.len(), width * height);
    let mut img: Vec<f32> = data.to_vec();
    if let Some(k) = params.glitch_sigma {
        remove_glitch_lines(&mut img, width, height, k);
    }
    let (mut img, width, height, scale) = match params.target_px {
        Some(t) if width.max(height) > t => {
            let f = width.max(height).div_ceil(t);
            let (w, h, out) = block_average(&img, width, height, f);
            (out, w, h, f as f64)
        }
        _ => (img, width, height, 1.),
    };
    if params.line_level {
        line_mean_subtract(&mut img, width, height);
    }
    plane_subtract(&mut img, width, height);
    if let Some(hp) = params.highpass_sigma {
        let bg = gaussian_blur(&img, width, height, hp);
        for (v, b) in img.iter_mut().zip(bg) {
            *v -= b;
        }
    }
    if params.blur_sigma > 0. {
        img = gaussian_blur(&img, width, height, params.blur_sigma);
    }
    Preprocessed { img, width, height, scale }
}

/// An edge pixel: position, unit gradient direction, and weight (gradient
/// magnitude on input; final IRLS weight in [`Analysis::edges`]).
pub type Edge = (DVec2, DVec2, f64);

/// A small, locally curved feature (an arm end cap, a blob) that was fitted
/// and removed before the ring fit. Analysed-image pixels.
#[derive(Clone, Copy, Debug)]
pub struct LocalFeature {
    pub center_px: DVec2,
    pub radius_px: f64,
    pub n_edges: usize,
}

pub struct Analysis {
    pub est: RingEstimate,
    /// Edge pixels used by the final ring fit, with their final weights.
    pub edges: Vec<Edge>,
    /// Features peeled off before the final fit, in order.
    pub features: Vec<LocalFeature>,
}

/// Run the estimator on an already-preprocessed image (`scale` = input pixels
/// per analysed pixel, from [`Preprocessed`]). Results are in analysed-image
/// pixel coordinates.
pub fn estimate_preprocessed(
    img: &[f32],
    width: usize,
    height: usize,
    scale: f64,
    params: &RingParams,
) -> Result<Analysis, RingError> {
    // Sobel gradients on interior pixels with a full valid 3x3 neighbourhood.
    let mut edges: Vec<Edge> = Vec::new();
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let p = |dx: isize, dy: isize| {
                img[(y as isize + dy) as usize * width + (x as isize + dx) as usize] as f64
            };
            let gx = (p(1, -1) + 2. * p(1, 0) + p(1, 1)) - (p(-1, -1) + 2. * p(-1, 0) + p(-1, 1));
            let gy = (p(-1, 1) + 2. * p(0, 1) + p(1, 1)) - (p(-1, -1) + 2. * p(0, -1) + p(1, -1));
            let mag = (gx * gx + gy * gy).sqrt();
            if mag.is_finite() && mag > 0. {
                edges.push((
                    DVec2::new(x as f64, y as f64),
                    DVec2::new(gx / mag, gy / mag),
                    mag,
                ));
            }
        }
    }
    let n_valid = edges.len();
    let keep = ((n_valid as f32 * params.edge_fraction) as usize).min(n_valid);
    if keep < params.min_edges {
        return Err(RingError::TooFewEdges(keep));
    }
    edges.select_nth_unstable_by(keep - 1, |a, b| b.2.total_cmp(&a.2));
    edges.truncate(keep);

    let frame_center = DVec2::new((width as f64 - 1.) / 2., (height as f64 - 1.) / 2.);
    let min_r = params.min_ring_radius_px.map(|r| r / scale);
    let mut features = Vec::new();
    loop {
        let (est, weights) = fit_center(&edges, frame_center, params)?;
        let Some(min_r) = min_r else {
            return Ok(Analysis { est, edges: with_weights(edges, weights), features });
        };
        if est.radius_px >= min_r || features.len() >= params.max_peel {
            return Ok(Analysis { est, edges: with_weights(edges, weights), features });
        }
        // Small radius: this fit locked onto a local feature. Remove the edges
        // that belong to it (radial to its centre and close to it) and refit.
        let c = est.center_px;
        let belongs = |e: &Edge| {
            angular_residual(c, e.0, e.1) < 3. * params.angle_scale
                && (c - e.0).length() < 2.5 * est.radius_px.max(1.)
        };
        let n_feat = edges.iter().filter(|e| belongs(e)).count();
        if n_feat == 0 || edges.len() - n_feat < params.min_edges {
            return Ok(Analysis { est, edges: with_weights(edges, weights), features });
        }
        features.push(LocalFeature { center_px: c, radius_px: est.radius_px, n_edges: n_feat });
        edges.retain(|e| !belongs(e));
    }
}

/// What one frame says to do next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// On the ring arcs: move along `dir` (unit, image pixel axes: +x right,
    /// +y down) toward the ring centre.
    Arc { dir: DVec2 },
    /// At the ends of ring arms (caps were peeled off): move sideways, along
    /// image ±x, toward the side the remaining arm curvature points to.
    ArmEnd { dir: DVec2 },
    Unsure(UnsureReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsureReason {
    /// Too few edges to fit anything.
    NoEdges,
    /// Edges don't share a common centre (noise, electrodes, no rings).
    NotRingLike,
    /// Everything that fitted was a small local feature; no ring left over.
    OnlyLocalFeatures,
}

impl UnsureReason {
    pub fn describe(self) -> &'static str {
        match self {
            UnsureReason::NoEdges => "too few edges",
            UnsureReason::NotRingLike => "edges are not ring-like",
            UnsureReason::OnlyLocalFeatures => "only small local features found",
        }
    }
}

/// Turn an analysis into a verdict. `scale` is input pixels per analysed pixel.
pub fn decide(res: &Result<Analysis, RingError>, scale: f64, params: &RingParams) -> Verdict {
    let a = match res {
        Ok(a) => a,
        Err(_) => return Verdict::Unsure(UnsureReason::NoEdges),
    };
    if a.est.inlier_fraction < params.min_inlier_fraction {
        return Verdict::Unsure(UnsureReason::NotRingLike);
    }
    if let Some(min_r) = params.min_ring_radius_px {
        if a.est.radius_px * scale < min_r {
            return Verdict::Unsure(UnsureReason::OnlyLocalFeatures);
        }
    }
    if a.features.is_empty() {
        return Verdict::Arc { dir: a.est.direction };
    }
    if a.est.direction.x == 0. {
        return Verdict::Unsure(UnsureReason::NotRingLike);
    }
    Verdict::ArmEnd { dir: DVec2::new(a.est.direction.x.signum(), 0.) }
}

fn with_weights(edges: Vec<Edge>, weights: Vec<f64>) -> Vec<Edge> {
    edges.into_iter().zip(weights).map(|((p, n, _), w)| (p, n, w)).collect()
}

/// Angle between the line through `p` along `n` and the direction from `p` to `c`.
fn angular_residual(c: DVec2, p: DVec2, n: DVec2) -> f64 {
    let r = c - p;
    let dist = r.length().max(1.);
    ((r.x * n.y - r.y * n.x).abs() / dist).min(1.).asin()
}

/// IRLS least-squares intersection of the edge lines. Returns the estimate
/// and the final per-edge weights.
fn fit_center(
    edges: &[Edge],
    frame_center: DVec2,
    params: &RingParams,
) -> Result<(RingEstimate, Vec<f64>), RingError> {
    // Dominant normal axis: principal eigenvector of sum w n n^T.
    let mut s = DMat2::ZERO;
    for &(_, n, w) in edges {
        s += outer(n, n) * w;
    }
    let (_, normal_axis) = sym_eig(s).1;

    let mut weights: Vec<f64> = edges.iter().map(|e| e.2).collect();
    let mut center = DVec2::ZERO;
    let mut conditioning = 0.;
    for iter in 0..=params.irls_iters {
        let mut a = DMat2::ZERO;
        let mut b = DVec2::ZERO;
        for (&(p, n, _), &w) in edges.iter().zip(&weights) {
            let proj = DMat2::IDENTITY - outer(n, n);
            a += proj * w;
            b += (proj * p) * w;
        }
        let ((lmin, lmax), _) = sym_eig(a);
        if !(lmax > 0.) || !(lmin > lmax * 1e-12) {
            return Err(RingError::Degenerate);
        }
        conditioning = lmin / lmax;
        center = a.inverse() * b;
        if iter == params.irls_iters {
            break;
        }
        for (&(p, n, mag), w) in edges.iter().zip(weights.iter_mut()) {
            let ang = angular_residual(center, p, n) / params.angle_scale;
            *w = mag / (1. + ang * ang);
        }
    }

    let inliers = edges
        .iter()
        .filter(|e| angular_residual(center, e.0, e.1) < 3. * params.angle_scale)
        .count();
    let inlier_fraction = inliers as f64 / edges.len() as f64;
    let direction = (center - frame_center).normalize_or_zero();

    let mut radii: Vec<(f64, f64)> = edges
        .iter()
        .zip(&weights)
        .map(|(e, &w)| ((center - e.0).length(), w))
        .collect();
    radii.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = radii.iter().map(|r| r.1).sum();
    let mut acc = 0.;
    let mut radius_px = radii.last().map(|r| r.0).unwrap_or(0.);
    for (r, w) in radii {
        acc += w;
        if acc >= total / 2. {
            radius_px = r;
            break;
        }
    }

    let est = RingEstimate {
        center_px: center,
        direction,
        normal_axis,
        radius_px,
        conditioning,
        inlier_fraction,
        n_edges: edges.len(),
    };
    Ok((est, weights))
}

fn block_average(img: &[f32], width: usize, height: usize, f: usize) -> (usize, usize, Vec<f32>) {
    let (w, h) = (width.div_ceil(f), height.div_ceil(f));
    let mut out = vec![f32::NAN; w * h];
    for by in 0..h {
        for bx in 0..w {
            let (mut s, mut n) = (0f64, 0usize);
            for y in by * f..((by + 1) * f).min(height) {
                for x in bx * f..((bx + 1) * f).min(width) {
                    let v = img[y * width + x];
                    if v.is_finite() {
                        s += v as f64;
                        n += 1;
                    }
                }
            }
            // Require most of the block to be valid so partial lines don't bias it.
            if n * 2 > f * f {
                out[by * w + bx] = (s / n as f64) as f32;
            }
        }
    }
    (w, h, out)
}

/// Detect scan lines that are offset from both neighbours (tip changes,
/// glitches) and set them to NaN. Each line's offset is measured as the median
/// over the line of `row - (above + below) / 2`. Lines whose offset exceeds
/// `k` robust sigmas (MAD) of all lines' offsets are dropped.
fn remove_glitch_lines(img: &mut [f32], width: usize, height: usize, k: f32) {
    if height < 5 {
        return;
    }
    let median = |v: &mut Vec<f32>| -> f32 {
        if v.is_empty() {
            return f32::NAN;
        }
        let m = v.len() / 2;
        *v.select_nth_unstable_by(m, f32::total_cmp).1
    };
    let mut offsets = vec![f32::NAN; height];
    let mut buf = Vec::with_capacity(width);
    for y in 1..height - 1 {
        buf.clear();
        for x in 0..width {
            let (a, r, b) = (img[(y - 1) * width + x], img[y * width + x], img[(y + 1) * width + x]);
            let d = r - (a + b) / 2.;
            if d.is_finite() {
                buf.push(d);
            }
        }
        offsets[y] = median(&mut buf);
    }
    let mut finite: Vec<f32> = offsets.iter().copied().filter(|v| v.is_finite()).collect();
    let med = median(&mut finite);
    let mut dev: Vec<f32> = finite.iter().map(|v| (v - med).abs()).collect();
    let mad = median(&mut dev) * 1.4826;
    if !(mad > 0.) {
        return;
    }
    let bad: Vec<usize> = (1..height - 1)
        .filter(|&y| (offsets[y] - med).abs() > k * mad)
        .collect();
    for y in bad {
        // A single-line glitch makes its neighbours look offset by half as
        // much in the other direction, so blank one line either side too.
        for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            img[yy * width..(yy + 1) * width].fill(f32::NAN);
        }
    }
}

fn outer(a: DVec2, b: DVec2) -> DMat2 {
    DMat2::from_cols(a * b.x, a * b.y)
}

/// Eigen decomposition of a symmetric 2x2 matrix: ((λmin, λmax), (v_min, v_max)).
fn sym_eig(m: DMat2) -> ((f64, f64), (DVec2, DVec2)) {
    let (a, b, d) = (m.x_axis.x, m.y_axis.x, m.y_axis.y);
    let tr = a + d;
    let disc = (((a - d) / 2.).powi(2) + b * b).sqrt();
    let lmax = tr / 2. + disc;
    let lmin = tr / 2. - disc;
    let vmax = if b.abs() > 1e-300 {
        DVec2::new(lmax - d, b).normalize()
    } else if a >= d {
        DVec2::X
    } else {
        DVec2::Y
    };
    let vmin = vmax.perp();
    ((lmin, lmax), (vmin, vmax))
}

fn line_mean_subtract(img: &mut [f32], width: usize, height: usize) {
    for row in img.chunks_mut(width).take(height) {
        let (sum, n) = row
            .iter()
            .filter(|v| v.is_finite())
            .fold((0f64, 0usize), |(s, n), &v| (s + v as f64, n + 1));
        if n > 0 {
            let mean = (sum / n as f64) as f32;
            row.iter_mut().for_each(|v| *v -= mean);
        }
    }
}

fn plane_subtract(img: &mut [f32], width: usize, height: usize) {
    // Least-squares z = c0 + c1 x + c2 y over the finite pixels.
    let mut m = [[0f64; 3]; 3];
    let mut r = [0f64; 3];
    for y in 0..height {
        for x in 0..width {
            let z = img[y * width + x];
            if !z.is_finite() {
                continue;
            }
            let f = [1., x as f64, y as f64];
            for i in 0..3 {
                r[i] += f[i] * z as f64;
                for j in 0..3 {
                    m[i][j] += f[i] * f[j];
                }
            }
        }
    }
    let m = nalgebra::Matrix3::from_fn(|i, j| m[i][j]);
    let Some(inv) = m.try_inverse() else { return };
    let c = inv * nalgebra::Vector3::new(r[0], r[1], r[2]);
    for y in 0..height {
        for x in 0..width {
            img[y * width + x] -= (c[0] + c[1] * x as f64 + c[2] * y as f64) as f32;
        }
    }
}

/// Separable, NaN-aware Gaussian blur: NaN pixels are excluded and the kernel
/// is renormalised, so unscanned regions don't bleed into valid data.
fn gaussian_blur(img: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let rad = (3. * sigma).ceil().max(1.) as isize;
    let kernel: Vec<f32> = (-rad..=rad)
        .map(|i| (-(i * i) as f32 / (2. * sigma * sigma)).exp())
        .collect();
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![f32::NAN; src.len()];
        for y in 0..height {
            for x in 0..width {
                if !src[y * width + x].is_finite() {
                    continue;
                }
                let (mut s, mut ws) = (0f32, 0f32);
                for (k, &kw) in kernel.iter().enumerate() {
                    let o = k as isize - rad;
                    let (xx, yy) = if horizontal {
                        (x as isize + o, y as isize)
                    } else {
                        (x as isize, y as isize + o)
                    };
                    if xx < 0 || yy < 0 || xx >= width as isize || yy >= height as isize {
                        continue;
                    }
                    let v = src[yy as usize * width + xx as usize];
                    if v.is_finite() {
                        s += v * kw;
                        ws += kw;
                    }
                }
                out[y * width + x] = s / ws;
            }
        }
        out
    };
    let h = pass(img, true);
    pass(&h, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic PRNG so the tests don't need a dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        }
    }

    /// Concentric rings with smoothed step edges (like implant boundaries),
    /// plus tilt, a straight diagonal terrace and noise.
    fn synth(size: usize, center: DVec2, period: f64, noise: f32, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let mut out = Vec::with_capacity(size * size);
        for y in 0..size {
            for x in 0..size {
                let p = DVec2::new(x as f64, y as f64);
                let r = (p - center).length();
                let phase = (r / period).fract();
                // Soft square wave: 1 on the first half of each period.
                let ring = 1. / (1. + (-(0.25 - (phase - 0.25).abs()) * period / 1.5).exp());
                let tilt = 0.02 * x as f64 + 0.01 * y as f64;
                let terrace = if (x + y) as f64 > size as f64 * 1.2 { 0.3 } else { 0. };
                out.push((ring + tilt + terrace) as f32 + noise * rng.next());
            }
        }
        out
    }

    /// Full-resolution settings matching the synthetic images' scale (ring
    /// period ~20 px). The tuned defaults downsample to 32 px, which would
    /// alias these rings.
    fn synth_params() -> RingParams {
        RingParams { target_px: None, edge_fraction: 0.08, angle_scale: 0.05, ..Default::default() }
    }

    fn check_direction(est: &RingEstimate, size: usize, truth: DVec2, max_deg: f64) {
        let fc = DVec2::splat((size as f64 - 1.) / 2.);
        let want = (truth - fc).normalize();
        let ang = want.angle_to(est.direction).to_degrees().abs();
        assert!(ang < max_deg, "direction off by {ang:.1}° ({est:?})");
    }

    #[test]
    fn center_inside_frame() {
        let size = 128;
        let truth = DVec2::new(70., 50.);
        let img = synth(size, truth, 20., 0.1, 1);
        let est = estimate_ring_center(&img, size, size, &synth_params()).unwrap();
        assert!((est.center_px - truth).length() < 3., "{est:?}");
        assert!(est.conditioning > 0.3, "{est:?}");
        assert!(est.inlier_fraction > 0.7, "{est:?}");
    }

    #[test]
    fn center_moderately_outside() {
        let size = 128;
        let truth = DVec2::new(-300., 90.);
        let img = synth(size, truth, 25., 0.1, 2);
        let est = estimate_ring_center(&img, size, size, &synth_params()).unwrap();
        check_direction(&est, size, truth, 5.);
        let rel = (est.center_px - truth).length() / (truth - DVec2::splat(63.5)).length();
        assert!(rel < 0.3, "distance error {rel:.2} ({est:?})");
    }

    #[test]
    fn center_far_away_direction_only() {
        // Sagitta over the frame is only ~1 px here; we still want the side right.
        for (i, truth) in [
            DVec2::new(2000., 64.),
            DVec2::new(64., -2000.),
            DVec2::new(-1400., -1400.),
        ]
        .into_iter()
        .enumerate()
        {
            let size = 128;
            let img = synth(size, truth, 25., 0.05, 10 + i as u64);
            let est = estimate_ring_center(&img, size, size, &synth_params()).unwrap();
            check_direction(&est, size, truth, 15.);
            assert!(est.conditioning < 0.1, "{est:?}");
        }
    }

    #[test]
    fn partial_frame_with_nans() {
        let size = 128;
        let truth = DVec2::new(-200., -150.);
        let mut img = synth(size, truth, 25., 0.1, 3);
        // Only the first 80 lines have been scanned.
        img[80 * size..].iter_mut().for_each(|v| *v = f32::NAN);
        let est = estimate_ring_center(&img, size, size, &synth_params()).unwrap();
        check_direction(&est, size, truth, 10.);
    }

    #[test]
    fn decide_arc_and_unsure() {
        let size = 128;
        let truth = DVec2::new(-300., 90.);
        let img = synth(size, truth, 25., 0.1, 2);
        // As in live use: anything under ~1.5 ring periods is a local feature.
        let p = RingParams { min_ring_radius_px: Some(40.), ..synth_params() };
        let pre = preprocess(&img, size, size, &p);
        let res = estimate_preprocessed(&pre.img, pre.width, pre.height, pre.scale, &p);
        match decide(&res, pre.scale, &p) {
            Verdict::Arc { dir } => assert!(dir.x < -0.9, "{dir:?}"),
            v => panic!("expected Arc, got {v:?}"),
        }

        let mut rng = Lcg(7);
        let noise: Vec<f32> = (0..64 * 64).map(|_| rng.next()).collect();
        let pre = preprocess(&noise, 64, 64, &p);
        let res = estimate_preprocessed(&pre.img, pre.width, pre.height, pre.scale, &p);
        let v = decide(&res, pre.scale, &p);
        assert!(matches!(v, Verdict::Unsure(_)), "{v:?}");
    }

    #[test]
    fn flat_noise_is_rejected_or_unconfident() {
        let size = 64;
        let mut rng = Lcg(7);
        let img: Vec<f32> = (0..size * size).map(|_| rng.next()).collect();
        let params = synth_params();
        if let Ok(est) = estimate_ring_center(&img, size, size, &params) {
            assert!(est.inlier_fraction < 0.5, "{est:?}");
        }
    }
}

/// Offline tuning harness for real data. Not run by default:
///
/// ```text
/// set RING_SXM_DIR=D:\path\to\scans
/// set RING_DEBUG_DIR=D:\path\to\out        (optional: writes a PNG per scan)
/// cargo test --bin scan_control ring_on_sxm -- --ignored --nocapture
/// ```
///
/// Parameters can be overridden with RING_TARGET, RING_BLUR, RING_HP,
/// RING_GLITCH, RING_EDGE_FRAC, RING_ANGLE (0 / "none" disables optional ones).
///
/// The printed direction is a compass angle in the displayed image: 0° = up
/// (+y), 90° = right (+x), measured in the scan frame (scan angle not applied).
#[cfg(test)]
pub(crate) mod sxm_harness {
    use super::*;

    fn env_opt<T: std::str::FromStr>(key: &str) -> Option<Option<T>> {
        let v = std::env::var(key).ok()?;
        if v == "none" || v == "0" {
            return Some(None);
        }
        Some(v.parse().ok())
    }

    pub(super) fn params_from_env() -> RingParams {
        let mut p = RingParams::default();
        if let Some(v) = env_opt("RING_TARGET") { p.target_px = v; }
        if let Some(v) = env_opt("RING_HP") { p.highpass_sigma = v; }
        if let Some(v) = env_opt("RING_GLITCH") { p.glitch_sigma = v; }
        if let Some(Some(v)) = env_opt("RING_BLUR") { p.blur_sigma = v; }
        if let Some(Some(v)) = env_opt("RING_EDGE_FRAC") { p.edge_fraction = v; }
        if let Some(Some(v)) = env_opt("RING_ANGLE") { p.angle_scale = v; }
        if let Some(Some(v)) = env_opt("RING_PEEL") { p.max_peel = v; }
        p
    }

    /// Load the forward Z channel with row 0 at the top of the displayed image.
    pub(crate) fn load_z(path: &std::path::Path) -> Result<(Vec<f32>, usize, usize, [f32; 2]), String> {
        let sxm = sxmfile::SXM::parse_file(path).map_err(|e| format!("parse error: {e:#}"))?;
        let [w, h] = sxm.get_image_size().map_err(|e| e.to_string())?;
        let range = sxm.get_scan_range().map_err(|e| e.to_string())?;
        let chans = sxm.get_channels().map_err(|e| e.to_string())?;
        let zi = chans
            .iter()
            .position(|(n, _)| n.trim() == "Z")
            .ok_or("no Z channel")?;
        // Nanonis stores the first-acquired line first. For an "up" scan that
        // is the bottom line, so flip to get row 0 = top.
        let up = sxm.get_metadata("SCAN_DIR").map(|d| d.trim() == "up").unwrap_or(false);
        let (w, h) = (w as usize, h as usize);
        let mut z = sxm.data[zi][0].to_vec();
        if up {
            z = z.chunks(w).rev().flatten().copied().collect();
        }
        Ok((z, w, h, range))
    }

    fn compass(dir_px: DVec2, sx: f64, sy: f64) -> f64 {
        let d = DVec2::new(dir_px.x * sx, -dir_px.y * sy);
        d.x.atan2(d.y).to_degrees().rem_euclid(360.)
    }

    fn write_debug(out: &std::path::Path, pre: &Preprocessed, res: &Result<Analysis, RingError>) {
        const UP: usize = 4;
        let (w, h) = (pre.width, pre.height);
        let mut vals: Vec<f32> = pre.img.iter().copied().filter(|v| v.is_finite()).collect();
        vals.sort_by(f32::total_cmp);
        let lo = vals.get(vals.len() / 100).copied().unwrap_or(0.);
        let hi = vals.get(vals.len() * 99 / 100).copied().unwrap_or(1.);
        let mut im = image::RgbImage::new((w * UP) as u32, (h * UP) as u32);
        for y in 0..h * UP {
            for x in 0..w * UP {
                let v = pre.img[(y / UP) * w + x / UP];
                let px = if v.is_finite() {
                    let g = (((v - lo) / (hi - lo)).clamp(0., 1.) * 255.) as u8;
                    [g, g, g]
                } else {
                    [60, 0, 60]
                };
                im.put_pixel(x as u32, y as u32, image::Rgb(px));
            }
        }
        let mut put = |x: f64, y: f64, c: [u8; 3]| {
            let (x, y) = ((x + 0.5) * UP as f64, (y + 0.5) * UP as f64);
            if x >= 0. && y >= 0. && (x as u32) < im.width() && (y as u32) < im.height() {
                im.put_pixel(x as u32, y as u32, image::Rgb(c));
            }
        };
        if let Ok(Analysis { est, edges, features }) = res {
            for f in features {
                for a in 0..128 {
                    let t = a as f64 / 128. * std::f64::consts::TAU;
                    put(f.center_px.x + f.radius_px * t.cos(), f.center_px.y + f.radius_px * t.sin(), [0, 200, 255]);
                }
            }
            let wmax = edges.iter().map(|e| e.2).fold(0., f64::max);
            for &(p, _, wgt) in edges {
                let t = (wgt / wmax).sqrt();
                put(p.x, p.y, [(255. * t) as u8, 0, (255. * (1. - t)) as u8]);
            }
            let fc = DVec2::new((w as f64 - 1.) / 2., (h as f64 - 1.) / 2.);
            for i in 0..(w.min(h) * UP / 3) {
                let q = fc + est.direction * (i as f64 / UP as f64);
                put(q.x, q.y, [255, 230, 0]);
            }
            for a in 0..64 {
                let t = a as f64 / 64. * std::f64::consts::TAU;
                for r in [1.5, 2.0] {
                    put(est.center_px.x + r * t.cos(), est.center_px.y + r * t.sin(), [0, 255, 0]);
                }
            }
        }
        im.save(out).unwrap();
    }

    #[test]
    #[ignore]
    fn ring_on_sxm() {
        let dir = std::env::var("RING_SXM_DIR").expect("set RING_SXM_DIR");
        let debug_dir = std::env::var("RING_DEBUG_DIR").ok().map(std::path::PathBuf::from);
        if let Some(d) = &debug_dir {
            std::fs::create_dir_all(d).unwrap();
        }
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("sxm")))
            .collect();
        paths.sort();
        let base = params_from_env();
        // Minimum ring radius is given in um and converted per file.
        let min_r_um: Option<f64> = std::env::var("RING_MIN_R_UM").ok().and_then(|v| v.parse().ok());
        println!("{base:?} min_r_um={min_r_um:?}");
        println!("file\tpx\tcompass\tdist_um\tradius_um\tcond\tinliers\tedges\tpeeled\tlateral\tverdict");
        for path in paths {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let (z, w, h, [rx, ry]) = match load_z(&path) {
                Ok(v) => v,
                Err(e) => {
                    println!("{name}\t{e}");
                    continue;
                }
            };
            let mut params = base;
            params.min_ring_radius_px = min_r_um.map(|r| r * 1e-6 / (rx as f64 / w as f64));
            let pre = preprocess(&z, w, h, &params);
            let res = estimate_preprocessed(&pre.img, pre.width, pre.height, pre.scale, &params);
            if let Some(d) = &debug_dir {
                write_debug(&d.join(format!("{name}.png")), &pre, &res);
            }
            // metres per analysed pixel
            let (sx, sy) = (rx as f64 / pre.width as f64, ry as f64 / pre.height as f64);
            let verdict = match decide(&res, pre.scale, &params) {
                Verdict::Arc { dir } => format!("ARC {:.0}", compass(dir, sx, sy)),
                Verdict::ArmEnd { dir } => format!("ARM_END {:.0}", compass(dir, sx, sy)),
                Verdict::Unsure(r) => format!("UNSURE ({})", r.describe()),
            };
            match res {
                Ok(Analysis { est: e, features, .. }) => {
                    let fc = DVec2::new((pre.width as f64 - 1.) / 2., (pre.height as f64 - 1.) / 2.);
                    let off = e.center_px - fc;
                    // At an arm end the move is sideways: the sign of the
                    // horizontal component of the ring direction.
                    let lateral = if e.direction.x < 0. { 270 } else { 90 };
                    println!(
                        "{name}\t{w}x{h}\t{:.0}\t{:.1}\t{:.1}\t{:.3}\t{:.2}\t{}\t{}\t{lateral}\t{verdict}",
                        compass(e.direction, sx, sy),
                        DVec2::new(off.x * sx, off.y * sy).length() * 1e6,
                        e.radius_px * sx * 1e6,
                        e.conditioning,
                        e.inlier_fraction,
                        e.n_edges,
                        features.len()
                    );
                }
                Err(err) => println!("{name}\t{w}x{h}\t{err:?}\t{verdict}"),
            }
        }
    }
}
