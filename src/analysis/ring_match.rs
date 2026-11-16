//! GDS-guided ring matching.
//!
//! The GDS says what a scan on the rings should look like: bands of known
//! width and spacing around a (roughly known) centre. This module uses that
//! as a matched filter:
//! - remove the background and clip blobs (as for the band direction),
//! - optionally flatten the scan to two tones (band / gap) at the level where
//!   the band fraction matches the GDS,
//! - take the median along circles around a candidate centre, giving a 1-D
//!   profile against radius. Only structure that runs along the expected
//!   circles survives; blobs and scan lines are a small part of any circle,
//!   so the median drops them,
//! - correlate that profile with the GDS bands, allowing the bands to be a
//!   bit wider or narrower than drawn and either raised or recessed,
//! - search the centre around the GDS position for the best match.
//!
//! The result is a refined ring centre for the frame, and a score that says
//! how well the scan looks like the expected rings at all.

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::analysis::gds_prior::{blur, detrend_clip};

/// Working resolution (pixels across) for matching.
const WORK_PX: usize = 96;
/// Coarse search grid half-size (steps each way along both axes).
const GRID: i32 = 8;
/// Score penalty at one search range from the GDS centre, so equally good
/// matches prefer the GDS position.
const PRIOR_PENALTY: f64 = 0.02;

/// The rings the GDS expects in a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Expected {
    /// Ring centre in frame pixels (+x right, +y down rows).
    pub center_px: DVec2,
    /// (inner, outer) radius of each band in frame pixels, innermost first.
    pub bands_px: Vec<(f64, f64)>,
}

impl Expected {
    fn narrowest(&self) -> f64 {
        self.bands_px.iter().map(|b| b.1 - b.0).fold(f64::INFINITY, f64::min)
    }
    /// Typical distance from one band to the next.
    fn pitch(&self) -> f64 {
        let mut d: Vec<f64> = self.bands_px.windows(2).map(|w| w[1].0 - w[0].0).collect();
        if d.is_empty() {
            return 2. * self.narrowest();
        }
        d.sort_by(f64::total_cmp);
        d[d.len() / 2]
    }
    fn scaled(&self, k: f64) -> Vec<(f64, f64)> {
        self.bands_px.iter().map(|b| (b.0 / k, b.1 / k)).collect()
    }
}

/// How the scan is flattened before matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatchMode {
    /// No matching: the plain GDS prediction.
    Off,
    /// Background removed and lightly blurred, heights kept.
    Smooth,
    /// Two tones with a soft step between them.
    SoftTwoTone,
    /// Two tones, hard threshold.
    HardTwoTone,
}

impl MatchMode {
    pub const ALL: [MatchMode; 4] =
        [MatchMode::Off, MatchMode::Smooth, MatchMode::SoftTwoTone, MatchMode::HardTwoTone];
    pub fn name(self) -> &'static str {
        match self {
            MatchMode::Off => "GDS only",
            MatchMode::Smooth => "smooth match",
            MatchMode::SoftTwoTone => "soft two-tone match",
            MatchMode::HardTwoTone => "hard two-tone match",
        }
    }
}

/// Whether the bands are higher or lower than the gaps in the scan. When
/// bands and gaps are about equally wide, raised bands look like recessed
/// ones shifted by a band width, so the polarity is best learned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Polarity {
    Auto,
    Raised,
    Recessed,
}

/// Matching settings (tuned by retraining).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MatchParams {
    pub mode: MatchMode,
    /// Soft two-tone step width, as a fraction of the interquartile range.
    pub softness: f64,
    /// Bands may be up to this fraction of the narrowest band wider than
    /// drawn (and a little narrower). Scanned rings usually come out wider
    /// than the GDS while the spacing matches.
    pub width_slack: f64,
    /// Below this correlation the match is not trusted and the plain GDS
    /// prediction is used.
    pub min_score: f64,
    pub polarity: Polarity,
    /// A frame's direction counts as sure when its side fits at least this
    /// much better than the opposite side (`RingMatch::dir_gap`). On the 42
    /// labelled frames, 0.1 was right 4 of 4 times; below it only about 2 in
    /// 3, so most frames need the check after a move.
    pub sure_gap: f64,
}

impl Default for MatchParams {
    fn default() -> Self {
        Self {
            mode: MatchMode::SoftTwoTone,
            softness: 0.25,
            width_slack: 0.6,
            min_score: 0.6,
            polarity: Polarity::Auto,
            sure_gap: 0.1,
        }
    }
}

impl MatchParams {
    /// Human-readable list of settings that differ from `other`.
    pub fn changes_from(&self, other: &MatchParams) -> Vec<String> {
        let mut out = Vec::new();
        if self.mode != other.mode {
            out.push(format!("GDS matching: {} \u{2192} {}", other.mode.name(), self.mode.name()));
        }
        let mut f = |name: &str, x: f64, y: f64| {
            if (x - y).abs() > 1e-9 {
                out.push(format!("{name} {x} \u{2192} {y}"));
            }
        };
        f("two-tone softness", other.softness, self.softness);
        f("band width slack", other.width_slack, self.width_slack);
        f("min match score", other.min_score, self.min_score);
        if self.polarity != other.polarity {
            out.push(format!("band polarity {:?} \u{2192} {:?}", other.polarity, self.polarity));
        }
        out
    }
}

/// Best match of a frame against the expected rings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingMatch {
    /// Refined ring centre, frame pixels.
    pub center_px: DVec2,
    /// Refined minus GDS centre, frame pixels.
    pub shift_px: DVec2,
    /// Bands are this much wider (negative: narrower) than drawn, pixels.
    pub width_bias_px: f64,
    /// Correlation of the radial profile with the GDS bands (−1..1).
    pub score: f64,
    /// Bands are higher than the gaps in the scan.
    pub raised: bool,
    /// Best score at least half a pitch away (radially) from the best match.
    pub second_score: f64,
    /// Best score with the centre on the other side of the frame (more
    /// than 90° away as seen from the frame centre). −1 when not searched.
    pub opposite_score: f64,
    /// Where that best opposite centre is, frame pixels.
    pub opposite_px: Option<DVec2>,
}

impl RingMatch {
    /// How much better the chosen side fits than the opposite side: the
    /// direction confidence. Near 0 the direction could be 180° off.
    pub fn dir_gap(&self) -> f64 {
        self.score - self.opposite_score.max(-1.)
    }
}

/// A frame flattened for matching, at the working resolution.
pub struct Prepared {
    pub w: usize,
    pub h: usize,
    /// Frame pixels per working pixel.
    pub k: usize,
    /// Value per working pixel, NaN where there is no data. Two-tone modes
    /// give 1 for band, 0 for gap.
    pub v: Vec<f64>,
}

fn to_work(p: DVec2, k: usize) -> DVec2 {
    (p - (k as f64 - 1.) / 2.) / k as f64
}

fn from_work(p: DVec2, k: usize) -> DVec2 {
    p * k as f64 + (k as f64 - 1.) / 2.
}

fn quantile(vals: &mut [f64], q: f64) -> f64 {
    let i = ((vals.len() - 1) as f64 * q.clamp(0., 1.)).round() as usize;
    *vals.select_nth_unstable_by(i, f64::total_cmp).1
}

/// How much of the frame should be band, for the two-tone threshold.
enum Frac<'a> {
    /// As the GDS has it at its nominal place.
    Nominal(&'a Expected),
    /// A fixed fraction (blind search: the place isn't known).
    Fixed(f64),
}

/// Area fraction of band within the outermost ring.
fn band_area_fraction(bands: &[(f64, f64)]) -> f64 {
    let outer = bands.last().map_or(1., |b| b.1);
    bands.iter().map(|b| b.1 * b.1 - b.0 * b.0).sum::<f64>() / (outer * outer)
}

fn prepare_with(
    data: &[f32],
    size: [u32; 2],
    narrowest_px: f64,
    frac: Frac,
    p: &MatchParams,
    raised: bool,
) -> Option<Prepared> {
    let (w, h) = (size[0] as usize, size[1] as usize);
    if w < 16 || h < 16 || data.len() != w * h {
        return None;
    }
    let res = detrend_clip(data, w, h)?;
    let k = w.max(h).div_ceil(WORK_PX).max(1);
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
    // Light blur (a fraction of the band width) against pixel noise,
    // normalised so gaps don't create edges.
    let sigma = (0.1 * narrowest_px / k as f64).clamp(0.5, 3.);
    let weighted: Vec<f64> = img.iter().zip(&mask).map(|(v, m)| v * m).collect();
    let bi = blur(&weighted, dw, dh, sigma);
    let bm = blur(&mask, dw, dh, sigma);
    let sign = if raised { 1. } else { -1. };
    let mut v: Vec<f64> = (0..dw * dh)
        .map(|i| if mask[i] > 0. && bm[i] > 1e-9 { sign * bi[i] / bm[i] } else { f64::NAN })
        .collect();

    if matches!(p.mode, MatchMode::SoftTwoTone | MatchMode::HardTwoTone) {
        // Threshold where the band fraction matches the GDS.
        let n = v.iter().filter(|x| x.is_finite()).count();
        if n < 16 {
            return None;
        }
        let f = match frac {
            Frac::Fixed(f) => f,
            Frac::Nominal(exp) => {
                let c = to_work(exp.center_px, k);
                let bands = exp.scaled(k as f64);
                let inside = v
                    .iter()
                    .enumerate()
                    .filter(|(i, x)| {
                        let r = (DVec2::new((i % dw) as f64, (i / dw) as f64) - c).length();
                        x.is_finite() && bands.iter().any(|b| r >= b.0 && r <= b.1)
                    })
                    .count();
                inside as f64 / n as f64
            }
        }
        .clamp(0.1, 0.9);
        let mut vals: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        let thr = quantile(&mut vals, 1. - f);
        let iqr = quantile(&mut vals, 0.75) - quantile(&mut vals, 0.25);
        let s = (p.softness * iqr).max(1e-12);
        for x in v.iter_mut().filter(|x| x.is_finite()) {
            *x = match p.mode {
                MatchMode::HardTwoTone => (*x > thr) as u8 as f64,
                _ => 0.5 + 0.5 * ((*x - thr) / s).tanh(),
            };
        }
    }
    Some(Prepared { w: dw, h: dh, k, v })
}

/// One flattened image, ready for profile scoring.
struct Image {
    pos: Vec<DVec2>,
    val: Vec<f64>,
    raised: bool,
    /// Heights kept (smooth): the other polarity is just the negative.
    symmetric: bool,
}

#[derive(Clone, Copy, Debug)]
struct Cand {
    off: DVec2,
    raw: f64,
    pen: f64,
    bias: f64,
    raised: bool,
}

struct Scorer {
    images: Vec<Image>,
    bands: Vec<(f64, f64)>,
    biases: Vec<f64>,
    r: Vec<f64>,
}

impl Scorer {
    /// Pixel-by-pixel correlation of the image with the rings drawn around
    /// `c`, best over images (polarity) and width biases. Unlike the radial
    /// profile this is sensitive to which way the bands curve. `stride`
    /// skips pixels for a faster coarse pass.
    fn score_2d(&mut self, c: DVec2, stride: usize) -> (f64, f64, bool) {
        let mut best = (-1., 0., true);
        let pos = &self.images[0].pos;
        self.r.clear();
        self.r.extend(pos.iter().step_by(stride).map(|p| (*p - c).length()));
        for img in &self.images {
            let vals: Vec<f64> = img.val.iter().step_by(stride).copied().collect();
            for &bias in &self.biases {
                let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0., 0., 0., 0., 0., 0.);
                for (&r, &x) in self.r.iter().zip(&vals) {
                    let y: f64 = self
                        .bands
                        .iter()
                        .map(|&(a, b)| ((r - (a - bias / 2.)).min((b + bias / 2.) - r) + 0.5).clamp(0., 1.))
                        .sum();
                    n += 1.;
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                }
                let cxx = sxx - sx * sx / n;
                let cyy = syy - sy * sy / n;
                if cxx <= 1e-12 || cyy <= 1e-12 {
                    continue;
                }
                let corr = (sxy - sx * sy / n) / (cxx * cyy).sqrt();
                let (corr, raised) = if img.symmetric && corr < 0. { (-corr, !img.raised) } else { (corr, img.raised) };
                if corr > best.0 {
                    best = (corr, bias, raised);
                }
            }
        }
        best
    }
}

/// Flattened images for scoring (one per polarity tried), and the working
/// size (w, h, k).
fn images_for(
    data: &[f32],
    size: [u32; 2],
    narrowest_px: f64,
    frac: Frac,
    p: &MatchParams,
) -> Option<(Vec<Image>, (usize, usize, usize))> {
    let polarities: &[bool] = match (p.mode, p.polarity) {
        (MatchMode::Off, _) => return None,
        (_, Polarity::Raised) => &[true],
        (_, Polarity::Recessed) => &[false],
        (MatchMode::Smooth, Polarity::Auto) => &[true],
        (_, Polarity::Auto) => &[true, false],
    };
    let mut images = Vec::new();
    let mut dims = (0, 0, 1);
    for &raised in polarities {
        let fr = match &frac {
            Frac::Nominal(e) => Frac::Nominal(*e),
            Frac::Fixed(f) => Frac::Fixed(*f),
        };
        let pr = prepare_with(data, size, narrowest_px, fr, p, raised)?;
        dims = (pr.w, pr.h, pr.k);
        let (pos, val) = pr
            .v
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (DVec2::new((i % pr.w) as f64, (i / pr.w) as f64), *v))
            .unzip();
        images.push(Image {
            pos,
            val,
            raised,
            symmetric: p.mode == MatchMode::Smooth && p.polarity == Polarity::Auto,
        });
    }
    Some((images, dims))
}

/// Band width offsets to try (working pixels). Scanned bands usually come
/// out wider than drawn, so the range is mostly on the wide side.
///
/// With the polarity unknown, bands may grow at most as wide as the gaps.
/// At equal widths raised and recessed look alike; that only moves the
/// centre by half a pitch, not the direction to it.
fn biases(p: &MatchParams, narrow: f64, pitch: f64) -> Vec<f64> {
    if p.width_slack > 0. {
        let mut hi = p.width_slack * narrow;
        if p.polarity == Polarity::Auto {
            hi = hi.min((pitch - 2. * narrow) / 2.).max(0.);
        }
        let lo = -NARROWER * narrow;
        (0..7).map(|i| lo + (hi - lo) * i as f64 / 6.).collect()
    } else {
        vec![0.]
    }
}

/// Bands may come out this fraction of their width narrower than drawn.
const NARROWER: f64 = 0.25;

/// Pattern search from `best` with starting step `step` (working pixels).
fn refine(mut best: Cand, step: f64, eval: &mut impl FnMut(DVec2) -> Cand) -> Cand {
    let mut s = step / 2.;
    while s >= 0.25 {
        let mut moved = false;
        for d in [
            DVec2::X,
            -DVec2::X,
            DVec2::Y,
            -DVec2::Y,
            DVec2::ONE,
            -DVec2::ONE,
            DVec2::new(1., -1.),
            DVec2::new(-1., 1.),
        ] {
            let c = eval(best.off + d * s);
            if c.pen > best.pen {
                best = c;
                moved = true;
            }
        }
        if !moved {
            s /= 2.;
        }
    }
    best
}

/// Match a frame against the expected rings. `None` when matching is off or
/// the frame can't be matched (too small, too little data).
pub fn match_rings(data: &[f32], size: [u32; 2], exp: &Expected, p: &MatchParams) -> Option<RingMatch> {
    if exp.bands_px.is_empty() {
        return None;
    }
    let (images, (w, h, k)) = images_for(data, size, exp.narrowest(), Frac::Nominal(exp), p)?;
    let kf = k as f64;
    let narrow = exp.narrowest() / kf;
    let pitch = exp.pitch() / kf;
    let mut sc = Scorer {
        images,
        bands: exp.scaled(kf),
        biases: biases(p, narrow, pitch),
        r: Vec::new(),
    };
    let c0 = to_work(exp.center_px, k);
    let fc = DVec2::new((w as f64 - 1.) / 2., (h as f64 - 1.) / 2.);
    let u = (c0 - fc).try_normalize().unwrap_or(DVec2::X);
    let t = u.perp();
    let range = pitch.max(3.);
    let step = range / GRID as f64;
    let mut eval = |off: DVec2| {
        let (raw, bias, raised) = sc.score_2d(c0 + off, 1);
        Cand { off, raw, pen: raw - PRIOR_PENALTY * (off.length() / range).powi(2), bias, raised }
    };

    let mut cands = Vec::new();
    for i in -GRID..=GRID {
        for j in -GRID..=GRID {
            cands.push(eval(u * (i as f64 * step) + t * (j as f64 * step)));
        }
    }
    let mut best = *cands.iter().max_by(|a, b| a.pen.total_cmp(&b.pen))?;
    let second = cands
        .iter()
        .filter(|c| (c.off - best.off).dot(u).abs() >= 0.5 * pitch)
        .map(|c| c.raw)
        .fold(-1., f64::max);
    best = refine(best, step, &mut eval);
    let center_px = from_work(c0 + best.off, k);
    Some(RingMatch {
        center_px,
        shift_px: center_px - exp.center_px,
        width_bias_px: best.bias * kf,
        score: best.raw,
        raised: best.raised,
        second_score: second,
        opposite_score: -1.,
        opposite_px: None,
    })
}

/// The flattened image the blind search scores, for inspection: frame
/// `bands` (pixels) give the blur and two-tone fraction. Working pixel
/// (x, y) covers frame pixels around `x * k + (k - 1) / 2`.
#[cfg(test)] // used by the tests only
pub fn blind_prepared(
    data: &[f32],
    size: [u32; 2],
    bands: &[(f64, f64)],
    p: &MatchParams,
    raised: bool,
) -> Option<Prepared> {
    let exp = Expected { center_px: DVec2::ZERO, bands_px: bands.to_vec() };
    prepare_with(data, size, exp.narrowest(), Frac::Fixed(band_area_fraction(bands)), p, raised)
}

/// Flatten a frame with the two-tone threshold set so that `band_fraction`
/// of it is band. `narrowest_px` (frame pixels) sets the blur.
pub fn prepare_fraction(
    data: &[f32],
    size: [u32; 2],
    narrowest_px: f64,
    band_fraction: f64,
    p: &MatchParams,
    raised: bool,
) -> Option<Prepared> {
    prepare_with(data, size, narrowest_px, Frac::Fixed(band_fraction), p, raised)
}

/// Coarse candidates refined in the blind search.
const BLIND_SEEDS: usize = 6;

/// Find the ring centre without knowing where the frame is on the GDS, only
/// what the ring systems look like: `kinds` holds the band radii (frame
/// pixels) of each kind of ring system. Every centre within reach of the
/// frame is tried; the best match over all kinds wins. Returns the kind and
/// the match (`shift_px` is from the frame centre).
pub fn blind_match(
    data: &[f32],
    size: [u32; 2],
    kinds: &[Vec<(f64, f64)>],
    p: &MatchParams,
) -> Option<(usize, RingMatch)> {
    // Each kind is searched on its own thread.
    let found: Vec<Option<RingMatch>> = std::thread::scope(|scope| {
        let handles: Vec<_> =
            kinds.iter().map(|bands| scope.spawn(move || blind_kind(data, size, bands, p))).collect();
        handles.into_iter().map(|h| h.join().ok().flatten()).collect()
    });
    found
        .into_iter()
        .enumerate()
        .filter_map(|(ki, m)| m.map(|m| (ki, m)))
        .max_by(|a, b| a.1.score.total_cmp(&b.1.score))
}

/// Coarse candidates refined on the opposite side.
const OPPOSITE_SEEDS: usize = 3;

/// Blind search for one kind of ring system.
fn blind_kind(data: &[f32], size: [u32; 2], bands: &[(f64, f64)], p: &MatchParams) -> Option<RingMatch> {
    if bands.is_empty() {
        return None;
    }
    let exp = Expected { center_px: DVec2::ZERO, bands_px: bands.to_vec() };
    let frac = Frac::Fixed(band_area_fraction(bands));
    let (images, (w, h, k)) = images_for(data, size, exp.narrowest(), frac, p)?;
    let kf = k as f64;
    let narrow = exp.narrowest() / kf;
    let pitch = exp.pitch() / kf;
    let fc = DVec2::new((w as f64 - 1.) / 2., (h as f64 - 1.) / 2.);
    let reach = bands.last().unwrap().1 / kf + 0.5 * DVec2::new(w as f64, h as f64).length();
    let step = (narrow / 2.).max(1.);
    let mut sc = Scorer {
        images,
        bands: exp.scaled(kf),
        biases: vec![0.],
        r: Vec::new(),
    };
    let eval = |sc: &mut Scorer, off: DVec2, stride: usize| {
        let (raw, bias, raised) = sc.score_2d(fc + off, stride);
        Cand { off, raw, pen: raw, bias, raised }
    };
    let n = (reach / step).ceil() as i32;
    let mut coarse = Vec::new();
    for i in -n..=n {
        for j in -n..=n {
            let off = DVec2::new(i as f64, j as f64) * step;
            if off.length() <= reach {
                coarse.push(eval(&mut sc, off, 3));
            }
        }
    }
    coarse.sort_by(|a, b| b.raw.total_cmp(&a.raw));
    // The best separated coarse candidates that `keep` allows.
    let seeds_from = |keep: &dyn Fn(&Cand) -> bool, max: usize| {
        let mut seeds: Vec<Cand> = Vec::new();
        for c in coarse.iter().filter(|c| keep(c)) {
            if seeds.len() >= max {
                break;
            }
            if seeds.iter().all(|s| (s.off - c.off).length() > 2. * step) {
                seeds.push(*c);
            }
        }
        seeds
    };
    let seeds = seeds_from(&|_| true, BLIND_SEEDS);
    sc.biases = biases(p, narrow, pitch);
    let refined: Vec<Cand> =
        seeds.into_iter().map(|s| refine(s, step, &mut |off| eval(&mut sc, off, 1))).collect();
    let best = refined.iter().copied().max_by(|a, b| a.raw.total_cmp(&b.raw))?;
    let second = refined
        .iter()
        .filter(|c| (c.off - best.off).length() > pitch / 2.)
        .map(|c| c.raw)
        .fold(-1., f64::max);
    // The best fit with the centre on the other side, refined the same way:
    // the direction confidence compares the two.
    let other_side = |c: &Cand| c.off.dot(best.off) < 0.;
    let mut opposite: Option<Cand> = None;
    let mut consider = |c: Cand| {
        // Refining may wander back across; only the other side counts.
        if other_side(&c) && opposite.is_none_or(|o| c.raw > o.raw) {
            opposite = Some(c);
        }
    };
    refined.iter().copied().for_each(&mut consider);
    for s in seeds_from(&other_side, OPPOSITE_SEEDS) {
        consider(refine(s, step, &mut |off| eval(&mut sc, off, 1)));
    }
    let center_px = from_work(fc + best.off, k);
    Some(RingMatch {
        center_px,
        shift_px: center_px - from_work(fc, k),
        width_bias_px: best.bias * kf,
        score: best.raw,
        raised: best.raised,
        second_score: second,
        opposite_score: opposite.map_or(-1., |o| o.raw),
        opposite_px: opposite.map(|o| from_work(fc + o.off, k)),
    })
}

/// How well the frame fits rings of `bands` (frame pixels) centred at
/// `center_px` (frame pixels), best over polarity and band widths. Used to
/// check one frame's centre in another frame.
pub fn score_at(
    data: &[f32],
    size: [u32; 2],
    bands: &[(f64, f64)],
    center_px: DVec2,
    p: &MatchParams,
) -> Option<f64> {
    if bands.is_empty() {
        return None;
    }
    let exp = Expected { center_px, bands_px: bands.to_vec() };
    let frac = Frac::Fixed(band_area_fraction(bands));
    let (images, (_, _, k)) = images_for(data, size, exp.narrowest(), frac, p)?;
    let kf = k as f64;
    let mut sc = Scorer {
        images,
        bands: exp.scaled(kf),
        biases: biases(p, exp.narrowest() / kf, exp.pitch() / kf),
        r: Vec::new(),
    };
    Some(sc.score_2d(to_work(center_px, k), 1).0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Simple LCG for repeatable noise.
    pub(crate) struct Lcg(pub u64);
    impl Lcg {
        pub fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Bands of 30 px every 75 px from radius 30, like the device rings
    /// (1 µm bands, 2.5 µm pitch).
    pub(crate) fn device_bands() -> Vec<(f64, f64)> {
        (0..8).map(|i| (30. + 75. * i as f64, 60. + 75. * i as f64)).collect()
    }

    /// A synthetic scan: rings around `center` (bands widened by `widen`),
    /// height `amp` (negative: recessed), with noise, a tilt and blobs.
    pub(crate) fn scan(w: usize, center: DVec2, bands: &[(f64, f64)], widen: f64, amp: f64, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let blobs: Vec<DVec2> = (0..6).map(|_| DVec2::new(rng.next() * w as f64, rng.next() * w as f64)).collect();
        (0..w * w)
            .map(|i| {
                let p = DVec2::new((i % w) as f64, (i / w) as f64);
                let r = (p - center).length();
                let on = bands.iter().any(|b| r >= b.0 - widen / 2. && r <= b.1 + widen / 2.);
                let mut v = if on { amp } else { 0. } + 0.01 * p.x - 0.005 * p.y + 0.4 * (rng.next() - 0.5);
                for b in &blobs {
                    let d = (p - *b).length();
                    if d < 6. {
                        v += 4.;
                    }
                }
                v as f32
            })
            .collect()
    }

    fn params(mode: MatchMode) -> MatchParams {
        MatchParams { mode, ..MatchParams::default() }
    }

    #[test]
    fn recovers_a_shift_in_every_mode_and_polarity() {
        let w = 192;
        let truth = DVec2::new(-150., 260.);
        let gds = truth + DVec2::new(10., -12.);
        let exp = Expected { center_px: gds, bands_px: device_bands() };
        for amp in [1., -1.] {
            let data = scan(w, truth, &device_bands(), 0., amp, 7);
            for mode in [MatchMode::Smooth, MatchMode::SoftTwoTone, MatchMode::HardTwoTone] {
                let m = match_rings(&data, [w as u32, w as u32], &exp, &params(mode)).unwrap();
                let radial = (gds - DVec2::splat(95.5)).normalize();
                let err = (m.center_px - truth).dot(radial);
                // A hard threshold rounds edges to whole pixels: less precise.
                let tol = if mode == MatchMode::HardTwoTone { 4. } else { 2.5 };
                assert!(err.abs() < tol, "{mode:?} amp {amp}: {m:?} err {err}");
                assert!(m.score > 0.6, "{mode:?} amp {amp}: {m:?}");
                assert_eq!(m.raised, amp > 0., "{mode:?}");
            }
        }
    }

    #[test]
    fn tolerates_wider_bands() {
        let w = 192;
        let truth = DVec2::new(96., -200.);
        let exp = Expected { center_px: truth + DVec2::new(0., 8.), bands_px: device_bands() };
        let data = scan(w, truth, &device_bands(), 8., 1., 3);
        let p = params(MatchMode::SoftTwoTone);
        let m = match_rings(&data, [w as u32, w as u32], &exp, &p).unwrap();
        assert!((m.center_px - truth).y.abs() < 2.5, "{m:?}");
        assert!(m.width_bias_px > 2., "{m:?}");
        // Without slack it still finds the place, with a lower score.
        let strict = match_rings(&data, [w as u32, w as u32], &exp, &MatchParams { width_slack: 0., ..p }).unwrap();
        assert!(strict.score < m.score);
    }

    #[test]
    fn centre_in_frame_is_found() {
        let w = 192;
        let truth = DVec2::new(100., 90.);
        let exp = Expected { center_px: DVec2::new(108., 84.), bands_px: device_bands() };
        let data = scan(w, truth, &device_bands(), 0., 1., 11);
        let m = match_rings(&data, [w as u32, w as u32], &exp, &params(MatchMode::HardTwoTone)).unwrap();
        assert!((m.center_px - truth).length() < 4., "{m:?}");
    }

    #[test]
    fn equal_bands_and_gaps_need_a_known_polarity() {
        // 30 px bands every 60 px: raised looks like recessed shifted by 30 px.
        let bands: Vec<(f64, f64)> = (0..8).map(|i| (30. + 60. * i as f64, 60. + 60. * i as f64)).collect();
        let w = 192;
        let truth = DVec2::new(-150., 260.);
        let exp = Expected { center_px: truth + DVec2::new(6., -6.), bands_px: bands.clone() };
        let data = scan(w, truth, &bands, 0., 1., 7);
        let p = MatchParams { polarity: Polarity::Raised, ..params(MatchMode::HardTwoTone) };
        let m = match_rings(&data, [w as u32, w as u32], &exp, &p).unwrap();
        let radial = (exp.center_px - DVec2::splat(95.5)).normalize();
        assert!((m.center_px - truth).dot(radial).abs() < 2.5, "{m:?}");
    }

    #[test]
    fn blind_search_finds_the_centre_without_a_placement() {
        let w = 192;
        // A second, different kind of ring system that must not win.
        let other: Vec<(f64, f64)> = (0..6).map(|i| (20. + 100. * i as f64, 50. + 100. * i as f64)).collect();
        let kinds = vec![other, device_bands()];
        for (k, truth) in [DVec2::new(-150., 260.), DVec2::new(300., 40.), DVec2::new(100., 90.)].into_iter().enumerate() {
            // Scanned bands 40 % wider than drawn, same spacing.
            let data = scan(w, truth, &device_bands(), 12., 1., k as u64 + 20);
            let t = std::time::Instant::now();
            let (kind, m) = blind_match(&data, [w as u32, w as u32], &kinds, &params(MatchMode::SoftTwoTone)).unwrap();
            let fc = DVec2::splat(95.5);
            let cos = (m.center_px - fc).normalize().dot((truth - fc).normalize());
            println!("{truth:?}: {m:?} kind {kind} cos {cos:.3} in {:?}", t.elapsed());
            assert_eq!(kind, 1, "{m:?}");
            assert!(cos > 30f64.to_radians().cos(), "{truth:?}: {m:?}");
            assert!(m.dir_gap() > 0.05, "{m:?}");
            // The matched centre scores well; its mirror across the frame
            // centre (the 180° mistake) clearly worse.
            let p = params(MatchMode::SoftTwoTone);
            let at = score_at(&data, [w as u32, w as u32], &device_bands(), m.center_px, &p).unwrap();
            let mirror = score_at(&data, [w as u32, w as u32], &device_bands(), 2. * fc - m.center_px, &p).unwrap();
            assert!(at > mirror + 0.1, "{at} vs {mirror}");
        }
    }

    #[test]
    fn noise_scores_low() {
        let w = 192;
        let mut rng = Lcg(5);
        let data: Vec<f32> = (0..w * w).map(|_| rng.next() as f32).collect();
        let exp = Expected { center_px: DVec2::new(-150., 260.), bands_px: device_bands() };
        for mode in [MatchMode::Smooth, MatchMode::SoftTwoTone, MatchMode::HardTwoTone] {
            let m = match_rings(&data, [w as u32, w as u32], &exp, &params(mode)).unwrap();
            // Many candidate centres: noise can reach ~0.5 by chance.
            assert!(m.score < MatchParams::default().min_score, "{mode:?}: {m:?}");
        }
        assert!(match_rings(&data, [w as u32, w as u32], &exp, &params(MatchMode::Off)).is_none());
    }
}
