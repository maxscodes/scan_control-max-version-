//! Lining up the tiles of "Polish images" (positions only).
//!
//! Every pair of tiles that overlaps gets a correction: how much tile `b`
//! must move relative to tile `a` so their shared strip matches. It comes
//! from image registration (automatic) or from the user lining the two up
//! by eye. All pairs are then fitted together (least squares), so each
//! tile gets one move. Hand-checked pairs count far more than automatic
//! ones, and automatic ones that disagree with the rest are dropped.
//!
//! Survey areas often repeat (lattices, arrays of rings), where a wrong
//! offset can match almost as well as the right one. The automatic match
//! therefore only searches a small range around where the map puts the
//! tile, and refuses a match that isn't clearly better than the runner-up.

use glam::{DAffine2, DVec2};
use nalgebra::DMatrix;

use crate::analysis::{
    gds_prior::{self, GdsRings},
    polish::Source,
    register::{shift_candidates, GOOD_SCORE},
    ring_match::{match_rings, MatchParams},
};

/// Smallest shared area (share of a tile) for a pair to be checked.
const MIN_SHARE: f64 = 0.02;
/// The best match must beat the runner-up by this much (correlation).
const MARGIN: f64 = 0.04;
/// Weight of a pair checked by eye against an automatic one.
const HAND_WEIGHT: f64 = 50.;

/// Why an automatic match was not used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// Different frame size or rotation: can't be compared directly.
    Different,
    /// Not enough in common.
    NoOverlap,
    Weak,
    /// Another offset matched about as well.
    LookAlike,
    /// Disagreed with the other pairs.
    Outlier,
}
impl Refused {
    pub fn label(self) -> &'static str {
        match self {
            Refused::Different => "different frame",
            Refused::NoOverlap => "too little overlap",
            Refused::Weak => "weak match",
            Refused::LookAlike => "look-alike",
            Refused::Outlier => "disagrees with the rest",
        }
    }
}

/// One pair of overlapping tiles (indices into the sources).
#[derive(Debug, Clone)]
pub struct Pair {
    pub a: usize,
    pub b: usize,
    /// Automatic correction (move of b relative to a, world nm) and score.
    pub auto: Option<(DVec2, f64)>,
    pub refused: Option<Refused>,
    /// Correction set by eye; overrides the automatic one.
    pub hand: Option<DVec2>,
}
impl Pair {
    /// The correction in use, if any.
    pub fn correction(&self) -> Option<DVec2> {
        self.hand.or_else(|| if self.refused.is_none() { self.auto.map(|a| a.0) } else { None })
    }
}

fn bbox(t: &DAffine2) -> (DVec2, DVec2) {
    let c = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].map(|(x, y)| t.transform_point2(DVec2::new(x, y)));
    (c.iter().fold(DVec2::INFINITY, |a, b| a.min(*b)), c.iter().fold(DVec2::NEG_INFINITY, |a, b| a.max(*b)))
}

/// Shared area of two tiles (bounding boxes), share of the smaller.
pub fn shared(a: &DAffine2, b: &DAffine2) -> f64 {
    let ((la, ha), (lb, hb)) = (bbox(a), bbox(b));
    let ov = (ha.min(hb) - la.max(lb)).max(DVec2::ZERO);
    let area = |l: DVec2, h: DVec2| ((h - l).x * (h - l).y).max(1e-12);
    ov.x * ov.y / area(la, ha).min(area(lb, hb))
}

/// Middle of the area two tiles share (bounding boxes).
pub fn shared_center(a: &DAffine2, b: &DAffine2) -> DVec2 {
    let ((la, ha), (lb, hb)) = (bbox(a), bbox(b));
    0.5 * (la.max(lb) + ha.min(hb))
}

/// Pairs of tiles that overlap on the map, in reading order (top row first,
/// left to right) so checking them sweeps across the survey.
pub fn find_pairs(sources: &[Source]) -> Vec<Pair> {
    let mut out = Vec::new();
    for a in 0..sources.len() {
        for b in a + 1..sources.len() {
            if shared(&sources[a].transform, &sources[b].transform) >= MIN_SHARE {
                out.push(Pair { a, b, auto: None, refused: None, hand: None });
            }
        }
    }
    let key = |p: &Pair| {
        let c = shared_center(&sources[p.a].transform, &sources[p.b].transform);
        (-(c.y / 50.).round() as i64, (c.x * 1000.) as i64)
    };
    out.sort_by_key(key);
    out
}

/// Automatic correction for `b` relative to `a`: registration of the two
/// scans within `radius` (share of a tile) of where the map puts them.
pub fn measure(a: &Source, b: &Source, radius: f64) -> Result<(DVec2, f64), Refused> {
    let (ma, mb) = (a.transform.matrix2, b.transform.matrix2);
    let scale = ma.x_axis.length().max(ma.y_axis.length());
    let [w, h] = a.forward.size;
    if b.forward.size != [w, h]
        || w < 8
        || h < 8
        || (ma.x_axis - mb.x_axis).length() > 1e-3 * scale
        || (ma.y_axis - mb.y_axis).length() > 1e-3 * scale
    {
        return Err(Refused::Different);
    }
    let (wf, hf) = (w as f64, h as f64);
    let delta = b.transform.translation - a.transform.translation;
    // Where the map has b's pixels in a's pixels: a(x + o) = b(x).
    let local = ma.inverse() * delta;
    let o = DVec2::new(local.x * wf, local.y * hf);
    let found = shift_candidates(
        &a.forward.data,
        &b.forward.data,
        w as usize,
        h as usize,
        (-o.x, -o.y),
        radius.clamp(0.005, 0.5) * wf.max(hf),
        MIN_SHARE,
        2,
    );
    let Some(best) = found.first() else { return Err(Refused::NoOverlap) };
    if best.score < GOOD_SCORE {
        return Err(Refused::Weak);
    }
    if found.get(1).is_some_and(|s| best.score - s.score < MARGIN) {
        return Err(Refused::LookAlike);
    }
    // image_shift convention: b(x) = a(x − d).
    let true_local = DVec2::new(-best.dx / wf, -best.dy / hf);
    Ok((ma * true_local - delta, best.score))
}

/// Where the GDS says a tile's rings belong. Moving the tile by `want`
/// (world nm, from where the map has it) puts the rings it shows on the GDS
/// rings. Away from the ring centre only the part along `dir` (towards the
/// centre) is known: along an arc a tile can slide without the rings
/// showing it, so that is left to the overlaps with its neighbours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RingAnchor {
    pub want: DVec2,
    /// `None`: the ring centre is in the tile, so both directions count.
    pub dir: Option<DVec2>,
    /// Match score (correlation).
    pub score: f64,
}

/// Why a tile got no [`RingAnchor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingMiss {
    /// No GDS with rings covers the tile.
    NoGds,
    /// Not enough data to match.
    NoData,
    /// The tile doesn't look enough like the rings.
    Weak,
    /// A ring further in or out fits about as well.
    Unsure,
}
impl RingMiss {
    pub const ALL: [RingMiss; 4] = [RingMiss::NoGds, RingMiss::NoData, RingMiss::Weak, RingMiss::Unsure];
    pub fn label(self) -> &'static str {
        match self {
            RingMiss::NoGds => "not on the GDS rings",
            RingMiss::NoData => "no data",
            RingMiss::Weak => "rings unclear",
            RingMiss::Unsure => "ring look-alike",
        }
    }
}

/// The best ring fit must beat a fit one ring further in or out by this much.
const RING_MARGIN: f64 = 0.03;
/// Weight of a ring anchor (times its score) against an automatic pair.
const RING_WEIGHT: f64 = 1.;

/// Match a tile (`data`, `w`x`h`, row 0 at local y -0.5, placed by
/// `transform`) against the rings the GDS expects there.
pub fn ring_anchor(data: &[f32], w: usize, h: usize, transform: DAffine2, gds: &[GdsRings]) -> Result<RingAnchor, RingMiss> {
    let size = [w as u32, h as u32];
    if w < 16 || h < 16 {
        return Err(RingMiss::NoData);
    }
    let exp = gds_prior::expected(gds, transform, size).ok_or(RingMiss::NoGds)?;
    let params = MatchParams::default();
    let m = match_rings(data, size, &exp, &params).ok_or(RingMiss::NoData)?;
    if m.score < params.min_score {
        return Err(RingMiss::Weak);
    }
    if m.score - m.second_score < RING_MARGIN {
        return Err(RingMiss::Unsure);
    }
    let to_world = |p: DVec2| transform.transform_point2(DVec2::new((p.x + 0.5) / w as f64 - 0.5, (p.y + 0.5) / h as f64 - 0.5));
    let (gds_c, found) = (to_world(exp.center_px), to_world(m.center_px));
    let to_c = gds_c - transform.translation;
    let half_diag = (transform.matrix2 * DVec2::new(0.5, 0.5)).length().max((transform.matrix2 * DVec2::new(0.5, -0.5)).length());
    let dir = if to_c.length() <= half_diag { None } else { to_c.try_normalize() };
    Ok(RingAnchor { want: gds_c - found, dir, score: m.score })
}

/// One move per tile (world nm) that best agrees with the pairs' corrections
/// and the ring anchors (`active[i]`: tile in use). Automatic pairs far off
/// the fit are marked [`Refused::Outlier`] and left out. A light pull
/// towards no move settles what nothing else does (lone tiles, and without
/// anchors the whole survey's position).
pub fn solve(n: usize, pairs: &mut [Pair], active: &[bool], anchors: &[Option<RingAnchor>], px: f64) -> Vec<DVec2> {
    for p in pairs.iter_mut() {
        if p.refused == Some(Refused::Outlier) {
            p.refused = None;
        }
    }
    let mut x = vec![DVec2::ZERO; n];
    for _round in 0..4 {
        // Unknowns: x of tile i at 2i, y at 2i + 1.
        let mut ata = DMatrix::<f64>::zeros(2 * n, 2 * n);
        let mut atb = DMatrix::<f64>::zeros(2 * n, 1);
        for i in 0..2 * n {
            ata[(i, i)] += 1e-3;
        }
        for p in pairs.iter() {
            if !(active[p.a] && active[p.b]) {
                continue;
            }
            let Some(d) = p.correction() else { continue };
            let w = if p.hand.is_some() { HAND_WEIGHT } else { p.auto.map_or(1., |a| a.1.max(0.1)) };
            // x_b - x_a = d
            for (c, v) in [(0, d.x), (1, d.y)] {
                let (a, b) = (2 * p.a + c, 2 * p.b + c);
                ata[(a, a)] += w;
                ata[(b, b)] += w;
                ata[(a, b)] -= w;
                ata[(b, a)] -= w;
                atb[(b, 0)] += w * v;
                atb[(a, 0)] -= w * v;
            }
        }
        for (i, a) in anchors.iter().enumerate().take(n) {
            let Some(a) = a.filter(|_| active[i]) else { continue };
            let w = RING_WEIGHT * a.score.max(0.1);
            let rows: Vec<DVec2> = match a.dir {
                Some(d) => vec![d],
                None => vec![DVec2::X, DVec2::Y],
            };
            // row . x_i = row . want
            for r in rows {
                let t = r.dot(a.want);
                let rv = [r.x, r.y];
                for j in 0..2 {
                    for k in 0..2 {
                        ata[(2 * i + j, 2 * i + k)] += w * rv[j] * rv[k];
                    }
                    atb[(2 * i + j, 0)] += w * rv[j] * t;
                }
            }
        }
        let Some(ch) = ata.cholesky() else { break };
        let sol = ch.solve(&atb);
        x = (0..n).map(|i| DVec2::new(sol[(2 * i, 0)], sol[(2 * i + 1, 0)])).collect();
        // Drop automatic pairs that disagree clearly with the rest.
        let resid: Vec<(usize, f64)> = pairs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.hand.is_none() && active[p.a] && active[p.b])
            .filter_map(|(k, p)| p.correction().map(|d| (k, ((x[p.b] - x[p.a]) - d).length())))
            .collect();
        if resid.is_empty() {
            break;
        }
        let mut r: Vec<f64> = resid.iter().map(|r| r.1).collect();
        r.sort_by(f64::total_cmp);
        let limit = (3. * r[r.len() / 2]).max(2. * px);
        let mut dropped = false;
        for (k, e) in resid {
            if e > limit {
                pairs[k].refused = Some(Refused::Outlier);
                dropped = true;
            }
        }
        if !dropped {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::connection::FrameData;

    /// A tile cut from a shared random surface; `true_at` is where it really
    /// is, `map_at` where the map puts it.
    fn tile(surface: &dyn Fn(f64, f64) -> f32, true_at: DVec2, map_at: DVec2) -> Source {
        let n = 64;
        let px = 100. / n as f64;
        let data: Vec<f32> = (0..n * n)
            .map(|k| {
                let (i, j) = ((k % n) as f64, (k / n) as f64);
                // Row 0 at local y −0.5, local y up = world y (scale +).
                surface(true_at.x - 50. + (i + 0.5) * px, true_at.y - 50. + (j + 0.5) * px)
            })
            .collect();
        let fd = FrameData { size: [n as u32, n as u32], data: Arc::new(data.into_boxed_slice()) };
        Source {
            id: uuid::Uuid::new_v4(),
            name: String::new(),
            transform: DAffine2::from_scale_angle_translation(DVec2::new(100., 100.), 0., map_at),
            forward: fd.clone(),
            backward: fd,
        }
    }

    fn surface(x: f64, y: f64) -> f32 {
        // Non-repeating bumps.
        let mut v = 0.;
        for k in 0..12 {
            let (cx, cy) = ((k * 37 % 230) as f64 - 40., (k * 53 % 190) as f64 - 40.);
            let r2 = (x - cx).powi(2) + (y - cy).powi(2);
            v += (-r2 / (60. + 10. * k as f64)).exp();
        }
        v as f32
    }

    #[test]
    fn finds_and_fits_the_true_offsets() {
        // Three tiles in a row, 80 nm apart really; the map has them off.
        let true_at = [DVec2::ZERO, DVec2::new(80., 0.), DVec2::new(160., 0.)];
        let map_at = [DVec2::ZERO, DVec2::new(84., 3.), DVec2::new(158., -2.)];
        let srcs: Vec<Source> = (0..3).map(|i| tile(&surface, true_at[i], map_at[i])).collect();
        let mut pairs = find_pairs(&srcs);
        assert!(pairs.len() >= 2);
        for p in pairs.iter_mut() {
            match measure(&srcs[p.a], &srcs[p.b], 0.15) {
                Ok(m) => p.auto = Some(m),
                Err(r) => p.refused = Some(r),
            }
        }
        let x = solve(3, &mut pairs, &[true; 3], &[None; 3], 100. / 64.);
        for i in 0..3 {
            let got = map_at[i] + x[i] - (map_at[0] + x[0]);
            let want = true_at[i] - true_at[0];
            assert!((got - want).length() < 1.0, "tile {i}: {got} vs {want}");
        }
    }

    #[test]
    fn hand_check_outweighs_automatic() {
        let mut pairs = vec![Pair { a: 0, b: 1, auto: Some((DVec2::new(5., 0.), 0.9)), refused: None, hand: Some(DVec2::new(-2., 1.)) }];
        let x = solve(2, &mut pairs, &[true; 2], &[None; 2], 1.);
        assert!(((x[1] - x[0]) - DVec2::new(-2., 1.)).length() < 0.01);
    }

    #[test]
    fn ring_anchor_moves_a_tile_onto_the_gds_rings() {
        // Rings around the origin, bands 600 nm wide every 1200 nm. A 4 µm
        // tile really at (3000, 2000) is put 250 nm too far out by the map.
        let bands: Vec<(f64, f64)> = (1..8).map(|k| (k as f64 * 1200., k as f64 * 1200. + 600.)).collect();
        let gds = vec![GdsRings {
            uuid: uuid::Uuid::new_v4(),
            transform: DAffine2::IDENTITY,
            systems: Arc::new(vec![gds_prior::RingSystem { center: DVec2::ZERO, bands: bands.clone(), covered: u64::MAX, layer: 0 }]),
            landing: None,
        }];
        let n = 128;
        let true_at = DVec2::new(3000., 2000.);
        let off = true_at.normalize() * 250.;
        let side = 4000.;
        let map = DAffine2::from_scale_angle_translation(DVec2::splat(side), 0., true_at + off);
        let data: Vec<f32> = (0..n * n)
            .map(|k| {
                let (i, j) = ((k % n) as f64, (k / n) as f64);
                // Row 0 at local y −0.5 (as the matcher expects).
                let w = true_at + DVec2::new((i + 0.5) / n as f64 - 0.5, (j + 0.5) / n as f64 - 0.5) * side;
                let r = w.length();
                if bands.iter().any(|b| r >= b.0 && r <= b.1) { 1. } else { 0. }
            })
            .collect();
        let a = ring_anchor(&data, n, n, map, &gds).expect("rings found");
        let d = a.dir.expect("centre is outside the tile");
        assert!((d.dot(a.want) - d.dot(-off)).abs() < 60., "radial move {} vs {}", d.dot(a.want), d.dot(-off));
    }

    #[test]
    fn ring_anchor_pulls_only_towards_the_centre() {
        // Two tiles agreeing they are 10 nm apart in x; tile 0 is told by the
        // rings (centre far off along +y) to move 4 nm in y, sideways ignored.
        let mut pairs = vec![Pair { a: 0, b: 1, auto: Some((DVec2::new(10., 0.), 0.9)), refused: None, hand: None }];
        let anchors = [Some(RingAnchor { want: DVec2::new(7., 4.), dir: Some(DVec2::Y), score: 0.9 }), None];
        let x = solve(2, &mut pairs, &[true; 2], &anchors, 1.);
        assert!((x[0].y - 4.).abs() < 0.1, "{:?}", x);
        assert!(((x[1] - x[0]) - DVec2::new(10., 0.)).length() < 0.1, "{:?}", x);
        assert!(x[0].x.abs() < 6., "sideways mostly left alone: {:?}", x);
    }
}
