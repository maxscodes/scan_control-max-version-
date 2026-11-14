//! Where has the tip landed on the GDS?
//!
//! The first frame can be anywhere over the ring structures, and the GDS
//! placement on the map is only a rough guess. This module finds the places
//! on the GDS where the frame would look like it does:
//! - the GDS is split into structures (a full ring system, or two half
//!   systems with bars between them). Each different kind of structure is
//!   drawn once, as a map of the distance to the nearest band edge,
//! - the flattened frame (two tones, as for ring matching) is slid over each
//!   map. Every place where the bands line up is a landing hypothesis. The
//!   drawn map has the band spacing and curvature, and also where arcs end,
//!   the bars, and how far out the rings go, so all of these count,
//! - one frame often fits several places (the mirror image across the ring
//!   centre, one ring further in or out, the other kind of structure). After
//!   a move, each hypothesis predicts where the new frame is on the GDS;
//!   checking the new frame there settles which one is right ([`Track`]).
//!
//! Positions are GDS local nm (before the GDS transform), relative to the
//! structure's reference point (the mean of its ring centres). The frame's
//! rotation and scale relative to the GDS come from the map; only where it
//! is on the GDS is unknown.

use std::{collections::BTreeMap, sync::Arc};

use glam::{DAffine2, DMat2, DVec2, Vec2};

use crate::analysis::{
    gds_prior::RingSystem,
    ring_live::Shot,
    ring_match::{prepare_fraction, MatchMode, MatchParams, Polarity},
};

/// Map resolution, local nm per pixel.
const MAP_NM: f64 = 125.;
/// Maps reach this many times the largest detected ring radius from the
/// structure's reference point (the GDS has more rings than are detected as
/// ring systems).
const MAP_REACH: f64 = 3.;
/// Coarse search points across the frame, per axis.
const COARSE_PTS: usize = 24;
/// Coarse candidates refined per kind of structure.
const SEEDS_PER_KIND: usize = 10;
/// Hypotheses kept.
const MAX_HYPS: usize = 16;
/// Band widenings tried when refining.
const FINE_BIASES: usize = 9;
/// Frames a hypothesis is scored on (the latest ones).
const MAX_SHOTS: usize = 4;
/// Coarse moves end within this fraction of their length of where they
/// were meant to, plus `MOVE_SLACK_NM` (local nm).
const MOVE_ERROR: f64 = 0.3;
const MOVE_SLACK_NM: f64 = 600.;
/// Score difference that makes a hypothesis e (2.7) times less likely, for
/// the rough odds shown to the user.
const ODDS_SCALE: f64 = 0.04;
/// Outside the map: far from any band.
const FAR: f32 = 1e6;

// ---------------------------------------------------------------------------
// Structure maps
// ---------------------------------------------------------------------------

/// Signed distance (local nm) to the nearest band edge, negative inside
/// bands, on a square grid centred on the structure's reference point.
pub struct Map {
    half: f64,
    n: usize,
    d: Vec<f32>,
}

impl Map {
    /// Continuous pixel coordinates of a local position.
    fn px(&self, p: DVec2) -> DVec2 {
        (p + self.half) / MAP_NM - 0.5
    }
    fn nearest(&self, p: DVec2) -> f32 {
        let q = self.px(p).round();
        if q.x < 0. || q.y < 0. || q.x >= self.n as f64 || q.y >= self.n as f64 {
            return FAR;
        }
        self.d[q.y as usize * self.n + q.x as usize]
    }
    fn bilinear(&self, p: DVec2) -> f32 {
        let q = self.px(p);
        let (x0, y0) = (q.x.floor(), q.y.floor());
        if x0 < 0. || y0 < 0. || x0 + 1. >= self.n as f64 || y0 + 1. >= self.n as f64 {
            return FAR;
        }
        let (fx, fy) = ((q.x - x0) as f32, (q.y - y0) as f32);
        let i = y0 as usize * self.n + x0 as usize;
        let (a, b, c, d) = (self.d[i], self.d[i + 1], self.d[i + self.n], self.d[i + self.n + 1]);
        (a * (1. - fx) + b * fx) * (1. - fy) + (c * (1. - fx) + d * fx) * fy
    }
    #[cfg(test)]
    fn inside(&self, p: DVec2) -> bool {
        self.nearest(p) < 0.
    }
}

/// Fill polygons (local nm, relative to the map centre) on a grid, even-odd
/// per polygon.
fn rasterize(polys: &[Vec<DVec2>], half: f64) -> (usize, Vec<bool>) {
    let n = (2. * half / MAP_NM).ceil() as usize;
    let mut inside = vec![false; n * n];
    let mut xs: Vec<f64> = Vec::new();
    for poly in polys {
        let (lo, hi) = poly.iter().fold((DVec2::INFINITY, DVec2::NEG_INFINITY), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        let row = |y: f64| (y + half) / MAP_NM - 0.5;
        let r0 = row(lo.y).ceil().max(0.) as usize;
        let r1 = (row(hi.y).floor() as i64).min(n as i64 - 1);
        if r1 < 0 {
            continue;
        }
        for r in r0..=r1 as usize {
            let y = -half + (r as f64 + 0.5) * MAP_NM;
            xs.clear();
            for i in 0..poly.len() {
                let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                if (a.y > y) != (b.y > y) {
                    xs.push(a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y));
                }
            }
            xs.sort_by(f64::total_cmp);
            for pair in xs.chunks_exact(2) {
                let c0 = ((pair[0] + half) / MAP_NM - 0.5).ceil().max(0.) as usize;
                let c1 = (((pair[1] + half) / MAP_NM - 0.5).floor() as i64).min(n as i64 - 1);
                for c in c0 as i64..=c1 {
                    inside[r * n + c as usize] = true;
                }
            }
        }
    }
    (n, inside)
}

/// Squared distance (pixels²) from each pixel to the nearest `true` one
/// (Felzenszwalb and Huttenlocher).
fn edt(feature: &[bool], n: usize) -> Vec<f64> {
    const BIG: f64 = 1e12;
    let mut g: Vec<f64> = feature.iter().map(|&f| if f { 0. } else { BIG }).collect();
    let mut f = vec![0.; n];
    let mut d = vec![0.; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.; n + 1];
    let mut pass = |get: &dyn Fn(usize) -> usize, g: &mut Vec<f64>| {
        for (q, fq) in f.iter_mut().enumerate() {
            *fq = g[get(q)];
        }
        let mut k = 0;
        v[0] = 0;
        z[0] = f64::NEG_INFINITY;
        z[1] = f64::INFINITY;
        let meet = |q: usize, p: usize, f: &[f64]| {
            let (qf, pf) = (q as f64, p as f64);
            ((f[q] + qf * qf) - (f[p] + pf * pf)) / (2. * qf - 2. * pf)
        };
        for q in 1..n {
            // z[0] is −∞, so this stops at k = 0 at the latest.
            let mut s = meet(q, v[k], &f);
            while s <= z[k] {
                k -= 1;
                s = meet(q, v[k], &f);
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
        }
        k = 0;
        for q in 0..n {
            while z[k + 1] < q as f64 {
                k += 1;
            }
            let dq = q as f64 - v[k] as f64;
            d[q] = dq * dq + f[v[k]];
        }
        for (q, dq) in d.iter().enumerate() {
            g[get(q)] = *dq;
        }
    };
    for x in 0..n {
        pass(&|q| q * n + x, &mut g);
    }
    for y in 0..n {
        pass(&|q| y * n + q, &mut g);
    }
    g
}

fn signed_distance(n: usize, inside: &[bool]) -> Vec<f32> {
    let to_in = edt(inside, n);
    let outside: Vec<bool> = inside.iter().map(|b| !b).collect();
    let to_out = edt(&outside, n);
    inside
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let px = if b { -(to_out[i].sqrt() - 0.5) } else { to_in[i].sqrt() - 0.5 };
            (px * MAP_NM).min(FAR as f64) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Library of structure kinds
// ---------------------------------------------------------------------------

/// One kind of ring structure, with every place it occurs on the GDS.
pub struct Kind {
    pub name: String,
    /// Ring systems, centres relative to the reference point.
    pub systems: Vec<RingSystem>,
    /// Reference point of every copy on the GDS (local nm).
    pub instances: Vec<DVec2>,
    map: Map,
    /// Fraction of the area within the rings that is band.
    band_fraction: f64,
    /// Narrowest band and typical band-to-band distance (local nm).
    pub narrowest: f64,
    pub pitch: f64,
}

/// Where a position is on a structure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    /// Ring system (index into `Kind::systems`) the position is on.
    pub system: usize,
    /// Ring number counted from the centre (1 = innermost). Rings past the
    /// detected ones are counted by the pitch.
    pub ring: usize,
    /// Distance from that system's centre (local nm).
    pub radius: f64,
    /// No system's arcs reach here (e.g. between the halves of split rings).
    pub between: bool,
}

impl Kind {
    /// Which system and ring a position (local nm from the reference point)
    /// is on.
    pub fn place(&self, at: DVec2) -> Place {
        let near = |i: usize| (at - self.systems[i].center).length();
        let covering = (0..self.systems.len())
            .filter(|&i| {
                let v = at - self.systems[i].center;
                let s = &self.systems[i];
                v.length() < s.bands[0].0 || s.covers(v.y.atan2(v.x))
            })
            .min_by(|&a, &b| near(a).total_cmp(&near(b)));
        let between = covering.is_none();
        let system = covering.unwrap_or_else(|| {
            (0..self.systems.len()).min_by(|&a, &b| near(a).total_cmp(&near(b))).unwrap_or(0)
        });
        let s = &self.systems[system];
        let radius = near(system);
        let mid = |b: &(f64, f64)| (b.0 + b.1) / 2.;
        let last = s.bands.last().map_or(0., mid);
        let ring = if radius <= last + self.pitch / 2. {
            1 + (0..s.bands.len())
                .min_by(|&a, &b| (radius - mid(&s.bands[a])).abs().total_cmp(&(radius - mid(&s.bands[b])).abs()))
                .unwrap_or(0)
        } else {
            s.bands.len() + ((radius - last) / self.pitch).round() as usize
        };
        Place { system, ring, radius, between }
    }

    /// Where to head from a position: the centre of the system it is on, or
    /// the middle of the structure when it is between systems.
    pub fn target(&self, at: DVec2) -> DVec2 {
        let p = self.place(at);
        if p.between {
            DVec2::ZERO
        } else {
            self.systems[p.system].center
        }
    }
}

pub struct Library {
    pub kinds: Vec<Kind>,
}

/// Group ring systems whose rings overlap (the halves of split rings).
fn group(systems: &[RingSystem]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..systems.len()).collect();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let mut order: Vec<usize> = (0..systems.len()).collect();
    order.sort_by(|&a, &b| systems[a].center.x.total_cmp(&systems[b].center.x));
    let reach = systems.iter().map(|s| s.outer_radius()).fold(0., f64::max);
    for (oi, &a) in order.iter().enumerate() {
        for &b in &order[oi + 1..] {
            let d = systems[b].center - systems[a].center;
            if d.x > reach {
                break;
            }
            if d.length() < systems[a].outer_radius().max(systems[b].outer_radius()) {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                parent[ra] = rb;
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..systems.len() {
        let r = root(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    groups.into_values().collect()
}

/// A group's systems relative to its reference point, in a fixed order.
fn relative(systems: &[RingSystem], g: &[usize]) -> (DVec2, Vec<RingSystem>) {
    let r = g.iter().map(|&i| systems[i].center).sum::<DVec2>() / g.len() as f64;
    let mut rel: Vec<RingSystem> =
        g.iter().map(|&i| RingSystem { center: systems[i].center - r, ..systems[i].clone() }).collect();
    rel.sort_by(|a, b| {
        let key = |s: &RingSystem| ((s.center.x / 100.).round() as i64, (s.center.y / 100.).round() as i64);
        key(a).cmp(&key(b))
    });
    (r, rel)
}

fn same_layout(a: &[RingSystem], b: &[RingSystem]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            (x.center - y.center).length() < 300.
                && x.bands.len() == y.bands.len()
                && x.bands.iter().zip(&y.bands).all(|(p, q)| {
                    let tol = (0.02 * q.1).max(50.);
                    (p.0 - q.0).abs() < tol && (p.1 - q.1).abs() < tol
                })
                && (x.covered ^ y.covered).count_ones() <= 3
        })
}

fn kind_name(systems: &[RingSystem]) -> String {
    match systems {
        [s] if s.covered.count_ones() >= 34 => "full rings".into(),
        [_] => "ring arcs".into(),
        [a, b] => {
            let d = b.center - a.center;
            if d.x.abs() > d.y.abs() {
                "split rings (side by side halves)".into()
            } else {
                "split rings (stacked halves)".into()
            }
        }
        s => format!("{} ring systems", s.len()),
    }
}

impl Library {
    /// Find the kinds of structure in a GDS and draw each once.
    pub fn build(polys: &BTreeMap<u16, Vec<Vec<Vec2>>>, systems: &[RingSystem]) -> Library {
        // (layout, absolute reference points)
        let mut found: Vec<(Vec<RingSystem>, Vec<DVec2>)> = Vec::new();
        for g in group(systems) {
            let (r, rel) = relative(systems, &g);
            match found.iter_mut().find(|k| same_layout(&k.0, &rel)) {
                Some(k) => k.1.push(r),
                None => found.push((rel, vec![r])),
            }
        }
        let kinds = std::thread::scope(|scope| {
            let hs: Vec<_> = found
                .into_iter()
                .map(|(rel, instances)| scope.spawn(move || Self::draw_kind(polys, rel, instances)))
                .collect();
            hs.into_iter().filter_map(|h| h.join().ok()).collect()
        });
        Library { kinds }
    }

    fn draw_kind(polys: &BTreeMap<u16, Vec<Vec<Vec2>>>, systems: Vec<RingSystem>, instances: Vec<DVec2>) -> Kind {
        let origin = instances[0];
        let outer = systems.iter().map(|s| s.outer_radius()).fold(0., f64::max);
        let half = MAP_REACH * outer;
        let layers: Vec<u16> = systems.iter().map(|s| s.layer).collect();
        let local: Vec<Vec<DVec2>> = polys
            .iter()
            .filter(|(l, _)| layers.contains(l))
            .flat_map(|(_, ps)| ps)
            .filter(|p| {
                let (lo, hi) = p.iter().fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
                let (lo, hi) = (lo.as_dvec2() - origin, hi.as_dvec2() - origin);
                hi.x >= -half && hi.y >= -half && lo.x <= half && lo.y <= half
            })
            .map(|p| p.iter().map(|v| v.as_dvec2() - origin).collect())
            .collect();
        let (n, inside) = rasterize(&local, half);
        let map = Map { half, n, d: signed_distance(n, &inside) };
        // Band fraction within the rings.
        let (mut band, mut all) = (0usize, 0usize);
        for y in 0..n {
            for x in 0..n {
                let p = DVec2::new(-half + (x as f64 + 0.5) * MAP_NM, -half + (y as f64 + 0.5) * MAP_NM);
                if systems.iter().any(|s| (p - s.center).length() <= s.outer_radius()) {
                    all += 1;
                    band += inside[y * n + x] as usize;
                }
            }
        }
        let narrowest = systems.iter().flat_map(|s| &s.bands).map(|b| b.1 - b.0).fold(f64::INFINITY, f64::min);
        let mut gaps: Vec<f64> = systems.iter().flat_map(|s| s.bands.windows(2).map(|w| w[1].0 - w[0].0)).collect();
        gaps.sort_by(f64::total_cmp);
        let pitch = gaps.get(gaps.len() / 2).copied().unwrap_or(2. * narrowest);
        Kind {
            name: kind_name(&systems),
            systems,
            instances,
            map,
            band_fraction: if all > 0 { band as f64 / all as f64 } else { 0.3 },
            narrowest,
            pitch,
        }
    }
}

/// A GDS image's library, built once on first use.
pub type LibraryCell = Arc<std::sync::OnceLock<Arc<Library>>>;

/// What it takes to build a GDS's library (cheap to clone).
#[derive(Clone)]
pub struct LandingSource {
    pub polys: Arc<BTreeMap<u16, Vec<Vec<Vec2>>>>,
    pub systems: Arc<Vec<RingSystem>>,
    pub cell: LibraryCell,
}

impl LandingSource {
    /// The library, building it if needed (about a second: call it off
    /// the UI thread).
    pub fn library(&self) -> Arc<Library> {
        self.cell.get_or_init(|| Arc::new(Library::build(&self.polys, &self.systems))).clone()
    }
}

// ---------------------------------------------------------------------------
// Matching a frame
// ---------------------------------------------------------------------------

/// Matching settings for landing: the ring-match settings, with the frame
/// flattened smoothly (heights kept). Two tones force the same band
/// fraction on every frame, which goes wrong where only a little of the
/// frame is band (arc ends, the outermost ring); on the 40 labelled frames
/// smooth was right on 29, two tones on 26.
pub fn params(matching: &MatchParams) -> MatchParams {
    MatchParams { mode: MatchMode::Smooth, ..*matching }
}

/// Frame points: displacement from the frame centre (local nm) and the
/// standardised two-tone value for each polarity tried.
struct Img {
    l: Vec<DVec2>,
    x: Vec<Vec<f64>>,
}

impl Img {
    fn standardise(mut x: Vec<f64>) -> Vec<f64> {
        let m = x.iter().sum::<f64>() / x.len().max(1) as f64;
        x.iter_mut().for_each(|v| *v -= m);
        let s = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        x.iter_mut().for_each(|v| *v /= s);
        x
    }
}

/// A frame prepared for one kind of structure.
struct Prep {
    full: Img,
    coarse: Img,
    /// Polarity of each image: bands raised.
    raised: Vec<bool>,
    /// Soft edge width (local nm): a working pixel, and wider for the
    /// coarse search so its score doesn't drop off between grid steps.
    edge: f64,
    coarse_edge: f64,
    /// Farthest point from the frame centre (local nm).
    reach: f64,
}

/// Local nm per frame pixel, along each pixel axis.
fn px_to_local(shot: &Shot, gds_linear: DMat2) -> DMat2 {
    let s = DVec2::new(shot.size[0] as f64, shot.size[1] as f64);
    gds_linear.inverse() * shot.frame.matrix2 * DMat2::from_diagonal(1. / s)
}

fn prepare(shot: &Shot, kind: &Kind, gds_linear: DMat2, p: &MatchParams) -> Option<Prep> {
    let a = px_to_local(shot, gds_linear);
    let local_per_px = (a.x_axis.length() + a.y_axis.length()) / 2.;
    let mut p = *p;
    if p.mode == MatchMode::Off {
        p.mode = MatchMode::SoftTwoTone;
    }
    let pols: &[bool] = match p.polarity {
        Polarity::Raised => &[true],
        Polarity::Recessed => &[false],
        Polarity::Auto => &[true, false],
    };
    let pc = DVec2::new(shot.size[0] as f64 / 2. - 0.5, shot.size[1] as f64 / 2. - 0.5);
    let mut l = Vec::new();
    let mut coarse_idx = Vec::new();
    let mut xs = Vec::new();
    let mut edge = 0.;
    for (pi, &raised) in pols.iter().enumerate() {
        let pr = prepare_fraction(&shot.data, shot.size, kind.narrowest / local_per_px, kind.band_fraction, &p, raised)?;
        let kf = pr.k as f64;
        let stride = (pr.w.max(pr.h) / COARSE_PTS).max(1);
        let mut x = Vec::new();
        for (i, v) in pr.v.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            x.push(*v);
            if pi == 0 {
                let (ix, iy) = (i % pr.w, i / pr.w);
                let frame_px = DVec2::new(ix as f64, iy as f64) * kf + (kf - 1.) / 2.;
                l.push(a * (frame_px - pc));
                if ix % stride == 0 && iy % stride == 0 {
                    coarse_idx.push(l.len() - 1);
                }
            }
        }
        if x.len() != l.len() || l.len() < 16 {
            return None;
        }
        edge = local_per_px * kf;
        xs.push(x);
    }
    let coarse = Img {
        l: coarse_idx.iter().map(|&i| l[i]).collect(),
        x: xs.iter().map(|x| Img::standardise(coarse_idx.iter().map(|&i| x[i]).collect())).collect(),
    };
    let reach = l.iter().map(|v| v.length()).fold(0., f64::max);
    let full = Img { l, x: xs.into_iter().map(Img::standardise).collect() };
    let coarse_edge = edge.max(kind.narrowest / 3.);
    Some(Prep { full, coarse, raised: pols.to_vec(), edge, coarse_edge, reach })
}

/// Band widenings to try (local nm). Scanned bands usually come out wider
/// than drawn; with the polarity unknown they may grow at most to the gap
/// width (as in ring matching).
fn biases(kind: &Kind, p: &MatchParams, n: usize) -> Vec<f64> {
    let narrow = kind.narrowest;
    let mut hi = p.width_slack * narrow;
    if p.polarity == Polarity::Auto {
        hi = hi.min((kind.pitch - 2. * narrow) / 2.).max(0.);
    }
    let lo = -0.25 * narrow;
    if n <= 1 || hi <= lo {
        return vec![0.];
    }
    (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect()
}

#[derive(Clone, Copy, Debug)]
struct Cand {
    at: DVec2,
    score: f64,
    raised: bool,
    bias: f64,
}

/// Correlation of the frame with the map with the frame centre at `t`,
/// best over polarity and band widening.
fn score(map: &Map, prep: &Prep, img: &Img, t: DVec2, biases: &[f64], fine: bool, ys: &mut Vec<f32>) -> Cand {
    let mut best = Cand { at: t, score: -1., raised: true, bias: 0. };
    ys.clear();
    ys.extend(img.l.iter().map(|l| if fine { map.bilinear(t + *l) } else { map.nearest(t + *l) }));
    let n = ys.len() as f64;
    let inv_edge = 1. / if fine { prep.edge } else { prep.coarse_edge };
    for &b in biases {
        let (mut sy, mut syy) = (0., 0.);
        let mut sxy = [0.; 2];
        for (i, &d) in ys.iter().enumerate() {
            let y = (0.5 - (d as f64 - b / 2.) * inv_edge).clamp(0., 1.);
            sy += y;
            syy += y * y;
            for (k, x) in img.x.iter().enumerate() {
                sxy[k] += x[i] * y;
            }
        }
        let var = syy - sy * sy / n;
        if var < 1e-9 {
            continue;
        }
        let sd = var.sqrt();
        for k in 0..img.x.len() {
            let c = sxy[k] / sd;
            if c > best.score {
                best = Cand { at: t, score: c, raised: prep.raised[k], bias: b };
            }
        }
    }
    best
}

/// Hill-climb from `c` on the full frame, halving the step down to a
/// fraction of a map pixel.
fn refine(kind: &Kind, prep: &Prep, mut c: Cand, step: f64, biases: &[f64]) -> Cand {
    let mut ys = Vec::new();
    c = score(&kind.map, prep, &prep.full, c.at, biases, true, &mut ys);
    let mut s = step / 2.;
    while s >= MAP_NM / 4. {
        let mut moved = false;
        for d in [DVec2::X, -DVec2::X, DVec2::Y, -DVec2::Y, DVec2::ONE, -DVec2::ONE, DVec2::new(1., -1.), DVec2::new(-1., 1.)] {
            let n = score(&kind.map, prep, &prep.full, c.at + d * s, biases, true, &mut ys);
            if n.score > c.score {
                c = n;
                moved = true;
            }
        }
        if !moved {
            s /= 2.;
        }
    }
    c
}

/// Every place on one kind of structure the frame could be, best first.
fn search_kind(kind: &Kind, prep: &Prep, p: &MatchParams) -> Vec<Cand> {
    let step = kind.narrowest / 3.;
    let lim = kind.map.half - prep.reach - MAP_NM;
    if lim <= 0. {
        return vec![];
    }
    let g = (lim / step).floor() as i64;
    let side = (2 * g + 1) as usize;
    let coarse_b = {
        let b = biases(kind, p, 3);
        if b.len() == 3 { vec![b[1], b[2]] } else { b }
    };
    // Coarse map of scores, rows split over threads.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let mut grid = vec![-1f32; side * side];
    std::thread::scope(|scope| {
        let rows_per = side.div_ceil(threads);
        for (ci, chunk) in grid.chunks_mut(rows_per * side).enumerate() {
            let coarse_b = &coarse_b;
            scope.spawn(move || {
                let mut ys = Vec::new();
                for (k, v) in chunk.iter_mut().enumerate() {
                    let (ix, iy) = (k % side, ci * rows_per + k / side);
                    let t = DVec2::new(ix as f64 - g as f64, iy as f64 - g as f64) * step;
                    *v = score(&kind.map, prep, &prep.coarse, t, coarse_b, false, &mut ys).score as f32;
                }
            });
        }
    });
    // Local maxima, best first, kept apart by about a band width.
    let mut peaks: Vec<(f32, usize, usize)> = Vec::new();
    for iy in 1..side - 1 {
        for ix in 1..side - 1 {
            let v = grid[iy * side + ix];
            if v <= 0. {
                continue;
            }
            let max = (-1..=1).all(|dy: i64| {
                (-1..=1).all(|dx: i64| grid[(iy as i64 + dy) as usize * side + (ix as i64 + dx) as usize] <= v)
            });
            if max {
                peaks.push((v, ix, iy));
            }
        }
    }
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    let apart = kind.narrowest.max(2. * step);
    let mut seeds: Vec<DVec2> = Vec::new();
    for (_, ix, iy) in peaks {
        let t = DVec2::new(ix as f64 - g as f64, iy as f64 - g as f64) * step;
        if seeds.iter().all(|s| (*s - t).length() > apart) {
            seeds.push(t);
            if seeds.len() >= SEEDS_PER_KIND {
                break;
            }
        }
    }
    let fine_b = biases(kind, p, FINE_BIASES);
    let mut out: Vec<Cand> = std::thread::scope(|scope| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&t| {
                let fine_b = &fine_b;
                scope.spawn(move || refine(kind, prep, Cand { at: t, score: -1., raised: true, bias: 0. }, step, fine_b))
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Cand> = Vec::new();
    for c in out {
        if kept.iter().all(|k| (k.at - c.at).length() > kind.narrowest / 2.) {
            kept.push(c);
        }
    }
    kept
}

/// Best fit within `radius` of `center` (local nm).
fn best_near(kind: &Kind, prep: &Prep, center: DVec2, radius: f64, p: &MatchParams) -> Cand {
    let step = kind.narrowest / 3.;
    let n = (radius / step).ceil() as i64;
    let coarse_b = biases(kind, p, 2);
    let mut ys = Vec::new();
    let mut best = Cand { at: center, score: -1., raised: true, bias: 0. };
    for iy in -n..=n {
        for ix in -n..=n {
            let off = DVec2::new(ix as f64, iy as f64) * step;
            if off.length() > radius + 1e-9 {
                continue;
            }
            let c = score(&kind.map, prep, &prep.coarse, center + off, &coarse_b, false, &mut ys);
            if c.score > best.score {
                best = c;
            }
        }
    }
    let c = refine(kind, prep, best, step, &biases(kind, p, FINE_BIASES));
    // Refining may wander off: keep it within reach of the prediction.
    if (c.at - center).length() <= radius + step {
        c
    } else {
        best
    }
}

// ---------------------------------------------------------------------------
// Hypotheses over several frames
// ---------------------------------------------------------------------------

/// One guess of where the tip is.
#[derive(Clone, Debug, PartialEq)]
pub struct Hypothesis {
    /// Index into `Library::kinds`.
    pub kind: usize,
    /// Centre of the latest frame relative to the structure's reference
    /// point (local nm).
    pub at: DVec2,
    /// Fit in each frame of the track, oldest first.
    pub scores: Vec<f64>,
    pub raised: bool,
    /// Bands are this much wider than drawn (local nm).
    pub bias_nm: f64,
}

impl Hypothesis {
    /// Mean fit over the frames.
    pub fn score(&self) -> f64 {
        self.scores.iter().sum::<f64>() / self.scores.len().max(1) as f64
    }
}

/// Rough odds of each hypothesis (they sum to 1).
pub fn odds(hyps: &[Hypothesis]) -> Vec<f64> {
    let best = hyps.iter().map(|h| h.score()).fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = hyps.iter().map(|h| ((h.score() - best) / ODDS_SCALE).exp()).collect();
    let s: f64 = w.iter().sum();
    w.into_iter().map(|x| x / s.max(1e-300)).collect()
}

/// Hypotheses whose way to the centre is within this angle agree.
pub const AGREE_DEG: f64 = 30.;

/// What the hypotheses say together.
#[derive(Clone, Copy, Debug)]
pub struct Consensus {
    /// Unit direction (local axes) from the frame to the centre the best
    /// hypothesis heads for.
    pub dir_local: DVec2,
    /// Share of the odds on hypotheses heading the same way (within
    /// `AGREE_DEG`).
    pub agree: f64,
    /// Range of distances to the centre among those hypotheses (local nm).
    pub dist_min: f64,
    pub dist_max: f64,
}

/// Which way the hypotheses point, and how much they agree.
pub fn consensus(lib: &Library, hyps: &[Hypothesis]) -> Option<Consensus> {
    let way = |h: &Hypothesis| lib.kinds[h.kind].target(h.at) - h.at;
    let best = hyps.first()?;
    let dir_local = way(best).try_normalize()?;
    let cos = AGREE_DEG.to_radians().cos();
    let (mut agree, mut dist_min, mut dist_max) = (0., f64::INFINITY, 0f64);
    for (h, o) in hyps.iter().zip(odds(hyps)) {
        let w = way(h);
        if w.try_normalize().is_some_and(|d| d.dot(dir_local) >= cos) {
            agree += o;
            // Only count hypotheses with real weight for the range.
            if o >= 0.02 {
                dist_min = dist_min.min(w.length());
                dist_max = dist_max.max(w.length());
            }
        }
    }
    Some(Consensus { dir_local, agree, dist_min, dist_max })
}

struct TrackShot {
    shot: Shot,
    preps: Vec<Option<Prep>>,
}

/// Landing hypotheses, kept up to date as frames come in after moves.
pub struct Track {
    pub library: Arc<Library>,
    /// GDS local → world (only the linear part is used for frame-to-frame
    /// moves, so moving the GDS on the map doesn't upset a track).
    gds_linear: DMat2,
    shots: Vec<TrackShot>,
    pub hyps: Vec<Hypothesis>,
}

impl Track {
    pub fn new(library: Arc<Library>, gds_transform: DAffine2) -> Self {
        Self { library, gds_linear: gds_transform.matrix2, shots: Vec::new(), hyps: Vec::new() }
    }

    pub fn latest(&self) -> Option<&Shot> {
        self.shots.last().map(|s| &s.shot)
    }

    /// The GDS scale on the map differs from the track's (a new GDS or a
    /// rescale): the track must start over.
    pub fn same_gds(&self, gds_transform: DAffine2) -> bool {
        (self.gds_linear - gds_transform.matrix2).to_cols_array().iter().all(|v| v.abs() < 1e-9 * self.gds_linear.determinant().abs().sqrt().max(1.))
    }

    /// Local displacement of a world move.
    fn local(&self, world: DVec2) -> DVec2 {
        self.gds_linear.inverse() * world
    }

    /// Add a complete frame: check the existing hypotheses at the places
    /// they predict, look for new ones, and check those in the earlier
    /// frames.
    pub fn add(&mut self, shot: Shot, p: &MatchParams) {
        let lib = self.library.clone();
        let preps: Vec<Option<Prep>> = std::thread::scope(|scope| {
            let hs: Vec<_> = lib
                .kinds
                .iter()
                .map(|k| {
                    let shot = &shot;
                    let g = self.gds_linear;
                    scope.spawn(move || prepare(shot, k, g, p))
                })
                .collect();
            hs.into_iter().map(|h| h.join().ok().flatten()).collect()
        });
        // Existing hypotheses, checked where they say the new frame is.
        let mut kept: Vec<Hypothesis> = Vec::new();
        if let Some(last) = self.latest() {
            let moved = self.local(shot.center() - last.center());
            let radius = MOVE_ERROR * moved.length() + MOVE_SLACK_NM;
            for h in &self.hyps {
                let (kind, Some(prep)) = (&lib.kinds[h.kind], &preps[h.kind]) else { continue };
                let c = best_near(kind, prep, h.at + moved, radius, p);
                let mut h = h.clone();
                h.at = c.at;
                h.scores.push(c.score);
                h.raised = c.raised;
                h.bias_nm = c.bias;
                kept.push(h);
            }
        }
        // New hypotheses from this frame, checked in the earlier frames.
        for (ki, kind) in lib.kinds.iter().enumerate() {
            let Some(prep) = &preps[ki] else { continue };
            for c in search_kind(kind, prep, p) {
                if kept.iter().any(|h| h.kind == ki && (h.at - c.at).length() < kind.narrowest) {
                    continue;
                }
                let mut scores = Vec::new();
                for old in &self.shots {
                    let back = self.local(old.shot.center() - shot.center());
                    let radius = MOVE_ERROR * back.length() + MOVE_SLACK_NM;
                    scores.push(match &old.preps[ki] {
                        Some(op) => best_near(kind, op, c.at + back, radius, p).score,
                        None => -1.,
                    });
                }
                scores.push(c.score);
                kept.push(Hypothesis { kind: ki, at: c.at, scores, raised: c.raised, bias_nm: c.bias });
            }
        }
        kept.sort_by(|a, b| b.score().total_cmp(&a.score()));
        kept.truncate(MAX_HYPS);
        self.shots.push(TrackShot { shot, preps });
        if self.shots.len() > MAX_SHOTS {
            self.shots.remove(0);
            for h in &mut kept {
                h.scores.remove(0);
            }
            kept.sort_by(|a, b| b.score().total_cmp(&a.score()));
        }
        self.hyps = kept;
    }

    /// A snapshot for display.
    pub fn view(&self) -> Option<View> {
        let shot = self.latest()?;
        Some(View {
            library: self.library.clone(),
            odds: odds(&self.hyps),
            consensus: consensus(&self.library, &self.hyps),
            hyps: self.hyps.clone(),
            frame_center: shot.center(),
            frame_width: shot.width(),
            frames: self.shots.len(),
            gds_linear: self.gds_linear,
        })
    }

    /// Where a hypothesis puts things on the map (world nm), with the GDS
    /// at `gds_transform`.
    #[cfg(test)]
    pub fn locate(&self, h: &Hypothesis, gds_transform: DAffine2) -> Option<Located> {
        self.view()?.locate(h, gds_transform)
    }

    /// The same frame again (e.g. re-analysed with other settings).
    pub fn has(&self, shot: &Shot) -> bool {
        self.latest().is_some_and(|l| Arc::ptr_eq(&l.data, &shot.data) && l.frame == shot.frame)
    }
}

/// What a track says, for display.
#[derive(Clone)]
pub struct View {
    pub library: Arc<Library>,
    /// Best first.
    pub hyps: Vec<Hypothesis>,
    pub odds: Vec<f64>,
    pub consensus: Option<Consensus>,
    /// Latest frame (world nm).
    pub frame_center: DVec2,
    pub frame_width: f64,
    pub frames: usize,
    gds_linear: DMat2,
}

impl View {
    /// Where a hypothesis puts things on the map (world nm), with the GDS
    /// at `gds_transform`.
    pub fn locate(&self, h: &Hypothesis, gds_transform: DAffine2) -> Option<Located> {
        let kind = self.library.kinds.get(h.kind)?;
        let c = self.frame_center;
        let (place, structure, target) = self.place_world(h)?;
        // The GDS shift that puts the nearest copy of this structure here.
        let q = gds_transform.inverse().transform_point2(c);
        let gds_shift = kind
            .instances
            .iter()
            .map(|&i| gds_transform.matrix2 * (q - i - h.at))
            .min_by(|a, b| a.length().total_cmp(&b.length()))?;
        Some(Located { place, structure_world: structure, target_world: target, gds_shift })
    }

    /// Where a hypothesis puts the frame on its structure, the structure's
    /// reference point and the centre to head for (world nm). Doesn't
    /// depend on where the GDS sits on the map.
    pub fn place_world(&self, h: &Hypothesis) -> Option<(Place, DVec2, DVec2)> {
        let kind = self.library.kinds.get(h.kind)?;
        let m = self.gds_linear;
        let structure = self.frame_center - m * h.at;
        Some((kind.place(h.at), structure, structure + m * kind.target(h.at)))
    }

    /// Consensus way to the centre (world, unit) and distance range (world nm).
    pub fn way_world(&self) -> Option<(DVec2, f64, f64)> {
        let c = self.consensus?;
        let k = self.gds_linear.determinant().abs().sqrt();
        Some(((self.gds_linear * c.dir_local).normalize_or_zero(), c.dist_min * k, c.dist_max * k))
    }

    /// Local nm → world nm, for lengths.
    pub fn scale(&self) -> f64 {
        self.gds_linear.determinant().abs().sqrt()
    }
}

/// A hypothesis on the map.
#[derive(Clone, Copy, Debug)]
pub struct Located {
    pub place: Place,
    /// The structure's reference point (world nm).
    pub structure_world: DVec2,
    /// The centre to head for (world nm).
    pub target_world: DVec2,
    /// Moving the GDS by this (world nm) lines its nearest copy of the
    /// structure up with this hypothesis.
    pub gds_shift: DVec2,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ring_match::tests::Lcg;

    /// Annular arc polygon (local nm) from angle a0 to a1 (radians).
    fn arc(c: DVec2, r_in: f64, r_out: f64, a0: f64, a1: f64) -> Vec<Vec2> {
        let n = 48;
        let mut v = Vec::new();
        for i in 0..=n {
            let a = a0 + (a1 - a0) * i as f64 / n as f64;
            v.push((c + DVec2::from_angle(a) * r_in).as_vec2());
        }
        for i in (0..=n).rev() {
            let a = a0 + (a1 - a0) * i as f64 / n as f64;
            v.push((c + DVec2::from_angle(a) * r_out).as_vec2());
        }
        v
    }

    fn rect(lo: DVec2, hi: DVec2) -> Vec<Vec2> {
        vec![lo.as_vec2(), DVec2::new(hi.x, lo.y).as_vec2(), hi.as_vec2(), DVec2::new(lo.x, hi.y).as_vec2()]
    }

    /// Full rings (1 µm bands every 4 µm) at two places, and split rings
    /// (halves 16 µm apart, 1 µm bands every 3.2 µm, three bars between).
    fn gds() -> BTreeMap<u16, Vec<Vec<Vec2>>> {
        let pi = std::f64::consts::PI;
        let mut p = Vec::new();
        for c in [DVec2::new(0., 0.), DVec2::new(500_000., 0.)] {
            for i in 0..6 {
                let r = 1000. + 4000. * i as f64;
                p.push(arc(c, r, r + 1000., 0., pi));
                p.push(arc(c, r, r + 1000., pi, 2. * pi));
            }
        }
        let (l, r) = (DVec2::new(-8000., 250_000.), DVec2::new(8000., 250_000.));
        for i in 0..7 {
            let a = 1800. + 3200. * i as f64;
            p.push(arc(l, a, a + 1000., pi / 2., 3. * pi / 2.));
            p.push(arc(r, a, a + 1000., -pi / 2., pi / 2.));
        }
        for x in [-4000., 0., 4000.] {
            p.push(rect(DVec2::new(x - 500., 250_000. - 30_000.), DVec2::new(x + 500., 250_000. + 30_000.)));
        }
        // Background on another layer, which must be ignored.
        let bg = vec![rect(DVec2::splat(-1e6), DVec2::splat(1e6))];
        BTreeMap::from([(1, bg), (3, p)])
    }

    pub(super) fn library() -> Library {
        let polys = gds();
        let systems = crate::analysis::gds_prior::find_ring_systems(&polys);
        Library::build(&polys, &systems)
    }

    /// A 3.6 µm frame (world, GDS scale 0.75 on the map) centred at `at`
    /// (local nm) on `kind`: bands raised, widened, with tilt, noise and blobs.
    pub(super) fn frame(kind: &Kind, at: DVec2, world_center: DVec2, seed: u64) -> Shot {
        let w = 160usize;
        let frame = DAffine2::from_scale_angle_translation(DVec2::splat(3600.), 0., world_center);
        let shot = Shot { data: Arc::new(Box::new([])), size: [w as u32; 2], frame };
        let a = px_to_local(&shot, gds_transform().matrix2);
        let pc = DVec2::splat(w as f64 / 2. - 0.5);
        let mut rng = Lcg(seed);
        let blobs: Vec<DVec2> = (0..5).map(|_| DVec2::new(rng.next(), rng.next()) * w as f64).collect();
        let data: Vec<f32> = (0..w * w)
            .map(|i| {
                let px = DVec2::new((i % w) as f64, (i / w) as f64);
                let on = kind.map.bilinear(at + a * (px - pc)) < 150.;
                let mut v = on as u8 as f64 + 0.004 * px.x - 0.003 * px.y + 0.5 * (rng.next() - 0.5);
                if blobs.iter().any(|b| (*b - px).length() < 5.) {
                    v += 3.;
                }
                v as f32
            })
            .collect();
        Shot { data: Arc::new(data.into_boxed_slice()), ..shot }
    }

    pub(super) fn gds_transform() -> DAffine2 {
        DAffine2::from_scale(DVec2::splat(0.75))
    }

    fn kind_of(lib: &Library, name: &str) -> usize {
        lib.kinds.iter().position(|k| k.name.starts_with(name)).unwrap()
    }

    #[test]
    fn library_finds_the_kinds_and_their_copies() {
        let lib = library();
        assert_eq!(lib.kinds.len(), 2, "{:?}", lib.kinds.iter().map(|k| &k.name).collect::<Vec<_>>());
        let full = &lib.kinds[kind_of(&lib, "full")];
        assert_eq!(full.instances.len(), 2);
        assert!((full.pitch - 4000.).abs() < 100., "{}", full.pitch);
        assert!((full.band_fraction - 0.25).abs() < 0.05, "{}", full.band_fraction);
        let split = &lib.kinds[kind_of(&lib, "split")];
        assert_eq!(split.systems.len(), 2);
        // The bars between the halves are drawn; the background is not.
        assert!(split.map.inside(DVec2::new(0., 10_000.)));
        assert!(!split.map.inside(DVec2::new(2000., 10_000.)));
        // Ring numbers and halves.
        let p = split.place(DVec2::new(-8000. - 8700., 0.));
        assert_eq!((p.ring, p.between), (3, false), "{p:?}");
        assert!(split.systems[p.system].center.x < 0.);
        assert!(split.place(DVec2::new(2000., 0.)).between);
        let far = full.place(DVec2::new(0., 41_500.));
        assert_eq!(far.ring, 11, "{far:?}");
    }

    #[test]
    fn signed_distance_is_right_at_a_square() {
        let half = 2000.;
        let (n, inside) = rasterize(&[vec![DVec2::new(-500., -500.), DVec2::new(500., -500.), DVec2::new(500., 500.), DVec2::new(-500., 500.)]], half);
        let map = Map { half, n, d: signed_distance(n, &inside) };
        assert!((map.bilinear(DVec2::ZERO) + 500.).abs() < MAP_NM as f32, "{}", map.bilinear(DVec2::ZERO));
        assert!((map.bilinear(DVec2::new(1500., 0.)) - 1000.).abs() < MAP_NM as f32);
        assert_eq!(map.nearest(DVec2::splat(5000.)), FAR);
    }

    /// The way to the centre for a hypothesis (local axes).
    fn way(lib: &Library, ki: usize, at: DVec2) -> DVec2 {
        lib.kinds[ki].target(at) - at
    }

    fn angle_deg(a: DVec2, b: DVec2) -> f64 {
        a.normalize().dot(b.normalize()).clamp(-1., 1.).acos().to_degrees()
    }

    /// One frame: the true place is among the best hypotheses, and the
    /// hypotheses agree on the way to the centre (a single band fits one
    /// ring further in or out, or the other kind, about as well).
    #[test]
    fn one_frame_finds_the_place_and_the_way() {
        let lib = Arc::new(library());
        let p = params(&MatchParams::default());
        let full = kind_of(&lib, "full");
        let split = kind_of(&lib, "split");
        // (kind, place, least agreement expected)
        for (seed, (ki, at, min_agree)) in [
            (full, DVec2::new(9500., 3000.), 0.7),
            (full, DVec2::new(-2500., -6000.), 0.7),
            (split, DVec2::new(-8000. - 9000., 4000.), 0.7),
            // Over the ends of the arcs of one half: near the cut the arcs
            // are almost straight, so the other half (up or down) fits
            // nearly as well and the hypotheses rightly disagree.
            (split, DVec2::new(-7000., 6000.), 0.4),
        ]
        .into_iter()
        .enumerate()
        {
            let mut track = Track::new(lib.clone(), gds_transform());
            let t = std::time::Instant::now();
            track.add(frame(&lib.kinds[ki], at, DVec2::new(1e5, -2e5), seed as u64), &p);
            let best = track.hyps[0].score();
            let truth = track.hyps.iter().position(|h| h.kind == ki && (h.at - at).length() < 400.);
            let c = consensus(&lib, &track.hyps).unwrap();
            println!(
                "{} {at:?}: truth rank {truth:?}, {:.1} s; agree {:.2}, dist {:.1}-{:.1} um; top {:?}",
                lib.kinds[ki].name,
                t.elapsed().as_secs_f64(),
                c.agree,
                c.dist_min * 1e-3,
                c.dist_max * 1e-3,
                track.hyps.iter().take(4).map(|h| (h.kind, h.at.round(), (h.score() * 1000.).round() / 1000.)).collect::<Vec<_>>()
            );
            let truth = truth.expect("true place among the hypotheses");
            assert!(track.hyps[truth].score() > best - 0.1, "{:?}", track.hyps[truth]);
            assert!(angle_deg(c.dir_local, way(&lib, ki, at)) < AGREE_DEG, "{c:?}");
            assert!(c.agree > min_agree, "{c:?}");
        }
    }

    /// A move along the rings turns the bands one way for the true place
    /// and the other way for its mirror image across the centre: after it,
    /// only hypotheses heading the right way are left near the top.
    #[test]
    fn a_move_along_the_rings_rules_out_the_mirror() {
        let lib = Arc::new(library());
        let p = params(&MatchParams::default());
        let ki = kind_of(&lib, "full");
        let kind = &lib.kinds[ki];
        let g = gds_transform();
        // A 3 µm move (world) along the rings that ended 15 % short.
        let at1 = DVec2::new(13_200., 1000.);
        let move_world = DVec2::new(0., 3000.);
        let at2 = at1 + g.matrix2.inverse() * move_world * 0.85;
        let c1 = DVec2::new(2e5, 1e5);
        let mut track = Track::new(lib.clone(), g);
        track.add(frame(kind, at1, c1, 3), &p);
        track.add(frame(kind, at2, c1 + move_world, 4), &p);
        let best = track.hyps[0].score();
        for h in track.hyps.iter().take(6) {
            println!("{} {:?} {:?}", h.kind, h.at.round(), h.scores);
        }
        let truth = track.hyps.iter().find(|h| h.kind == ki && (h.at - at2).length() < 400.).expect("truth kept");
        assert_eq!(truth.scores.len(), 2);
        assert!(truth.score() > best - 0.1, "{truth:?}");
        for h in &track.hyps {
            if angle_deg(way(&lib, h.kind, h.at), way(&lib, ki, at2)) > 90. {
                assert!(h.score() < best - 0.05, "wrong way still fits: {h:?}");
            }
        }
        let c = consensus(&lib, &track.hyps).unwrap();
        assert!(c.agree > 0.9, "{c:?}");
        // On the map: the true hypothesis heads for the ring centre.
        let loc = track.locate(truth, g).unwrap();
        let centre = c1 + move_world - g.matrix2 * at2;
        assert!((loc.target_world - centre).length() < 400., "{loc:?} vs {centre:?}");
    }

    #[test]
    fn gds_shift_lines_up_the_nearest_copy() {
        let lib = Arc::new(library());
        let ki = kind_of(&lib, "full");
        let kind = &lib.kinds[ki];
        // The GDS on the map is 5 µm off from where the frame really is.
        let at = DVec2::new(9500., 3000.);
        let inst = kind.instances[1];
        let truth = DAffine2::from_scale(DVec2::splat(0.75));
        let world = truth.transform_point2(inst + at);
        let placed = DAffine2::from_translation(DVec2::new(4000., -3000.)) * truth;
        let mut track = Track::new(lib.clone(), placed);
        track.add(frame(kind, at, world, 9), &params(&MatchParams::default()));
        let h = track.hyps.iter().find(|h| h.kind == ki && (h.at - at).length() < 400.).unwrap();
        let loc = track.locate(h, placed).unwrap();
        assert!((loc.gds_shift - DVec2::new(-4000., 3000.)).length() < 300., "{loc:?}");
    }
}
#[cfg(test)]
mod real {
    use super::*;
    use crate::analysis::{
        gds_prior::{find_ring_systems, ring_kinds},
        ring_match::blind_match,
        ring_training::{Dataset, LabelKind, TuneParams},
    };

    fn angle_deg(a: DVec2, b: DVec2) -> f64 {
        a.normalize().dot(b.normalize()).clamp(-1., 1.).acos().to_degrees()
    }

    /// Landing on the labelled frames (one frame each, no moves), next to
    /// the blind ring match. Read-only:
    /// `MODE=soft|hard FLIP=1 cargo test --release --bin scan_control landing_report -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn landing_report() {
        let scale: f64 = std::env::var("GDS_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.75);
        let flip = std::env::var("FLIP").is_ok_and(|v| v == "1");
        let mut p = TuneParams::load().matching;
        p.mode = match std::env::var("MODE").as_deref() {
            Ok("smooth") => MatchMode::Smooth,
            Ok("hard") => MatchMode::HardTwoTone,
            Ok("soft") => MatchMode::SoftTwoTone,
            _ => MatchMode::Smooth,
        };
        println!("matching {p:?}, GDS scale {scale}, flip {flip}");
        let t = std::time::Instant::now();
        let polys = crate::scan_view::gds_image::read_polys("As_Implanted_MLA150.GDS");
        let systems = find_ring_systems(&polys);
        let lib = Arc::new(Library::build(&polys, &systems));
        println!("library in {:?}:", t.elapsed());
        for k in &lib.kinds {
            println!(
                "  {}: {} copies, pitch {:.0} nm, band fraction {:.2}, map {} px",
                k.name, k.instances.len(), k.pitch, k.band_fraction, k.map.n
            );
        }
        let kinds_nm = ring_kinds(&systems, scale);
        let gds = DAffine2::from_scale(DVec2::new(scale, if flip { -scale } else { scale }));
        let samples = Dataset::load(Dataset::default_dir()).samples();
        let (mut n, mut cons_ok, mut best_ok, mut blind_ok, mut sure_n, mut sure_ok) = (0, 0, 0, 0, 0, 0);
        let mut time = std::time::Duration::ZERO;
        for s in &samples {
            let r = &s.record;
            if !matches!(r.kind, LabelKind::Direction | LabelKind::Good) || r.dir().length() < 0.5 {
                continue;
            }
            let (w, h) = (r.w as f64, r.h as f64);
            let frame = DAffine2::from_scale(DVec2::new(w * r.nm_per_px, h * r.nm_per_px));
            let shot = Shot { data: Arc::new(s.data.clone().into_boxed_slice()), size: [r.w, r.h], frame };
            let t = std::time::Instant::now();
            let mut track = Track::new(lib.clone(), gds);
            track.add(shot, &p);
            time += t.elapsed();
            let label = r.dir();
            // World and frame pixel axes are the same here (square pixels).
            let to_world = |d: DVec2| gds.matrix2 * d;
            let c = consensus(&lib, &track.hyps);
            let best = track.hyps.first();
            let best_dir = best.map(|h| to_world(lib.kinds[h.kind].target(h.at) - h.at));
            let c_ok = c.is_some_and(|c| angle_deg(to_world(c.dir_local), label) < AGREE_DEG);
            let b_ok = best_dir.is_some_and(|d| angle_deg(d, label) < AGREE_DEG);
            let kinds_px: Vec<Vec<(f64, f64)>> =
                kinds_nm.iter().map(|k| k.iter().map(|b| (b.0 / r.nm_per_px, b.1 / r.nm_per_px)).collect()).collect();
            let blind = blind_match(&s.data, [r.w, r.h], &kinds_px, &p).filter(|m| m.1.score >= p.min_score);
            let bl_ok = blind.is_some_and(|(_, m)| angle_deg(m.center_px - DVec2::new(w / 2. - 0.5, h / 2. - 0.5), label) < AGREE_DEG);
            n += 1;
            cons_ok += c_ok as usize;
            best_ok += b_ok as usize;
            blind_ok += bl_ok as usize;
            let sure = c.is_some_and(|c| c.agree >= 0.8);
            sure_n += sure as usize;
            sure_ok += (sure && c_ok) as usize;
            let tag = |b: bool| if b { "right" } else { "WRONG" };
            println!(
                "{:16} {:9} agree {:.2} dist {:>4.1}-{:<4.1} um | consensus {} best {} ({} {:.3}) | blind {}",
                r.source,
                format!("{:?}", r.kind),
                c.map_or(0., |c| c.agree),
                c.map_or(0., |c| c.dist_min * scale * 1e-3),
                c.map_or(0., |c| c.dist_max * scale * 1e-3),
                tag(c_ok),
                tag(b_ok),
                best.map_or("-".into(), |h| lib.kinds[h.kind].name.split(' ').next().unwrap_or("").to_string()),
                best.map_or(0., |h| h.score()),
                if blind.is_some() { tag(bl_ok) } else { "-" },
            );
        }
        println!("{n} frames, {:?} each", time / n.max(1) as u32);
        println!("landing consensus right {cons_ok}/{n}, best hypothesis right {best_ok}/{n}, blind match right {blind_ok}/{n}");
        println!("landing sure (agree >= 0.8): {sure_ok}/{sure_n} right");
    }
}
