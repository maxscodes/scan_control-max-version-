//! Polishing several stamps (e.g. the tiles of a survey grid) into one clean
//! picture: each tile is cleaned up on its own (background, line fixes,
//! filters, ragged edges trimmed), the tiles' levels and tilts are matched
//! where they overlap so the seams disappear, the tiles are blended into one
//! mosaic, large hills can be flattened, and one shared contrast stretch and
//! colour scale are applied to the whole picture.
//!
//! Nothing here changes the stamps: everything works on copies.

use std::sync::{Arc, OnceLock};

use glam::{DAffine2, DVec2};
use nalgebra::{DMatrix, DVector};
use uuid::Uuid;

use crate::{
    connection::FrameData,
    scan_view::processing::{self, Background, ColorMap, Filters},
};

/// A stamp to polish, copied from the map.
#[derive(Clone)]
pub struct Source {
    pub id: Uuid,
    pub name: String,
    /// Map transform of the image (local −0.5..0.5 → world nm).
    pub transform: DAffine2,
    pub forward: FrameData,
    pub backward: FrameData,
}

/// Which recorded scan direction to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Forward,
    Backward,
    /// Mean of forward and backward: less noise.
    Average,
}
impl Channel {
    pub const ALL: [Channel; 3] = [Channel::Forward, Channel::Backward, Channel::Average];
    pub fn label(self) -> &'static str {
        match self {
            Channel::Forward => "Forward",
            Channel::Backward => "Backward",
            Channel::Average => "Average fwd + bwd",
        }
    }
}

/// How tiles are made to agree where they overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelMatch {
    Off,
    /// One height offset per tile.
    Offset,
    /// Offset plus a tilt per tile.
    Tilt,
    /// Offset, tilt and a curved bow (piezo creep, scanner bow) per tile.
    Bow,
}
impl LevelMatch {
    pub const ALL: [LevelMatch; 4] = [LevelMatch::Off, LevelMatch::Offset, LevelMatch::Tilt, LevelMatch::Bow];
    pub fn label(self) -> &'static str {
        match self {
            LevelMatch::Off => "Off",
            LevelMatch::Offset => "Height",
            LevelMatch::Tilt => "Height + tilt",
            LevelMatch::Bow => "Height + tilt + bow",
        }
    }
}

/// How overlapping tiles are joined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blend {
    /// The topmost tile wins, as on the map.
    Topmost,
    /// Overlaps fade from one tile into the other.
    Feather,
}

/// What empty parts of the picture look like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    Transparent,
    Black,
    White,
}
impl Fill {
    pub const ALL: [Fill; 3] = [Fill::Transparent, Fill::Black, Fill::White];
    pub fn label(self) -> &'static str {
        match self {
            Fill::Transparent => "Transparent",
            Fill::Black => "Black",
            Fill::White => "White",
        }
    }
}

/// Per-tile clean-up and matching. The filters are cached, so changing the
/// matching or a tile's hand edits doesn't redo them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileSettings {
    pub channel: Channel,
    pub background: Background,
    pub filters: Filters,
    /// Share of each tile cut away at every edge (ragged first lines, edge
    /// artefacts), 0..=0.25.
    pub trim: f32,
    pub level: LevelMatch,
    /// Tiles this close (share of a tile) without overlapping are still
    /// matched, edge strip to edge strip, 0..=0.2.
    pub bridge: f32,
    /// Also match each tile's contrast (a gain) to its neighbours, for tiles
    /// scanned with a sharper or blunter tip.
    pub match_contrast: bool,
    /// Heal every seam: each tile edge is bent half way to meet its
    /// neighbour, fading back to unchanged deeper in.
    pub heal_all: bool,
    /// How far into a tile a healed seam's correction reaches, share of the
    /// tile, 0.05..=0.5.
    pub heal_reach: f32,
}
impl Default for TileSettings {
    fn default() -> Self {
        Self {
            channel: Channel::Forward,
            background: Background::LineLinear,
            filters: Filters::default(),
            trim: 0.,
            level: LevelMatch::Tilt,
            bridge: 0.05,
            match_contrast: false,
            heal_all: false,
            heal_reach: 0.3,
        }
    }
}

/// Hand adjustments to one tile, made by eye. Edges are the tile's own:
/// 0 local −x, 1 +x, 2 −y, 3 +y (see [`edge_name`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TileEdit {
    pub hidden: bool,
    /// Drawn above the other tiles.
    pub on_top: bool,
    /// Move on the map, nm.
    pub shift: DVec2,
    /// Share of the tile cut off at each edge.
    pub crop: [f32; 4],
    /// Height added, data units.
    pub offset: f32,
    /// Contrast multiplier.
    pub gain: f32,
    /// Slope added along local x and y, data units across the tile.
    pub tilt: [f32; 2],
    /// Edges bent to meet their neighbour (seams clicked on).
    pub heals: Vec<u8>,
}
impl Default for TileEdit {
    fn default() -> Self {
        Self { hidden: false, on_top: false, shift: DVec2::ZERO, crop: [0.; 4], offset: 0., gain: 1., tilt: [0.; 2], heals: Vec::new() }
    }
}

/// Where a source tile is drawn, with its hand-made shift.
pub fn edit_transform(src: &Source, e: Option<&TileEdit>) -> DAffine2 {
    DAffine2::from_translation(e.map_or(DVec2::ZERO, |e| e.shift)) * src.transform
}

const EDGE_NORMALS: [DVec2; 4] = [DVec2::new(-1., 0.), DVec2::new(1., 0.), DVec2::new(0., -1.), DVec2::new(0., 1.)];

/// Which way edge `e` of a frame faces on the map, as a word.
pub fn edge_name(transform: &DAffine2, e: u8) -> &'static str {
    let n = transform.matrix2 * EDGE_NORMALS[e as usize];
    if n.x.abs() >= n.y.abs() {
        if n.x > 0. { "right" } else { "left" }
    } else if n.y > 0. {
        "top"
    } else {
        "bottom"
    }
}

/// One cleaned tile.
pub struct Tile {
    /// Index of the source it came from.
    pub src: usize,
    transform: DAffine2,
    inv: DAffine2,
    w: usize,
    h: usize,
    data: Vec<f32>,
    /// How far the kept data reaches towards each edge (local units, 0.5 =
    /// the whole tile), after trimming and cropping.
    inner: [f64; 4],
}
impl Tile {
    fn local_of_pixel(&self, i: usize, j: usize) -> DVec2 {
        DVec2::new((i as f64 + 0.5) / self.w as f64 - 0.5, (j as f64 + 0.5) / self.h as f64 - 0.5)
    }
    /// Nearest pixel at local `l` (inside the tile), NaN if missing.
    fn nearest(&self, l: DVec2) -> f32 {
        let i = (((l.x + 0.5) * self.w as f64) as usize).min(self.w - 1);
        let j = (((l.y + 0.5) * self.h as f64) as usize).min(self.h - 1);
        self.data[j * self.w + i]
    }
    fn bilinear(&self, l: DVec2) -> f32 {
        bilinear(&self.data, self.w, self.h, l)
    }
    fn corners(&self) -> [DVec2; 4] {
        [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
            .map(|(x, y)| self.transform.transform_point2(DVec2::new(x, y)))
    }
    /// For edge `e`: how far `l` is inside the kept data from that edge,
    /// and where along the edge it is.
    fn edge_coords(&self, e: u8, l: DVec2) -> (f64, f64) {
        match e {
            0 => (l.x + self.inner[0], l.y),
            1 => (self.inner[1] - l.x, l.y),
            2 => (l.y + self.inner[2], l.x),
            _ => (self.inner[3] - l.y, l.x),
        }
    }
    /// Local point `t` inside edge `e`, at `s` along it.
    fn edge_point(&self, e: u8, s: f64, t: f64) -> DVec2 {
        match e {
            0 => DVec2::new(-self.inner[0] + t, s),
            1 => DVec2::new(self.inner[1] - t, s),
            2 => DVec2::new(s, -self.inner[2] + t),
            _ => DVec2::new(s, self.inner[3] - t),
        }
    }
    /// Extent of edge `e` (along-edge coordinates).
    fn edge_span(&self, e: u8) -> (f64, f64) {
        if e < 2 {
            (-self.inner[2], self.inner[3])
        } else {
            (-self.inner[0], self.inner[1])
        }
    }
    /// Distance from the nearest edge of the kept data, local units.
    fn edge_distance(&self, l: DVec2) -> f64 {
        (0..4).map(|e| self.edge_coords(e, l).0).fold(f64::INFINITY, f64::min)
    }
}

/// Bilinear sample at local `l`, ignoring missing pixels.
fn bilinear(data: &[f32], w: usize, h: usize, l: DVec2) -> f32 {
    let fx = (l.x + 0.5) * w as f64 - 0.5;
    let fy = (l.y + 0.5) * h as f64 - 0.5;
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = ((fx - x0) as f32, (fy - y0) as f32);
    let (mut s, mut ws) = (0f32, 0f32);
    for (dx, dy, wt) in [
        (0, 0, (1. - tx) * (1. - ty)),
        (1, 0, tx * (1. - ty)),
        (0, 1, (1. - tx) * ty),
        (1, 1, tx * ty),
    ] {
        let (x, y) = (x0 as i64 + dx, y0 as i64 + dy);
        let x = x.clamp(0, w as i64 - 1) as usize;
        let y = y.clamp(0, h as i64 - 1) as usize;
        let v = data[y * w + x];
        if v.is_finite() && wt > 0. {
            s += wt * v;
            ws += wt;
        }
    }
    if ws > 0.25 {
        s / ws
    } else {
        f32::NAN
    }
}

/// The tiles cleaned up and matched, ready to be drawn (topmost first).
pub struct Prepared {
    pub tiles: Vec<Tile>,
    /// Overlaps used to match the tiles.
    pub pairs: usize,
    /// Finest pixel size among the tiles, nm.
    pub finest_px: f64,
    /// Map area covered: lower-left and upper-right corners, nm.
    pub lo: DVec2,
    pub hi: DVec2,
}

type CleanKey = (usize, usize, Channel, Background, Filters);

/// Tiles after the background and filters, before anything else: the slow
/// part, kept between previews.
#[derive(Default)]
pub struct Cleaned {
    key: Option<CleanKey>,
    tiles: Vec<Option<(Arc<Vec<f32>>, usize, usize)>>,
}

/// One source's data after the channel choice, background and filters
/// (for showing it on its own, e.g. when checking a seam by eye).
pub fn clean_one(src: &Source, s: &TileSettings) -> Option<(Vec<f32>, usize, usize)> {
    let (raw, w, h) = channel_data(src, s.channel)?;
    let mut d = processing::process(&raw, [w as u32, h as u32], s.background, s.filters).unwrap_or(raw);
    remove_background(&mut d, w, h, s.background);
    Some((d, w, h))
}

/// Clean up every tile and match them, then apply the hand edits (`edits[i]`
/// for `sources[i]`). `None` if there's nothing to draw or `cancelled` says
/// to stop.
pub fn prepare(
    sources: &[Source],
    s: &TileSettings,
    edits: &[TileEdit],
    cache: &mut Cleaned,
    cancelled: &dyn Fn() -> bool,
) -> Option<Prepared> {
    let key = (sources.as_ptr() as usize, sources.len(), s.channel, s.background, s.filters);
    if cache.key != Some(key) {
        let mut tiles = Vec::with_capacity(sources.len());
        for src in sources {
            if cancelled() {
                return None;
            }
            tiles.push(clean_one(src, s).map(|(d, w, h)| (Arc::new(d), w, h)));
        }
        *cache = Cleaned { key: Some(key), tiles };
    }
    let none = TileEdit::default();
    let edit = |i: usize| edits.get(i).unwrap_or(&none);
    // Tiles put on top first, otherwise as on the map.
    let mut order: Vec<usize> = (0..sources.len()).collect();
    order.sort_by_key(|&i| !edit(i).on_top);
    let trim = s.trim.clamp(0., 0.25) as f64;
    let mut tiles = Vec::new();
    for i in order {
        let e = edit(i);
        if e.hidden {
            continue;
        }
        let Some((d, w, h)) = &cache.tiles[i] else { continue };
        let tf = edit_transform(&sources[i], Some(e));
        if tf.matrix2.determinant().abs() <= 0. {
            continue;
        }
        let inner: [f64; 4] = std::array::from_fn(|k| (0.5 - trim - e.crop[k].clamp(0., 0.45) as f64).max(0.02));
        let mut t = Tile { src: i, transform: tf, inv: tf.inverse(), w: *w, h: *h, data: d.to_vec(), inner };
        if inner.iter().any(|&x| x < 0.5) {
            for j in 0..t.h {
                for i in 0..t.w {
                    let l = t.local_of_pixel(i, j);
                    if l.x < -inner[0] || l.x > inner[1] || l.y < -inner[2] || l.y > inner[3] {
                        t.data[j * t.w + i] = f32::NAN;
                    }
                }
            }
        }
        if let Some(m) = median_of(&t.data) {
            t.data.iter_mut().for_each(|v| *v -= m);
        }
        tiles.push(t);
    }
    if tiles.is_empty() {
        return None;
    }
    let bridge = s.bridge.clamp(0., 0.2) as f64;
    if s.match_contrast {
        match_gains(&mut tiles, bridge);
    }
    let pairs = match s.level {
        LevelMatch::Off => 0,
        mode => match_levels(&mut tiles, mode, bridge, cancelled)?,
    };
    // Hand adjustments.
    for t in &mut tiles {
        let e = edit(t.src);
        if (e.gain, e.offset, e.tilt) != (1., 0., [0., 0.]) {
            for j in 0..t.h {
                for i in 0..t.w {
                    let l = t.local_of_pixel(i, j);
                    let v = &mut t.data[j * t.w + i];
                    *v = *v * e.gain + e.offset + e.tilt[0] * l.x as f32 + e.tilt[1] * l.y as f32;
                }
            }
        }
    }
    if cancelled() {
        return None;
    }
    let reach = s.heal_reach.clamp(0.05, 0.5) as f64;
    if s.heal_all {
        // All profiles from the same state, then each side goes half way.
        let profiles: Vec<(usize, u8, Vec<f32>)> = (0..tiles.len())
            .flat_map(|a| (0..4u8).map(move |e| (a, e)))
            .filter_map(|(a, e)| edge_profile(&tiles, a, e).map(|p| (a, e, p)))
            .collect();
        for (a, e, p) in profiles {
            apply_heal(&mut tiles[a], e, &p, 0.5, reach);
        }
    }
    for a in 0..tiles.len() {
        for &e in &edit(tiles[a].src).heals {
            if let Some(p) = edge_profile(&tiles, a, e) {
                apply_heal(&mut tiles[a], e, &p, 1., reach);
            }
        }
    }
    let (mut lo, mut hi) = (DVec2::INFINITY, DVec2::NEG_INFINITY);
    for t in &tiles {
        for c in t.corners() {
            lo = lo.min(c);
            hi = hi.max(c);
        }
    }
    let finest_px = tiles
        .iter()
        .map(|t| {
            let m = t.transform.matrix2;
            (m.x_axis.length() / t.w as f64).min(m.y_axis.length() / t.h as f64)
        })
        .fold(f64::INFINITY, f64::min);
    Some(Prepared { tiles, pairs, finest_px, lo, hi })
}

/// Segments along an edge for seam healing.
const HEAL_SEGMENTS: usize = 24;
/// Depth of the strip compared at a seam, local units.
const HEAL_DEPTH: f64 = 0.04;

/// What tile `a` must add along its edge `e` to meet whatever lies across
/// the seam (`HEAL_SEGMENTS` values along the edge), or `None` if nothing
/// is there.
fn edge_profile(tiles: &[Tile], a: usize, e: u8) -> Option<Vec<f32>> {
    let ta = &tiles[a];
    let other = |w: DVec2| -> f32 {
        for (k, t) in tiles.iter().enumerate() {
            if k == a {
                continue;
            }
            let l = t.inv.transform_point2(w);
            if l.x.abs() <= 0.5 && l.y.abs() <= 0.5 {
                let v = t.bilinear(l);
                if v.is_finite() {
                    return v;
                }
            }
        }
        f32::NAN
    };
    let (s0, s1) = ta.edge_span(e);
    let n = HEAL_SEGMENTS;
    let mut prof = vec![f32::NAN; n];
    for (seg, p) in prof.iter_mut().enumerate() {
        let mut diffs = Vec::new();
        for u in 0..4 {
            let s = s0 + (seg as f64 + (u as f64 + 0.5) / 4.) / n as f64 * (s1 - s0);
            for k in 0..5 {
                let t = (k as f64 + 0.5) / 5. * HEAL_DEPTH;
                let la = ta.edge_point(e, s, t);
                let va = ta.bilinear(la);
                if !va.is_finite() {
                    continue;
                }
                // Overlapping tile here, or else just across a small gap.
                let mut vb = other(ta.transform.transform_point2(la));
                for gap in [0.01, 0.03, 0.06] {
                    if vb.is_finite() {
                        break;
                    }
                    vb = other(ta.transform.transform_point2(ta.edge_point(e, s, -gap)));
                }
                if vb.is_finite() {
                    diffs.push(vb - va);
                }
            }
        }
        if diffs.len() >= 5 {
            let m = diffs.len() / 2;
            *p = *diffs.select_nth_unstable_by(m, f32::total_cmp).1;
        }
    }
    if prof.iter().filter(|v| v.is_finite()).count() < 3 {
        return None;
    }
    // Gaps along the edge take the nearest value, then a light smoothing.
    let known: Vec<usize> = (0..n).filter(|&i| prof[i].is_finite()).collect();
    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            if prof[i].is_finite() {
                prof[i]
            } else {
                prof[*known.iter().min_by_key(|&&k| k.abs_diff(i)).unwrap()]
            }
        })
        .collect();
    for _ in 0..2 {
        let prev = out.clone();
        for i in 0..n {
            let (l, r) = (prev[i.saturating_sub(1)], prev[(i + 1).min(n - 1)]);
            out[i] = 0.25 * l + 0.5 * prev[i] + 0.25 * r;
        }
    }
    Some(out)
}

/// Add `factor` × `profile` at edge `e` of `t`, fading to nothing `reach`
/// into the tile.
fn apply_heal(t: &mut Tile, e: u8, profile: &[f32], factor: f32, reach: f64) {
    let (s0, s1) = t.edge_span(e);
    let n = profile.len();
    for j in 0..t.h {
        for i in 0..t.w {
            let l = t.local_of_pixel(i, j);
            let (depth, s) = t.edge_coords(e, l);
            if !(0. ..reach).contains(&depth) {
                continue;
            }
            let x = (depth / reach) as f32;
            let fade = 1. - x * x * (3. - 2. * x);
            let pos = (((s - s0) / (s1 - s0)) as f32 * n as f32 - 0.5).clamp(0., (n - 1) as f32);
            let k = pos.floor() as usize;
            let f = pos - k as f32;
            let p = profile[k] * (1. - f) + profile[(k + 1).min(n - 1)] * f;
            let v = &mut t.data[j * t.w + i];
            if v.is_finite() {
                *v += factor * fade * p;
            }
        }
    }
}

fn channel_data(src: &Source, ch: Channel) -> Option<(Vec<f32>, usize, usize)> {
    let ok = |d: &FrameData| {
        d.size[0] > 0 && d.size[1] > 0 && d.data.len() == (d.size[0] * d.size[1]) as usize
    };
    let (f, b) = (&src.forward, &src.backward);
    let pick = |d: &FrameData| Some((d.data.to_vec(), d.size[0] as usize, d.size[1] as usize));
    match ch {
        Channel::Forward if ok(f) => pick(f),
        Channel::Backward if ok(b) => pick(b),
        Channel::Average if ok(f) && ok(b) && f.size == b.size => {
            let d = f
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(&a, &b)| match (a.is_finite(), b.is_finite()) {
                    (true, true) => 0.5 * (a + b),
                    (true, false) => a,
                    (false, true) => b,
                    _ => f32::NAN,
                })
                .collect();
            Some((d, f.size[0] as usize, f.size[1] as usize))
        }
        _ if ok(f) => pick(f),
        _ if ok(b) => pick(b),
        _ => None,
    }
}

/// The background fits the map does on the GPU, done here on the CPU (the
/// parabola fits are already done by [`processing::process`]).
fn remove_background(d: &mut [f32], w: usize, h: usize, bg: Background) {
    match bg {
        Background::Raw | Background::Parabola | Background::LineParabola => {}
        Background::LineMean => {
            for row in d.chunks_mut(w) {
                let (s, n) = row.iter().filter(|v| v.is_finite()).fold((0f64, 0usize), |(s, n), &v| (s + v as f64, n + 1));
                if n > 0 {
                    let m = (s / n as f64) as f32;
                    row.iter_mut().for_each(|v| *v -= m);
                }
            }
        }
        Background::LineLinear => {
            for row in d.chunks_mut(w) {
                let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
                for (i, &v) in row.iter().enumerate() {
                    if v.is_finite() {
                        let (x, y) = (i as f64, v as f64);
                        n += 1.;
                        sx += x;
                        sy += y;
                        sxx += x * x;
                        sxy += x * y;
                    }
                }
                if n < 2. {
                    continue;
                }
                let den = n * sxx - sx * sx;
                let slope = if den.abs() > 0. { (n * sxy - sx * sy) / den } else { 0. };
                let (mean, xm) = (sy / n, sx / n);
                for (i, v) in row.iter_mut().enumerate() {
                    *v = (*v as f64 - mean - slope * (i as f64 - xm)) as f32;
                }
            }
        }
        Background::Plane => {
            let mut ata = nalgebra::Matrix3::<f64>::zeros();
            let mut atb = nalgebra::Vector3::<f64>::zeros();
            let at = |i: usize, j: usize| {
                nalgebra::Vector3::new(1., i as f64 / w as f64 - 0.5, j as f64 / h as f64 - 0.5)
            };
            for j in 0..h {
                for i in 0..w {
                    let v = d[j * w + i];
                    if v.is_finite() {
                        let a = at(i, j);
                        ata += a * a.transpose();
                        atb += a * v as f64;
                    }
                }
            }
            if let Some(c) = ata.cholesky().map(|c| c.solve(&atb)) {
                for j in 0..h {
                    for i in 0..w {
                        d[j * w + i] -= c.dot(&at(i, j)) as f32;
                    }
                }
            }
        }
    }
}

/// Median of the finite values (of a sample of at most ~100k of them).
fn median_of(d: &[f32]) -> Option<f32> {
    let step = (d.len() / 100_000).max(1);
    let mut v: Vec<f32> = d.iter().step_by(step).copied().filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    let k = v.len() / 2;
    Some(*v.select_nth_unstable_by(k, f32::total_cmp).1)
}

fn bounding_box(t: &Tile) -> (DVec2, DVec2) {
    let c = t.corners();
    (c.iter().fold(DVec2::INFINITY, |a, b| a.min(*b)), c.iter().fold(DVec2::NEG_INFINITY, |a, b| a.max(*b)))
}

/// Overlapping, or apart by less than `bridge` (share of a tile).
fn near(a: (DVec2, DVec2), b: (DVec2, DVec2), bridge: f64) -> bool {
    let ((la, ha), (lb, hb)) = (a, b);
    let gap = bridge * (ha - la).max(hb - lb).max_element();
    !(la.x >= hb.x + gap || lb.x >= ha.x + gap || la.y >= hb.y + gap || lb.y >= ha.y + gap)
}

/// Pairs of values where `ta` and `tb` show the same place (or, across a
/// gap of up to `bridge`, `ta`'s edge strip against `tb`'s nearest edge),
/// on a grid of at most ~80×80 of `ta`'s pixels: (a, b, local in a, local
/// in b).
fn overlap_samples(ta: &Tile, tb: &Tile, bridge: f64) -> Vec<(f32, f32, DVec2, DVec2)> {
    let step_i = (ta.w / 80).max(1);
    let step_j = (ta.h / 80).max(1);
    let mut out = Vec::new();
    for j in (0..ta.h).step_by(step_j) {
        for i in (0..ta.w).step_by(step_i) {
            let va = ta.data[j * ta.w + i];
            if !va.is_finite() {
                continue;
            }
            let l_a = ta.local_of_pixel(i, j);
            let mut l_b = tb.inv.transform_point2(ta.transform.transform_point2(l_a));
            if l_b.x.abs() >= 0.5 || l_b.y.abs() >= 0.5 {
                let out_by = (l_b.x.abs() - 0.5).max(l_b.y.abs() - 0.5);
                // The strip is `bridge` wide and the gap up to `bridge`.
                if bridge <= 0. || ta.edge_distance(l_a) > bridge || out_by > 2. * bridge {
                    continue;
                }
                l_b = l_b.clamp(DVec2::splat(-0.4999), DVec2::splat(0.4999));
            }
            let vb = tb.nearest(l_b);
            if vb.is_finite() {
                out.push((va, vb, l_a, l_b));
            }
        }
    }
    out
}

/// Per-tile gains (contrast) that make overlapping tiles equally contrasty:
/// the spread of each tile's values in a shared area is compared, and the
/// log ratios fitted together.
fn match_gains(tiles: &mut [Tile], bridge: f64) {
    let n = tiles.len();
    let boxes: Vec<_> = tiles.iter().map(bounding_box).collect();
    let mut ata = DMatrix::<f64>::zeros(n, n);
    let mut atb = DVector::<f64>::zeros(n);
    let mut any = false;
    let mad = |mut v: Vec<f32>| -> f32 {
        let m = v.len() / 2;
        let med = *v.select_nth_unstable_by(m, f32::total_cmp).1;
        let mut d: Vec<f32> = v.iter().map(|x| (x - med).abs()).collect();
        *d.select_nth_unstable_by(m, f32::total_cmp).1
    };
    for a in 0..n {
        for b in a + 1..n {
            if !near(boxes[a], boxes[b], bridge) {
                continue;
            }
            let smp = overlap_samples(&tiles[a], &tiles[b], bridge);
            if smp.len() < 30 {
                continue;
            }
            let ma = mad(smp.iter().map(|s| s.0).collect());
            let mb = mad(smp.iter().map(|s| s.1).collect());
            if !(ma > 0. && mb > 0.) {
                continue;
            }
            // g_a − g_b = −ln(ma / mb)
            let r = -((ma / mb) as f64).ln();
            let wt = (smp.len() as f64).sqrt();
            ata[(a, a)] += wt;
            ata[(b, b)] += wt;
            ata[(a, b)] -= wt;
            ata[(b, a)] -= wt;
            atb[a] += wt * r;
            atb[b] -= wt * r;
            any = true;
        }
    }
    if !any {
        return;
    }
    let mean_diag = (0..n).map(|i| ata[(i, i)]).sum::<f64>() / n as f64;
    for i in 0..n {
        ata[(i, i)] += 1e-2 * mean_diag.max(1e-9);
    }
    let Some(ch) = ata.cholesky() else { return };
    let x = ch.solve(&atb);
    for (t, tile) in tiles.iter_mut().enumerate() {
        let g = x[t].exp().clamp(0.2, 5.) as f32;
        tile.data.iter_mut().for_each(|v| *v *= g);
    }
}

/// Least squares of per-tile corrections (offset, and with `Tilt`/`Bow`
/// slopes and curves along local x and y) that make overlapping tiles
/// agree. Each overlap is split into patches and each patch gives the
/// median difference, so a few odd pixels (a tip change, a spike) don't
/// pull the fit. Returns the number of overlaps used.
fn match_levels(tiles: &mut [Tile], mode: LevelMatch, bridge: f64, cancelled: &dyn Fn() -> bool) -> Option<usize> {
    let n = tiles.len();
    let (k, bins) = match mode {
        LevelMatch::Off | LevelMatch::Offset => (1, 1),
        LevelMatch::Tilt => (3, 3),
        LevelMatch::Bow => (6, 4),
    };
    let mut ata = DMatrix::<f64>::zeros(n * k, n * k);
    let mut atb = DVector::<f64>::zeros(n * k);
    let mut pairs = 0;
    let boxes: Vec<_> = tiles.iter().map(bounding_box).collect();
    let coeffs = |l: DVec2| -> [f64; 6] { [1., l.x, l.y, l.x * l.x - 1. / 12., l.y * l.y - 1. / 12., l.x * l.y] };
    for a in 0..n {
        if cancelled() {
            return None;
        }
        for b in a + 1..n {
            if !near(boxes[a], boxes[b], bridge) {
                continue;
            }
            let smp = overlap_samples(&tiles[a], &tiles[b], bridge);
            if smp.len() < 30 {
                continue;
            }
            let mut binned: Vec<Vec<(f32, DVec2, DVec2)>> = vec![Vec::new(); bins * bins];
            for (va, vb, l_a, l_b) in smp {
                let bx = (((l_a.x + 0.5) * bins as f64) as usize).min(bins - 1);
                let by = (((l_a.y + 0.5) * bins as f64) as usize).min(bins - 1);
                binned[by * bins + bx].push((va - vb, l_a, l_b));
            }
            pairs += 1;
            for bin in binned.iter_mut().filter(|b| b.len() >= 12) {
                let m = bin.len() / 2;
                let d = bin.select_nth_unstable_by(m, |x, y| x.0.total_cmp(&y.0)).1 .0 as f64;
                let la = bin.iter().map(|s| s.1).sum::<DVec2>() / bin.len() as f64;
                let lb = bin.iter().map(|s| s.2).sum::<DVec2>() / bin.len() as f64;
                let wt = (bin.len() as f64).sqrt();
                // (c_a + g_a·l_a) − (c_b + g_b·l_b) = −d
                let mut row: Vec<(usize, f64)> = Vec::with_capacity(2 * k);
                let (ca, cb) = (coeffs(la), coeffs(lb));
                for q in 0..k {
                    row.push((a * k + q, ca[q]));
                    row.push((b * k + q, -cb[q]));
                }
                for &(r, vr) in &row {
                    atb[r] += wt * vr * -d;
                    for &(c, vc) in &row {
                        ata[(r, c)] += wt * vr * vc;
                    }
                }
            }
        }
    }
    if pairs == 0 {
        return Some(0);
    }
    // A light pull of every correction towards none settles what the
    // overlaps can't (the whole mosaic's level and tilt, lone tiles).
    let mean_diag = (0..n * k).map(|i| ata[(i, i)]).sum::<f64>() / (n * k) as f64;
    for t in 0..n {
        for q in 0..k {
            let ridge = match q {
                0 => 1e-3,
                1 | 2 => 2e-2,
                _ => 5e-2,
            };
            ata[(t * k + q, t * k + q)] += ridge * mean_diag.max(1e-9);
        }
    }
    let x = ata.cholesky()?.solve(&atb);
    for (t, tile) in tiles.iter_mut().enumerate() {
        for j in 0..tile.h {
            for i in 0..tile.w {
                let cf = coeffs(tile.local_of_pixel(i, j));
                let corr: f64 = (0..k).map(|q| x[t * k + q] * cf[q]).sum();
                tile.data[j * tile.w + i] += corr as f32;
            }
        }
    }
    Some(pairs)
}

/// How the cleaned tiles become a picture (fast, redone on every change).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    pub blend: Blend,
    /// Width of the fade at tile edges, share of the tile, 0..=0.5.
    pub feather: f32,
    /// Two-band blending: the broad shading (hills, brightness) of the tiles
    /// fades over this much of a tile (share, 0..=0.5; 0 = off) while fine
    /// detail keeps the narrow `feather`, so tiles flow into each other
    /// without ghosting.
    pub seam_blend: f32,
    /// How much of the large-scale hills is removed, 0..=1.
    pub flatten: f32,
    /// Size of the hills removed, nm.
    pub flatten_nm: f64,
    /// Percent of values clipped to black / white.
    pub clip_low: f32,
    pub clip_high: f32,
    /// Contrast curve exponent: below 1 brightens the darks.
    pub gamma: f32,
    /// Mix of histogram equalisation, 0..=1.
    pub equalize: f32,
    /// Even out the contrast across the picture (areas of weak and strong
    /// contrast brought together, over `flatten_nm`), 0..=1.
    pub even: f32,
    pub color_map: ColorMap,
    pub invert: bool,
    pub fill: Fill,
    pub scale_bar: bool,
}
impl Default for Look {
    fn default() -> Self {
        Self {
            blend: Blend::Feather,
            feather: 0.1,
            seam_blend: 0.3,
            flatten: 0.,
            flatten_nm: 100.,
            clip_low: 0.5,
            clip_high: 0.5,
            gamma: 1.,
            equalize: 0.,
            even: 0.,
            color_map: ColorMap::Gray,
            invert: false,
            fill: Fill::Transparent,
            scale_bar: true,
        }
    }
}

/// The tiles drawn into one grid of heights (NaN where empty). Row 0 is the
/// top of the map (largest y).
pub struct Composite {
    pub w: usize,
    pub h: usize,
    /// Pixel size, nm.
    pub px: f64,
    pub data: Vec<f32>,
    /// Map position of the top-left corner of the picture, nm.
    pub top_left: DVec2,
}

/// Pixel size for a picture of `p` at most `max_side` pixels on a side.
pub fn pixel_size(p: &Prepared, max_side: u32) -> f64 {
    let span = p.hi - p.lo;
    let longest = (span.x.max(span.y) / p.finest_px).ceil();
    if longest > max_side as f64 {
        p.finest_px * longest / max_side as f64
    } else {
        p.finest_px
    }
}

pub fn compose(p: &Prepared, look: &Look, max_side: u32, cancelled: &dyn Fn() -> bool) -> Option<Composite> {
    let px = pixel_size(p, max_side);
    let span = p.hi - p.lo;
    let (w, h) = ((span.x / px).ceil().max(1.) as usize, (span.y / px).ceil().max(1.) as usize);
    let narrow = if look.blend == Blend::Feather { look.feather } else { 0. };
    let mut data = if look.seam_blend > 0. {
        // Two bands: each tile split into its broad shading (blurred) and
        // the detail on top; the shading fades over a wide zone, the detail
        // over the narrow one.
        let mut lows = Vec::with_capacity(p.tiles.len());
        let mut highs = Vec::with_capacity(p.tiles.len());
        for t in &p.tiles {
            if cancelled() {
                return None;
            }
            let sigma = 0.05 * t.w.max(t.h) as f32;
            let low = large_blur(&t.data, t.w, t.h, sigma.max(1.));
            let high: Vec<f32> = t.data.iter().zip(&low).map(|(v, l)| v - l).collect();
            lows.push(low);
            highs.push(high);
        }
        let wide = look.seam_blend.max(narrow);
        let low = draw(p, &lows, wide, px, w, h, cancelled)?;
        let high = draw(p, &highs, narrow, px, w, h, cancelled)?;
        low.iter().zip(&high).map(|(l, h)| l + h).collect()
    } else {
        draw_tiles(p, narrow, px, w, h, cancelled)?
    };
    if look.flatten > 0. {
        let sigma = (look.flatten_nm / px) as f32;
        let bg = large_blur(&data, w, h, sigma.max(0.5));
        for (v, b) in data.iter_mut().zip(&bg) {
            if b.is_finite() {
                *v -= look.flatten * b;
            }
        }
    }
    if look.even > 0. {
        // Divide out the local spread (relative to the typical one).
        let sigma = ((look.flatten_nm / px) as f32).max(0.5);
        let mean = large_blur(&data, w, h, sigma);
        let sq: Vec<f32> = data.iter().zip(&mean).map(|(v, m)| (v - m) * (v - m)).collect();
        let var = large_blur(&sq, w, h, sigma);
        let sd: Vec<f32> = var.iter().map(|v| v.max(0.).sqrt()).collect();
        if let Some(typical) = median_of(&sd).filter(|t| *t > 0.) {
            for ((v, m), s) in data.iter_mut().zip(&mean).zip(&sd) {
                if m.is_finite() && *s > 0. {
                    *v = m + (*v - m) * (typical / s).powf(look.even).clamp(0.2, 5.);
                }
            }
        }
    }
    Some(Composite { w, h, px, data, top_left: DVec2::new(p.lo.x, p.hi.y) })
}

fn draw_tiles(p: &Prepared, feather: f32, px: f64, w: usize, h: usize, cancelled: &dyn Fn() -> bool) -> Option<Vec<f32>> {
    let datas: Vec<&[f32]> = p.tiles.iter().map(|t| t.data.as_slice()).collect();
    draw_slices(p, &datas, feather, px, w, h, cancelled)
}

fn draw(p: &Prepared, datas: &[Vec<f32>], feather: f32, px: f64, w: usize, h: usize, cancelled: &dyn Fn() -> bool) -> Option<Vec<f32>> {
    let datas: Vec<&[f32]> = datas.iter().map(|d| d.as_slice()).collect();
    draw_slices(p, &datas, feather, px, w, h, cancelled)
}

/// Draw one value per tile pixel (`datas[k]` laid out like tile `k`) into a
/// `w`×`h` grid. `feather` 0: the topmost tile wins; otherwise overlaps fade
/// over that share of a tile from its (trimmed) edges.
fn draw_slices(p: &Prepared, datas: &[&[f32]], feather: f32, px: f64, w: usize, h: usize, cancelled: &dyn Fn() -> bool) -> Option<Vec<f32>> {
    let fade = feather > 0.;
    let mut sum = vec![0f32; if fade { w * h } else { 0 }];
    let mut wsum = vec![0f32; if fade { w * h } else { 0 }];
    let mut out = vec![f32::NAN; if fade { 0 } else { w * h }];
    // Bottom tile first, so the topmost is drawn last.
    for (t, data) in p.tiles.iter().zip(datas).rev() {
        if cancelled() {
            return None;
        }
        let c = t.corners();
        let clo = c.iter().fold(DVec2::INFINITY, |a, b| a.min(*b));
        let chi = c.iter().fold(DVec2::NEG_INFINITY, |a, b| a.max(*b));
        let x0 = ((clo.x - p.lo.x) / px).floor().max(0.) as usize;
        let x1 = (((chi.x - p.lo.x) / px).ceil().max(0.) as usize).min(w);
        let y0 = ((p.hi.y - chi.y) / px).floor().max(0.) as usize;
        let y1 = (((p.hi.y - clo.y) / px).ceil().max(0.) as usize).min(h);
        for y in y0..y1 {
            for x in x0..x1 {
                let world = DVec2::new(p.lo.x + (x as f64 + 0.5) * px, p.hi.y - (y as f64 + 0.5) * px);
                let l = t.inv.transform_point2(world);
                if l.x.abs() > 0.5 || l.y.abs() > 0.5 {
                    continue;
                }
                let v = bilinear(data, t.w, t.h, l);
                if !v.is_finite() {
                    continue;
                }
                let i = y * w + x;
                if fade {
                    let dist = t.edge_distance(l) / feather as f64;
                    let s = dist.clamp(0., 1.) as f32;
                    let wt = (s * s * (3. - 2. * s)).max(1e-3);
                    sum[i] += wt * v;
                    wsum[i] += wt;
                } else {
                    out[i] = v;
                }
            }
        }
    }
    Some(if fade {
        sum.iter().zip(&wsum).map(|(s, w)| if *w > 0. { s / w } else { f32::NAN }).collect()
    } else {
        out
    })
}

/// Gaussian blur for large radii: done on a shrunk copy, then scaled back.
fn large_blur(data: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 6. {
        return processing::gaussian_blur(data, w, h, sigma);
    }
    let k = (sigma / 3.).ceil() as usize;
    let (cw, ch) = (w.div_ceil(k), h.div_ceil(k));
    let mut small = vec![f32::NAN; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut s, mut n) = (0f32, 0u32);
            for y in cy * k..((cy + 1) * k).min(h) {
                for x in cx * k..((cx + 1) * k).min(w) {
                    let v = data[y * w + x];
                    if v.is_finite() {
                        s += v;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                small[cy * cw + cx] = s / n as f32;
            }
        }
    }
    let small = processing::gaussian_blur(&small, cw, ch, sigma / k as f32);
    let mut out = vec![f32::NAN; w * h];
    for y in 0..h {
        let fy = (y as f32 + 0.5) / k as f32 - 0.5;
        let y0 = fy.floor();
        let ty = fy - y0;
        for x in 0..w {
            if !data[y * w + x].is_finite() {
                continue;
            }
            let fx = (x as f32 + 0.5) / k as f32 - 0.5;
            let x0 = fx.floor();
            let tx = fx - x0;
            let (mut s, mut ws) = (0f32, 0f32);
            for (dx, dy, wt) in [(0, 0, (1. - tx) * (1. - ty)), (1, 0, tx * (1. - ty)), (0, 1, (1. - tx) * ty), (1, 1, tx * ty)] {
                let sx = (x0 as i64 + dx).clamp(0, cw as i64 - 1) as usize;
                let sy = (y0 as i64 + dy).clamp(0, ch as i64 - 1) as usize;
                let v = small[sy * cw + sx];
                if v.is_finite() {
                    s += wt * v;
                    ws += wt;
                }
            }
            if ws > 0. {
                out[y * w + x] = s / ws;
            }
        }
    }
    out
}

/// Value distribution of the picture, for the stretch histogram.
#[derive(Clone, Default)]
pub struct Stats {
    pub hist: Vec<u32>,
    /// Values the histogram spans.
    pub range: (f32, f32),
    /// Values mapped to the ends of the colour scale.
    pub lo: f32,
    pub hi: f32,
}

/// Stretch the contrast and colour the picture.
pub fn colorize(c: &Composite, look: &Look) -> (image::RgbaImage, Stats) {
    let step = (c.data.len() / 1_000_000).max(1);
    let mut sample: Vec<f32> = c.data.iter().step_by(step).copied().filter(|v| v.is_finite()).collect();
    sample.sort_by(f32::total_cmp);
    let q = |f: f64| -> f32 {
        if sample.is_empty() {
            0.
        } else {
            sample[(((sample.len() - 1) as f64) * f.clamp(0., 1.)).round() as usize]
        }
    };
    let lo = q(look.clip_low as f64 / 100.);
    let hi = q(1. - look.clip_high as f64 / 100.).max(lo + f32::MIN_POSITIVE);
    let span = (hi - lo).max(f32::MIN_POSITIVE);
    // Histogram-equalisation curve between lo and hi.
    const EQ_BINS: usize = 1024;
    let mut cdf = vec![0f32; EQ_BINS];
    if look.equalize > 0. {
        for &v in &sample {
            let b = ((((v - lo) / span).clamp(0., 1.)) * (EQ_BINS - 1) as f32) as usize;
            cdf[b] += 1.;
        }
        let total: f32 = cdf.iter().sum::<f32>().max(1.);
        let mut acc = 0.;
        for x in cdf.iter_mut() {
            acc += *x;
            *x = acc / total;
        }
    }
    let table = look.color_map.table(look.invert);
    let empty = match look.fill {
        Fill::Transparent => image::Rgba([0, 0, 0, 0]),
        Fill::Black => image::Rgba([0, 0, 0, 255]),
        Fill::White => image::Rgba([255, 255, 255, 255]),
    };
    let gamma = look.gamma.clamp(0.1, 10.);
    let img = image::RgbaImage::from_fn(c.w as u32, c.h as u32, |x, y| {
        let v = c.data[y as usize * c.w + x as usize];
        if !v.is_finite() {
            return empty;
        }
        let mut t = ((v - lo) / span).clamp(0., 1.);
        if look.equalize > 0. {
            let e = cdf[(t * (EQ_BINS - 1) as f32) as usize];
            t = (1. - look.equalize) * t + look.equalize * e;
        }
        if gamma != 1. {
            t = t.powf(gamma);
        }
        let col = table[((t * (table.len() - 1) as f32).round() as usize).min(table.len() - 1)];
        image::Rgba([col.r(), col.g(), col.b(), 255])
    });
    // Histogram of the middle 99.8 % of values.
    let (r0, r1) = (q(0.001), q(0.999).max(q(0.001) + f32::MIN_POSITIVE));
    let mut hist = vec![0u32; 128];
    for &v in &sample {
        if v >= r0 && v <= r1 {
            let b = (((v - r0) / (r1 - r0)) * 127.) as usize;
            hist[b.min(127)] += 1;
        }
    }
    (img, Stats { hist, range: (r0, r1), lo, hi })
}

/// A round length (1, 2 or 5 × 10ⁿ nm) near `target` nm.
fn nice_length(target: f64) -> f64 {
    let p = 10f64.powf(target.log10().floor());
    [5., 2., 1.].into_iter().map(|m| m * p).find(|&l| l <= target).unwrap_or(p)
}

pub fn format_nm(nm: f64) -> String {
    if nm >= 1000. {
        let um = nm / 1000.;
        if um.fract() == 0. { format!("{um:.0} µm") } else { format!("{um} µm") }
    } else {
        format!("{nm:.0} nm")
    }
}

fn fonts() -> Arc<resvg::usvg::fontdb::Database> {
    static DB: OnceLock<Arc<resvg::usvg::fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = resvg::usvg::fontdb::Database::new();
        db.load_system_fonts();
        Arc::new(db)
    })
    .clone()
}

/// Draw a labelled scale bar in the bottom-right corner (pixel size `px`
/// nm). Its size follows the picture's, so a preview looks like the saved
/// image.
pub fn draw_scale_bar(img: &mut image::RgbaImage, px: f64) {
    use resvg::{tiny_skia, usvg};
    let (w, h) = (img.width() as f64, img.height() as f64);
    if w < 40. || h < 40. {
        return;
    }
    let len_nm = nice_length(0.2 * w * px);
    let bar = len_nm / px;
    let unit = (w.min(h) / 40.).max(4.);
    let font = (unit * 2.2).max(10.);
    let pad = unit;
    let (bw, bh) = (bar + 2. * pad, font + unit + 3. * pad);
    let label = format_nm(len_nm);
    let svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{bw}" height="{bh}">
<rect x="0" y="0" width="{bw}" height="{bh}" rx="{r}" fill="#000" fill-opacity="0.45"/>
<text x="{cx}" y="{ty}" font-family="Segoe UI, Arial, sans-serif" font-size="{font}" fill="#fff" text-anchor="middle">{label}</text>
<rect x="{pad}" y="{by}" width="{bar}" height="{unit}" fill="#fff"/>
</svg>"##,
        r = unit * 0.6,
        cx = bw / 2.,
        ty = pad + font * 0.85,
        by = 2. * pad + font,
    );
    let mut opt = usvg::Options::default();
    opt.fontdb = fonts();
    let Ok(tree) = usvg::Tree::from_str(&svg, &opt) else { return };
    let Some(mut pm) = tiny_skia::Pixmap::new(bw.ceil() as u32, bh.ceil() as u32) else { return };
    resvg::render(&tree, tiny_skia::Transform::identity(), &mut pm.as_mut());
    let margin = (unit * 1.5) as i64;
    let ox = img.width() as i64 - pm.width() as i64 - margin;
    let oy = img.height() as i64 - pm.height() as i64 - margin;
    for y in 0..pm.height() {
        for x in 0..pm.width() {
            let (ix, iy) = (ox + x as i64, oy + y as i64);
            if ix < 0 || iy < 0 || ix >= img.width() as i64 || iy >= img.height() as i64 {
                continue;
            }
            // Premultiplied source over the image.
            let s = pm.pixel(x, y).unwrap();
            let a = s.alpha() as f32 / 255.;
            if a <= 0. {
                continue;
            }
            let d = img.get_pixel_mut(ix as u32, iy as u32);
            let da = d[3] as f32 / 255.;
            let oa = a + da * (1. - a);
            let mix = |sc: u8, dc: u8| -> u8 {
                let c = sc as f32 / 255. + dc as f32 / 255. * da * (1. - a);
                ((c / oa.max(1e-6)) * 255.).round().clamp(0., 255.) as u8
            };
            *d = image::Rgba([mix(s.red(), d[0]), mix(s.green(), d[1]), mix(s.blue(), d[2]), (oa * 255.).round() as u8]);
        }
    }
}

/// The finished picture.
pub struct Rendered {
    pub image: image::RgbaImage,
    pub stats: Stats,
    /// Pixel size, nm.
    pub px: f64,
    /// Map position of the top-left corner, nm.
    pub top_left: DVec2,
}

/// Compose, stretch and colour; with `for_map` no scale bar is drawn and
/// empty parts stay transparent.
pub fn render(p: &Prepared, look: &Look, max_side: u32, for_map: bool, cancelled: &dyn Fn() -> bool) -> Option<Rendered> {
    let mut look = *look;
    if for_map {
        look.fill = Fill::Transparent;
        look.scale_bar = false;
    }
    let c = compose(p, &look, max_side, cancelled)?;
    let (mut image, stats) = colorize(&c, &look);
    if look.scale_bar {
        draw_scale_bar(&mut image, c.px);
    }
    Some(Rendered { image, stats, px: c.px, top_left: c.top_left })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(at: DVec2, offset: f32, tilt: f32) -> Source {
        let n = 20;
        // A shared surface (a slope across the map) seen by every tile, plus
        // the tile's own offset and tilt along local x.
        let data: Vec<f32> = (0..n * n)
            .map(|k| {
                let (i, j) = (k % n, k / n);
                let wx = at.x + (i as f64 + 0.5) * 5. - 50.;
                let _ = j;
                (0.01 * wx) as f32 + offset + tilt * (i as f32 / n as f32 - 0.5)
            })
            .collect();
        let fd = FrameData { size: [n as u32, n as u32], data: Arc::new(data.into_boxed_slice()) };
        Source {
            id: Uuid::new_v4(),
            name: String::new(),
            transform: DAffine2::from_scale_angle_translation(DVec2::new(100., -100.), 0., at),
            forward: fd.clone(),
            backward: fd,
        }
    }

    #[test]
    fn matched_tiles_agree_in_their_overlap() {
        // Two tiles overlapping by 30 nm, with different offsets and tilts.
        let srcs = [source(DVec2::ZERO, 0., 0.), source(DVec2::new(70., 0.), 3., 0.5)];
        let s = TileSettings { background: Background::Raw, level: LevelMatch::Tilt, ..Default::default() };
        let p = prepare(&srcs, &s, &[], &mut Cleaned::default(), &|| false).unwrap();
        assert_eq!(p.pairs, 1);
        // Compare the two tiles at a point in the overlap.
        let world = DVec2::new(35., 0.);
        let va = p.tiles[0].bilinear(p.tiles[0].inv.transform_point2(world));
        let vb = p.tiles[1].bilinear(p.tiles[1].inv.transform_point2(world));
        assert!((va - vb).abs() < 0.05, "{va} vs {vb}");
    }

    #[test]
    fn composes_at_the_finest_pixel_and_leaves_gaps_empty() {
        let srcs = [source(DVec2::ZERO, 0., 0.), source(DVec2::new(120., 0.), 0., 0.)];
        let p = prepare(&srcs, &TileSettings::default(), &[], &mut Cleaned::default(), &|| false).unwrap();
        let look = Look { blend: Blend::Topmost, scale_bar: false, ..Default::default() };
        let r = render(&p, &look, 4000, false, &|| false).unwrap();
        assert_eq!((r.image.width(), r.image.height()), (44, 20));
        assert_eq!(r.image.get_pixel(4, 10)[3], 255);
        assert_eq!(r.image.get_pixel(22, 10)[3], 0);
        let small = render(&p, &look, 22, false, &|| false).unwrap();
        assert_eq!((small.image.width(), small.image.height()), (22, 10));
    }

    #[test]
    fn tiles_with_a_small_gap_are_still_matched() {
        // 4 nm apart (4 % of a tile): matched only with the bridge on.
        let srcs = [source(DVec2::ZERO, 0., 0.), source(DVec2::new(104., 0.), 3., 0.)];
        let off = TileSettings { background: Background::Raw, level: LevelMatch::Offset, bridge: 0., ..Default::default() };
        assert_eq!(prepare(&srcs, &off, &[], &mut Cleaned::default(), &|| false).unwrap().pairs, 0);
        // Test tiles are only 20 px: a 10 % strip gives enough samples.
        let on = TileSettings { bridge: 0.1, ..off };
        let p = prepare(&srcs, &on, &[], &mut Cleaned::default(), &|| false).unwrap();
        assert_eq!(p.pairs, 1);
        let va = p.tiles[0].bilinear(DVec2::new(0.49, 0.));
        let vb = p.tiles[1].bilinear(DVec2::new(-0.49, 0.));
        assert!((va - vb).abs() < 0.2, "{va} vs {vb}");
    }

    #[test]
    fn healing_a_seam_closes_the_step() {
        // Overlapping tiles, the second 2 higher; no level matching.
        let srcs = [source(DVec2::ZERO, 0., 0.), source(DVec2::new(70., 0.), 2., 0.)];
        let s = TileSettings { background: Background::Raw, level: LevelMatch::Off, ..Default::default() };
        let step = |edits: &[TileEdit]| {
            let p = prepare(&srcs, &s, edits, &mut Cleaned::default(), &|| false).unwrap();
            // Tile 0 is on top; its right edge (local +x, edge 1) is the seam.
            let a = p.tiles.iter().find(|t| t.src == 0).unwrap();
            let b = p.tiles.iter().find(|t| t.src == 1).unwrap();
            let w = a.transform.transform_point2(DVec2::new(0.48, 0.));
            (b.bilinear(b.inv.transform_point2(w)) - a.bilinear(a.inv.transform_point2(w))).abs()
        };
        assert!(step(&[]) > 0.5);
        let healed = [TileEdit { heals: vec![1], ..Default::default() }, TileEdit::default()];
        assert!(step(&healed) < 0.1, "{}", step(&healed));
    }

    #[test]
    fn nice_lengths() {
        assert_eq!(nice_length(730.), 500.);
        assert_eq!(nice_length(190.), 100.);
        assert_eq!(nice_length(2500.), 2000.);
        assert_eq!(format_nm(2000.), "2 µm");
    }
}
