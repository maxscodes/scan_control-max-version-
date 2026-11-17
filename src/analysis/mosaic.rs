//! Line up overlapping stamps (e.g. survey tiles) by their image content.
//!
//! Coarse moves land a little differently each time, so tiles placed from
//! the step count alone don't quite meet. Each pair of stamps that overlaps
//! on the map is registered within its shared strip, and all the measured
//! offsets are fitted together (least squares), so errors don't build up
//! along a row. Stamps keep their rotation and size; only their position
//! changes.
//!
//! With the stretch fix on, drift during each scan is undone as well. Drift
//! moves the tip steadily while the lines are recorded, so a tile comes out
//! sheared and stretched along its slow-scan axis (local y): what the map
//! should show at local `u` is `T(u) + d + v·u.y`. Several patches along
//! each shared strip are matched on their own; where they disagree from one
//! end of the strip to the other, that is `v`. All tiles' `d` and `v` are
//! fitted together. The overlaps only tell tiles' stretches apart, not the
//! whole mosaic's, so a light pull towards no stretch settles that: with
//! alternating scans (drift one way, then the other) it lands on the average,
//! which is the true shape.

use std::sync::Arc;

use crossbeam::channel::Receiver;
use glam::{DAffine2, DVec2};
use uuid::Uuid;

use crate::{
    analysis::register::{shift_near, GOOD_SCORE},
    components::selectable_list::SelectableList,
    view_object::Object,
};

/// One stamp as the aligner sees it.
#[derive(Clone)]
pub struct Tile {
    pub id: Uuid,
    pub data: Arc<Box<[f32]>>,
    pub size: [u32; 2],
    /// Map transform of the image (local −0.5..0.5 → world nm).
    pub transform: DAffine2,
}

/// What the alignment found.
#[derive(Debug, Clone, Default)]
pub struct Alignment {
    /// Move of each stamp, world nm.
    pub moves: Vec<(Uuid, DVec2)>,
    /// Overlaps that matched well and were used, and ones too weak to use.
    pub pairs_used: usize,
    pub pairs_weak: usize,
    /// Stamps that couldn't be tied to any other (no good overlap).
    pub unlinked: usize,
    /// Drift stretch of each stamp: world nm added per unit of its local y
    /// (added to its transform's y axis).
    pub stretch: Vec<(Uuid, DVec2)>,
    /// Groups of stamps whose stretch fit looked wrong (only shifted).
    pub stretch_refused: usize,
}
impl Alignment {
    pub fn largest_nm(&self) -> f64 {
        self.moves.iter().map(|(_, d)| d.length()).fold(0., f64::max)
    }
    /// Largest stretch: how far the top edge moved against the bottom, nm.
    pub fn largest_stretch_nm(&self) -> f64 {
        self.stretch.iter().map(|(_, v)| v.length()).fold(0., f64::max)
    }
    pub fn summary(&self, tiles: usize) -> String {
        let mut s = format!(
            "Aligned {} of {tiles} stamps from {} overlaps (largest move {:.0} nm)",
            tiles - self.unlinked,
            self.pairs_used,
            self.largest_nm()
        );
        if !self.stretch.is_empty() {
            s += &format!(
                "; drift stretch undone on {} (largest {:.0} nm top to bottom)",
                self.stretch.len(),
                self.largest_stretch_nm()
            );
        }
        if self.stretch_refused > 0 {
            s += &format!(
                "; stretch fit looked wrong for {} group{}, only shifted",
                self.stretch_refused,
                if self.stretch_refused == 1 { "" } else { "s" }
            );
        }
        if self.pairs_weak > 0 {
            s += &format!("; {} weak overlaps ignored", self.pairs_weak);
        }
        if self.unlinked > 0 {
            s += &format!("; {} stamps had no usable overlap and were left as they were", self.unlinked);
        }
        s
    }
}

/// Smallest shared area (share of a tile) worth registering.
const MIN_OVERLAP: f64 = 0.03;
/// Search this far (share of the tile) around where the map puts a tile.
const SEARCH: f64 = 0.25;

/// Largest drift stretch accepted, share of the tile size: more is a bad fit.
const MAX_STRETCH: f64 = 0.15;
/// Pull of each tile's stretch towards none, relative to the overlaps.
const STRETCH_RIDGE: f64 = 1e-3;
/// Side of the patches matched along a shared strip, pixels.
const PATCH_MAX: f64 = 96.;
const PATCH_MIN: f64 = 16.;

/// Same frame shape (size and rotation) and resolution, so the two differ
/// only by a shift (and, after a stretch fix, a little along local y).
fn comparable(a: &Tile, b: &Tile) -> bool {
    let (ma, mb) = (a.transform.matrix2, b.transform.matrix2);
    let scale = ma.x_axis.length().max(ma.y_axis.length());
    a.size == b.size
        && (ma.x_axis - mb.x_axis).length() < 1e-3 * scale
        && (ma.y_axis - mb.y_axis).length() < 2. * MAX_STRETCH * scale
}

/// Measured world offset of `b` from `a` and the match score, if they
/// overlap enough to tell.
fn measure(a: &Tile, b: &Tile) -> Option<(DVec2, f64)> {
    if !comparable(a, b) {
        return None;
    }
    let [w, h] = a.size;
    let (wf, hf) = (w as f64, h as f64);
    let inv = a.transform.matrix2.inverse();
    // Where the map has b's pixels in a's pixels: a(x + o) = b(x).
    let local = inv * (b.transform.translation - a.transform.translation);
    let o = DVec2::new(local.x * wf, local.y * hf);
    // Could they overlap, allowing for the search range?
    let share = (1. + SEARCH - local.x.abs()).max(0.) * (1. + SEARCH - local.y.abs()).max(0.);
    if share < MIN_OVERLAP + SEARCH {
        return None;
    }
    // image_shift convention: b(x) = a(x − d), so d = −o.
    let s = shift_near(&a.data, &b.data, w as usize, h as usize, (-o.x, -o.y), SEARCH * wf.max(hf), MIN_OVERLAP)?;
    let true_local = DVec2::new(-s.dx / wf, -s.dy / hf);
    Some((a.transform.matrix2 * true_local, s.score))
}

/// Pixel position (in pixel indices, may be fractional) → local −0.5..0.5.
fn to_local(p: DVec2, size: [u32; 2]) -> DVec2 {
    DVec2::new((p.x + 0.5) / size[0] as f64 - 0.5, (p.y + 0.5) / size[1] as f64 - 0.5)
}

/// Slope along each scan line (lightly smoothed down the image): line
/// offsets differ between tiles, slopes don't. NaN where unknown.
fn line_slope(d: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut g = vec![f32::NAN; w * h];
    for y in 0..h {
        for x in 1..w - 1 {
            g[y * w + x] = d[y * w + x + 1] - d[y * w + x - 1];
        }
    }
    g
}

/// Where `b`'s `k`×`k` patch with top-left (`bx`, `by`) is found in `a`,
/// searching `r` px around top-left (`ax`, `ay`): the patch is always
/// compared whole (template matching), so no shift is favoured over
/// another. Returns the sub-pixel top-left in `a` and the correlation.
#[allow(clippy::too_many_arguments)]
fn find_patch(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    (bx, by): (usize, usize),
    k: usize,
    (ax, ay): (i64, i64),
    r: i64,
) -> Option<(DVec2, f64)> {
    let score = |x0: i64, y0: i64| -> Option<f64> {
        if x0 < 0 || y0 < 0 || x0 as usize + k > w || y0 as usize + k > h {
            return None;
        }
        let (x0, y0) = (x0 as usize, y0 as usize);
        let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for y in 0..k {
            let (ra, rb) = ((y0 + y) * w + x0, (by + y) * w + bx);
            for x in 0..k {
                let (va, vb) = (a[ra + x], b[rb + x]);
                if va.is_finite() && vb.is_finite() {
                    let (va, vb) = (va as f64, vb as f64);
                    n += 1.;
                    sa += va;
                    sb += vb;
                    saa += va * va;
                    sbb += vb * vb;
                    sab += va * vb;
                }
            }
        }
        if n < (k * k / 2) as f64 {
            return None;
        }
        let (caa, cbb) = (saa - sa * sa / n, sbb - sb * sb / n);
        (caa > 0. && cbb > 0.).then(|| (sab - sa * sb / n) / (caa * cbb).sqrt())
    };
    let mut best: Option<(f64, i64, i64)> = None;
    for dy in -r..=r {
        for dx in -r..=r {
            if let Some(c) = score(ax + dx, ay + dy) {
                if best.is_none_or(|b| c > b.0) {
                    best = Some((c, ax + dx, ay + dy));
                }
            }
        }
    }
    let (c, x, y) = best?;
    // A best match on the edge of the search may lie beyond it.
    if (x - ax).abs() == r || (y - ay).abs() == r {
        return None;
    }
    let at = |x: i64, y: i64| score(x, y).unwrap_or(c);
    let sub_x = parabola(at(x - 1, y), c, at(x + 1, y));
    let sub_y = parabola(at(x, y - 1), c, at(x, y + 1));
    Some((DVec2::new(x as f64 + sub_x, y as f64 + sub_y), c))
}

/// Peak offset (−0.5..0.5) of a parabola through three samples.
fn parabola(l: f64, c: f64, r: f64) -> f64 {
    let d = l - 2. * c + r;
    if d.abs() < 1e-12 {
        0.
    } else {
        (0.5 * (l - r) / d).clamp(-0.5, 0.5)
    }
}

/// Matches spread over the area `a` and `b` share, given that `b`'s
/// content sits `offset` (world) from `a`'s: each patch is registered on
/// its own, so a tile stretched by drift shows as matches that disagree
/// along the strip. `sa`, `sb`: the tiles' [`line_slope`]s. Each match is
/// (local in `a`, local in `b`, weight).
fn local_links(a: &Tile, b: &Tile, sa: &[f32], sb: &[f32], offset: DVec2) -> Vec<(DVec2, DVec2, f64)> {
    let [w, h] = a.size;
    let (wf, hf) = (w as f64, h as f64);
    let local = a.transform.matrix2.inverse() * offset;
    // Pixel q of b shows what pixel q + p of a shows.
    let p = DVec2::new(local.x * wf, local.y * hf);
    let margin = 2.;
    let lo = DVec2::new((-p.x).max(0.), (-p.y).max(0.)) + margin;
    let hi = DVec2::new((wf - p.x).min(wf), (hf - p.y).min(hf)) - margin;
    let span = hi - lo;
    let k = span.x.min(span.y).min(PATCH_MAX).floor();
    if k < PATCH_MIN {
        return Vec::new();
    }
    let ku = k as usize;
    let count = |len: f64| ((len / k).floor() as usize).clamp(1, 6);
    let (nx, ny) = (count(span.x), count(span.y));
    let spread = |i: usize, n: usize| if n == 1 { 0.5 } else { i as f64 / (n - 1) as f64 };
    // Search room: drift moves the patches along a strip by a few % of the
    // tile, and the whole-tile match it starts from is itself a little off.
    let search = (0.5 * k).clamp(6., 24.) as i64;
    let mut out = Vec::new();
    for iy in 0..ny {
        for ix in 0..nx {
            let bx = (lo.x + (span.x - k) * spread(ix, nx)).floor();
            let by = (lo.y + (span.y - k) * spread(iy, ny)).floor();
            if bx < 0. || by < 0. || bx + k > wf || by + k > hf {
                continue;
            }
            let guess = ((bx + p.x).round() as i64, (by + p.y).round() as i64);
            let found =
                find_patch(sa, sb, w as usize, h as usize, (bx as usize, by as usize), ku, guess, search);
            let Some((at_a, score)) = found else { continue };
            if score < GOOD_SCORE {
                continue;
            }
            let c = DVec2::splat((k - 1.) / 2.);
            out.push((to_local(at_a + c, a.size), to_local(DVec2::new(bx, by) + c, b.size), score));
        }
    }
    out
}

/// One feature seen in tiles `i` and `j`, at these local positions.
struct Link {
    i: usize,
    j: usize,
    ui: DVec2,
    uj: DVec2,
    w: f64,
}

/// Shift `d` and stretch `v` of each tile in `group` (the anchor's shift
/// fixed at 0) so that for every link
/// `T_j(u_j) + d_j + v_j·u_j.y = T_i(u_i) + d_i + v_i·u_i.y`.
/// Links that disagree with the rest are dropped and the fit redone.
fn fit_stretch(tiles: &[Tile], links: &[Link], group: &[usize], anchor: usize) -> Option<Vec<(usize, DVec2, DVec2)>> {
    // Columns: a shift for every tile but the anchor, then a stretch for each.
    let n_d = group.len() - 1;
    let col_d = |t: usize| group.iter().filter(|&&g| g != anchor).position(|&g| g == t);
    let col_v = |t: usize| group.iter().position(|&g| g == t).map(|k| n_d + k);
    let m = n_d + group.len();
    let mut used: Vec<&Link> =
        links.iter().filter(|l| col_v(l.i).is_some() && col_v(l.j).is_some()).collect();
    if used.is_empty() {
        return None;
    }
    let row = |l: &Link| {
        let mut c = Vec::with_capacity(4);
        if let Some(k) = col_d(l.j) {
            c.push((k, 1.));
        }
        if let Some(k) = col_d(l.i) {
            c.push((k, -1.));
        }
        c.push((col_v(l.j).unwrap(), l.uj.y));
        c.push((col_v(l.i).unwrap(), -l.ui.y));
        // What has to be made up: where j puts the feature minus where i does.
        let gap = tiles[l.j].transform.transform_point2(l.uj) - tiles[l.i].transform.transform_point2(l.ui);
        (c, -gap)
    };
    let px = tiles[anchor].transform.matrix2.x_axis.length() / tiles[anchor].size[0] as f64;
    let mut x = nalgebra::DMatrix::<f64>::zeros(m, 2);
    for round in 0..3 {
        let mut ata = nalgebra::DMatrix::<f64>::zeros(m, m);
        let mut atb = nalgebra::DMatrix::<f64>::zeros(m, 2);
        let mut total = 0.;
        for l in &used {
            let (c, rhs) = row(l);
            for &(a, ca) in &c {
                for &(b, cb) in &c {
                    ata[(a, b)] += l.w * ca * cb;
                }
                atb[(a, 0)] += l.w * ca * rhs.x;
                atb[(a, 1)] += l.w * ca * rhs.y;
            }
            total += l.w;
        }
        let ridge = STRETCH_RIDGE * total / group.len() as f64;
        for k in n_d..m {
            ata[(k, k)] += ridge;
        }
        x = ata.lu().solve(&atb)?;
        if round == 2 {
            break;
        }
        // Drop links much worse than the typical one (false matches).
        let resid = |l: &Link| {
            let (c, rhs) = row(l);
            let mut r = -rhs;
            for (k, ck) in c {
                r += ck * DVec2::new(x[(k, 0)], x[(k, 1)]);
            }
            r.length()
        };
        let mut rs: Vec<f64> = used.iter().map(|l| resid(l)).collect();
        rs.sort_by(f64::total_cmp);
        let limit = (3. * rs[rs.len() / 2]).max(px);
        let before = used.len();
        used.retain(|l| resid(l) <= limit);
        if used.len() == before || used.is_empty() {
            if used.is_empty() {
                return None;
            }
            break;
        }
    }
    let at = |k: usize| DVec2::new(x[(k, 0)], x[(k, 1)]);
    let out: Vec<(usize, DVec2, DVec2)> =
        group.iter().map(|&t| (t, col_d(t).map_or(DVec2::ZERO, at), at(col_v(t).unwrap()))).collect();
    // A stretch beyond what drift could do is a bad fit.
    let ok = out.iter().all(|&(t, d, v)| {
        let size = tiles[t].transform.matrix2.y_axis.length();
        d.is_finite() && v.is_finite() && v.length() <= MAX_STRETCH * size
    });
    ok.then_some(out)
}

/// Solve `x_j − x_i ≈ m_ij` (weighted) for every tile linked to `anchor`,
/// with `x_anchor` = 0. Returns positions for the linked tiles.
fn solve(n: usize, edges: &[(usize, usize, DVec2, f64)], anchor: usize) -> Vec<Option<DVec2>> {
    // Tiles reachable from the anchor.
    let mut linked = vec![false; n];
    linked[anchor] = true;
    let mut stack = vec![anchor];
    while let Some(i) = stack.pop() {
        for &(a, b, _, _) in edges {
            for (p, q) in [(a, b), (b, a)] {
                if p == i && !linked[q] {
                    linked[q] = true;
                    stack.push(q);
                }
            }
        }
    }
    let idx: Vec<usize> = (0..n).filter(|&i| linked[i] && i != anchor).collect();
    let mut pos = vec![None; n];
    pos[anchor] = Some(DVec2::ZERO);
    if idx.is_empty() {
        return pos;
    }
    let col = |i: usize| idx.iter().position(|&k| k == i);
    let m = idx.len();
    let mut lap = nalgebra::DMatrix::<f64>::zeros(m, m);
    let mut rhs = nalgebra::DMatrix::<f64>::zeros(m, 2);
    for &(a, b, d, w) in edges {
        if !(linked[a] && linked[b]) {
            continue;
        }
        // w · (x_b − x_a − d)² summed: normal equations.
        let (ca, cb) = (col(a), col(b));
        if let Some(ca) = ca {
            lap[(ca, ca)] += w;
            rhs[(ca, 0)] -= w * d.x;
            rhs[(ca, 1)] -= w * d.y;
        }
        if let Some(cb) = cb {
            lap[(cb, cb)] += w;
            rhs[(cb, 0)] += w * d.x;
            rhs[(cb, 1)] += w * d.y;
        }
        if let (Some(ca), Some(cb)) = (ca, cb) {
            lap[(ca, cb)] -= w;
            lap[(cb, ca)] -= w;
        }
    }
    if let Some(x) = lap.lu().solve(&rhs) {
        for (k, &i) in idx.iter().enumerate() {
            pos[i] = Some(DVec2::new(x[(k, 0)], x[(k, 1)]));
        }
    }
    pos
}

/// New positions for `tiles` (the first one stays put), from their
/// overlaps; with `stretch`, each tile's drift stretch is undone too (see
/// the module notes).
pub fn align_with(tiles: &[Tile], stretch: bool) -> Alignment {
    let n = tiles.len();
    let mut out = Alignment::default();
    // Measured relative position of each overlapping pair, as a correction
    // to where the map has them now.
    let mut edges = Vec::new();
    // The same pairs, measured offset as is (for the stretch fit).
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let Some((offset, score)) = measure(&tiles[i], &tiles[j]) else { continue };
            let size = tiles[i].transform.matrix2.x_axis.length();
            let now = tiles[j].transform.translation - tiles[i].transform.translation;
            // A match that would move a tile by over a third of its size is
            // more likely a false match than a coarse-move error.
            if score < GOOD_SCORE || (offset - now).length() > 0.35 * size {
                out.pairs_weak += 1;
                continue;
            }
            out.pairs_used += 1;
            edges.push((i, j, offset - now, score));
            pairs.push((i, j, offset, score));
        }
    }
    // Features matched along each shared strip (stretch fit only).
    let mut links = Vec::new();
    if stretch {
        let slopes: Vec<Vec<f32>> =
            tiles.iter().map(|t| line_slope(&t.data, t.size[0] as usize, t.size[1] as usize)).collect();
        for &(i, j, offset, score) in &pairs {
            let found = local_links(&tiles[i], &tiles[j], &slopes[i], &slopes[j], offset);
            if found.is_empty() {
                // Strip too narrow for patches: the whole-tile match, at b's centre.
                let ui = tiles[i].transform.matrix2.inverse() * offset;
                links.push(Link { i, j, ui, uj: DVec2::ZERO, w: score });
            }
            for (ui, uj, w) in found {
                links.push(Link { i, j, ui, uj, w });
            }
        }
    }
    // Each group of linked tiles moves relative to its first tile.
    let mut done = vec![false; n];
    for anchor in 0..n {
        if done[anchor] {
            continue;
        }
        let pos = solve(n, &edges, anchor);
        let linked: Vec<usize> = (0..n).filter(|&i| pos[i].is_some()).collect();
        for &i in &linked {
            done[i] = true;
        }
        if linked.len() == 1 {
            out.unlinked += 1;
            continue;
        }
        if stretch {
            match fit_stretch(tiles, &links, &linked, anchor) {
                Some(fit) => {
                    for (i, d, v) in fit {
                        if d.length() > 1e-6 {
                            out.moves.push((tiles[i].id, d));
                        }
                        if v.length() > 1e-6 {
                            out.stretch.push((tiles[i].id, v));
                        }
                    }
                    continue;
                }
                None => out.stretch_refused += 1,
            }
        }
        for &i in &linked {
            let d = pos[i].unwrap();
            if d.length() > 1e-6 {
                out.moves.push((tiles[i].id, d));
            }
        }
    }
    out
}

/// The stamps in `ids` that are scan images, in that order.
pub fn tiles_of(objects: &SelectableList<Object>, ids: &[Uuid]) -> Vec<Tile> {
    ids.iter()
        .filter_map(|id| {
            objects.iter().find_map(|o| match &**o {
                Object::ScanImage { image, .. } if image.uuid() == *id && image.forward_data.size[0] > 0 => {
                    Some(Tile {
                        id: *id,
                        data: image.forward_data.data.clone(),
                        size: image.forward_data.size,
                        transform: image.transform,
                    })
                }
                _ => None,
            })
        })
        .collect()
}

/// Runs an alignment in the background and applies it to the map.
#[derive(Default)]
pub struct Aligner {
    job: Option<(Receiver<Alignment>, Vec<(Uuid, DAffine2)>)>,
    /// The last alignment applied: (stamp, before, after), to undo it.
    last: Vec<(Uuid, DAffine2, DAffine2)>,
}
impl Aligner {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }
    /// Start aligning these stamps (ignored while one is running); with
    /// `stretch`, undo each one's drift stretch too.
    pub fn start(&mut self, objects: &SelectableList<Object>, ids: &[Uuid], stretch: bool) -> bool {
        if self.job.is_some() {
            return false;
        }
        let tiles = tiles_of(objects, ids);
        if tiles.len() < 2 {
            return false;
        }
        let before = tiles.iter().map(|t| (t.id, t.transform)).collect();
        let (tx, rx) = crossbeam::channel::bounded(1);
        std::thread::spawn(move || {
            let _ = tx.send(align_with(&tiles, stretch));
            crate::app::request_repaint();
        });
        self.job = Some((rx, before));
        true
    }
    /// Apply a finished alignment; returns its summary. Stamps moved by
    /// hand meanwhile are left where they were put.
    pub fn poll(&mut self, objects: &mut SelectableList<Object>) -> Option<String> {
        let (rx, _) = self.job.as_ref()?;
        let result = rx.try_recv().ok()?;
        let (_, before) = self.job.take()?;
        self.last.clear();
        for &(id, was) in &before {
            let d = result.moves.iter().find(|(m, _)| *m == id).map(|m| m.1);
            let v = result.stretch.iter().find(|(m, _)| *m == id).map(|m| m.1);
            if d.is_none() && v.is_none() {
                continue;
            }
            for o in objects.iter_mut() {
                if let Object::ScanImage { image, .. } = &mut **o {
                    if image.uuid() == id && image.transform == was {
                        image.transform.translation += d.unwrap_or_default();
                        image.transform.matrix2.y_axis += v.unwrap_or_default();
                        self.last.push((id, was, image.transform));
                    }
                }
            }
        }
        let summary = result.summary(before.len());
        tracing::info!("{summary}");
        Some(summary)
    }

    /// The last alignment can be undone.
    pub fn can_undo(&self) -> bool {
        !self.last.is_empty() && self.job.is_none()
    }
    /// Put the stamps of the last alignment back where they were (those not
    /// moved since); returns how many.
    pub fn undo(&mut self, objects: &mut SelectableList<Object>) -> usize {
        let mut n = 0;
        for (id, before, after) in self.last.drain(..) {
            for o in objects.iter_mut() {
                if let Object::ScanImage { image, .. } = &mut **o {
                    if image.uuid() == id && image.transform == after {
                        image.transform = before;
                        n += 1;
                    }
                }
            }
        }
        tracing::info!("alignment undone: {n} stamps put back");
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(x: f64, y: f64) -> f32 {
        let mut v = 0.;
        for k in 1..7 {
            let k = k as f64;
            v += (x * 0.011 * k + (y * 0.007 * k).sin() * 3.).sin() / k
                + (y * 0.013 * k + (x * 0.005 * k).cos() * 2.).cos() / k;
        }
        v as f32
    }

    /// A 1000 nm, 128 px tile whose image is the sample at `true_pos`, but
    /// placed on the map at `map_pos` (as a coarse move would leave it).
    fn tile(true_pos: DVec2, map_pos: DVec2) -> Tile {
        let n = 128;
        let m = DAffine2::from_scale(DVec2::new(1000., -1000.));
        let mut d = vec![0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let local = DVec2::new((i as f64 + 0.5) / n as f64 - 0.5, (j as f64 + 0.5) / n as f64 - 0.5);
                let p = m.transform_point2(local) + true_pos;
                d[j * n + i] = surface(p.x, p.y);
            }
        }
        Tile {
            id: Uuid::new_v4(),
            data: Arc::new(d.into_boxed_slice()),
            size: [n as u32, n as u32],
            transform: DAffine2::from_translation(map_pos) * m,
        }
    }

    #[test]
    fn survey_tiles_line_up() {
        // A 3 × 2 snake with 15 % overlap (850 nm moves); every move after
        // the first tile landed up to ~90 nm off, so errors add up.
        let step = 850.;
        let errors = [
            DVec2::ZERO,
            DVec2::new(60., -25.),
            DVec2::new(-40., 70.),
            DVec2::new(90., 10.),
            DVec2::new(-30., -60.),
            DVec2::new(20., 45.),
        ];
        let grid = [(0., 0.), (1., 0.), (2., 0.), (2., 1.), (1., 1.), (0., 1.)];
        let mut truth = DVec2::ZERO;
        let mut tiles = Vec::new();
        let mut truths = Vec::new();
        for (k, &(c, r)) in grid.iter().enumerate() {
            let planned = DVec2::new(c * step, -r * step);
            truth += errors[k];
            let real = planned + truth;
            truths.push(real);
            tiles.push(tile(real, planned));
        }
        let a = align_with(&tiles, false);
        assert_eq!(a.unlinked, 0, "{a:?}");
        assert!(a.pairs_used >= 5, "{a:?}");
        for (k, t) in tiles.iter().enumerate() {
            let d = a.moves.iter().find(|(id, _)| *id == t.id).map_or(DVec2::ZERO, |m| m.1);
            let placed = t.transform.translation + d;
            // Relative to the first tile, which stays put.
            let want = truths[k] - truths[0];
            assert!((placed - want).length() < 12., "tile {k}: placed {placed:?}, want {want:?}");
        }
    }

    /// Like [`tile`], but drift during the scan put the feature at local
    /// `u` at `T(u) + drift·u.y` on the sample.
    fn drifted_tile(true_pos: DVec2, map_pos: DVec2, drift: DVec2) -> Tile {
        let n = 128;
        let m = DAffine2::from_scale(DVec2::new(1000., -1000.));
        let mut d = vec![0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let local = DVec2::new((i as f64 + 0.5) / n as f64 - 0.5, (j as f64 + 0.5) / n as f64 - 0.5);
                let p = m.transform_point2(local) + true_pos + drift * local.y;
                d[j * n + i] = surface(p.x, p.y);
            }
        }
        Tile {
            id: Uuid::new_v4(),
            data: Arc::new(d.into_boxed_slice()),
            size: [n as u32, n as u32],
            transform: DAffine2::from_translation(map_pos) * m,
        }
    }

    /// Alternating scans in a drifting microscope: every other tile is
    /// stretched and slanted the other way. Moving tiles can only meet in
    /// the middle of a seam; undoing the stretch makes them meet along it.
    #[test]
    fn drift_stretch_is_undone() {
        // 3 × 2 snake, 30 % overlap, moves up to ~50 nm off.
        let step = 700.;
        let errors = [
            DVec2::ZERO,
            DVec2::new(40., -20.),
            DVec2::new(-30., 50.),
            DVec2::new(50., 10.),
            DVec2::new(-20., -40.),
            DVec2::new(10., 30.),
        ];
        let grid = [(0., 0.), (1., 0.), (2., 0.), (2., 1.), (1., 1.), (0., 1.)];
        // 1.5 % slant and 3 % stretch, opposite in neighbours. (Much more —
        // neighbours ~10 % apart — blurs the patch matches and is only
        // partly undone.)
        let drift = DVec2::new(15., 30.);
        let mut truth = DVec2::ZERO;
        let (mut tiles, mut truths, mut drifts) = (Vec::new(), Vec::new(), Vec::new());
        for (k, &(c, r)) in grid.iter().enumerate() {
            let planned = DVec2::new(c * step, -r * step);
            truth += errors[k];
            let real = planned + truth;
            // Up, down, up, …: the drift shows with opposite signs.
            let v = if k % 2 == 0 { drift } else { -drift };
            truths.push(real);
            drifts.push(v);
            tiles.push(drifted_tile(real, planned, v));
        }
        // Worst distance, over each tile's corners and centre, between
        // where the map puts a feature and where it really is (both
        // relative to the first tile's centre).
        let worst = |a: &Alignment| {
            let mut worst: f64 = 0.;
            for (k, t) in tiles.iter().enumerate() {
                let d = a.moves.iter().find(|(id, _)| *id == t.id).map_or(DVec2::ZERO, |m| m.1);
                let v = a.stretch.iter().find(|(id, _)| *id == t.id).map_or(DVec2::ZERO, |m| m.1);
                let mut placed = t.transform;
                placed.translation += d;
                placed.matrix2.y_axis += v;
                for u in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (0., 0.)] {
                    let u = DVec2::new(u.0, u.1);
                    let real = DAffine2::from_scale(DVec2::new(1000., -1000.)).transform_point2(u)
                        + truths[k]
                        + drifts[k] * u.y
                        - truths[0];
                    worst = worst.max((placed.transform_point2(u) - real).length());
                }
            }
            worst
        };
        let shifted = align_with(&tiles, false);
        let fixed = align_with(&tiles, true);
        println!("{}", fixed.summary(tiles.len()));
        assert_eq!(fixed.stretch_refused, 0, "{fixed:?}");
        assert_eq!(fixed.stretch.len(), tiles.len(), "{fixed:?}");
        // Shifting alone leaves ~30 nm at the corners; one pixel is ~8 nm.
        let (ws, wf) = (worst(&shifted), worst(&fixed));
        println!("worst shift only {ws:.1} nm, with stretch {wf:.1} nm");
        assert!(ws > 25., "shift only: {ws:.1} nm");
        assert!(wf < 10., "with stretch: {wf:.1} nm");
    }

    /// Undistorted tiles: the stretch fit finds next to none.
    #[test]
    fn no_drift_no_stretch() {
        let tiles = [tile(DVec2::ZERO, DVec2::ZERO), tile(DVec2::new(700., 20.), DVec2::new(700., 0.))];
        let a = align_with(&tiles, true);
        assert!(a.largest_stretch_nm() < 3., "{a:?}");
        let d = a.moves.iter().find(|(id, _)| *id == tiles[1].id).map_or(DVec2::ZERO, |m| m.1);
        assert!((d - DVec2::new(0., 20.)).length() < 4., "{a:?}");
    }

    #[test]
    fn tiles_without_overlap_stay_put() {
        let a = tile(DVec2::ZERO, DVec2::ZERO);
        let far = tile(DVec2::new(5000., 0.), DVec2::new(5000., 0.));
        let r = align_with(&[a, far], false);
        assert!(r.moves.is_empty());
        assert_eq!(r.unlinked, 2);
    }

    /// Real STM images: two tiles cut from one scan with a 20 % (and a
    /// 10 %) shared strip, the second misplaced, must be put back.
    /// `SXM_DIR=dir cargo test --release --bin scan_control tiles_from_real_scans -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn tiles_from_real_scans() {
        let dir = std::env::var("SXM_DIR").unwrap_or(r"F:\Data\LT SPM\2026\November".into());
        let files = crate::analysis::sxm_match::list_sxm(std::path::Path::new(&dir));
        let (mut n, mut ok) = (0, 0);
        for f in files.iter().step_by(7) {
            let Ok(img) = crate::analysis::sxm_match::load_frame(f, Some("Z")) else { continue };
            let (w, h) = (img.size[0] as usize, img.size[1] as usize);
            if w < 256 || w != h || img.forward.iter().any(|v| !v.is_finite()) {
                continue;
            }
            for overlap in [0.2, 0.1] {
                // Tiles t px wide: A = columns 0..t, B = columns s..s+t.
                let t = (w as f64 / (2. - overlap)) as usize;
                let s0 = t - (t as f64 * overlap) as usize;
                let cut = |x0: usize| -> Vec<f32> {
                    (0..t).flat_map(|j| (0..t).map(move |i| (j, i))).map(|(j, i)| img.forward[j * w + x0 + i]).collect()
                };
                let px = 10.; // nm per pixel
                let m = DAffine2::from_scale(DVec2::new(t as f64 * px, -(t as f64) * px));
                let mk = |data: Vec<f32>, at: DVec2| Tile {
                    id: Uuid::new_v4(),
                    data: Arc::new(data.into_boxed_slice()),
                    size: [t as u32, t as u32],
                    transform: DAffine2::from_translation(at) * m,
                };
                let truth = DVec2::new(s0 as f64 * px, 0.);
                let error = DVec2::new(0.08, -0.06) * t as f64 * px;
                let a = mk(cut(0), DVec2::ZERO);
                let b = mk(cut(s0), truth + error);
                let r = align_with(&[a, b.clone()], false);
                let d = r.moves.iter().find(|(id, _)| *id == b.id).map_or(DVec2::ZERO, |m| m.1);
                let off = (b.transform.translation + d - truth).length() / px;
                n += 1;
                ok += (off < 1.5) as usize;
                println!("{} overlap {:.0} %: off by {off:.2} px, {:?}", img.name, overlap * 100., r.summary(2));
            }
            if n >= 16 {
                break;
            }
        }
        println!("{ok} of {n} put back within 1.5 px");
        assert!(n > 0);
    }
}
