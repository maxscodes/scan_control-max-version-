//! Line up overlapping stamps (e.g. survey tiles) by their image content.
//!
//! Coarse moves land a little differently each time, so tiles placed from
//! the step count alone don't quite meet. Each pair of stamps that overlaps
//! on the map is registered within its shared strip, and all the measured
//! offsets are fitted together (least squares), so errors don't build up
//! along a row. Stamps keep their rotation and size; only their position
//! changes.

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
}
impl Alignment {
    pub fn largest_nm(&self) -> f64 {
        self.moves.iter().map(|(_, d)| d.length()).fold(0., f64::max)
    }
    pub fn summary(&self, tiles: usize) -> String {
        let mut s = format!(
            "Aligned {} of {tiles} stamps from {} overlaps (largest move {:.0} nm)",
            tiles - self.unlinked,
            self.pairs_used,
            self.largest_nm()
        );
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

/// Same frame shape (size and rotation) and resolution, so the two differ
/// only by a shift.
fn comparable(a: &Tile, b: &Tile) -> bool {
    let (ma, mb) = (a.transform.matrix2, b.transform.matrix2);
    let scale = ma.x_axis.length().max(ma.y_axis.length());
    a.size == b.size && (ma - mb).to_cols_array().iter().all(|v| v.abs() < 1e-3 * scale)
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

/// New positions for `tiles` (the first one stays put), from their overlaps.
pub fn align(tiles: &[Tile]) -> Alignment {
    let n = tiles.len();
    let mut out = Alignment::default();
    // Measured relative position of each overlapping pair, as a correction
    // to where the map has them now.
    let mut edges = Vec::new();
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
}
impl Aligner {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }
    /// Start aligning these stamps (ignored while one is running).
    pub fn start(&mut self, objects: &SelectableList<Object>, ids: &[Uuid]) -> bool {
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
            let _ = tx.send(align(&tiles));
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
        for (id, d) in &result.moves {
            let Some(&(_, was)) = before.iter().find(|(b, _)| b == id) else { continue };
            for o in objects.iter_mut() {
                if let Object::ScanImage { image, .. } = &mut **o {
                    if image.uuid() == *id && image.transform == was {
                        image.transform.translation += *d;
                    }
                }
            }
        }
        let summary = result.summary(before.len());
        tracing::info!("{summary}");
        Some(summary)
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
        let a = align(&tiles);
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

    #[test]
    fn tiles_without_overlap_stay_put() {
        let a = tile(DVec2::ZERO, DVec2::ZERO);
        let far = tile(DVec2::new(5000., 0.), DVec2::new(5000., 0.));
        let r = align(&[a, far]);
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
                let r = align(&[a, b.clone()]);
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
