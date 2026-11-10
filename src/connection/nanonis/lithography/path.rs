//! Lithography tip-path planning, ported from the Python lithography GUI
//! (v1.18). Everything here works in *frame-local* nm: origin at the scan
//! frame centre, x along the frame width, y along the frame height (y up).
//!
//! A path is a continuous list of [`Segment`]s. Active segments are written
//! (lithography bias/current); inactive ones are travel.

use std::f64;

use glam::DVec2;
use serde::{Deserialize, Serialize};

pub type Polygon = Vec<DVec2>;

/// Snap tolerance for coordinates (nm), as in the Python planner.
const EPS: f64 = 1e-10;
/// Refuse rasters with more rows than this (pitch far too small).
const MAX_ROWS: f64 = 200_000.;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathMode {
    /// Serpentine through each row's active intervals; patches turn at their
    /// own edge, separate patches are joined by same-row connectors or jumps.
    PatchLocal,
    /// Patch-local rasters reordered (nearest neighbour + 2-opt) to shorten travel.
    Shortest,
    /// Write the perimeter of every polygon fully inside the frame.
    Boundary,
    /// Write left→right only; return right→left over the row's active extent.
    Comb,
    /// Like comb, but every row traverses the complete frame width.
    CombFrame,
}
impl PathMode {
    pub const ALL: [PathMode; 5] = [
        PathMode::PatchLocal,
        PathMode::Shortest,
        PathMode::Boundary,
        PathMode::Comb,
        PathMode::CombFrame,
    ];
    pub fn label(self) -> &'static str {
        match self {
            PathMode::PatchLocal => "Patch-local raster",
            PathMode::Shortest => "Shortest patch travel (2-opt)",
            PathMode::Boundary => "Polygon boundary",
            PathMode::Comb => "Comb raster (L→R write)",
            PathMode::CombFrame => "Full-frame comb",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            PathMode::PatchLocal => {
                "Serpentine raster of each patch; never sweeps the empty frame. \
                 Rows of one patch turn at its edge; separate patches are joined by \
                 inactive same-row connectors or patch jumps."
            }
            PathMode::Shortest => {
                "Same patch rasters, with the patch order and direction chosen to \
                 shorten inactive travel (nearest neighbour + 2-opt)."
            }
            PathMode::Boundary => {
                "Writes the perimeter of each polygon fully inside the frame, one \
                 polygon at a time. Polygons crossing the frame edge are ignored."
            }
            PathMode::Comb => {
                "Each row is written left→right only, then returned right→left \
                 (inactive) across that row's active extent. Requires the bias/current ramp."
            }
            PathMode::CombFrame => {
                "Every row traverses the full frame width left→right (only polygon \
                 intervals active) and back. Requires the bias/current ramp."
            }
        }
    }
    /// Comb returns are made at scan values with feedback on, so Z-lift is not allowed.
    pub fn requires_ramp(self) -> bool {
        matches!(self, PathMode::Comb | PathMode::CombFrame)
    }
    /// The full-frame comb uses the exact frame edges; the others keep a
    /// half-pitch clearance from them.
    pub fn uses_edge_clearance(self) -> bool {
        self != PathMode::CombFrame
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegKind {
    LithoRaster,
    PatchTurn,
    SameYConnector,
    PatchJump,
    PathfinderJump,
    PolygonBoundary,
    PolygonJump,
    CombLitho,
    CombGap,
    CombReturn,
    CombStepUp,
    FrameLitho,
    FrameTravel,
    FrameReturn,
    FrameStepUp,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub start: DVec2,
    pub end: DVec2,
    pub active: bool,
    pub kind: SegKind,
    /// Raster row (comb modes only).
    pub row: Option<usize>,
}
impl Segment {
    fn new(start: DVec2, end: DVec2, active: bool, kind: SegKind) -> Self {
        Self {
            start,
            end,
            active,
            kind,
            row: None,
        }
    }
    fn row(mut self, row: usize) -> Self {
        self.row = Some(row);
        self
    }
    pub fn length(&self) -> f64 {
        self.start.distance(self.end)
    }
    fn reversed(&self) -> Self {
        Self {
            start: self.end,
            end: self.start,
            ..*self
        }
    }
}

/// Edge clearance (nm) kept from the frame border by the clipped modes.
pub fn edge_clearance(pitch_nm: f64) -> f64 {
    (pitch_nm * 0.5).max(0.1)
}

/// Plan a path for `mode` over `polygons` (frame-local nm) in a frame of
/// `size` nm. Points are clamped inside the frame (minus the edge clearance)
/// for every mode except the full-frame comb.
pub fn plan(
    mode: PathMode,
    polygons: &[Polygon],
    size: DVec2,
    pitch_nm: f64,
) -> Result<Vec<Segment>, String> {
    if !(pitch_nm > 0.) || !pitch_nm.is_finite() {
        return Err("Raster pitch must be positive".into());
    }
    if !(size.x > 0. && size.y > 0.) {
        return Err("Scan frame dimensions must be positive".into());
    }
    let clearance = edge_clearance(pitch_nm);
    let inner = if mode.uses_edge_clearance() {
        size - DVec2::splat(2. * clearance)
    } else {
        size
    };
    if inner.x <= 0. || inner.y <= 0. {
        return Err("Scan frame is too small for the raster edge clearance".into());
    }
    if inner.y / pitch_nm > MAX_ROWS {
        return Err(format!(
            "Pitch {pitch_nm} nm gives more than {MAX_ROWS} rows; use a larger pitch"
        ));
    }
    let mut path = match mode {
        PathMode::PatchLocal => local_raster(polygons, inner.x, inner.y, pitch_nm)?,
        PathMode::Shortest => shortest_path(polygons, inner.x, inner.y, pitch_nm)?,
        PathMode::Boundary => boundary_path(polygons, inner.x, inner.y)?,
        PathMode::Comb => comb_raster(polygons, inner.x, inner.y, pitch_nm)?,
        PathMode::CombFrame => comb_frame_raster(polygons, inner.x, inner.y, pitch_nm)?,
    };
    if mode.uses_edge_clearance() {
        let half = (size / 2. - DVec2::splat(clearance)).max(DVec2::ZERO);
        for s in &mut path {
            s.start = s.start.clamp(-half, half);
            s.end = s.end.clamp(-half, half);
        }
    }
    Ok(path)
}

fn inside(p: DVec2, poly: &[DVec2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a.y > p.y) != (b.y > p.y) {
            let x_cross = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if p.x < x_cross {
                inside = !inside;
            }
        }
    }
    inside
}

fn inside_any(p: DVec2, polygons: &[Polygon]) -> bool {
    polygons.iter().any(|poly| inside(p, poly))
}

/// X positions where the polygon's edges cross the scanline `y` (half-open).
fn scan_events(poly: &[DVec2], y: f64, out: &mut Vec<f64>) {
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if a.y == b.y {
            continue;
        }
        if a.y.min(b.y) <= y && y < a.y.max(b.y) {
            let t = (y - a.y) / (b.y - a.y);
            out.push(a.x + t * (b.x - a.x));
        }
    }
}

type Rows = Vec<(f64, Vec<(f64, f64)>)>;

/// Active scanline intervals inside the frame, bottom row first. Empty rows
/// are omitted; intervals are clipped to the frame.
fn raster_intervals(polygons: &[Polygon], width: f64, height: f64, pitch: f64) -> Rows {
    let (xmin, xmax) = (-width / 2., width / 2.);
    let (ymin, ymax) = (-height / 2., height / 2.);
    let mut rows = Vec::new();
    let mut y = ymin;
    let mut events = Vec::new();
    while y <= ymax + pitch * 0.5 {
        events.clear();
        events.extend([xmin, xmax]);
        for poly in polygons {
            scan_events(poly, y, &mut events);
        }
        events.sort_by(f64::total_cmp);
        let mut xs: Vec<f64> = Vec::with_capacity(events.len());
        for &v in &events {
            if xs.last().is_none_or(|&last| v - last > EPS) {
                xs.push(v);
            }
        }
        let mut intervals = Vec::new();
        for w in xs.windows(2) {
            let (left, right) = (w[0], w[1]);
            if right - left <= EPS || !inside_any(DVec2::new((left + right) / 2., y), polygons) {
                continue;
            }
            // Clip before accepting, so polygons outside the frame don't
            // leave zero-length intervals at the frame edges.
            let (l, r) = (left.max(xmin), right.min(xmax));
            if r - l > EPS {
                intervals.push((l, r));
            }
        }
        if !intervals.is_empty() {
            rows.push((y, intervals));
        }
        y += pitch;
    }
    rows
}

fn no_geometry() -> String {
    "No GDS geometry intersects the lithography scan frame".into()
}

/// Row-extent comb: each row from its leftmost to its rightmost active
/// extent, written L→R, returned R→L; rows joined at their left extents.
fn comb_raster(polygons: &[Polygon], width: f64, height: f64, pitch: f64) -> Result<Vec<Segment>, String> {
    let rows = raster_intervals(polygons, width, height, pitch);
    if rows.is_empty() {
        return Err(no_geometry());
    }
    let mut path = Vec::new();
    let mut prev: Option<(f64, f64)> = None; // (y, left)
    for (ri, (y, intervals)) in rows.iter().enumerate() {
        let y = *y;
        let row_left = intervals.iter().map(|i| i.0).fold(f64::INFINITY, f64::min);
        let row_right = intervals.iter().map(|i| i.1).fold(f64::NEG_INFINITY, f64::max);
        if let Some((py, pl)) = prev {
            if y != py {
                path.push(
                    Segment::new(DVec2::new(pl, py), DVec2::new(row_left, y), false, SegKind::CombStepUp)
                        .row(ri),
                );
            }
        }
        let mut cursor = row_left;
        let mut sorted = intervals.clone();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (l, r) in sorted {
            if l > cursor + EPS {
                path.push(Segment::new(DVec2::new(cursor, y), DVec2::new(l, y), false, SegKind::CombGap).row(ri));
            }
            if r - l > EPS {
                path.push(Segment::new(DVec2::new(l, y), DVec2::new(r, y), true, SegKind::CombLitho).row(ri));
                cursor = r;
            }
        }
        if row_right - row_left > EPS {
            path.push(
                Segment::new(DVec2::new(row_right, y), DVec2::new(row_left, y), false, SegKind::CombReturn)
                    .row(ri),
            );
        }
        prev = Some((y, row_left));
    }
    Ok(path)
}

/// Full-frame comb: every non-empty row xmin→xmax (polygon intervals
/// active) then xmax→xmin inactive; rows joined at xmin.
fn comb_frame_raster(
    polygons: &[Polygon],
    width: f64,
    height: f64,
    pitch: f64,
) -> Result<Vec<Segment>, String> {
    let rows = raster_intervals(polygons, width, height, pitch);
    if rows.is_empty() {
        return Err(no_geometry());
    }
    let (xmin, xmax) = (-width / 2., width / 2.);
    let mut path = Vec::new();
    let mut prev_y: Option<f64> = None;
    for (ri, (y, intervals)) in rows.iter().enumerate() {
        let y = *y;
        if let Some(py) = prev_y {
            if y != py {
                path.push(
                    Segment::new(DVec2::new(xmin, py), DVec2::new(xmin, y), false, SegKind::FrameStepUp)
                        .row(ri),
                );
            }
        }
        let mut cursor = xmin;
        let mut sorted = intervals.clone();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (l, r) in sorted {
            if l > cursor + EPS {
                path.push(Segment::new(DVec2::new(cursor, y), DVec2::new(l, y), false, SegKind::FrameTravel).row(ri));
            }
            if r - l > EPS {
                path.push(Segment::new(DVec2::new(l, y), DVec2::new(r, y), true, SegKind::FrameLitho).row(ri));
                cursor = r;
            }
        }
        if xmax > cursor + EPS {
            path.push(Segment::new(DVec2::new(cursor, y), DVec2::new(xmax, y), false, SegKind::FrameTravel).row(ri));
        }
        path.push(Segment::new(DVec2::new(xmax, y), DVec2::new(xmin, y), false, SegKind::FrameReturn).row(ri));
        prev_y = Some(y);
    }
    Ok(path)
}

/// Perimeters of the polygons completely inside the frame, one at a time.
fn boundary_path(polygons: &[Polygon], width: f64, height: f64) -> Result<Vec<Segment>, String> {
    let half = DVec2::new(width / 2., height / 2.);
    let enclosed = polygons.iter().filter(|poly| {
        poly.len() >= 3 && poly.iter().all(|p| p.x >= -half.x && p.x <= half.x && p.y >= -half.y && p.y <= half.y)
    });
    let mut path = Vec::new();
    let mut found = false;
    let mut prev_end: Option<DVec2> = None;
    for poly in enclosed {
        found = true;
        let mut verts = poly.clone();
        if verts.len() > 1 && verts.first() == verts.last() {
            verts.pop();
        }
        if verts.len() < 3 {
            continue;
        }
        if let Some(prev) = prev_end {
            path.push(Segment::new(prev, verts[0], false, SegKind::PolygonJump));
        }
        let n = verts.len();
        for i in 0..n {
            let (a, b) = (verts[i], verts[(i + 1) % n]);
            if a.distance(b) <= EPS {
                continue;
            }
            path.push(Segment::new(a, b, true, SegKind::PolygonBoundary));
        }
        prev_end = path.last().map(|s| s.end);
    }
    if !found {
        return Err("No complete GDS polygon lies inside the lithography scan frame".into());
    }
    if path.is_empty() {
        return Err("No non-degenerate polygon boundary lies inside the lithography scan frame".into());
    }
    Ok(path)
}

/// Row-bundled patch raster: serpentine over all rows; consecutive rows of
/// the same connected patch turn actively at its edge, otherwise the tip
/// travels (same-row connector or patch jump).
fn local_raster(polygons: &[Polygon], width: f64, height: f64, pitch: f64) -> Result<Vec<Segment>, String> {
    let rows = raster_intervals(polygons, width, height, pitch);
    if rows.is_empty() {
        return Err(no_geometry());
    }
    // One node per interval; union overlapping intervals on adjacent rows.
    let mut nodes: Vec<(usize, f64, f64)> = Vec::new();
    let mut by_row: Vec<Vec<usize>> = Vec::with_capacity(rows.len());
    for (ri, (_, intervals)) in rows.iter().enumerate() {
        let mut ids = Vec::new();
        for &(l, r) in intervals {
            ids.push(nodes.len());
            nodes.push((ri, l, r));
        }
        by_row.push(ids);
    }
    let mut uf = UnionFind::new(nodes.len());
    for ri in 1..rows.len() {
        if rows[ri].0 - rows[ri - 1].0 > pitch * 1.5 {
            continue;
        }
        for &a in &by_row[ri - 1] {
            for &b in &by_row[ri] {
                let (_, l0, r0) = nodes[a];
                let (_, l1, r1) = nodes[b];
                if r0.min(r1) >= l0.max(l1) - 1e-9 {
                    uf.union(a, b);
                }
            }
        }
    }

    let mut path = Vec::new();
    let mut prev: Option<(DVec2, usize, usize)> = None; // (end, node, row)
    for (ri, (y, _)) in rows.iter().enumerate() {
        let y = *y;
        let forward = ri % 2 == 0;
        let mut ordered = by_row[ri].clone();
        ordered.sort_by(|&a, &b| nodes[a].1.total_cmp(&nodes[b].1));
        if !forward {
            ordered.reverse();
        }
        for node in ordered {
            let (_, l, r) = nodes[node];
            let (start, end) = if forward {
                (DVec2::new(l, y), DVec2::new(r, y))
            } else {
                (DVec2::new(r, y), DVec2::new(l, y))
            };
            if let Some((prev_end, prev_node, prev_row)) = prev {
                let same_row = ri == prev_row;
                let same_patch_turn = !same_row
                    && uf.find(prev_node) == uf.find(node)
                    && y - rows[prev_row].0 <= pitch * 1.5;
                if same_patch_turn {
                    path.push(Segment::new(prev_end, start, true, SegKind::PatchTurn));
                } else {
                    let kind = if same_row { SegKind::SameYConnector } else { SegKind::PatchJump };
                    path.push(Segment::new(prev_end, start, false, kind));
                }
            }
            path.push(Segment::new(start, end, true, SegKind::LithoRaster));
            prev = Some((end, node, ri));
        }
    }
    Ok(path)
}

struct UnionFind(Vec<usize>);
impl UnionFind {
    fn new(n: usize) -> Self {
        Self((0..n).collect())
    }
    fn find(&mut self, mut i: usize) -> usize {
        while self.0[i] != i {
            self.0[i] = self.0[self.0[i]];
            i = self.0[i];
        }
        i
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[rb] = ra;
        }
    }
}

/// Split a path into its runs of active segments.
fn split_components(path: &[Segment]) -> Vec<Vec<Segment>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    for s in path {
        if s.active {
            current.push(*s);
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// (start, end) of each component in its forward orientation.
type Ends = Vec<(DVec2, DVec2)>;

/// End point of component `i` in `orientation` (0 forward, 1 reversed).
fn exit(ends: &Ends, i: usize, o: usize) -> DVec2 {
    if o == 0 { ends[i].1 } else { ends[i].0 }
}
fn entry(ends: &Ends, i: usize, o: usize) -> DVec2 {
    if o == 0 { ends[i].0 } else { ends[i].1 }
}

/// Extend the per-orientation DP costs of an order ending in `prev` by `next`.
fn extend_cost(ends: &Ends, costs: [f64; 2], prev: usize, next: usize) -> [f64; 2] {
    let step = |o: usize| {
        (0..2)
            .map(|po| costs[po] + exit(ends, prev, po).distance(entry(ends, next, o)))
            .fold(f64::INFINITY, f64::min)
    };
    [step(0), step(1)]
}

/// Best travel cost of `order` with each component oriented optimally.
fn order_cost(ends: &Ends, order: &[usize]) -> f64 {
    let mut costs = [0., 0.];
    for w in order.windows(2) {
        costs = extend_cost(ends, costs, w[0], w[1]);
    }
    costs[0].min(costs[1])
}

/// Orientation (0/1) of each component in `order` minimising travel.
fn orient(ends: &Ends, order: &[usize]) -> Vec<usize> {
    if order.is_empty() {
        return vec![];
    }
    let mut costs = [0., 0.];
    let mut parents: Vec<[usize; 2]> = Vec::new();
    for w in order.windows(2) {
        let (prev, cur) = (w[0], w[1]);
        let mut next = [0.; 2];
        let mut parent = [0; 2];
        for o in 0..2 {
            let start = entry(ends, cur, o);
            // Ties keep the earlier (forward) orientation, as in Python's min().
            let c0 = costs[0] + exit(ends, prev, 0).distance(start);
            let c1 = costs[1] + exit(ends, prev, 1).distance(start);
            (next[o], parent[o]) = if c1 < c0 { (c1, 1) } else { (c0, 0) };
        }
        costs = next;
        parents.push(parent);
    }
    let mut o = if costs[1] < costs[0] { 1 } else { 0 };
    let mut out = vec![o];
    for parent in parents.iter().rev() {
        o = parent[o];
        out.push(o);
    }
    out.reverse();
    out
}

/// Patch-local rasters with the patch order and direction chosen to shorten
/// travel: nearest-neighbour start, bounded open-route 2-opt.
fn shortest_path(polygons: &[Polygon], width: f64, height: f64, pitch: f64) -> Result<Vec<Segment>, String> {
    let base = local_raster(polygons, width, height, pitch)?;
    let components = split_components(&base);
    if components.len() < 2 {
        return Ok(base);
    }
    let n = components.len();
    let ends: Ends = components
        .iter()
        .map(|c| (c[0].start, c[c.len() - 1].end))
        .collect();

    // Nearest neighbour, scored by the best-oriented cost of the order so far.
    let mut order = vec![0];
    let mut remaining: Vec<usize> = (1..n).collect();
    let mut costs = [0., 0.];
    while !remaining.is_empty() {
        let last = *order.last().unwrap();
        let (pos, next_costs) = remaining
            .iter()
            .enumerate()
            .map(|(pos, &i)| (pos, extend_cost(&ends, costs, last, i)))
            .fold(None, |best: Option<(usize, [f64; 2])>, (pos, c)| match best {
                Some((_, b)) if b[0].min(b[1]) <= c[0].min(c[1]) => best,
                _ => Some((pos, c)),
            })
            .unwrap();
        order.push(remaining.remove(pos));
        costs = next_costs;
    }

    // Open-route 2-opt, bounded so large layouts stay quick.
    let mut best = order_cost(&ends, &order);
    let max_evaluations = (n * 4).max(80).min(200);
    let mut evaluations = 0;
    let mut changed = true;
    while changed && evaluations < max_evaluations {
        changed = false;
        'search: for left in 1..n.saturating_sub(2) {
            for right in left + 1..n - 1 {
                evaluations += 1;
                let mut candidate = order.clone();
                candidate[left..=right].reverse();
                let cost = order_cost(&ends, &candidate);
                if cost + 1e-9 < best {
                    order = candidate;
                    best = cost;
                    changed = true;
                    break 'search;
                }
            }
        }
    }

    let orientations = orient(&ends, &order);
    let mut path = Vec::new();
    let mut prev_end: Option<DVec2> = None;
    for (&i, &o) in order.iter().zip(&orientations) {
        let comp: Vec<Segment> = if o == 0 {
            components[i].clone()
        } else {
            components[i].iter().rev().map(Segment::reversed).collect()
        };
        if let Some(prev) = prev_end {
            path.push(Segment::new(prev, comp[0].start, false, SegKind::PathfinderJump));
        }
        prev_end = Some(comp[comp.len() - 1].end);
        path.extend(comp);
    }
    Ok(path)
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PathStats {
    pub segments: usize,
    pub active_nm: f64,
    pub travel_nm: f64,
    /// Changes between active and inactive along the path.
    pub transitions: usize,
}
impl PathStats {
    pub fn of(path: &[Segment]) -> Self {
        Self {
            segments: path.len(),
            active_nm: path.iter().filter(|s| s.active).map(Segment::length).sum(),
            travel_nm: path.iter().filter(|s| !s.active).map(Segment::length).sum(),
            transitions: path.windows(2).filter(|w| w[0].active != w[1].active).count(),
        }
    }
    /// Rough run time (s) at `speed_nm_s`, as estimated by the Python GUI:
    /// path length / speed + 50 ms per transition (+ the ramp, if used).
    pub fn estimate_s(&self, speed_nm_s: f64, ramp_s: Option<f64>) -> f64 {
        (self.active_nm + self.travel_nm) / speed_nm_s
            + self.transitions as f64 * (0.05 + ramp_s.unwrap_or(0.))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon {
        vec![
            DVec2::new(x0, y0),
            DVec2::new(x1, y0),
            DVec2::new(x1, y1),
            DVec2::new(x0, y1),
        ]
    }
    fn continuous(path: &[Segment]) -> bool {
        path.windows(2).all(|w| w[0].end == w[1].start)
    }
    fn three() -> Vec<Polygon> {
        vec![
            rect(-8., -5., -2., 0.),
            rect(2., -5., 8., 0.),
            rect(-4., 2., 4., 5.),
        ]
    }

    // --- from test_comb_path.py (CombPathTests) ---

    #[test]
    fn comb_active_segments_are_left_to_right() {
        let path = comb_raster(&three(), 20., 12., 2.).unwrap();
        let active: Vec<_> = path.iter().filter(|s| s.active).collect();
        assert!(!active.is_empty());
        assert!(active.iter().all(|s| s.start.x < s.end.x && s.kind == SegKind::CombLitho));
    }

    #[test]
    fn comb_rows_start_and_return_at_active_extents() {
        let path = comb_raster(&three(), 20., 12., 2.).unwrap();
        let rows: std::collections::BTreeSet<_> = path.iter().filter_map(|s| s.row).collect();
        for r in rows {
            let items: Vec<_> = path.iter().filter(|s| s.row == Some(r)).collect();
            let active: Vec<_> = items.iter().filter(|s| s.active).collect();
            let left = active.iter().map(|s| s.start.x).fold(f64::INFINITY, f64::min);
            let right = active.iter().map(|s| s.end.x).fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(items.iter().find(|s| s.active).unwrap().start.x, left);
            let returns: Vec<_> = items.iter().filter(|s| s.kind == SegKind::CombReturn).collect();
            assert_eq!(returns.len(), 1);
            assert_eq!(returns[0].start.x, right);
            assert_eq!(returns[0].end.x, left);
        }
    }

    #[test]
    fn comb_returns_inactive_and_steps_at_left_extents() {
        let path = comb_raster(&three(), 20., 12., 2.).unwrap();
        let returns: Vec<_> = path.iter().filter(|s| s.kind == SegKind::CombReturn).collect();
        assert!(returns.len() >= 2);
        assert!(returns.iter().all(|s| !s.active && s.start.x > s.end.x && s.start.y == s.end.y));
        let steps: Vec<_> = path.iter().filter(|s| s.kind == SegKind::CombStepUp).collect();
        assert!(!steps.is_empty());
        for s in steps {
            assert!(!s.active && s.end.y > s.start.y);
            let r = s.row.unwrap();
            let left_of = |row| {
                path.iter()
                    .filter(|p| p.row == Some(row) && p.active)
                    .map(|p| p.start.x)
                    .fold(f64::INFINITY, f64::min)
            };
            assert_eq!(s.start.x, left_of(r - 1));
            assert_eq!(s.end.x, left_of(r));
        }
        assert!(continuous(&path));
        assert!(!path.iter().any(|s| s.active && s.start.x >= s.end.x));
    }

    #[test]
    fn comb_uses_full_frame_only_when_geometry_reaches_it() {
        let path = comb_raster(&[rect(-10., -5., 10., 0.)], 20., 12., 2.).unwrap();
        let returns: Vec<_> = path.iter().filter(|s| s.kind == SegKind::CombReturn).collect();
        assert!(returns.iter().any(|s| s.start.x == 10.));
        assert!(returns.iter().any(|s| s.end.x == -10.));
    }

    // --- from test_comb_path.py (FullFrameCombPathTests) ---

    #[test]
    fn full_frame_comb_rows_span_exact_frame() {
        let path = comb_frame_raster(&three(), 20., 12., 2.).unwrap();
        let rows: std::collections::BTreeSet<_> = path.iter().filter_map(|s| s.row).collect();
        for &r in &rows {
            let items: Vec<_> = path.iter().filter(|s| s.row == Some(r)).collect();
            let forward: Vec<_> = items
                .iter()
                .filter(|s| s.start.y == s.end.y && s.start.x < s.end.x)
                .collect();
            assert_eq!(forward[0].start.x, -10.);
            assert_eq!(forward.last().unwrap().end.x, 10.);
            let fwd_len: f64 = forward.iter().map(|s| s.length()).sum();
            assert!((fwd_len - 20.).abs() < 1e-9);
            let rev: Vec<_> = items.iter().filter(|s| s.kind == SegKind::FrameReturn).collect();
            assert_eq!(rev.len(), 1);
            assert!(!rev[0].active && rev[0].start.x == 10. && rev[0].end.x == -10.);
        }
        let steps: Vec<_> = path.iter().filter(|s| s.kind == SegKind::FrameStepUp).collect();
        assert_eq!(steps.len(), rows.len() - 1);
        assert!(steps.iter().all(|s| !s.active && s.start.x == -10. && s.end.x == -10.));
        assert!(path.iter().filter(|s| s.active).all(|s| s.start.x < s.end.x && s.kind == SegKind::FrameLitho));
        assert!(continuous(&path));
    }

    // --- from the GUI --self-test ---

    #[test]
    fn local_raster_single_patch_is_continuous() {
        let path = local_raster(&[rect(-5., -5., 5., 5.)], 20., 20., 0.77).unwrap();
        assert!(!path.is_empty() && continuous(&path));
    }

    #[test]
    fn outside_polygons_give_no_rows() {
        let polys = [rect(-20., -5., -12., 5.), rect(12., -5., 20., 5.)];
        assert!(raster_intervals(&polys, 20., 20., 0.77).is_empty());
    }

    #[test]
    fn local_raster_split_patches_jump_once() {
        let path = local_raster(&[rect(-3., -5., 3., -2.), rect(-3., 2., 3., 5.)], 20., 20., 0.77).unwrap();
        let jumps = path.iter().filter(|s| s.kind == SegKind::PatchJump).count();
        assert_eq!(jumps, 1);
        assert!(path.iter().all(|s| s.active || s.kind == SegKind::PatchJump));
    }

    #[test]
    fn local_raster_side_by_side_uses_same_row_connectors() {
        let path = local_raster(&[rect(-9., -4., -3., 0.), rect(3., -4., 9., 0.)], 20., 20., 0.77).unwrap();
        assert!(path.iter().filter(|s| s.kind == SegKind::SameYConnector).count() >= 2);
        assert!(!path.iter().any(|s| s.kind == SegKind::PatchJump));
    }

    #[test]
    fn shortest_path_is_continuous_with_jumps() {
        let polys = [
            rect(-8., 4., -5., 9.),
            rect(-8., -1., -5., 2.),
            rect(-8., -9., -5., -5.),
            rect(3., -9., 9., 7.),
        ];
        let local = local_raster(&polys, 20., 20., 0.77).unwrap();
        assert!(local.iter().filter(|s| !s.active && s.kind == SegKind::SameYConnector).count() >= 2);
        let path = shortest_path(&polys, 20., 20., 0.77).unwrap();
        assert!(continuous(&path));
        assert!(path.iter().any(|s| s.kind == SegKind::PathfinderJump));
        // Reordering only changes travel, never the written length.
        let written = |p: &[Segment]| PathStats::of(p).active_nm;
        let base_written: f64 = split_components(&local).iter().flatten().map(|s| s.length()).sum();
        assert!((written(&path) - base_written).abs() < 1e-6);
        assert!(PathStats::of(&path).travel_nm <= PathStats::of(&local).travel_nm + 1e-6);
    }

    #[test]
    fn boundary_writes_enclosed_polygons_only() {
        let a = rect(-10., -10., 0., 0.);
        let b = rect(10., 10., 20., 20.);
        let path = boundary_path(&[a.clone(), b], 40., 40.).unwrap();
        assert_eq!(path.iter().filter(|s| s.active).count(), 8);
        assert_eq!(path.iter().filter(|s| s.kind == SegKind::PolygonJump).count(), 1);

        let complete = rect(-5., -5., 5., 5.);
        let partial = vec![DVec2::new(-30., 0.), DVec2::new(0., 0.), DVec2::new(0., 1.)];
        let path = boundary_path(&[complete, partial.clone()], 40., 40.).unwrap();
        assert_eq!(path.iter().filter(|s| s.active).count(), 4);
        assert!(boundary_path(&[partial], 40., 40.).is_err());
    }

    #[test]
    fn plan_clamps_inside_clearance_except_full_frame() {
        let polys = [rect(-50., -50., 50., 50.)];
        let size = DVec2::new(20., 20.);
        let half = 10. - edge_clearance(1.);
        let p = plan(PathMode::PatchLocal, &polys, size, 1.).unwrap();
        assert!(p.iter().all(|s| s.start.abs().max_element() <= half + 1e-12));
        let f = plan(PathMode::CombFrame, &polys, size, 1.).unwrap();
        assert!(f.iter().any(|s| s.start.x == -10.));
        assert!(plan(PathMode::Comb, &polys, size, 0.).is_err());
        assert!(plan(PathMode::Comb, &polys, DVec2::new(0.5, 0.5), 1.).is_err());
    }

    #[test]
    fn stats_and_estimate() {
        let path = comb_raster(&[rect(-5., -5., 5., 5.)], 20., 20., 5.).unwrap();
        let s = PathStats::of(&path);
        assert!(s.active_nm > 0. && s.travel_nm > 0. && s.transitions > 0);
        let t = s.estimate_s(10., Some(0.2));
        assert!((t - ((s.active_nm + s.travel_nm) / 10. + s.transitions as f64 * 0.25)).abs() < 1e-9);
    }
}
