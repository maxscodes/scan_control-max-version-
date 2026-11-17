//! Closed-loop survey grids. Where each tile sits on the map is taken as
//! where the tip really was: a new tile is placed by matching it with the
//! tile before (the move that brought the tip there), and the user can drag
//! any tile to the right spot. From those positions the grid
//!
//! * knows where the tip is now, and corrects the next move so the next
//!   tile lands where it was planned, and
//! * learns how far each motor steps (a calibration to save at the end).
//!
//! Stick-slip steps vary by 10–30 % from run to run, so fixed steps per tile
//! let the error pile up along a row; this keeps each tile's error to that
//! of one move.
//!
//! Tiles share only a strip, and the surface can repeat, so a look-alike can
//! match about as well as the real overlap. A match is only taken when it is
//! near where the move should have gone and clearly better than any other
//! spot there; one move can only nudge the learned step; and a corrected
//! move stays near the planned one. Every automatic fix can be undone.
//! Tiles already stamped are never moved automatically: what is learned
//! changes where the next tiles go.

use glam::{DMat2, DVec2, IVec2};
use uuid::Uuid;

use crate::{
    analysis::register::{shift_candidates, GOOD_SCORE},
    connection::nanonis::{
        coarse_cal::{shift_to_tip, CalFrame},
        course_motion::CoarseCalibration,
    },
};

/// A measured step this far from the calibration (ratio) is not learned from.
const MAX_RATIO: f64 = 1.6;
/// A measured step turned further than this from the calibration's
/// direction (degrees) is not learned from: one motor can't do that.
const MAX_TURN_DEG: f64 = 20.;
/// The starting calibration counts as this many moves in each fit, so one
/// move only nudges the learned step.
const PRIOR_MOVES: f64 = 0.5;
/// Learned step ÷ starting calibration is kept within this range.
const LEARN_RANGE: (f64, f64) = (0.75, 1.33);
/// Largest correction added to a move, share of the tile spacing.
const MAX_CORRECTION: f64 = 0.5;
/// A corrected move differs from the planned one by at most this share of
/// the planned move's largest step count (at least 2 steps) per motor.
const MAX_STEP_CHANGE: f64 = 0.5;
/// Share of the image two tiles must have in common to be matched: less
/// leaves too little to tell a real match from a look-alike.
const MIN_OVERLAP: f64 = 0.15;
/// A match is only looked for this far (share of the predicted move) from
/// where the move should have gone. Steps vary by 10–30 %.
const MAX_DEVIATION: f64 = 0.35;
/// Another spot scoring within this of the best one makes the match unclear.
const AMBIGUOUS_MARGIN: f64 = 0.05;

/// Why a tile was not matched with the one before.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unmatched {
    /// No convincing match near where the move should have gone.
    NoMatch,
    /// Two different spots matched about equally well (a repeating surface).
    Ambiguous,
}

/// How a tile was placed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    /// The first tile: where it was planned, by definition.
    First,
    /// Matched with the tile before (correlation score).
    Matched { score: f64 },
    /// Placed where the grid's calibration predicts.
    Predicted(Unmatched),
}

struct Tile {
    /// Its stamp, once taken.
    id: Option<Uuid>,
    /// Sequence entry that scanned it.
    entry: usize,
    /// Planned position, relative to the first tile (piezo nm).
    target: DVec2,
    /// Steps of the move that brought the tip here.
    steps_in: Option<IVec2>,
    /// Where it really was (piezo nm, relative to the first tile's plan).
    pos: DVec2,
    /// Its stamp's position on the map when last looked at.
    world: Option<DVec2>,
    placement: Placement,
    /// How far the automatic placement put the stamp from where the main
    /// coarse calibration alone would have (map nm).
    auto_shift: DVec2,
    /// The automatic placement was undone: the stamp sits where the main
    /// calibration put it, and moves into or out of it are not learned from.
    undone: bool,
    /// Dragged on the map by the user (its position is then known even if
    /// it wasn't matched).
    dragged: bool,
}

impl Tile {
    /// Where it is was measured: matched with the tile before, or dragged
    /// into place (and its automatic placement not undone).
    fn measured(&self) -> bool {
        !self.undone && (self.dragged || matches!(self.placement, Placement::Matched { .. } | Placement::First))
    }
}

/// One tile, for the list of automatic fixes.
#[derive(Debug, Clone, Copy)]
pub struct TileInfo {
    pub index: usize,
    pub entry: usize,
    pub placement: Placement,
    /// Map nm the automatic placement moved the stamp.
    pub auto_shift_nm: f64,
    pub undone: bool,
    /// Distance from its planned position, nm (none for the first tile).
    pub error_nm: Option<f64>,
}

/// A stamp to move on the map (undo or redo of an automatic placement).
#[derive(Debug, Clone, Copy)]
pub struct StampMove {
    pub id: Uuid,
    /// Map nm.
    pub delta: DVec2,
    /// The latest tile: the tip (scan area) moves with it.
    pub latest: bool,
}

pub struct GridTrack {
    /// Calibration the run started with.
    base: CoarseCalibration,
    /// Calibration fitted to the tiles' positions (used for the moves).
    pub calib: CoarseCalibration,
    volts: DVec2,
    /// Map (world) displacement → piezo nm.
    to_piezo: DMat2,
    /// Piezo nm → map displacement.
    to_world: DMat2,
    tiles: Vec<Tile>,
    prev_frame: Option<CalFrame>,
    /// The last coarse move, as the map placed it (piezo nm), once the scan
    /// area was moved with the grid's calibration.
    moved_on_map: Option<(IVec2, DVec2)>,
    /// Learn the step size from the tiles (off: the starting calibration).
    learn: bool,
    pub matched: usize,
    pub unmatched: usize,
    pub dragged: usize,
}

impl GridTrack {
    /// `world_m2`: the scan area's piezo → map matrix.
    pub fn new(calib: CoarseCalibration, volts: DVec2, world_m2: DMat2) -> Self {
        let ok = world_m2.determinant().abs() > 1e-12;
        Self {
            base: calib,
            calib,
            volts,
            to_piezo: if ok { world_m2.inverse() } else { DMat2::IDENTITY },
            to_world: if ok { world_m2 } else { DMat2::IDENTITY },
            tiles: Vec::new(),
            prev_frame: None,
            moved_on_map: None,
            learn: true,
            matched: 0,
            unmatched: 0,
            dragged: 0,
        }
    }

    /// Where a move of `steps` goes with what the grid has learned (piezo nm).
    pub fn predict(&self, steps: IVec2) -> DVec2 {
        self.calib.steps_to_real(steps, self.volts)
    }

    /// A coarse move to the next tile finished, and the map moved the scan
    /// area with `map_calib`. Returns how much further (piezo nm) to move
    /// it so it sits where the grid's calibration puts it.
    pub fn moved(&mut self, steps: IVec2, map_calib: &CoarseCalibration) -> DVec2 {
        let grid = self.predict(steps);
        self.moved_on_map = Some((steps, grid));
        grid - map_calib.steps_to_real(steps, self.volts)
    }

    /// A tile was scanned by `entry` (planned at `target`; `steps_in`: the
    /// move that brought the tip here). `map_calib`: the main calibration,
    /// which the map moves the scan area with. Returns how far (piezo nm)
    /// the tip is from where the scan area is now, to place the stamp right.
    pub fn tile_scanned(
        &mut self,
        frame: CalFrame,
        entry: usize,
        target: DVec2,
        steps_in: Option<IVec2>,
        map_calib: &CoarseCalibration,
    ) -> DVec2 {
        let moved_on_map = self.moved_on_map.take();
        let (pos, off, placement, auto) = match (self.tiles.last(), self.prev_frame.take(), steps_in) {
            (Some(last), Some(prev), Some(steps)) => {
                let predicted = self.predict(steps);
                let by_main = map_calib.steps_to_real(steps, self.volts);
                // Where the scan area is now: moved with the grid's
                // calibration if `moved` was told, else the main one.
                let on_map = match moved_on_map {
                    Some((s, at)) if s == steps => at,
                    _ => by_main,
                };
                let (real, placement) = match measure(&prev, &frame, predicted) {
                    Ok((real, score)) => {
                        self.matched += 1;
                        (real, Placement::Matched { score })
                    }
                    Err(why) => {
                        self.unmatched += 1;
                        (predicted, Placement::Predicted(why))
                    }
                };
                (last.pos + real, real - on_map, placement, self.to_world * (real - by_main))
            }
            _ => (target, DVec2::ZERO, Placement::First, DVec2::ZERO),
        };
        self.tiles.push(Tile {
            id: None,
            entry,
            target,
            steps_in,
            pos,
            world: None,
            placement,
            auto_shift: auto,
            undone: false,
            dragged: false,
        });
        self.prev_frame = Some(frame);
        self.refit();
        off
    }

    /// The latest tile's stamp.
    pub fn set_latest_id(&mut self, id: Uuid) {
        if let Some(t) = self.tiles.last_mut() {
            t.id = Some(id);
        }
    }
    pub fn ids(&self) -> Vec<Uuid> {
        self.tiles.iter().filter_map(|t| t.id).collect()
    }
    /// Entry of the latest tile.
    pub fn latest_entry(&self) -> Option<usize> {
        self.tiles.last().map(|t| t.entry)
    }

    /// Take in where the stamps are on the map (`world`: stamp id → map
    /// position). Returns `Some(map displacement)` of the latest tile if a
    /// stamp was moved (the tip moved with it), `None` if nothing changed.
    pub fn sync(&mut self, world: impl Fn(Uuid) -> Option<DVec2>) -> Option<DVec2> {
        let mut changed = false;
        let mut latest_delta = DVec2::ZERO;
        let n = self.tiles.len();
        for (k, t) in self.tiles.iter_mut().enumerate() {
            let Some(w) = t.id.and_then(&world) else { continue };
            if let Some(old) = t.world {
                if (w - old).length() > 1e-6 {
                    changed = true;
                    t.dragged = true;
                    if k + 1 == n {
                        latest_delta = w - old;
                    }
                }
            }
            t.world = Some(w);
        }
        if !changed {
            return None;
        }
        self.dragged += 1;
        self.positions_from_map();
        self.refit();
        Some(latest_delta)
    }

    /// Positions from the map, relative to the first tile.
    fn positions_from_map(&mut self) {
        if let Some(w0) = self.tiles.first().and_then(|t| t.world) {
            let p0 = self.tiles[0].pos;
            for t in &mut self.tiles {
                if let Some(w) = t.world {
                    t.pos = p0 + self.to_piezo * (w - w0);
                }
            }
        }
    }

    /// Undo (or redo) the automatic placement of tile `k`: its stamp goes
    /// back to where the main calibration put it (or to where the match
    /// put it), and the moves into and out of it stop (or start again)
    /// teaching the step size. Returns the stamp move to make on the map.
    pub fn set_undone(&mut self, k: usize, undone: bool) -> Option<StampMove> {
        let latest = k + 1 == self.tiles.len();
        let to_piezo = self.to_piezo;
        let t = self.tiles.get_mut(k)?;
        if t.undone == undone || t.placement == Placement::First {
            return None;
        }
        t.undone = undone;
        let delta = if undone { -t.auto_shift } else { t.auto_shift };
        t.pos += to_piezo * delta;
        // The map is updated by the caller: this is not a drag.
        if let Some(w) = &mut t.world {
            *w += delta;
        }
        let id = t.id;
        self.refit();
        Some(StampMove { id: id?, delta, latest })
    }

    /// Learn the step size from the tiles (off: moves use the calibration
    /// the run started with).
    pub fn set_learning(&mut self, on: bool) {
        self.learn = on;
        self.refit();
    }
    pub fn learning(&self) -> bool {
        self.learn
    }

    /// Steps to the tile planned at `next_target`, from where the latest
    /// tile really is; `planned`: the steps the grid was planned with.
    pub fn next_steps(&self, next_target: DVec2, planned: IVec2) -> Option<IVec2> {
        let last = self.tiles.last()?;
        // Nothing measured about where the latest tile is (not matched, not
        // dragged, or its placement undone): no reason to change the plan.
        if !last.measured() {
            return (planned != IVec2::ZERO).then_some(planned);
        }
        // The planned step plus a correction for where the latest tile
        // really is, at most half a tile (a bad position can't throw the
        // grid far off).
        let plan = next_target - last.target;
        let mut correction = last.target - last.pos;
        let cap = MAX_CORRECTION * plan.length();
        if correction.length() > cap {
            correction = correction.normalize_or_zero() * cap;
        }
        let mut steps = self.calib.real_to_steps(plan + correction, self.volts);
        // And never far from the planned steps, whatever was learned.
        let lim = ((MAX_STEP_CHANGE * planned.abs().max_element() as f64).ceil() as i32).max(2);
        steps = steps.clamp(planned - IVec2::splat(lim), planned + IVec2::splat(lim));
        (steps != IVec2::ZERO).then_some(steps)
    }

    /// Distance of the latest tile from its plan, nm.
    pub fn latest_error_nm(&self) -> Option<f64> {
        let t = self.tiles.last()?;
        (self.tiles.len() > 1).then(|| (t.pos - t.target).length())
    }

    pub fn tiles_info(&self) -> Vec<TileInfo> {
        self.tiles
            .iter()
            .enumerate()
            .map(|(index, t)| TileInfo {
                index,
                entry: t.entry,
                placement: t.placement,
                auto_shift_nm: t.auto_shift.length(),
                undone: t.undone,
                error_nm: (index > 0).then(|| (t.pos - t.target).length()),
            })
            .collect()
    }

    /// Moves between consecutive tiles: (steps, real move in piezo nm).
    /// Only moves whose end was measured (matched, or dragged into place)
    /// teach anything; moves into or out of a tile whose placement was
    /// undone are left out.
    fn moves(&self) -> Vec<(IVec2, DVec2)> {
        self.tiles
            .windows(2)
            .filter(|p| !p[0].undone && p[1].measured())
            .filter_map(|p| Some((p[1].steps_in?, p[1].pos - p[0].pos)))
            .collect()
    }
    pub fn n_moves(&self) -> usize {
        self.moves().len()
    }

    /// Fit each motor direction's step to the moves between tiles (least
    /// squares, with the starting calibration counting as a move; directions
    /// without moves keep it).
    fn refit(&mut self) {
        if !self.learn {
            self.calib = self.base;
            return;
        }
        // Per motor and direction: Σ column·n², Σ n², moves.
        let mut sums = [[(DVec2::ZERO, 0f64, 0usize); 2]; 2];
        let max_turn = MAX_TURN_DEG.to_radians().cos();
        for (steps, real) in self.moves() {
            let part = |axis: usize| {
                let mut s = IVec2::ZERO;
                s[axis] = steps[axis];
                self.base.steps_to_real(s, self.volts)
            };
            let (main, other) = if part(0).length() >= part(1).length() { (0, 1) } else { (1, 0) };
            let (n, v) = (steps[main] as f64, self.volts[main]);
            if n == 0. || v <= 0. {
                continue;
            }
            // What the main motor did: the move minus the other motor's share.
            let column = (real - part(other)) / (n * v) * 1e-9;
            let old = self.base.column(main, n > 0.);
            let ratio = column.length() / old.length().max(f64::MIN_POSITIVE);
            let turn = column.dot(old) / (column.length() * old.length()).max(f64::MIN_POSITIVE);
            if !(1. / MAX_RATIO..=MAX_RATIO).contains(&ratio) || turn < max_turn || !column.is_finite() {
                continue;
            }
            // Weighted by n², as fitting real = column · n · V.
            let s = &mut sums[main][(n > 0.) as usize];
            s.0 += column * n * n;
            s.1 += n * n;
            s.2 += 1;
        }
        let mut c = self.base;
        for axis in 0..2 {
            for pos in [false, true] {
                let (num, den, k) = sums[axis][pos as usize];
                if k == 0 {
                    continue;
                }
                let old = self.base.column(axis, pos);
                // The calibration as PRIOR_MOVES typical moves.
                let w = PRIOR_MOVES * den / k as f64;
                let fit = (num + old * w) / (den + w);
                let ratio = (fit.length() / old.length().max(f64::MIN_POSITIVE)).clamp(LEARN_RANGE.0, LEARN_RANGE.1);
                c.set_column(axis, pos, fit.normalize_or_zero() * old.length() * ratio);
            }
        }
        self.calib = c;
    }

    /// Step of each motor direction learned so far ÷ the starting
    /// calibration's (for the summary).
    fn learned_ratios(&self) -> Vec<(&'static str, f64)> {
        let mut out = Vec::new();
        for (axis, pos, name) in [(0, true, "+X"), (0, false, "\u{2212}X"), (1, true, "+Y"), (1, false, "\u{2212}Y")] {
            let (a, b) = (self.calib.column(axis, pos), self.base.column(axis, pos));
            if a != b && b.length() > 0. {
                out.push((name, a.length() / b.length()));
            }
        }
        out
    }

    pub fn summary(&self) -> String {
        let mut s = format!("Grid tracking: {} tiles, {} moves matched automatically", self.tiles.len(), self.matched);
        if self.unmatched > 0 {
            s += &format!(", {} not (placed as predicted: drag them into place)", self.unmatched);
        }
        if self.dragged > 0 {
            s += &format!(", tiles moved by hand {}\u{d7}", self.dragged);
        }
        if let Some(e) = self.latest_error_nm() {
            s += &format!(" \u{b7} latest tile {e:.0} nm from plan");
        }
        if !self.learn {
            s += " \u{b7} step size not learned (calibration as at the start)";
        } else {
            let r = self.learned_ratios();
            if !r.is_empty() {
                let parts: Vec<String> = r.iter().map(|(n, x)| format!("{n} {:.0} %", x * 100.)).collect();
                s += &format!(" \u{b7} steps vs calibration: {}", parts.join(", "));
            }
        }
        s
    }
}

/// Tip move (nm) between two tiles of the same frame, looked for only near
/// the predicted move, with its score; an error when there is no clear match
/// there.
fn measure(prev: &CalFrame, cur: &CalFrame, predicted: DVec2) -> Result<(DVec2, f64), Unmatched> {
    if prev.data.size != cur.data.size {
        return Err(Unmatched::NoMatch);
    }
    let [w, h] = prev.data.size;
    let m = prev.frame.matrix2;
    if m.determinant().abs() < 1e-12 || w == 0 || h == 0 {
        return Err(Unmatched::NoMatch);
    }
    // Inverse of `shift_to_tip`: content moves opposite to the tip.
    let local = m.inverse() * predicted;
    let guess = (-local.x * w as f64, -local.y * h as f64);
    let width = m.x_axis.length().max(m.y_axis.length());
    let tol = (MAX_DEVIATION * predicted.length()).max(0.05 * width);
    // Pixels per nm along the finer axis.
    let px_per_nm = (w as f64 / m.x_axis.length()).max(h as f64 / m.y_axis.length());
    let found = shift_candidates(
        &prev.data.data,
        &cur.data.data,
        w as usize,
        h as usize,
        guess,
        tol * px_per_nm,
        MIN_OVERLAP,
        3,
    );
    let tips: Vec<(DVec2, f64)> = found
        .iter()
        .map(|s| (shift_to_tip(prev, DVec2::new(s.dx, s.dy)), s.score))
        .filter(|(tip, _)| (*tip - predicted).length() <= tol)
        .collect();
    let Some(&(best, score)) = tips.first() else { return Err(Unmatched::NoMatch) };
    if score < GOOD_SCORE {
        return Err(Unmatched::NoMatch);
    }
    let rival = tips[1..].iter().any(|&(tip, sc)| {
        sc >= GOOD_SCORE && sc >= score - AMBIGUOUS_MARGIN && (tip - best).length() > 0.03 * width
    });
    if rival {
        return Err(Unmatched::Ambiguous);
    }
    Ok((best, score))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::DAffine2;

    use super::*;
    use crate::connection::FrameData;

    fn surface(x: f64, y: f64) -> f32 {
        let mut v = 0.;
        for k in 1..6 {
            let k = k as f64;
            v += (x * 0.011 * k + (y * 0.007 * k).sin() * 3.).sin() / k
                + (y * 0.013 * k + (x * 0.005 * k).cos() * 2.).cos() / k;
        }
        v as f32
    }

    fn scan(frame: DAffine2, tip: DVec2) -> CalFrame {
        let n = 128;
        let mut d = vec![0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let local = DVec2::new((i as f64 + 0.5) / n as f64 - 0.5, (j as f64 + 0.5) / n as f64 - 0.5);
                let p = frame.transform_point2(local) + tip;
                d[j * n + i] = surface(p.x, p.y);
            }
        }
        CalFrame {
            data: FrameData { size: [n as u32, n as u32], data: Arc::new(d.into_boxed_slice()) },
            frame,
            source: String::new(),
        }
    }

    fn calib() -> CoarseCalibration {
        // X motor moves along −y, Y motor along +x, 10 nm/step.
        CoarseCalibration::from_matrix(DMat2::from_cols(DVec2::new(0., -1e-8), DVec2::new(1e-8, 0.)))
    }

    fn flat() -> CalFrame {
        CalFrame {
            data: FrameData { size: [16, 16], data: Arc::new(vec![0f32; 256].into_boxed_slice()) },
            frame: DAffine2::from_scale(DVec2::new(1000., -1000.)),
            source: String::new(),
        }
    }

    /// 4 × 3 snake, tiles 600 nm apart (40 % overlap of a 1 µm frame).
    fn plan() -> Vec<DVec2> {
        let mut plan = Vec::new();
        for r in 0..3 {
            let cs: Vec<i32> = if r % 2 == 0 { (0..4).collect() } else { (0..4).rev().collect() };
            for c in cs {
                plan.push(DVec2::new(c as f64 * 600., -(r as f64) * 600.));
            }
        }
        plan
    }

    /// The motor steps 20 % further than calibrated (X back only 85 %):
    /// with tracking every tile lands near its plan; without it, the error
    /// grows along the row.
    #[test]
    fn corrects_moves_and_learns_the_step() {
        let frame = DAffine2::from_scale_angle_translation(DVec2::new(1000., -1000.), 0., DVec2::ZERO);
        let volts = DVec2::ONE;
        let calib = calib();
        let mut truth = calib;
        truth.set_column(0, true, calib.column(0, true) * 1.2);
        truth.set_column(0, false, calib.column(0, false) * 0.85);
        truth.set_column(1, true, calib.column(1, true) * 1.2);
        truth.set_column(1, false, calib.column(1, false) * 1.2);
        let plan = plan();
        let run = |track_on: bool| -> (f64, CoarseCalibration, usize) {
            let mut track = GridTrack::new(calib, volts, DMat2::IDENTITY);
            let mut tip = DVec2::ZERO;
            let mut worst: f64 = 0.;
            let mut steps: Option<IVec2> = None;
            for (k, &target) in plan.iter().enumerate() {
                if let Some(s) = steps {
                    tip += truth.steps_to_real(s, volts);
                }
                worst = worst.max((tip - target).length());
                track.tile_scanned(scan(frame, tip), k, target, steps, &calib);
                let next = plan.get(k + 1).copied();
                let planned = next.map(|n| calib.real_to_steps(n - target, volts));
                steps = if track_on { next.and_then(|n| track.next_steps(n, planned.unwrap())) } else { planned };
            }
            (worst, track.calib, track.matched)
        };
        let (open_loop, _, _) = run(false);
        let (closed, learned, matched) = run(true);
        assert!(open_loop > 300., "open loop worst {open_loop}");
        assert!(closed < 160., "closed loop worst {closed}");
        assert_eq!(matched, plan.len() - 1);
        let ratio = learned.column(1, true).length() / truth.column(1, true).length();
        assert!((ratio - 1.).abs() < 0.05, "learned +Y ratio {ratio}");
    }

    /// A tile dragged on the map moves the tip with it and corrects the
    /// learned step and the next move.
    #[test]
    fn dragging_a_tile_corrects_the_next_move() {
        let volts = DVec2::ONE;
        let calib = calib();
        let mut track = GridTrack::new(calib, volts, DMat2::IDENTITY);
        let ids = [Uuid::new_v4(), Uuid::new_v4()];
        // Unmatchable images (flat): every tile is placed as predicted.
        track.tile_scanned(flat(), 0, DVec2::ZERO, None, &calib);
        track.set_latest_id(ids[0]);
        // 60 +Y steps, predicted 600 nm along +x.
        track.tile_scanned(flat(), 1, DVec2::new(600., 0.), Some(IVec2::new(0, 60)), &calib);
        track.set_latest_id(ids[1]);
        let mut world = [DVec2::new(5000., 5000.), DVec2::new(5600., 5000.)];
        let lookup = |w: [DVec2; 2]| move |id: Uuid| ids.iter().position(|i| *i == id).map(|k| w[k]);
        assert_eq!(track.sync(lookup(world)), None);
        let planned = IVec2::new(0, 60);
        assert_eq!(track.next_steps(DVec2::new(1200., 0.), planned), Some(IVec2::new(0, 60)));
        // The user drags tile 2 to where it really was: 780 nm from tile 1.
        world[1] = DVec2::new(5780., 5000.);
        assert_eq!(track.sync(lookup(world)), Some(DVec2::new(180., 0.)));
        // 13 nm/step measured, blended with the calibration's 10: 12.
        assert!((track.calib.column(1, true).x - 1.2e-8).abs() < 1e-12, "{:?}", track.calib.column(1, true));
        // The next tile is 1200 − 780 = 420 nm away: 35 steps.
        assert_eq!(track.next_steps(DVec2::new(1200., 0.), planned), Some(IVec2::new(0, 35)));
    }

    /// A move measured at half the calibrated step (a look-alike match or a
    /// misplaced tile) is not learned from, and the next move stays near
    /// the plan.
    #[test]
    fn one_odd_move_does_not_rescale_the_grid() {
        let volts = DVec2::ONE;
        let calib = calib();
        let mut track = GridTrack::new(calib, volts, DMat2::IDENTITY);
        let ids = [Uuid::new_v4(), Uuid::new_v4()];
        track.tile_scanned(flat(), 0, DVec2::ZERO, None, &calib);
        track.set_latest_id(ids[0]);
        track.tile_scanned(flat(), 1, DVec2::new(600., 0.), Some(IVec2::new(0, 60)), &calib);
        track.set_latest_id(ids[1]);
        let mut world = [DVec2::ZERO, DVec2::new(600., 0.)];
        let lookup = |w: [DVec2; 2]| move |id: Uuid| ids.iter().position(|i| *i == id).map(|k| w[k]);
        track.sync(lookup(world));
        // Tile 2 put at 300 nm and turned 40°: not a step of one motor.
        world[1] = DVec2::new(230., 193.);
        track.sync(lookup(world));
        assert_eq!(track.calib, calib);
        let s = track.next_steps(DVec2::new(1200., 0.), IVec2::new(0, 60)).unwrap();
        assert!((s - IVec2::new(0, 60)).abs().max_element() <= 30, "{s:?}");
    }

    /// The real move was half the predicted one: the image that matches is
    /// outside the window around the prediction, so the tile is placed as
    /// predicted rather than at a look-alike.
    #[test]
    fn far_off_matches_are_not_taken() {
        let frame = DAffine2::from_scale(DVec2::new(1000., -1000.));
        let volts = DVec2::ONE;
        let calib = calib();
        let mut track = GridTrack::new(calib, volts, DMat2::IDENTITY);
        track.tile_scanned(scan(frame, DVec2::ZERO), 0, DVec2::ZERO, None, &calib);
        // Predicted 700 nm along +x; really 300 nm.
        track.tile_scanned(scan(frame, DVec2::new(300., 0.)), 1, DVec2::new(700., 0.), Some(IVec2::new(0, 70)), &calib);
        let info = track.tiles_info();
        assert!(matches!(info[1].placement, Placement::Predicted(_)), "{:?}", info[1]);
    }

    /// Replay of the survey of 2026-11-16 (row 1, tiles 1 → 2 = scans 0014
    /// → 0015), where the old matcher took a look-alike (score 0.52, tile 2
    /// "only" 1.9 µm from its plan) and sent tile 3 off with (30, 59) steps
    /// instead of (−1, 20). Calibration as saved that day, ≈ 221 V.
    /// `GRID_DIR=dir cargo test --release --bin scan_control replay_real_grid -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn replay_real_grid() {
        let dir = std::env::var("GRID_DIR")
            .unwrap_or(r"F:\Data\LT SPM\2026\November\20261116_STM225_As3Nov13".into());
        let load = |n: &str| {
            let f = crate::analysis::sxm_match::load_frame(&std::path::Path::new(&dir).join(n), Some("Z")).unwrap();
            CalFrame { data: FrameData { size: f.size, data: Arc::new(f.forward.into_boxed_slice()) }, frame: f.frame, source: n.into() }
        };
        let mut calib = CoarseCalibration::from_matrix(DMat2::from_cols(
            DVec2::new(-1.3486649619747819e-11, -6.408809709858327e-10),
            DVec2::new(6.726159963193429e-10, 4.2427200836619344e-12),
        ));
        calib.set_column(0, false, DVec2::new(-1.0059935303222996e-11, -5.535418769285185e-10));
        calib.set_column(1, false, DVec2::new(6.241652507908547e-10, 6.034038355950515e-12));
        let volts = DVec2::splat(221.);
        let plan = [DVec2::ZERO, DVec2::new(2973.8256112056465, 83.58514779310269), DVec2::new(5947.651222411293, 167.17029558620538)];
        let planned = IVec2::new(-1, 20);
        println!("planned move {:?} predicts {:?} nm", planned, calib.steps_to_real(planned, volts));
        let mut track = GridTrack::new(calib, volts, DMat2::IDENTITY);
        track.tile_scanned(load("20261116_0014.sxm"), 0, plan[0], None, &calib);
        let off = track.tile_scanned(load("20261116_0015.sxm"), 1, plan[1], Some(planned), &calib);
        let info = track.tiles_info()[1];
        let next = track.next_steps(plan[2], planned).unwrap();
        println!("tile 2: {:?}, stamp moved {off:?} nm, {:.0} nm from plan", info.placement, info.error_nm.unwrap());
        println!("next move {next:?} (planned {planned:?}); {}", track.summary());
        // The look-alike is not taken, and the next move stays near the plan.
        assert!(info.error_nm.unwrap() < 0.35 * plan[1].length(), "{info:?}");
        assert!((next - planned).abs().max_element() <= 10, "{next:?}");
        assert_eq!(track.calib, calib, "nothing learned from an unmatched tile");

        // Tile 3 (0016) after the (30, 59) move the old matcher asked for.
        let mut track = GridTrack::new(calib, volts, DMat2::IDENTITY);
        track.tile_scanned(load("20261116_0015.sxm"), 1, plan[1], None, &calib);
        if std::path::Path::new(&dir).join("20261116_0016.sxm").exists() {
            let t3 = plan[1] + calib.steps_to_real(IVec2::new(30, 59), volts);
            track.tile_scanned(load("20261116_0016.sxm"), 2, t3, Some(IVec2::new(30, 59)), &calib);
            let info = track.tiles_info()[1];
            println!("0015 -> 0016 after (30, 59): {:?}, stamp moved {:.0} nm", info.placement, info.auto_shift_nm);
        }

        // An earlier pair with the same moves (0012 → 0013): reported only.
        let mut track = GridTrack::new(calib, volts, DMat2::IDENTITY);
        track.tile_scanned(load("20261116_0012.sxm"), 0, plan[0], None, &calib);
        track.tile_scanned(load("20261116_0013.sxm"), 1, plan[1], Some(planned), &calib);
        let info = track.tiles_info()[1];
        println!(
            "0012 -> 0013: {:?}, {:.0} nm from plan, next {:?}",
            info.placement,
            info.error_nm.unwrap(),
            track.next_steps(plan[2], planned)
        );
    }

    /// Undoing a tile's automatic placement moves its stamp back by what the
    /// placement added, stops learning from it, and redo puts it back.
    #[test]
    fn undo_and_redo_a_placement() {
        let frame = DAffine2::from_scale(DVec2::new(1000., -1000.));
        let volts = DVec2::ONE;
        let calib = calib();
        let mut track = GridTrack::new(calib, volts, DMat2::IDENTITY);
        let ids = [Uuid::new_v4(), Uuid::new_v4()];
        track.tile_scanned(scan(frame, DVec2::ZERO), 0, DVec2::ZERO, None, &calib);
        track.set_latest_id(ids[0]);
        // 60 steps predicted 600 nm; the tip really went 700.
        let off = track.tile_scanned(
            scan(frame, DVec2::new(700., 0.)),
            1,
            DVec2::new(600., 0.),
            Some(IVec2::new(0, 60)),
            &calib,
        );
        track.set_latest_id(ids[1]);
        assert!((off - DVec2::new(100., 0.)).length() < 10., "{off:?}");
        assert!(track.calib != calib);
        let m = track.set_undone(1, true).unwrap();
        assert_eq!(m.id, ids[1]);
        assert!(m.latest);
        assert!((m.delta + off).length() < 1e-9, "{m:?}");
        assert_eq!(track.calib, calib);
        assert!(track.latest_error_nm().unwrap() < 1e-6);
        let m = track.set_undone(1, false).unwrap();
        assert!((m.delta - off).length() < 1e-9);
        assert!(track.calib != calib);
        assert!(track.set_undone(0, true).is_none());
    }
}
