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

use glam::{DMat2, DVec2, IVec2};
use uuid::Uuid;

use crate::{
    analysis::register::{shift_near, GOOD_SCORE},
    connection::nanonis::{
        coarse_cal::{shift_to_tip, CalFrame},
        course_motion::CoarseCalibration,
    },
};

/// A measured step this far from the calibration (ratio) is not learned from.
const MAX_RATIO: f64 = 2.5;
/// Largest correction added to a move, share of the tile spacing.
const MAX_CORRECTION: f64 = 0.5;
/// Share of the image two tiles must have in common to be matched: less
/// leaves too little to tell a real match from a look-alike.
const MIN_OVERLAP: f64 = 0.15;

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
}

pub struct GridTrack {
    /// Calibration the run started with.
    base: CoarseCalibration,
    /// Calibration fitted to the tiles' positions (used for the moves).
    pub calib: CoarseCalibration,
    volts: DVec2,
    /// Map (world) displacement → piezo nm.
    to_piezo: DMat2,
    tiles: Vec<Tile>,
    prev_frame: Option<CalFrame>,
    pub matched: usize,
    pub unmatched: usize,
    pub dragged: usize,
}

impl GridTrack {
    /// `world_m2`: the scan area's piezo → map matrix.
    pub fn new(calib: CoarseCalibration, volts: DVec2, world_m2: DMat2) -> Self {
        let to_piezo = if world_m2.determinant().abs() > 1e-12 { world_m2.inverse() } else { DMat2::IDENTITY };
        Self {
            base: calib,
            calib,
            volts,
            to_piezo,
            tiles: Vec::new(),
            prev_frame: None,
            matched: 0,
            unmatched: 0,
            dragged: 0,
        }
    }

    /// A tile was scanned by `entry` (planned at `target`; `steps_in`: the
    /// move that brought the tip here). `map_calib`: the calibration the map
    /// moved the scan area with for that move. Returns how far (piezo nm)
    /// the tip is from where the map put it, to place the stamp right.
    pub fn tile_scanned(
        &mut self,
        frame: CalFrame,
        entry: usize,
        target: DVec2,
        steps_in: Option<IVec2>,
        map_calib: &CoarseCalibration,
    ) -> DVec2 {
        let (pos, off) = match (self.tiles.last(), self.prev_frame.take(), steps_in) {
            (Some(last), Some(prev), Some(steps)) => {
                let predicted = self.calib.steps_to_real(steps, self.volts);
                let on_map = map_calib.steps_to_real(steps, self.volts);
                let real = match measure(&prev, &frame, predicted) {
                    Some(real) => {
                        self.matched += 1;
                        real
                    }
                    None => {
                        // Not matched: the grid's own calibration is the best guess.
                        self.unmatched += 1;
                        predicted
                    }
                };
                (last.pos + real, real - on_map)
            }
            // First tile: where it was planned, by definition.
            _ => (target, DVec2::ZERO),
        };
        self.tiles.push(Tile { id: None, entry, target, steps_in, pos, world: None });
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
        // Positions from the map, relative to the first tile.
        if let Some(w0) = self.tiles.first().and_then(|t| t.world) {
            let p0 = self.tiles[0].pos;
            for t in &mut self.tiles {
                if let Some(w) = t.world {
                    t.pos = p0 + self.to_piezo * (w - w0);
                }
            }
        }
        self.refit();
        Some(latest_delta)
    }

    /// Steps to the tile planned at `next_target`, from where the latest
    /// tile really is.
    pub fn next_steps(&self, next_target: DVec2) -> Option<IVec2> {
        let last = self.tiles.last()?;
        // The planned step plus a correction for where the latest tile
        // really is, at most half a tile (a bad position can't throw the
        // grid far off).
        let planned = next_target - last.target;
        let mut correction = last.target - last.pos;
        let cap = MAX_CORRECTION * planned.length();
        if correction.length() > cap {
            correction = correction.normalize_or_zero() * cap;
        }
        let steps = self.calib.real_to_steps(planned + correction, self.volts);
        (steps != IVec2::ZERO).then_some(steps)
    }

    /// Distance of the latest tile from its plan, nm.
    pub fn latest_error_nm(&self) -> Option<f64> {
        let t = self.tiles.last()?;
        (self.tiles.len() > 1).then(|| (t.pos - t.target).length())
    }

    /// Moves between consecutive tiles: (steps, real move in piezo nm).
    fn moves(&self) -> Vec<(IVec2, DVec2)> {
        self.tiles.windows(2).filter_map(|p| Some((p[1].steps_in?, p[1].pos - p[0].pos))).collect()
    }
    pub fn n_moves(&self) -> usize {
        self.moves().len()
    }

    /// Fit each motor direction's step to the moves between tiles (least
    /// squares; directions without moves keep the starting calibration).
    fn refit(&mut self) {
        let mut sums = [[(DVec2::ZERO, 0f64); 2]; 2];
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
            if !(1. / MAX_RATIO..=MAX_RATIO).contains(&ratio) || !column.is_finite() {
                continue;
            }
            // Weighted by n², as fitting real = column · n · V.
            let s = &mut sums[main][(n > 0.) as usize];
            s.0 += column * n * n;
            s.1 += n * n;
        }
        let mut c = self.base;
        for axis in 0..2 {
            for pos in [false, true] {
                let (num, den) = sums[axis][pos as usize];
                if den > 0. {
                    c.set_column(axis, pos, num / den);
                }
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
        let r = self.learned_ratios();
        if !r.is_empty() {
            let parts: Vec<String> = r.iter().map(|(n, x)| format!("{n} {:.0} %", x * 100.)).collect();
            s += &format!(" \u{b7} steps vs calibration: {}", parts.join(", "));
        }
        s
    }
}

/// Tip move (nm) between two tiles of the same frame, searched near the
/// predicted move; `None` without a convincing match.
fn measure(prev: &CalFrame, cur: &CalFrame, predicted: DVec2) -> Option<DVec2> {
    if prev.data.size != cur.data.size {
        return None;
    }
    let [w, h] = prev.data.size;
    let m = prev.frame.matrix2;
    if m.determinant().abs() < 1e-12 {
        return None;
    }
    // Inverse of `shift_to_tip`: content moves opposite to the tip.
    let local = m.inverse() * predicted;
    let guess = (-local.x * w as f64, -local.y * h as f64);
    let radius = 0.4 * w.max(h) as f64;
    let s = shift_near(&prev.data.data, &cur.data.data, w as usize, h as usize, guess, radius, MIN_OVERLAP)?;
    (s.score >= GOOD_SCORE).then(|| shift_to_tip(prev, DVec2::new(s.dx, s.dy)))
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
        let run = |track_on: bool| -> (f64, CoarseCalibration) {
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
                steps = if track_on {
                    next.and_then(|n| track.next_steps(n))
                } else {
                    next.map(|n| calib.real_to_steps(n - target, volts))
                };
            }
            (worst, track.calib)
        };
        let (open_loop, _) = run(false);
        let (closed, learned) = run(true);
        assert!(open_loop > 300., "open loop worst {open_loop}");
        assert!(closed < 160., "closed loop worst {closed}");
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
        // Unmatchable images (flat): every tile is placed as predicted.
        let flat = CalFrame {
            data: FrameData { size: [16, 16], data: Arc::new(vec![0f32; 256].into_boxed_slice()) },
            frame: DAffine2::from_scale(DVec2::new(1000., -1000.)),
            source: String::new(),
        };
        let ids = [Uuid::new_v4(), Uuid::new_v4()];
        track.tile_scanned(flat.clone(), 0, DVec2::ZERO, None, &calib);
        track.set_latest_id(ids[0]);
        // 60 +Y steps, predicted 600 nm along +x.
        track.tile_scanned(flat, 1, DVec2::new(600., 0.), Some(IVec2::new(0, 60)), &calib);
        track.set_latest_id(ids[1]);
        let mut world = [DVec2::new(5000., 5000.), DVec2::new(5600., 5000.)];
        let lookup = |w: [DVec2; 2]| move |id: Uuid| ids.iter().position(|i| *i == id).map(|k| w[k]);
        assert_eq!(track.sync(lookup(world)), None);
        assert_eq!(track.next_steps(DVec2::new(1200., 0.)), Some(IVec2::new(0, 60)));
        // The user drags tile 2 to where it really was: 780 nm from tile 1.
        world[1] = DVec2::new(5780., 5000.);
        assert_eq!(track.sync(lookup(world)), Some(DVec2::new(180., 0.)));
        // Learned 13 nm/step for +Y.
        assert!((track.calib.column(1, true).x - 1.3e-8).abs() < 1e-12);
        // The next tile is 1200 − 780 = 420 nm away: 32 steps.
        assert_eq!(track.next_steps(DVec2::new(1200., 0.)), Some(IVec2::new(0, 32)));
    }
}
