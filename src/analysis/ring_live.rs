//! Ring matching on live frames, checking a guess across frames, and planning
//! the move toward the ring centre.
//!
//! One frame often can't tell which side of the bands the centre is on (the
//! two sides can fit almost equally well). A move settles it: after moving,
//! the true centre still fits both frames, while the mirror-image guess
//! doesn't. So each new frame is checked jointly with the frame before the
//! move.

use std::sync::Arc;

use glam::{DAffine2, DVec2};

use crate::analysis::ring_match::{blind_match, score_at, MatchMode, MatchParams};

/// One analysed frame: data and where it is (quad −0.5..0.5 to world nm).
#[derive(Clone)]
pub struct Shot {
    pub data: Arc<Box<[f32]>>,
    pub size: [u32; 2],
    pub frame: DAffine2,
}

impl Shot {
    pub fn nm_per_px(&self) -> f64 {
        self.frame.matrix2.x_axis.length() / self.size[0] as f64
    }
    pub fn center(&self) -> DVec2 {
        self.frame.translation
    }
    pub fn width(&self) -> f64 {
        self.frame.matrix2.x_axis.length()
    }
    pub fn px_to_world(&self, px: DVec2) -> DVec2 {
        let s = DVec2::new(self.size[0] as f64, self.size[1] as f64);
        self.frame.transform_point2((px + 0.5) / s - 0.5)
    }
    pub fn world_to_px(&self, w: DVec2) -> DVec2 {
        let s = DVec2::new(self.size[0] as f64, self.size[1] as f64);
        (self.frame.inverse().transform_point2(w) + 0.5) * s - 0.5
    }
    /// Ring kinds (world nm) in this frame's pixels.
    fn bands_px(&self, kind_nm: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let k = self.nm_per_px();
        kind_nm.iter().map(|b| (b.0 / k, b.1 / k)).collect()
    }
    fn complete(&self) -> bool {
        self.data.len() == self.size[0] as usize * self.size[1] as usize
            && self.data.iter().filter(|v| v.is_finite()).count() as f64 >= 0.98 * self.data.len() as f64
    }
}

/// The ring match of one live frame.
#[derive(Clone, Debug)]
pub struct LiveMatch {
    /// Which ring kind fitted.
    pub kind: usize,
    pub center_world: DVec2,
    /// Best centre on the other side of the frame.
    pub opposite_world: Option<DVec2>,
    pub score: f64,
    /// Direction confidence: how much better this side fits than the other.
    pub gap: f64,
    /// The backward scan fitted better than the forward one.
    pub backward: bool,
}

impl LiveMatch {
    /// The direction from this frame alone is sure enough to act on.
    pub fn sure(&self, p: &MatchParams) -> bool {
        self.gap >= p.sure_gap
    }
}

/// Match a complete frame (forward and, if given, backward; the better one
/// is kept). `None` when matching is off, there are no ring kinds, the frame
/// is incomplete, or nothing fits well enough to trust.
pub fn match_shot(fwd: &Shot, bwd: Option<&Shot>, kinds_nm: &[Vec<(f64, f64)>], p: &MatchParams) -> Option<LiveMatch> {
    if p.mode == MatchMode::Off || kinds_nm.is_empty() {
        return None;
    }
    let one = |s: &Shot| {
        if !s.complete() {
            return None;
        }
        let kinds: Vec<Vec<(f64, f64)>> = kinds_nm.iter().map(|k| s.bands_px(k)).collect();
        blind_match(&s.data, s.size, &kinds, p)
    };
    let bwd = bwd.filter(|b| b.size == fwd.size);
    let (f, b) = std::thread::scope(|scope| {
        let hb = bwd.map(|b| scope.spawn(move || one(b)));
        (one(fwd), hb.and_then(|h| h.join().ok().flatten()))
    });
    let (backward, (kind, m)) = match (f, b) {
        (Some(f), Some(b)) if b.1.score > f.1.score => (true, b),
        (Some(f), _) => (false, f),
        (None, Some(b)) => (true, b),
        (None, None) => return None,
    };
    if m.score < p.min_score {
        return None;
    }
    let shot = if backward { bwd? } else { fwd };
    Some(LiveMatch {
        kind,
        center_world: shot.px_to_world(m.center_px),
        opposite_world: m.opposite_px.map(|c| shot.px_to_world(c)),
        score: m.score,
        gap: m.dir_gap(),
        backward,
    })
}

/// Result of checking the current frame against the one before a move.
#[derive(Clone, Debug)]
pub struct JointCheck {
    /// The centre that fits both frames best.
    pub center_world: DVec2,
    /// Both frames' own guesses point at about the same centre.
    pub agrees: bool,
    /// The joint winner is not the current frame's own guess (its direction
    /// was probably flipped).
    pub overrides_current: bool,
    /// How much better the winner fits both frames than the best centre on
    /// the other side of the current frame.
    pub gap: f64,
}

/// Score candidate centres (each frame's guess and its opposite) in both
/// frames together; the true centre should fit both.
pub fn joint_check(
    prev: &Shot,
    prev_m: &LiveMatch,
    cur: &Shot,
    cur_m: &LiveMatch,
    kinds_nm: &[Vec<(f64, f64)>],
    p: &MatchParams,
) -> Option<JointCheck> {
    let kind = kinds_nm.get(cur_m.kind)?;
    let (bp, bc) = (prev.bands_px(kind), cur.bands_px(kind));
    let both = |c: DVec2| {
        let a = score_at(&prev.data, prev.size, &bp, prev.world_to_px(c), p)?;
        let b = score_at(&cur.data, cur.size, &bc, cur.world_to_px(c), p)?;
        Some(a + b)
    };
    let cands: Vec<DVec2> = [Some(cur_m.center_world), cur_m.opposite_world, Some(prev_m.center_world), prev_m.opposite_world]
        .into_iter()
        .flatten()
        .collect();
    let scored: Vec<(DVec2, f64)> = std::thread::scope(|scope| {
        let hs: Vec<_> = cands.iter().map(|&c| scope.spawn(move || both(c).map(|s| (c, s)))).collect();
        hs.into_iter().filter_map(|h| h.join().ok().flatten()).collect()
    });
    let &(best, best_score) = scored.iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    let side = |c: DVec2| (c - cur.center()).dot(best - cur.center()) > 0.;
    let other = scored.iter().filter(|(c, _)| !side(*c)).map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    let near = |a: DVec2, b: DVec2| {
        let d = (a - cur.center()).length().max(cur.width());
        (a - b).length() < 0.3 * d
    };
    Some(JointCheck {
        center_world: best,
        agrees: near(cur_m.center_world, prev_m.center_world),
        overrides_current: !near(best, cur_m.center_world),
        // Scores of two frames: halve to compare with one frame's gap.
        gap: if other.is_finite() { (best_score - other) / 2. } else { 1. },
    })
}

/// What to do next from this frame.
#[derive(Clone, Debug, PartialEq)]
pub struct MovePlan {
    /// World displacement (nm).
    pub world: DVec2,
    /// Distance from the frame centre to the ring centre (nm).
    pub dist_nm: f64,
    /// The ring centre is in or next to this frame.
    pub at_center: bool,
    /// A short test move (direction not sure yet), to be checked on the
    /// next frame.
    pub test: bool,
}

/// Move settings, in frame widths.
#[derive(Clone, Copy, Debug)]
pub struct MoveLimits {
    /// Largest move toward a sure centre.
    pub max_frames: f64,
    /// Test move when the direction isn't sure.
    pub test_frames: f64,
}

impl Default for MoveLimits {
    fn default() -> Self {
        Self { max_frames: 1.5, test_frames: 0.5 }
    }
}

/// Plan the move from frame `shot` toward `center` (world nm). `sure` says
/// whether the direction is trusted (sure gap, or confirmed by the frame
/// before the move). The ring centre is "here" once it lies inside the
/// frame; then the move centres it.
pub fn plan_move(shot: &Shot, center: DVec2, sure: bool, lim: MoveLimits) -> MovePlan {
    let d = center - shot.center();
    let dist = d.length();
    let w = shot.width();
    let q = shot.frame.inverse().transform_point2(center);
    let at_center = q.x.abs() <= 0.5 && q.y.abs() <= 0.5;
    let len = if at_center {
        dist
    } else if sure {
        dist.min(lim.max_frames * w)
    } else {
        dist.min(lim.test_frames * w)
    };
    MovePlan { world: d.normalize_or_zero() * len, dist_nm: dist, at_center, test: !sure && !at_center }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ring_match::tests::{device_bands, scan};

    const W: usize = 192;

    /// A frame at `center` (world nm, 1 nm per px) of rings around `ring`.
    fn shot(center: DVec2, ring: DVec2, seed: u64) -> Shot {
        let frame = DAffine2::from_scale_angle_translation(DVec2::splat(W as f64), 0., center);
        let s = Shot { data: Arc::new(Box::new([])), size: [W as u32; 2], frame };
        let ring_px = s.world_to_px(ring);
        Shot { data: Arc::new(scan(W, ring_px, &device_bands(), 10., 1., seed).into_boxed_slice()), ..s }
    }

    #[test]
    fn pixel_world_round_trip() {
        let s = shot(DVec2::new(100., -40.), DVec2::ZERO, 1);
        let p = DVec2::new(12.3, 150.);
        assert!((s.world_to_px(s.px_to_world(p)) - p).length() < 1e-9);
        assert!((s.px_to_world(DVec2::splat(95.5)) - s.center()).length() < 1e-9);
    }

    #[test]
    fn live_match_finds_the_side() {
        let ring = DVec2::new(-260., 150.);
        let s = shot(DVec2::ZERO, ring, 3);
        let m = match_shot(&s, None, &[device_bands()], &MatchParams::default()).unwrap();
        assert!((m.center_world - s.center()).normalize().dot(ring.normalize()) > 0.85, "{m:?}");
        assert!(m.opposite_world.is_some());
    }

    #[test]
    fn joint_check_overrides_a_flipped_guess() {
        let ring = DVec2::new(-260., 150.);
        let p = MatchParams::default();
        let kinds = [device_bands()];
        // Before the move, at the origin; after a move toward the ring.
        let a = shot(DVec2::ZERO, ring, 3);
        let b = shot(DVec2::new(-60., 30.), ring, 4);
        let ma = match_shot(&a, None, &kinds, &p).unwrap();
        let mut mb = match_shot(&b, None, &kinds, &p).unwrap();
        // The current frame got it wrong: swap its guess with the opposite.
        let flipped = 2. * b.center() - ring;
        mb.opposite_world = Some(mb.center_world);
        mb.center_world = flipped;
        let j = joint_check(&a, &ma, &b, &mb, &kinds, &p).unwrap();
        assert!(j.overrides_current, "{j:?}");
        assert!((j.center_world - ring).length() < 60., "{j:?}");
        assert!(!j.agrees);
        assert!(j.gap > 0., "{j:?}");
    }

    #[test]
    fn plans_test_moves_when_unsure_and_centres_at_the_end() {
        let s = shot(DVec2::ZERO, DVec2::ZERO, 1);
        let lim = MoveLimits::default();
        let far = DVec2::new(1000., 0.);
        let unsure = plan_move(&s, far, false, lim);
        assert!(unsure.test && (unsure.world.length() - 0.5 * W as f64).abs() < 1e-9);
        let sure = plan_move(&s, far, true, lim);
        assert!(!sure.test && (sure.world.length() - 1.5 * W as f64).abs() < 1e-9);
        let here = plan_move(&s, DVec2::new(30., -20.), false, lim);
        assert!(here.at_center && !here.test && (here.world - DVec2::new(30., -20.)).length() < 1e-9);
    }
}
