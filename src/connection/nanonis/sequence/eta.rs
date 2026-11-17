//! Time-to-completion estimate for a running sequence.
//!
//! Scans and waits are predicted from the plan (lines × time per line, wait
//! lengths). Steps whose length depends on the hardware (coarse moves,
//! approach, I(V) series, …) start from rough defaults and are replaced by
//! how long they actually took earlier in the run.

use std::time::{Duration, Instant};

use super::model::{Frame, Speed, Step};

/// What the scan will look like when a step runs: changes as the plan sets
/// resolution, frame and speed.
#[derive(Debug, Clone, Copy)]
pub struct ScanState {
    /// Lines per frame.
    pub lines: u32,
    /// Forward and backward time per line, s.
    pub fwd_s: f64,
    pub bwd_s: f64,
    /// Frame width, nm (for speeds given in nm/s).
    pub width_nm: f64,
    /// Extra time per scan for the repeated Starts, s.
    pub restart_s: f64,
}
impl ScanState {
    pub fn frame_s(&self) -> f64 {
        self.lines as f64 * (self.fwd_s + self.bwd_s)
    }
    /// Follow the plan's setting changes.
    pub fn apply(&mut self, step: &Step) {
        let ratio = if self.fwd_s > 0. { self.bwd_s / self.fwd_s } else { 1. };
        match step {
            Step::SetResolution([_, lines]) => self.lines = *lines,
            Step::SetFrame(spec) => {
                let f = Frame { center_x: 0., center_y: 0., width: self.width_nm, height: 0., angle: 0. };
                self.width_nm = spec.apply(f).width;
            }
            Step::SetSpeed(Speed::LineTime(t)) if *t > 0. => {
                self.fwd_s = *t;
                self.bwd_s = t * ratio;
            }
            Step::SetSpeed(Speed::Linear(v)) if *v > 0. && self.width_nm > 0. => {
                self.fwd_s = self.width_nm / v;
                self.bwd_s = self.fwd_s * ratio;
            }
            _ => {}
        }
    }
}

/// Kinds of step whose length is learned while running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Withdraw,
    Approach,
    /// Per motor step.
    Coarse,
    Stamp,
    MoveTip,
    /// Per I(V) curve.
    Iv,
}

/// Learned (or default) lengths of hardware-bound steps, s.
#[derive(Debug, Clone)]
pub struct StepTimes {
    withdraw: f64,
    approach: f64,
    coarse_per_step: f64,
    stamp: f64,
    move_tip: f64,
    iv_per_curve: f64,
}
impl Default for StepTimes {
    fn default() -> Self {
        Self {
            withdraw: 2.,
            approach: 30.,
            coarse_per_step: 0.002,
            stamp: 2.,
            move_tip: 2.,
            iv_per_curve: 3.,
        }
    }
}

fn kind(step: &Step) -> Option<(Kind, f64)> {
    Some(match step {
        Step::Withdraw => (Kind::Withdraw, 1.),
        Step::Approach => (Kind::Approach, 1.),
        Step::CoarseMove { steps } => (Kind::Coarse, (steps[0].abs() + steps[1].abs()).max(1) as f64),
        Step::Stamp => (Kind::Stamp, 1.),
        Step::MoveTip(_) => (Kind::MoveTip, 1.),
        Step::Spectroscopy(iv) => (Kind::Iv, iv.num_curves.max(1) as f64),
        _ => return None,
    })
}

impl StepTimes {
    fn slot(&mut self, k: Kind) -> &mut f64 {
        match k {
            Kind::Withdraw => &mut self.withdraw,
            Kind::Approach => &mut self.approach,
            Kind::Coarse => &mut self.coarse_per_step,
            Kind::Stamp => &mut self.stamp,
            Kind::MoveTip => &mut self.move_tip,
            Kind::Iv => &mut self.iv_per_curve,
        }
    }
    fn get(&self, k: Kind) -> f64 {
        match k {
            Kind::Withdraw => self.withdraw,
            Kind::Approach => self.approach,
            Kind::Coarse => self.coarse_per_step,
            Kind::Stamp => self.stamp,
            Kind::MoveTip => self.move_tip,
            Kind::Iv => self.iv_per_curve,
        }
    }
    /// Learn from a finished step that took `secs`.
    pub fn record(&mut self, step: &Step, secs: f64) {
        if let Some((k, n)) = kind(step) {
            let per = secs / n;
            let slot = self.slot(k);
            // Lean on recent runs, but don't let one odd step take over.
            *slot = 0.5 * *slot + 0.5 * per;
        }
    }
    /// Predicted length of `step` with the scan set up as `scan`.
    pub fn predict(&self, step: &Step, scan: &ScanState) -> f64 {
        match step {
            Step::Scan(_) => scan.frame_s() + scan.restart_s,
            Step::AwaitScan { .. } => 0.,
            Step::Wait(s) => s.max(0.),
            _ => match kind(step) {
                Some((k, n)) => n * self.get(k),
                // Setting a value: a round trip to Nanonis.
                None => 0.1,
            },
        }
    }
}

/// Clock for one run: active (unpaused) time overall and in the current step.
pub struct RunClock {
    pub times: StepTimes,
    started: Instant,
    paused_total: Duration,
    paused_since: Option<Instant>,
    step_index: usize,
    step_started: Instant,
    step_paused: Duration,
}
impl RunClock {
    pub fn new(times: StepTimes) -> Self {
        let now = Instant::now();
        Self {
            times,
            started: now,
            paused_total: Duration::ZERO,
            paused_since: None,
            step_index: 0,
            step_started: now,
            step_paused: Duration::ZERO,
        }
    }
    /// Starting at step `index` (a run picked up part way).
    pub fn at(mut self, index: usize) -> Self {
        self.step_index = index;
        self
    }
    fn paused_at(&self, now: Instant) -> Duration {
        self.paused_since
            .map_or(Duration::ZERO, |t| now.saturating_duration_since(t))
    }
    /// Unpaused time since the run started, s.
    pub fn elapsed(&self) -> f64 {
        // One clock read for every term, and saturate: a run paused from the
        // start would otherwise subtract a slightly later reading and panic.
        let now = Instant::now();
        now.saturating_duration_since(self.started)
            .saturating_sub(self.paused_total + self.paused_at(now))
            .as_secs_f64()
    }
    /// Unpaused time in the current step, s.
    pub fn step_elapsed(&self) -> f64 {
        let now = Instant::now();
        now.saturating_duration_since(self.step_started)
            .saturating_sub(self.step_paused + self.paused_at(now))
            .as_secs_f64()
    }
    pub fn pause(&mut self) {
        self.paused_since.get_or_insert_with(Instant::now);
    }
    pub fn resume(&mut self) {
        if let Some(t) = self.paused_since.take() {
            let paused = t.elapsed();
            self.paused_total += paused;
            self.step_paused += paused;
        }
    }
    /// Step `index` started: learn from the one that just ended.
    pub fn step_started(&mut self, index: usize, steps: &[(usize, Step)]) {
        if index == self.step_index {
            return;
        }
        if index == self.step_index + 1 {
            if let Some((_, step)) = steps.get(self.step_index) {
                let secs = self.step_elapsed();
                self.times.record(step, secs);
            }
        }
        self.step_index = index;
        self.step_started = Instant::now();
        self.step_paused = Duration::ZERO;
    }
}

/// Seconds left: the rest of the current step plus every later one.
/// `scan` is the scan set-up when the run started; `await_left` is what is
/// left of a scan already running in Nanonis (for an `AwaitScan` step).
pub fn remaining(
    steps: &[(usize, Step)],
    current: usize,
    mut scan: ScanState,
    clock: &RunClock,
    wait_left: Option<f64>,
    await_left: Option<f64>,
) -> f64 {
    let mut left = 0.;
    for (i, (_, step)) in steps.iter().enumerate() {
        if i < current {
            scan.apply(step);
            continue;
        }
        let full = clock.times.predict(step, &scan);
        left += if i == current {
            match step {
                Step::Wait(_) => wait_left.unwrap_or((full - clock.step_elapsed()).max(0.)),
                Step::AwaitScan { .. } => await_left.unwrap_or(0.),
                _ => (full - clock.step_elapsed()).max(0.),
            }
        } else {
            full
        };
        scan.apply(step);
    }
    left
}

/// "45 s", "3 min 05 s", "1 h 02 min".
pub fn fmt_duration(secs: f64) -> String {
    let s = secs.round().max(0.) as u64;
    if s < 60 {
        format!("{s} s")
    } else if s < 3600 {
        format!("{} min {:02} s", s / 60, s % 60)
    } else {
        format!("{} h {:02} min", s / 3600, (s % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::nanonis::sequence::model::Dir;

    #[test]
    fn scans_follow_resolution_and_speed() {
        let scan = ScanState { lines: 100, fwd_s: 1., bwd_s: 1., width_nm: 1000., restart_s: 0. };
        let steps = vec![
            (0, Step::Scan(Dir::Down)),
            (1, Step::SetResolution([256, 50])),
            (1, Step::SetSpeed(Speed::LineTime(2.))),
            (1, Step::Scan(Dir::Up)),
            (1, Step::Wait(10.)),
        ];
        let clock = RunClock::new(StepTimes::default());
        // 100 × 2 s + 0.1 + 0.1 + 50 × 4 s + 10 s
        let left = remaining(&steps, 0, scan, &clock, None, None);
        assert!((left - 410.2).abs() < 0.5, "{left}");
        // From the second scan on: settings before it already applied.
        let left = remaining(&steps, 3, scan, &clock, None, None);
        assert!((left - 210.).abs() < 0.5, "{left}");
    }

    #[test]
    fn restarts_add_to_each_scan() {
        use crate::connection::nanonis::sequence::model::ScanStart;
        let restart_s = ScanStart { starts: 3, gap_s: 3., park_s: 10. }.extra_s();
        let scan = ScanState { lines: 10, fwd_s: 1., bwd_s: 1., width_nm: 1000., restart_s };
        let clock = RunClock::new(StepTimes::default());
        let left = remaining(&[(0, Step::Scan(Dir::Down))], 0, scan, &clock, None, None);
        // 20 s frame + 10 s parked + 2 × 3 s restarts.
        assert!((left - 36.).abs() < 0.1, "{left}");
    }

    #[test]
    fn learns_from_finished_steps() {
        let mut t = StepTimes::default();
        let approach = Step::Approach;
        t.record(&approach, 60.);
        assert!((t.predict(&approach, &ScanState { lines: 0, fwd_s: 0., bwd_s: 0., width_nm: 0., restart_s: 0. }) - 45.).abs() < 1e-9);
    }
}
