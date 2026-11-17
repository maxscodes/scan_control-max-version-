use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, RecvTimeoutError, Sender};
use eyre::Context as _;
use glam::{DVec2, IVec2};
use nanonis_tcp::{
    blocking::NanonisTcp,
    commands,
    error::{NanonisTcpError, NanonisTcpResult},
    ActionType, ScanDir,
};
use tracing::{debug, error, info, warn};

use crate::connection::nanonis::{
    sequence::{
        checkpoint::{self, Checkpoint, SavedScan},
        model::{restart_index, scan_start_point, Dir, Frame, ScanStart, Sequence, Speed, Step},
    },
    spectroscopy::{
        nanonis_saved_files, run_iv_series, save_batch, IvCurve, IvSeriesSettings, SeriesHost,
        SeriesOutcome, SharedPlotChoice,
    },
    worker::{move_args_from_steps, Worker},
};

/// After resuming a paused scan, how long to wait before checking whether
/// it is actually running (it may have been finished or stopped meanwhile).
const RESUME_CHECK: Duration = Duration::from_millis(300);

/// How long the runner waits for the UI to take a stamp before moving on.
const STAMP_TIMEOUT: Duration = Duration::from_secs(15);
/// Slice length used for waits and polling so Abort/Pause stay responsive.
const SLICE: Duration = Duration::from_millis(100);
/// Timeout passed to Scan.WaitEndOfScan per poll.
const SCAN_POLL: Duration = Duration::from_millis(500);
/// The tip counts as arrived within this distance of its target (m).
const TIP_TOLERANCE_M: f64 = 0.5e-9;
/// Longest a single tip move may take.
const TIP_MAX_TIME: Duration = Duration::from_secs(600);
/// How often Nanonis is checked for changes made outside the sequence.
const CHECK_EVERY: Duration = Duration::from_secs(1);
/// A scan that ends before this share of its expected time was stopped,
/// not finished.
const SCAN_DONE_FRACTION: f64 = 0.8;

/// The Nanonis settings a sequence controls. A change the sequence didn't
/// make means someone else is using the instrument.
#[derive(Debug, Clone, Copy)]
struct Snapshot {
    bias: f32,
    setpoint: f32,
    frame: Frame,
    px: usize,
    lines: usize,
    fwd_line_time: f32,
    bwd_line_time: f32,
}
impl Snapshot {
    fn read(conn: &mut NanonisTcp) -> NanonisTcpResult<Self> {
        let buffer = conn.scan_buffer_get()?;
        let speed = conn.scan_speed_get()?;
        Ok(Self {
            bias: conn.bias_get()?,
            setpoint: conn.zctrl_setpnt_get()?,
            frame: Frame::from_get(&conn.scan_frame_get()?),
            px: buffer.px_per_line,
            lines: buffer.num_lines,
            fwd_line_time: speed.fwd_line_time,
            bwd_line_time: speed.bwd_line_time,
        })
    }
    /// Seconds for one full frame (forward and backward lines).
    fn frame_s(&self) -> f64 {
        self.lines as f64 * (self.fwd_line_time + self.bwd_line_time) as f64
    }
    /// What changed from `self` to `now`, if anything.
    fn change(&self, now: &Self) -> Option<String> {
        let rel = |a: f64, b: f64| (a - b).abs() > 1e-3 * a.abs().max(b.abs());
        if (self.bias - now.bias).abs() > 1e-3_f32.max(1e-3 * self.bias.abs()) {
            return Some(format!("bias changed to {:.3} V", now.bias));
        }
        if rel(self.setpoint as f64, now.setpoint as f64) && (self.setpoint - now.setpoint).abs() > 1e-14 {
            return Some(format!("setpoint changed to {:.1} pA", now.setpoint * 1e12));
        }
        let (a, b) = (self.frame, now.frame);
        if (a.center_x - b.center_x).abs() > 1. || (a.center_y - b.center_y).abs() > 1. {
            return Some("scan frame was moved".into());
        }
        if rel(a.width, b.width) || rel(a.height, b.height) || (a.angle - b.angle).abs() > 0.1 {
            return Some("scan frame size or angle changed".into());
        }
        if self.px != now.px || self.lines != now.lines {
            return Some(format!("resolution changed to {}×{}", now.px, now.lines));
        }
        if (self.fwd_line_time - now.fwd_line_time).abs() > 0.01 * self.fwd_line_time.abs() {
            return Some("scan speed changed".into());
        }
        None
    }
}

#[derive(Debug)]
pub enum SequenceCommand {
    Run {
        steps: Vec<(usize, Step)>,
        /// Motor group index (0-based) for coarse moves.
        group: u32,
        /// How each scan is started.
        scan_start: ScanStart,
        /// The editor's plan. `Some`: the run is saved to disk as it goes,
        /// so it survives the app restarting (not for calibration runs).
        plan: Option<Sequence>,
    },
    /// Pause once the scan in progress (or the next one) is done and
    /// stamped. Cleared when it pauses.
    PauseAfterScan(bool),
    /// The app wants to close (to be rebuilt): at the next safe point (a
    /// scan keeps running in Nanonis) save the run so the next start picks
    /// it up, then report [`SequenceEvent::Detached`].
    Detach,
    CancelDetach,
    /// Hold now. A scan in progress is paused in Nanonis and resumed (not
    /// restarted) on Resume. A wait in progress is cut short and restarts
    /// from the beginning on resume. Other steps finish first.
    Pause,
    /// Someone used the instrument from the app (reason): hold at once and
    /// send nothing more to Nanonis, leaving it as the user set it.
    Interrupt(String),
    Resume,
    /// From a pause: go back to the start of the current entry (after a
    /// finished coarse move) and run it again.
    RestartEntry,
    Abort,
    /// UI has finished stamping the image.
    StampDone,
    /// Use these steps for the coarse move of entry `entry`, if it hasn't
    /// run yet (a survey grid correcting its next move).
    AdjustCoarse { entry: usize, steps: [i32; 2] },
    /// The run's survey tiles so far and their layer group (saved with the
    /// run, so a restart carries on the same survey).
    SurveyTiles { group: Option<String>, tiles: Vec<uuid::Uuid> },
}

#[derive(Debug, Clone)]
pub enum SequenceEvent {
    Started,
    StepStarted { index: usize },
    WaitProgress { remaining: f64 },
    CoarseMoved { steps: IVec2 },
    StampRequest,
    ScanDone { file_path: String },
    /// An I(V) series started (its base name).
    SpecStarted { base: String },
    SpecStatus(String),
    SpecCurve(IvCurve),
    /// The series ended; `message` says how and where it was saved. `files`
    /// are the .dat files Nanonis saved.
    SpecDone { ok: bool, message: String, files: Vec<std::path::PathBuf> },
    Paused { index: usize },
    /// Paused because the instrument was used outside the sequence.
    Interrupted { index: usize, reason: String },
    Resumed,
    Finished,
    Aborted,
    Error { index: usize, message: String },
    /// A run saved by the previous app session was picked up (paused,
    /// unless `auto_resume`: it continues once Nanonis is connected).
    Restored {
        steps: Vec<(usize, Step)>,
        index: usize,
        plan: Sequence,
        auto_resume: bool,
        note: String,
        /// The survey the run was adding tiles to (see [`SequenceCommand::SurveyTiles`]).
        survey_group: Option<String>,
        survey_tiles: Vec<uuid::Uuid>,
    },
    /// "Pause after scan" is armed or not (it clears itself once it fires).
    PauseAfterScan(bool),
    /// The run is saved and the app may close.
    Detached,
}

struct RunState {
    steps: Vec<(usize, Step)>,
    group: u32,
    scan_start: ScanStart,
    index: usize,
    paused: bool,
    pause_requested: bool,
    /// State of the scan started by the current Scan step.
    scan: ScanPhase,
    /// The app asked to hold (someone used a control), with the reason.
    interrupt: Option<String>,
    /// Restart the current entry (set by RestartEntry, done with a connection).
    restart: bool,
    /// Nanonis settings as the sequence last left them; `None`: read afresh.
    baseline: Option<Snapshot>,
    last_check: Option<Instant>,
    /// Scanning time of the current scan, and how long a full frame takes
    /// (0: unknown, e.g. a scan started by hand).
    scan_active: Duration,
    scan_since: Option<Instant>,
    scan_expected_s: f64,
    /// Saved to disk as it goes (see [`checkpoint`]), with this plan.
    plan: Option<Sequence>,
    /// A scan finished since the run last paused (for "pause after scan").
    scanned: bool,
    /// Step `index` has begun.
    step_started: bool,
    /// Handed off for a rebuild while running: continue after the restart.
    resume_after_restart: bool,
    /// Survey tiles taken so far and their layer group (from the editor).
    survey_group: Option<String>,
    survey_tiles: Vec<uuid::Uuid>,
}
impl RunState {
    fn new(steps: Vec<(usize, Step)>, group: u32, scan_start: ScanStart, plan: Option<Sequence>) -> Self {
        Self {
            steps,
            group,
            scan_start,
            index: 0,
            paused: false,
            pause_requested: false,
            scan: ScanPhase::NotStarted,
            interrupt: None,
            restart: false,
            baseline: None,
            last_check: None,
            scan_active: Duration::ZERO,
            scan_since: None,
            scan_expected_s: 0.,
            plan,
            scanned: false,
            step_started: false,
            resume_after_restart: false,
            survey_group: None,
            survey_tiles: Vec::new(),
        }
    }
    /// What is saved to disk (`None`: this run isn't saved).
    fn checkpoint(&self, pause_after_scan: bool) -> Option<Checkpoint> {
        let plan = self.plan.clone()?;
        let scan = match self.scan {
            ScanPhase::NotStarted => SavedScan::NotStarted,
            ScanPhase::Running(d) => SavedScan::Running(d.into()),
            ScanPhase::Paused(d) => SavedScan::Paused(d.into()),
        };
        let now = checkpoint::unix_now();
        Some(Checkpoint {
            steps: self.steps.clone(),
            group: self.group,
            scan_start: self.scan_start,
            index: self.index,
            paused: self.paused,
            step_started: self.step_started,
            scan,
            scan_active_s: self.scan_active.as_secs_f64(),
            // Whole seconds, so an unchanged run writes the same file.
            scan_since_unix: self.scan_since.map(|t| now.saturating_sub(t.elapsed().as_secs())),
            scan_expected_s: self.scan_expected_s,
            pause_after_scan,
            resume_after_restart: self.resume_after_restart,
            plan,
            survey_group: self.survey_group.clone(),
            survey_tiles: self.survey_tiles.iter().map(|id| id.to_string()).collect(),
        })
    }
    /// A run saved by an earlier session, held (paused).
    fn from_checkpoint(c: Checkpoint) -> Self {
        let now = checkpoint::unix_now();
        let dir = |d: Dir| -> ScanDir { d.into() };
        let scan = match c.scan {
            SavedScan::NotStarted => ScanPhase::NotStarted,
            SavedScan::Running(d) => ScanPhase::Running(dir(d)),
            SavedScan::Paused(d) => ScanPhase::Paused(dir(d)),
        };
        // Time the scan ran while the app was closed counts as scanned.
        let mut scan_active = Duration::from_secs_f64(c.scan_active_s.max(0.));
        let scan_since = c.scan_since_unix.map(|since| {
            scan_active += Duration::from_secs(now.saturating_sub(since));
            Instant::now()
        });
        let mut s = Self::new(c.steps, c.group, c.scan_start, Some(c.plan));
        s.index = c.index;
        s.paused = true;
        s.scan = scan;
        s.scan_active = scan_active;
        s.scan_since = scan_since;
        s.scan_expected_s = c.scan_expected_s;
        s.step_started = c.step_started;
        s.survey_group = c.survey_group;
        s.survey_tiles = c.survey_tiles.iter().filter_map(|id| id.parse().ok()).collect();
        s
    }
    fn scan_time(&self) -> f64 {
        (self.scan_active + self.scan_since.map_or(Duration::ZERO, |t| t.elapsed())).as_secs_f64()
    }
    /// The scan ran (nearly) its full time, so it finished rather than
    /// being stopped.
    fn scan_complete(&self) -> bool {
        self.scan_expected_s <= 0. || self.scan_time() >= SCAN_DONE_FRACTION * self.scan_expected_s
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanPhase {
    /// No scan started by this step yet.
    NotStarted,
    Running(ScanDir),
    /// Paused by a sequence Pause; Resume continues it.
    Paused(ScanDir),
}

enum Outcome {
    Completed,
    /// Step was interrupted by a pause; do not advance.
    Interrupted,
    /// The instrument was used outside the sequence (reason): hold.
    External(String),
    Aborted,
    /// Stopped at a safe point to hand off to a restarted app (see
    /// [`SequenceCommand::Detach`]); a scan in progress keeps running.
    Detach,
}

pub struct SequenceWorker {
    ctx: egui::Context,
    commands: Receiver<SequenceCommand>,
    events: Sender<SequenceEvent>,
    /// Plots chosen in the I(V) window; saved series use the same.
    plot_choice: SharedPlotChoice,
    state: Option<RunState>,
    abort: bool,
    stamp_done: bool,
    /// Pause once a scan is done (one-shot).
    pause_after_scan: bool,
    /// Asked to hand off for a restart at the next safe point.
    detach: bool,
    /// A run restored after a rebuild continues once connected.
    resume_on_connect: bool,
    /// The checkpoint last written, so an unchanged run isn't rewritten.
    last_saved: String,
}

impl SequenceWorker {
    pub fn new(
        ctx: &egui::Context,
        commands: Receiver<SequenceCommand>,
        events: Sender<SequenceEvent>,
        plot_choice: &SharedPlotChoice,
    ) -> Self {
        let mut worker = Self {
            ctx: ctx.clone(),
            commands,
            events,
            plot_choice: plot_choice.clone(),
            state: None,
            abort: false,
            stamp_done: false,
            pause_after_scan: false,
            detach: false,
            resume_on_connect: false,
            last_saved: String::new(),
        };
        worker.restore();
        worker
    }

    /// Pick up the run the previous app session saved, if any.
    fn restore(&mut self) {
        let Some(c) = checkpoint::load() else { return };
        let auto_resume = c.resume_after_restart && !c.paused;
        let pause_after_scan = c.pause_after_scan;
        let state = RunState::from_checkpoint(c);
        let (index, steps) = (state.index, state.steps.clone());
        let entry = match steps.get(index) {
            Some((crate::connection::nanonis::sequence::model::CURRENT_SCAN, _)) => "the current scan".to_string(),
            Some((e, _)) => format!("entry {}", e + 1),
            None => "the end".to_string(),
        };
        let step = &steps[index].1;
        let scan_note = match state.scan {
            ScanPhase::Running(_) => ", its scan was left running in Nanonis",
            ScanPhase::Paused(_) => ", its scan is paused in Nanonis",
            ScanPhase::NotStarted => "",
        };
        let mut note = format!("picked up after the app restarted, at {entry} ({step}){scan_note}");
        let half_done = state.step_started
            && matches!(step, Step::CoarseMove { .. } | Step::MoveTip(_) | Step::Spectroscopy(_));
        if half_done {
            note.push_str(
                ". The app closed during this step, so it may be partly done: \
                 check before Continue, which runs it again",
            );
        }
        let auto_resume = auto_resume && !half_done;
        info!("restored the saved sequence at step {index} of {} (auto resume: {auto_resume})", steps.len());
        self.resume_on_connect = auto_resume;
        self.pause_after_scan = pause_after_scan;
        let plan = state.plan.clone().unwrap_or_default();
        let (survey_group, survey_tiles) = (state.survey_group.clone(), state.survey_tiles.clone());
        self.state = Some(state);
        self.emit(SequenceEvent::Restored { steps, index, plan, auto_resume, note, survey_group, survey_tiles });
        self.emit(SequenceEvent::PauseAfterScan(pause_after_scan));
        self.save();
    }

    /// Save the run to disk if it changed (see [`checkpoint`]).
    fn save(&mut self) {
        let Some(c) = self.state.as_ref().and_then(|s| s.checkpoint(self.pause_after_scan)) else {
            return;
        };
        match serde_json::to_string(&c) {
            Ok(json) if json != self.last_saved => {
                checkpoint::write(&json);
                self.last_saved = json;
            }
            Ok(_) => {}
            Err(e) => warn!("could not encode the sequence checkpoint: {e}"),
        }
    }

    /// The run ended: nothing to pick up any more.
    fn clear_saved(&mut self) {
        checkpoint::clear();
        self.last_saved.clear();
    }

    /// Save the run for the restarted app and tell the UI it may close.
    /// A running sequence continues by itself after the restart.
    fn detach_now(&mut self) {
        self.detach = false;
        if let Some(state) = &mut self.state {
            state.resume_after_restart = !state.paused;
            let index = state.index;
            if !state.paused {
                state.paused = true;
                state.pause_requested = false;
                self.emit(SequenceEvent::Paused { index });
            }
            info!("sequence handed off at step {index} for a restart");
        }
        self.save();
        self.emit(SequenceEvent::Detached);
    }

    fn emit(&self, event: SequenceEvent) {
        let _ = self.events.send(event);
        self.ctx.request_repaint();
    }

    fn handle_command(&mut self, cmd: SequenceCommand) {
        debug!("sequence command {cmd:?}");
        match cmd {
            SequenceCommand::Run { steps, group, scan_start, plan } => {
                if self.state.as_ref().is_some_and(|s| !s.paused) {
                    warn!("ignoring Run while a sequence is active");
                    return;
                }
                if steps.is_empty() {
                    self.emit(SequenceEvent::Finished);
                    return;
                }
                info!("starting sequence with {} steps", steps.len());
                self.abort = false;
                self.resume_on_connect = false;
                if plan.is_none() {
                    // Not saved: don't leave an older run to be picked up.
                    self.clear_saved();
                }
                self.state = Some(RunState::new(steps, group, scan_start, plan));
                self.emit(SequenceEvent::Started);
                self.save();
            }
            SequenceCommand::PauseAfterScan(on) => {
                self.pause_after_scan = on;
                self.save();
            }
            SequenceCommand::Detach => {
                // Idle: nothing to save, the app may close at once.
                if self.state.is_some() {
                    self.detach = true;
                } else {
                    self.emit(SequenceEvent::Detached);
                }
            }
            SequenceCommand::CancelDetach => self.detach = false,
            SequenceCommand::Pause => {
                if let Some(state) = &mut self.state {
                    state.pause_requested = true;
                }
            }
            SequenceCommand::Interrupt(reason) => {
                if let Some(state) = &mut self.state {
                    if !state.paused {
                        state.interrupt = Some(reason);
                    }
                }
            }
            SequenceCommand::Resume => {
                if let Some(state) = &mut self.state {
                    if state.paused {
                        state.paused = false;
                        state.pause_requested = false;
                        state.interrupt = None;
                        state.resume_after_restart = false;
                        self.resume_on_connect = false;
                        // Whatever was changed meanwhile is accepted.
                        state.baseline = None;
                        self.emit(SequenceEvent::Resumed);
                    }
                }
            }
            SequenceCommand::RestartEntry => {
                if let Some(state) = &mut self.state {
                    if state.paused {
                        state.restart = true;
                    }
                }
            }
            SequenceCommand::Abort => {
                if self.state.is_some() {
                    self.abort = true;
                }
            }
            SequenceCommand::StampDone => self.stamp_done = true,
            SequenceCommand::SurveyTiles { group, tiles } => {
                if let Some(state) = &mut self.state {
                    state.survey_group = group;
                    state.survey_tiles = tiles;
                }
                self.save();
            }
            SequenceCommand::AdjustCoarse { entry, steps } => {
                if let Some(state) = &mut self.state {
                    let from = state.index + 1;
                    for (e, step) in state.steps.iter_mut().skip(from) {
                        if *e == entry {
                            if let Step::CoarseMove { steps: s } = step {
                                info!("entry {}: coarse move {s:?} -> {steps:?} steps", entry + 1);
                                *s = steps;
                            }
                        }
                    }
                }
            }
        }
    }

    fn drain_commands(&mut self) {
        while let Ok(cmd) = self.commands.try_recv() {
            self.handle_command(cmd);
        }
    }

    fn pause_requested(&self) -> bool {
        self.state.as_ref().is_some_and(|s| s.pause_requested)
    }

    /// Did the app report someone using the instrument?
    fn take_interrupt(&mut self) -> Option<String> {
        self.state.as_mut().and_then(|s| s.interrupt.take())
    }

    /// Compare Nanonis with how the sequence left it (at most every
    /// [`CHECK_EVERY`] unless `now`). The first check after a reset just
    /// records the settings.
    fn external_change(&mut self, conn: &mut NanonisTcp, now: bool) -> NanonisTcpResult<Option<String>> {
        let Some(state) = &self.state else {
            return Ok(None);
        };
        if !now && state.last_check.is_some_and(|t| t.elapsed() < CHECK_EVERY) {
            return Ok(None);
        }
        let snap = Snapshot::read(conn)?;
        let state = self.state.as_mut().unwrap();
        state.last_check = Some(Instant::now());
        match &state.baseline {
            None => {
                state.baseline = Some(snap);
                Ok(None)
            }
            Some(base) => Ok(base.change(&snap)),
        }
    }

    /// A UI interrupt or a change in Nanonis, checked while a step runs.
    fn interrupted(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<Option<String>> {
        if let Some(reason) = self.take_interrupt() {
            return Ok(Some(reason));
        }
        self.external_change(conn, false)
    }

    /// A wait step, checking commands every slice and stopping when the
    /// instrument is used outside the sequence.
    fn wait_step(&mut self, conn: &mut NanonisTcp, dur: Duration, report: bool) -> NanonisTcpResult<Outcome> {
        let end = Instant::now() + dur;
        let mut last_report = Instant::now() - Duration::from_secs(1);
        loop {
            self.drain_commands();
            if self.abort {
                return Ok(Outcome::Aborted);
            }
            if self.detach {
                // Restarts from the beginning after the hand-off.
                return Ok(Outcome::Detach);
            }
            if self.pause_requested() {
                return Ok(Outcome::Interrupted);
            }
            if let Some(reason) = self.interrupted(conn)? {
                return Ok(Outcome::External(reason));
            }
            let now = Instant::now();
            if now >= end {
                if report {
                    self.emit(SequenceEvent::WaitProgress { remaining: 0. });
                }
                return Ok(Outcome::Completed);
            }
            if report && now - last_report >= Duration::from_millis(250) {
                last_report = now;
                self.emit(SequenceEvent::WaitProgress { remaining: (end - now).as_secs_f64() });
            }
            std::thread::sleep(SLICE.min(end - now));
        }
    }

    fn execute(&mut self, conn: &mut NanonisTcp, step: Step, group: u32) -> NanonisTcpResult<Outcome> {
        match step {
            Step::Withdraw => conn.zctrl_withdraw(true, None)?,
            Step::Approach => conn.zctrl_onoffset(true)?,
            Step::CoarseMove { steps } => {
                let steps = IVec2::from_array(steps);
                for (dir, n) in move_args_from_steps(steps).into_iter().flatten() {
                    conn.motor_start_move(dir, n, group, true)?;
                }
                self.emit(SequenceEvent::CoarseMoved { steps });
            }
            Step::Wait(secs) => {
                return self.wait_step(conn, Duration::from_secs_f64(secs.max(0.)), true);
            }
            Step::SetFrame(spec) => {
                let current = Frame::from_get(&conn.scan_frame_get()?);
                let frame = spec.apply(current);
                conn.call::<commands::scan::FrameSet>(&frame.to_args())?;
            }
            Step::SetResolution([px, lines]) => {
                let buffer = conn.scan_buffer_get()?;
                conn.scan_buffer_set(buffer.channel_indexes, px as usize, lines as usize)?;
            }
            Step::SetSpeed(speed) => {
                let current = conn.scan_speed_get()?;
                let width = conn.scan_frame_get()?.width as f64;
                if let Some(args) = speed_args(speed, width, current.speed_ratio as f64) {
                    conn.scan_speed_set(&args)?;
                } else {
                    warn!("ignoring invalid scan speed {speed} (frame width {width} m)");
                }
            }
            Step::MoveTip([x, y]) => return self.move_tip(conn, DVec2::new(x, y) * 1e-9),
            Step::Spectroscopy(iv) => return self.spectroscopy(conn, &iv),
            Step::SetBias(bias) => conn.bias_set(bias)?,
            Step::SetCurrent(current) => conn.zctrl_setpnt_set(current)?,
            Step::Scan(dir) => return self.execute_scan(conn, dir.into(), false),
            Step::AwaitScan { dir, include } => return self.await_scan(conn, dir.into(), include),
            Step::Stamp => {
                self.stamp_done = false;
                self.emit(SequenceEvent::StampRequest);
                let end = Instant::now() + STAMP_TIMEOUT;
                loop {
                    match self.commands.recv_timeout(SLICE) {
                        Ok(cmd) => self.handle_command(cmd),
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                    if self.abort {
                        return Ok(Outcome::Aborted);
                    }
                    if self.stamp_done {
                        break;
                    }
                    if Instant::now() >= end {
                        warn!("stamp not acknowledged within {STAMP_TIMEOUT:?}, continuing");
                        break;
                    }
                }
            }
        }
        Ok(Outcome::Completed)
    }

    /// Follow Me to `target` (m) and wait until the tip arrives. Abort stops
    /// the tip where it is; a pause waits for the move to finish.
    fn move_tip(&mut self, conn: &mut NanonisTcp, target: DVec2) -> NanonisTcpResult<Outcome> {
        if conn.scan_status_get()?.running {
            return Err(NanonisTcpError::Api(
                "a scan is running, so the tip can't be moved".into(),
            ));
        }
        let xy = |conn: &mut NanonisTcp| -> NanonisTcpResult<DVec2> {
            let r = conn.fol_me_xy_pos_get(true)?;
            Ok(DVec2::new(r.x_pos, r.y_pos))
        };
        let distance = xy(conn)?.distance(target);
        conn.fol_me_xy_pos_set(target.x, target.y, false)?;
        // Generous timeout from the Follow Me speed.
        let speed = (conn.fol_me_speed_get()?.speed as f64).max(1e-10);
        let timeout = Duration::from_secs_f64((5. * distance / speed + 5.).min(TIP_MAX_TIME.as_secs_f64()));
        let deadline = Instant::now() + timeout;
        loop {
            self.drain_commands();
            if self.abort {
                let here = xy(conn)?;
                conn.fol_me_xy_pos_set(here.x, here.y, false)?;
                return Ok(Outcome::Aborted);
            }
            if let Some(reason) = self.take_interrupt() {
                let here = xy(conn)?;
                conn.fol_me_xy_pos_set(here.x, here.y, false)?;
                return Ok(Outcome::External(reason));
            }
            let pos = xy(conn)?;
            if pos.distance(target) <= TIP_TOLERANCE_M {
                return Ok(Outcome::Completed);
            }
            if Instant::now() >= deadline {
                return Err(NanonisTcpError::Api(format!(
                    "the tip did not reach ({:.1}, {:.1}) nm within {:.0} s; it is at ({:.1}, {:.1}) nm",
                    target.x * 1e9,
                    target.y * 1e9,
                    timeout.as_secs_f64(),
                    pos.x * 1e9,
                    pos.y * 1e9
                )));
            }
            std::thread::sleep(SLICE / 2);
        }
    }

    /// Run an I(V) series where the tip is (same steps as the I(V) window),
    /// then save it to the Nanonis session folder. A pause waits for the
    /// series to finish; abort stops after the curve in progress.
    fn spectroscopy(&mut self, conn: &mut NanonisTcp, iv: &IvSeriesSettings) -> NanonisTcpResult<Outcome> {
        iv.validate().map_err(NanonisTcpError::Api)?;
        self.emit(SequenceEvent::SpecStarted { base: iv.base_name.clone() });
        let since = std::time::SystemTime::now();
        let mut host = RunnerHost { worker: self, curves: Vec::new() };
        let outcome = run_iv_series(conn, iv, &mut host);
        let curves = host.curves;
        if let SeriesOutcome::ConnectionLost(e) = outcome {
            self.emit(SequenceEvent::SpecDone {
                ok: false,
                message: format!("I(V) series {} lost the connection: {e}", iv.base_name),
                files: Vec::new(),
            });
            return Err(e);
        }
        // Save whatever was taken, also after an abort or error.
        let session = conn.util_session_path_get().unwrap_or_default();
        let files = nanonis_saved_files(&session, iv, since);
        let saved = if curves.is_empty() {
            Ok(String::new())
        } else {
            if session.trim().is_empty() {
                Err("Nanonis reported no session folder, so the curves were not saved".to_string())
            } else {
                let choice = self.plot_choice.lock().clone();
                save_batch(std::path::Path::new(&session), &iv.base_name, &curves, &choice, &files)
                    .map(|(_, msg)| msg)
            }
        };
        let (ok, mut message) = match &outcome {
            SeriesOutcome::Finished => (true, format!("I(V) series {} done.", iv.base_name)),
            SeriesOutcome::Aborted => (false, format!("I(V) series {} aborted; Z feedback left OFF.", iv.base_name)),
            SeriesOutcome::Failed(msg) => (false, format!("I(V) series {} failed: {msg}. Z feedback left OFF.", iv.base_name)),
            SeriesOutcome::ConnectionLost(_) => unreachable!(),
        };
        let save_failed = saved.is_err();
        match saved {
            Ok(msg) if !msg.is_empty() => message = format!("{message} {msg}"),
            Ok(_) => {}
            Err(e) => message = format!("{message} {e}"),
        }
        self.emit(SequenceEvent::SpecDone { ok: ok && !save_failed, message: message.clone(), files });
        match outcome {
            SeriesOutcome::Finished if save_failed => {
                // Don't carry on losing data: hold after this step.
                self.emit(SequenceEvent::Error { index: self.state.as_ref().map_or(0, |s| s.index), message });
                if let Some(state) = &mut self.state {
                    state.pause_requested = true;
                }
                Ok(Outcome::Completed)
            }
            SeriesOutcome::Finished => Ok(Outcome::Completed),
            SeriesOutcome::Aborted => Ok(Outcome::Aborted),
            _ => Err(NanonisTcpError::Api(message)),
        }
    }

    fn set_scan_phase(&mut self, phase: ScanPhase) {
        if let Some(state) = &mut self.state {
            state.scan = phase;
        }
    }

    /// Wait for a scan started outside the sequence to end. Included: it is
    /// treated like a sequence scan (Pause pauses it, Abort stops it).
    /// Otherwise Pause/Abort only stop waiting and the scan keeps running.
    fn await_scan(&mut self, conn: &mut NanonisTcp, dir: ScanDir, include: bool) -> NanonisTcpResult<Outcome> {
        let phase = self.state.as_ref().map_or(ScanPhase::NotStarted, |s| s.scan);
        if phase == ScanPhase::NotStarted {
            if !conn.scan_status_get()?.running {
                info!("no scan running when the sequence reached it; continuing");
                return Ok(Outcome::Completed);
            }
            if include {
                self.set_scan_phase(ScanPhase::Running(dir));
            }
        }
        if include {
            // Adopted: from here on it behaves like a scan this sequence started.
            return self.execute_scan(conn, dir, true);
        }
        loop {
            self.drain_commands();
            if self.abort {
                return Ok(Outcome::Aborted);
            }
            if self.detach {
                return Ok(Outcome::Detach);
            }
            if self.pause_requested() {
                return Ok(Outcome::Interrupted);
            }
            if let Some(reason) = self.take_interrupt() {
                return Ok(Outcome::External(reason));
            }
            let resp = conn.scan_wait_end_of_scan(Some(SCAN_POLL))?;
            if !resp.timed_out || !conn.scan_status_get()?.running {
                if let Some(state) = &mut self.state {
                    state.scanned = true;
                }
                return Ok(Outcome::Completed);
            }
        }
    }

    /// Start the scan and note how long a full frame should take.
    /// Pressing Start again a few seconds later lets the scanner settle
    /// after jumping to the frame start ([`ScanStart`]). Returns an outcome
    /// if the run was stopped, paused or held during the restarts.
    fn start_scan(&mut self, conn: &mut NanonisTcp, dir: ScanDir) -> NanonisTcpResult<Option<Outcome>> {
        let expected = Snapshot::read(conn).map(|s| s.frame_s()).unwrap_or(0.);
        let scan_start = self.state.as_ref().map_or(ScanStart::default(), |s| s.scan_start);
        if scan_start.park_s > 0. {
            if let Some(outcome) = self.park_at_start(conn, dir, scan_start.park_s)? {
                return Ok(Some(outcome));
            }
        }
        let starts = scan_start.starts.max(1);
        for k in 0..starts {
            conn.scan_action(ActionType::Start, dir)?;
            if let Some(state) = &mut self.state {
                state.scan = ScanPhase::Running(dir);
                state.scan_active = Duration::ZERO;
                state.scan_since = Some(Instant::now());
                state.scan_expected_s = expected;
            }
            if k + 1 == starts {
                // The scan that is kept: if the app closes now, the next
                // start finds it running.
                self.save();
                break;
            }
            debug!("scan start {} of {starts}", k + 1);
            match self.wait_step(conn, Duration::from_secs_f64(scan_start.gap_s.max(0.)), false)? {
                Outcome::Completed => {}
                Outcome::Aborted => {
                    conn.scan_action(ActionType::Stop, dir)?;
                    self.set_scan_phase(ScanPhase::NotStarted);
                    return Ok(Some(Outcome::Aborted));
                }
                o @ (Outcome::Interrupted | Outcome::Detach) => {
                    // Paused (or handed off) before the kept start: start
                    // over on resume.
                    conn.scan_action(ActionType::Stop, dir)?;
                    self.set_scan_phase(ScanPhase::NotStarted);
                    return Ok(Some(o));
                }
                other => return Ok(Some(other)),
            }
        }
        Ok(None)
    }

    /// Move the tip to where the scan will begin and wait there, so the
    /// creep after that jump is over before the first line. A tip that
    /// can't get there is not fatal: the scan just starts as before.
    fn park_at_start(&mut self, conn: &mut NanonisTcp, dir: ScanDir, wait_s: f64) -> NanonisTcpResult<Option<Outcome>> {
        let frame = Frame::from_get(&conn.scan_frame_get()?);
        let target = scan_start_point(&frame, matches!(dir, ScanDir::Up));
        debug!("parking at the scan start ({:.1}, {:.1}) nm for {wait_s} s", target.x * 1e9, target.y * 1e9);
        let moved = match self.move_tip(conn, target) {
            Ok(o) => o,
            Err(NanonisTcpError::Api(m)) => {
                warn!("could not park the tip at the scan start ({m}); starting anyway");
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        let outcome = match moved {
            Outcome::Completed => self.wait_step(conn, Duration::from_secs_f64(wait_s), false)?,
            other => other,
        };
        match outcome {
            Outcome::Completed => Ok(None),
            other => {
                // Not started yet: on resume the scan starts (and parks) afresh.
                self.set_scan_phase(ScanPhase::NotStarted);
                Ok(Some(other))
            }
        }
    }

    /// The scan is not running where the sequence expected it to be (paused
    /// or stopped while the sequence held). Try resuming it; if that fails,
    /// it counts as done when it ran its full time, else it is scanned again.
    fn recover_scan(&mut self, conn: &mut NanonisTcp, dir: ScanDir) -> NanonisTcpResult<Option<Outcome>> {
        conn.scan_action(ActionType::Resume, dir)?;
        std::thread::sleep(RESUME_CHECK);
        if conn.scan_status_get()?.running {
            return Ok(None);
        }
        if self.state.as_ref().is_some_and(|s| s.scan_complete()) {
            info!("scan had finished while the sequence was held; continuing");
            self.set_scan_phase(ScanPhase::NotStarted);
            if let Some(state) = &mut self.state {
                state.scanned = true;
            }
            return Ok(Some(Outcome::Completed));
        }
        info!("scan was stopped while the sequence was held; scanning it again");
        self.start_scan(conn, dir)
    }

    /// Start (or resume) the scan and wait for it to end, reacting to Pause,
    /// Abort and use of the instrument outside the sequence while it runs.
    /// `adopted`: a scan the sequence didn't start (its length is unknown).
    fn execute_scan(&mut self, conn: &mut NanonisTcp, dir: ScanDir, adopted: bool) -> NanonisTcpResult<Outcome> {
        let phase = self.state.as_ref().map_or(ScanPhase::NotStarted, |s| s.scan);
        let dir = match phase {
            ScanPhase::NotStarted => {
                if let Some(outcome) = self.start_scan(conn, dir)? {
                    return Ok(outcome);
                }
                dir
            }
            ScanPhase::Paused(dir) => {
                if let Some(state) = &mut self.state {
                    state.scan_since = Some(Instant::now());
                }
                if let Some(outcome) = self.recover_scan(conn, dir)? {
                    return Ok(outcome);
                }
                dir
            }
            ScanPhase::Running(dir) => {
                if adopted {
                    if let Some(state) = &mut self.state {
                        state.scan_expected_s = 0.;
                    }
                }
                // Held while the scan ran: it may have been paused or stopped.
                if !adopted && !conn.scan_status_get()?.running {
                    if let Some(outcome) = self.recover_scan(conn, dir)? {
                        return Ok(outcome);
                    }
                }
                dir
            }
        };
        self.set_scan_phase(ScanPhase::Running(dir));
        loop {
            self.drain_commands();
            if self.abort {
                conn.scan_action(ActionType::Stop, dir)?;
                self.set_scan_phase(ScanPhase::NotStarted);
                return Ok(Outcome::Aborted);
            }
            if self.detach {
                // The scan keeps running; the restarted app waits for it.
                return Ok(Outcome::Detach);
            }
            if self.pause_requested() {
                conn.scan_action(ActionType::Pause, dir)?;
                self.set_scan_phase(ScanPhase::Paused(dir));
                if let Some(state) = &mut self.state {
                    if let Some(t) = state.scan_since.take() {
                        state.scan_active += t.elapsed();
                    }
                }
                return Ok(Outcome::Interrupted);
            }
            // Someone else is using the instrument: hold, and leave the scan
            // as it is (they may have paused, stopped or changed it).
            if let Some(reason) = self.interrupted(conn)? {
                return Ok(Outcome::External(reason));
            }
            let resp = conn.scan_wait_end_of_scan(Some(SCAN_POLL))?;
            let ended = !resp.timed_out || !conn.scan_status_get()?.running;
            if !ended {
                continue;
            }
            if adopted || self.state.as_ref().is_some_and(|s| s.scan_complete()) {
                self.set_scan_phase(ScanPhase::NotStarted);
                if let Some(state) = &mut self.state {
                    state.scanned = true;
                }
                self.emit(SequenceEvent::ScanDone { file_path: resp.file_path });
                return Ok(Outcome::Completed);
            }
            // Ended long before a full frame: stopped (or paused) outside
            // the sequence.
            return Ok(Outcome::External("the scan was stopped or paused outside the sequence".into()));
        }
    }

    /// Restart the current entry: stop a scan it left running or paused,
    /// go back to the entry's first step (after a finished coarse move).
    fn restart_entry(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        let Some(state) = &mut self.state else {
            return Ok(());
        };
        state.restart = false;
        if let ScanPhase::Paused(dir) | ScanPhase::Running(dir) = state.scan {
            conn.scan_action(ActionType::Stop, dir)?;
        }
        let state = self.state.as_mut().unwrap();
        state.scan = ScanPhase::NotStarted;
        state.index = restart_index(&state.steps, state.index);
        state.paused = false;
        state.pause_requested = false;
        state.interrupt = None;
        state.baseline = None;
        info!("restarting the entry at step {}", state.index);
        self.emit(SequenceEvent::Resumed);
        Ok(())
    }

    /// Abort, stopping a scan this sequence started and left paused/running.
    fn abort_with_conn(&mut self, conn: &mut NanonisTcp) {
        if let Some(ScanPhase::Paused(dir) | ScanPhase::Running(dir)) =
            self.state.as_ref().map(|s| s.scan)
        {
            if let Err(e) = conn.scan_action(ActionType::Stop, dir) {
                warn!("failed to stop scan on abort: {e}");
            }
        }
        self.finish_aborted();
    }

    fn finish_aborted(&mut self) {
        info!("sequence aborted");
        self.state = None;
        self.clear_saved();
        self.abort = false;
        self.emit(SequenceEvent::Aborted);
    }
}

/// Lets the shared I(V) series report to the sequence UI and see Abort.
struct RunnerHost<'a> {
    worker: &'a mut SequenceWorker,
    curves: Vec<IvCurve>,
}
impl SeriesHost for RunnerHost<'_> {
    fn aborted(&mut self) -> bool {
        self.worker.drain_commands();
        self.worker.abort
    }
    fn status(&mut self, text: String) {
        self.worker.emit(SequenceEvent::SpecStatus(text));
    }
    fn curve(&mut self, curve: IvCurve) {
        self.curves.push(curve.clone());
        self.worker.emit(SequenceEvent::SpecCurve(curve));
    }
}

impl Worker for SequenceWorker {
    fn name(&self) -> String {
        "Sequence Worker".to_string()
    }

    fn init(&mut self, _conn: &mut NanonisTcp) -> eyre::Result<()> {
        // Handed off for a rebuild while running: carry on where it was.
        if std::mem::take(&mut self.resume_on_connect) {
            if let Some(state) = &mut self.state {
                info!("continuing the sequence after the restart");
                state.paused = false;
                state.resume_after_restart = false;
                state.baseline = None;
                self.emit(SequenceEvent::Resumed);
                self.save();
                return Ok(());
            }
        }
        // After a (re)connect, never continue blindly: hold a running sequence.
        if let Some(state) = &mut self.state {
            if !state.paused {
                state.paused = true;
                let index = state.index;
                self.emit(SequenceEvent::Error {
                    index,
                    message: "connection to Nanonis was lost; sequence paused".into(),
                });
                self.emit(SequenceEvent::Paused { index });
            }
        }
        Ok(())
    }

    fn work(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        if self.state.as_ref().is_some_and(|s| s.paused && s.restart) {
            self.restart_entry(conn)?;
        }
        let active = self.state.as_ref().is_some_and(|s| !s.paused);
        if !active {
            // Idle or paused: block briefly instead of spinning.
            match self.commands.recv_timeout(Duration::from_millis(200)) {
                Ok(cmd) => self.handle_command(cmd),
                Err(_) => {}
            }
            self.drain_commands();
            if self.abort {
                self.abort_with_conn(conn);
            } else if self.detach {
                self.detach_now();
            } else {
                self.save();
            }
            return Ok(());
        }
        self.drain_commands();
        if self.abort {
            self.abort_with_conn(conn);
            return Ok(());
        }
        if self.detach {
            self.detach_now();
            return Ok(());
        }

        let (index, step, group) = {
            let s = self.state.as_ref().unwrap();
            (s.index, s.steps[s.index].1.clone(), s.group)
        };
        // Before each step: nothing may have changed behind the sequence.
        let our_scan = self.state.as_ref().is_some_and(|s| s.scan != ScanPhase::NotStarted);
        let reason = match self.take_interrupt() {
            Some(r) => Some(r),
            None if matches!(step, Step::AwaitScan { include: false, .. }) => None,
            None => match self.external_change(conn, true)? {
                Some(r) => Some(r),
                None if !our_scan
                    && !matches!(step, Step::AwaitScan { .. })
                    && conn.scan_status_get()?.running =>
                {
                    Some("a scan was started outside the sequence".into())
                }
                None => None,
            },
        };
        let result = match reason {
            Some(reason) => Ok(Outcome::External(reason)),
            None => {
                self.emit(SequenceEvent::StepStarted { index });
                debug!("sequence step {index}: {step}");
                if let Some(state) = &mut self.state {
                    state.step_started = true;
                }
                self.save();
                self.execute(conn, step.clone(), group)
            }
        };
        // Steps that change watched settings: take them as the new normal.
        if matches!(
            step,
            Step::SetFrame(_)
                | Step::SetResolution(_)
                | Step::SetSpeed(_)
                | Step::SetBias(_)
                | Step::SetCurrent(_)
                | Step::Spectroscopy(_)
                | Step::AwaitScan { .. }
        ) {
            if let Some(state) = &mut self.state {
                state.baseline = None;
            }
        }

        match result {
            Ok(Outcome::External(reason)) => {
                warn!("sequence held at step {index}: {reason}");
                let state = self.state.as_mut().unwrap();
                state.paused = true;
                state.pause_requested = false;
                self.emit(SequenceEvent::Paused { index });
                self.emit(SequenceEvent::Interrupted { index, reason });
            }
            Ok(Outcome::Completed) => {
                let state = self.state.as_mut().unwrap();
                state.index += 1;
                state.step_started = false;
                // "Pause after scan": once the scan is done and stamped
                // (never between a scan and its stamp).
                let next_is_stamp = matches!(state.steps.get(state.index), Some((_, Step::Stamp)));
                // A warm-up scan is followed by its entry's real scan: wait for that.
                let entry_done = state.index.checked_sub(1).and_then(|i| state.steps.get(i)).map(|(e, _)| *e);
                let warm_up = matches!(state.steps.get(state.index), Some((e, Step::Scan(_))) if Some(*e) == entry_done);
                let after_scan = self.pause_after_scan && state.scanned && !next_is_stamp && !warm_up;
                if !next_is_stamp {
                    // Only the scan just finished counts, not an older one.
                    state.scanned = false;
                }
                if state.index >= state.steps.len() {
                    info!("sequence finished");
                    self.state = None;
                    self.clear_saved();
                    self.emit(SequenceEvent::Finished);
                } else if state.pause_requested || after_scan {
                    state.paused = true;
                    let index = state.index;
                    if after_scan {
                        info!("pausing after the scan, as asked");
                        self.pause_after_scan = false;
                        self.emit(SequenceEvent::PauseAfterScan(false));
                    }
                    self.emit(SequenceEvent::Paused { index });
                }
            }
            Ok(Outcome::Interrupted) => {
                let state = self.state.as_mut().unwrap();
                state.paused = true;
                self.emit(SequenceEvent::Paused { index });
            }
            Ok(Outcome::Detach) => self.detach_now(),
            Ok(Outcome::Aborted) => self.finish_aborted(),
            Err(e @ NanonisTcpError::Io(_)) => {
                // Leave the step un-advanced; init() pauses on reconnect.
                return Err(e).context(format!("sequence step {index} ({step})"));
            }
            Err(e) => {
                error!("sequence step {index} ({step}) failed: {e}");
                let state = self.state.as_mut().unwrap();
                state.paused = true;
                self.emit(SequenceEvent::Error {
                    index,
                    message: format!("{step}: {e}"),
                });
                self.emit(SequenceEvent::Paused { index });
            }
        }
        self.save();
        Ok(())
    }
}

/// Consistent Scan.SpeedSet arguments for `speed` on a frame `width` m wide,
/// keeping Nanonis's backward/forward speed `ratio`. `None` if invalid.
fn speed_args(speed: Speed, width: f64, ratio: f64) -> Option<commands::scan::SpeedSetArgs> {
    let ratio = if ratio > 0. { ratio } else { 1. };
    let (line_time, linear, keep_constant) = match speed {
        Speed::LineTime(t) if t > 0. => (t, width / t, 2),
        Speed::Linear(v) if v > 0. => {
            let v = v * 1e-9;
            (width / v, v, 1)
        }
        _ => return None,
    };
    if !(line_time.is_finite() && linear.is_finite() && width > 0.) {
        return None;
    }
    Some(commands::scan::SpeedSetArgs {
        fwd_speed: linear as f32,
        bwd_speed: (linear * ratio) as f32,
        fwd_line_time: line_time as f32,
        bwd_line_time: (line_time / ratio) as f32,
        keep_constant,
        speed_ratio: ratio as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trips() {
        use crate::connection::nanonis::sequence::model::CURRENT_SCAN;
        let steps = vec![
            (CURRENT_SCAN, Step::AwaitScan { dir: Dir::Down, include: true }),
            (0, Step::CoarseMove { steps: [3, -2] }),
            (0, Step::Scan(Dir::Up)),
            (0, Step::Stamp),
        ];
        let mut s = RunState::new(steps.clone(), 1, ScanStart::default(), Some(Sequence::default()));
        s.index = 2;
        s.scan = ScanPhase::Running(ScanDir::Up);
        s.scan_active = Duration::from_secs(5);
        s.scan_expected_s = 100.;
        s.resume_after_restart = true;
        let json = serde_json::to_string(&s.checkpoint(true).unwrap()).unwrap();
        let c: Checkpoint = serde_json::from_str(&json).unwrap();
        assert!(c.pause_after_scan && c.resume_after_restart);
        let r = RunState::from_checkpoint(c);
        assert_eq!(r.steps, steps);
        assert_eq!(r.index, 2);
        assert!(r.paused);
        assert_eq!(r.scan, ScanPhase::Running(ScanDir::Up));
        assert!(r.scan_active >= Duration::from_secs(5));
        // Not saved without a plan (calibration runs).
        assert!(RunState::new(steps, 1, ScanStart::default(), None).checkpoint(false).is_none());
    }

    #[test]
    fn speed_args_are_consistent() {
        // 100 nm frame at 1 s/line -> 100 nm/s, backward twice as fast.
        let a = speed_args(Speed::LineTime(1.), 100e-9, 2.).unwrap();
        assert!((a.fwd_speed - 100e-9).abs() < 1e-12);
        assert!((a.bwd_speed - 200e-9).abs() < 1e-12);
        assert!((a.bwd_line_time - 0.5).abs() < 1e-6);
        assert_eq!(a.keep_constant, 2);
        let b = speed_args(Speed::Linear(50.), 100e-9, 1.).unwrap();
        assert!((b.fwd_line_time - 2.).abs() < 1e-6);
        assert_eq!(b.keep_constant, 1);
        assert!(speed_args(Speed::LineTime(0.), 100e-9, 1.).is_none());
        assert!(speed_args(Speed::Linear(10.), 0., 1.).is_none());
    }
}
