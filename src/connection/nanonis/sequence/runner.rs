use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, RecvTimeoutError, Sender};
use eyre::Context as _;
use glam::IVec2;
use nanonis_tcp::{
    blocking::NanonisTcp,
    commands,
    error::{NanonisTcpError, NanonisTcpResult},
    ActionType, ScanDir,
};
use tracing::{debug, error, info, warn};

use crate::connection::nanonis::{
    sequence::model::{Frame, Speed, Step},
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

#[derive(Debug)]
pub enum SequenceCommand {
    Run {
        steps: Vec<(usize, Step)>,
        /// Motor group index (0-based) for coarse moves.
        group: u32,
    },
    /// Hold now. A scan in progress is paused in Nanonis and resumed (not
    /// restarted) on Resume. A wait in progress is cut short and restarts
    /// from the beginning on resume. Other steps finish first.
    Pause,
    Resume,
    Abort,
    /// UI has finished stamping the image.
    StampDone,
}

#[derive(Debug, Clone)]
pub enum SequenceEvent {
    Started,
    StepStarted { index: usize },
    WaitProgress { remaining: f64 },
    CoarseMoved { steps: IVec2 },
    StampRequest,
    ScanDone { file_path: String },
    Paused { index: usize },
    Resumed,
    Finished,
    Aborted,
    Error { index: usize, message: String },
}

struct RunState {
    steps: Vec<(usize, Step)>,
    group: u32,
    index: usize,
    paused: bool,
    pause_requested: bool,
    /// State of the scan started by the current Scan step.
    scan: ScanPhase,
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
    Aborted,
}

pub struct SequenceWorker {
    ctx: egui::Context,
    commands: Receiver<SequenceCommand>,
    events: Sender<SequenceEvent>,
    state: Option<RunState>,
    abort: bool,
    stamp_done: bool,
}

impl SequenceWorker {
    pub fn new(
        ctx: &egui::Context,
        commands: Receiver<SequenceCommand>,
        events: Sender<SequenceEvent>,
    ) -> Self {
        Self {
            ctx: ctx.clone(),
            commands,
            events,
            state: None,
            abort: false,
            stamp_done: false,
        }
    }

    fn emit(&self, event: SequenceEvent) {
        let _ = self.events.send(event);
        self.ctx.request_repaint();
    }

    fn handle_command(&mut self, cmd: SequenceCommand) {
        debug!("sequence command {cmd:?}");
        match cmd {
            SequenceCommand::Run { steps, group } => {
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
                self.state = Some(RunState {
                    steps,
                    group,
                    index: 0,
                    paused: false,
                    pause_requested: false,
                    scan: ScanPhase::NotStarted,
                });
                self.emit(SequenceEvent::Started);
            }
            SequenceCommand::Pause => {
                if let Some(state) = &mut self.state {
                    state.pause_requested = true;
                }
            }
            SequenceCommand::Resume => {
                if let Some(state) = &mut self.state {
                    if state.paused {
                        state.paused = false;
                        state.pause_requested = false;
                        self.emit(SequenceEvent::Resumed);
                    }
                }
            }
            SequenceCommand::Abort => {
                if self.state.is_some() {
                    self.abort = true;
                }
            }
            SequenceCommand::StampDone => self.stamp_done = true,
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

    /// Sleep for `dur`, checking commands every slice.
    fn interruptible_sleep(&mut self, dur: Duration, report: bool) -> Outcome {
        let end = Instant::now() + dur;
        let mut last_report = Instant::now() - Duration::from_secs(1);
        loop {
            self.drain_commands();
            if self.abort {
                return Outcome::Aborted;
            }
            if report && self.pause_requested() {
                return Outcome::Interrupted;
            }
            let now = Instant::now();
            if now >= end {
                if report {
                    self.emit(SequenceEvent::WaitProgress { remaining: 0. });
                }
                return Outcome::Completed;
            }
            if report && now - last_report >= Duration::from_millis(250) {
                last_report = now;
                self.emit(SequenceEvent::WaitProgress {
                    remaining: (end - now).as_secs_f64(),
                });
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
                return Ok(self.interruptible_sleep(Duration::from_secs_f64(secs.max(0.)), true));
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
            Step::SetBias(bias) => conn.bias_set(bias)?,
            Step::SetCurrent(current) => conn.zctrl_setpnt_set(current)?,
            Step::Scan(dir) => return self.execute_scan(conn, dir.into()),
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

    fn set_scan_phase(&mut self, phase: ScanPhase) {
        if let Some(state) = &mut self.state {
            state.scan = phase;
        }
    }

    /// Start (or resume) the scan and wait for it to end, reacting to Pause
    /// and Abort while it runs.
    fn execute_scan(&mut self, conn: &mut NanonisTcp, dir: ScanDir) -> NanonisTcpResult<Outcome> {
        let phase = self.state.as_ref().map_or(ScanPhase::NotStarted, |s| s.scan);
        let dir = match phase {
            ScanPhase::NotStarted => {
                conn.scan_action(ActionType::Start, dir)?;
                dir
            }
            ScanPhase::Paused(dir) => {
                conn.scan_action(ActionType::Resume, dir)?;
                std::thread::sleep(RESUME_CHECK);
                if !conn.scan_status_get()?.running {
                    // Finished or stopped by hand while the sequence was paused.
                    info!("scan was no longer running after resume; continuing");
                    self.set_scan_phase(ScanPhase::NotStarted);
                    return Ok(Outcome::Completed);
                }
                dir
            }
            // Reconnected mid-scan: just keep waiting for it.
            ScanPhase::Running(dir) => dir,
        };
        self.set_scan_phase(ScanPhase::Running(dir));
        loop {
            self.drain_commands();
            if self.abort {
                conn.scan_action(ActionType::Stop, dir)?;
                self.set_scan_phase(ScanPhase::NotStarted);
                return Ok(Outcome::Aborted);
            }
            if self.pause_requested() {
                conn.scan_action(ActionType::Pause, dir)?;
                self.set_scan_phase(ScanPhase::Paused(dir));
                return Ok(Outcome::Interrupted);
            }
            let resp = conn.scan_wait_end_of_scan(Some(SCAN_POLL))?;
            if !resp.timed_out {
                self.set_scan_phase(ScanPhase::NotStarted);
                self.emit(SequenceEvent::ScanDone {
                    file_path: resp.file_path,
                });
                return Ok(Outcome::Completed);
            }
        }
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
        self.abort = false;
        self.emit(SequenceEvent::Aborted);
    }
}

impl Worker for SequenceWorker {
    fn name(&self) -> String {
        "Sequence Worker".to_string()
    }

    fn init(&mut self, _conn: &mut NanonisTcp) -> eyre::Result<()> {
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
            }
            return Ok(());
        }
        self.drain_commands();
        if self.abort {
            self.abort_with_conn(conn);
            return Ok(());
        }

        let (index, step, group) = {
            let s = self.state.as_ref().unwrap();
            (s.index, s.steps[s.index].1, s.group)
        };
        self.emit(SequenceEvent::StepStarted { index });
        debug!("sequence step {index}: {step}");

        match self.execute(conn, step, group) {
            Ok(Outcome::Completed) => {
                let state = self.state.as_mut().unwrap();
                state.index += 1;
                if state.index >= state.steps.len() {
                    info!("sequence finished");
                    self.state = None;
                    self.emit(SequenceEvent::Finished);
                } else if state.pause_requested {
                    state.paused = true;
                    let index = state.index;
                    self.emit(SequenceEvent::Paused { index });
                }
            }
            Ok(Outcome::Interrupted) => {
                let state = self.state.as_mut().unwrap();
                state.paused = true;
                self.emit(SequenceEvent::Paused { index });
            }
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
