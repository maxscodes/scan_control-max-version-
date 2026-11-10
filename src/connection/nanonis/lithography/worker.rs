//! Executes a lithography path on its own Nanonis connection.
//!
//! Ported from the Python GUI's lithography worker (v1.13+):
//! 1. Capture the current bias/setpoint (the "scan values") and Follow Me speed.
//! 2. Feedback ON, travel to the path start, then apply the lithography
//!    bias/setpoint and the lithography Follow Me speed.
//! 3. For each segment:
//!    - entering an active segment: Z-lift mode restores the surface Z
//!      (verified) and turns feedback on; ramp mode ramps scan → litho values;
//!    - each inactive segment: Z-lift mode switches feedback off, lifts by
//!      `lift_pm` from the current Z and verifies the readback before any XY
//!      move; ramp mode ramps litho → scan values (feedback stays on);
//!    - move to the segment end and wait until the tip arrives.
//! 4. Finally (also after stop/error): halt XY motion, restore the surface
//!    (Z-lift) and feedback ON, restore the scan values and Follow Me speed,
//!    and, only after a complete run, optionally start a scan of the frame.

use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender};
use glam::DVec2;
use nanonis_tcp::{blocking::NanonisTcp, error::NanonisTcpError, ActionType, ScanDir};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::connection::nanonis::{lithography::path::PathMode, worker::Worker};

const Z_OFF: u16 = 1;
const Z_ON: u16 = 2;
const STATUS_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a commanded Z is given to settle before the readback is judged.
const Z_SETTLE_TIMEOUT: Duration = Duration::from_secs(1);
const Z_TOLERANCE_M: f64 = 2e-12;
/// XY arrival tolerance.
const XY_TOLERANCE_M: f64 = 0.5e-9;
const POLL: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transition {
    /// Feedback stays ON; bias/setpoint ramp between litho and scan values.
    Ramp,
    /// Feedback OFF and tip lifted for every inactive segment; no ramp.
    ZLift,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LithoSettings {
    pub mode: PathMode,
    pub pitch_nm: f64,
    pub litho_bias_v: f64,
    pub litho_setpoint_pa: f64,
    /// Follow Me speed while following the path.
    pub litho_speed_nm_s: f64,
    /// Follow Me speed for the move to the path start (at scan values).
    pub travel_speed_nm_s: f64,
    pub transition: Transition,
    pub ramp_steps: u32,
    pub ramp_ms: f64,
    pub lift_pm: f64,
    /// Continue (with a warning) when the lifted Z readback is off target.
    pub allow_z_mismatch: bool,
    /// Start an upward scan of the current frame after a complete run.
    pub scan_after: bool,
}
impl Default for LithoSettings {
    fn default() -> Self {
        Self {
            mode: PathMode::PatchLocal,
            pitch_nm: 0.77,
            litho_bias_v: 4.0,
            litho_setpoint_pa: 2000.,
            litho_speed_nm_s: 10.,
            travel_speed_nm_s: 100.,
            transition: Transition::Ramp,
            ramp_steps: 20,
            ramp_ms: 200.,
            lift_pm: 100.,
            allow_z_mismatch: false,
            scan_after: true,
        }
    }
}
impl LithoSettings {
    pub fn validate(&self) -> Result<(), String> {
        let finite = [
            self.pitch_nm,
            self.litho_bias_v,
            self.litho_setpoint_pa,
            self.litho_speed_nm_s,
            self.travel_speed_nm_s,
            self.ramp_ms,
            self.lift_pm,
        ];
        if finite.iter().any(|v| !v.is_finite()) {
            return Err("Lithography values must be finite".into());
        }
        if self.litho_setpoint_pa <= 0. {
            return Err("Lithography setpoint must be positive".into());
        }
        if self.litho_speed_nm_s <= 0. || self.travel_speed_nm_s <= 0. {
            return Err("Speeds must be positive".into());
        }
        if self.mode.requires_ramp() && self.transition != Transition::Ramp {
            return Err(
                "Comb modes require the bias/current ramp for returns and step-ups; \
                 Z-lift is not allowed"
                    .into(),
            );
        }
        match self.transition {
            Transition::Ramp if self.ramp_steps == 0 || self.ramp_ms <= 0. => {
                Err("Ramp steps and duration must be positive".into())
            }
            Transition::ZLift if self.lift_pm < 0. => {
                Err("Travel lift (+Z) must be non-negative".into())
            }
            _ => Ok(()),
        }
    }
    pub fn ramp_s(&self) -> Option<f64> {
        (self.transition == Transition::Ramp).then_some(self.ramp_ms / 1000.)
    }
}

/// One segment in Nanonis XY coordinates (m).
#[derive(Debug, Clone, Copy)]
pub struct JobSegment {
    pub start: DVec2,
    pub end: DVec2,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct LithoJob {
    pub settings: LithoSettings,
    pub path: Vec<JobSegment>,
}

#[derive(Debug)]
pub enum LithoCommand {
    Run(LithoJob),
    /// Move the tip to a point (m) at `speed` m/s, without lithography.
    MoveTo { point: DVec2, speed: f64 },
    Stop,
}

#[derive(Debug, Clone)]
pub enum LithoEvent {
    Status(String),
    Progress { done: usize, total: usize },
    /// Completed. `scan` reports the post-lithography scan, if one was requested.
    Finished { scan: Option<Result<(), String>> },
    Stopped,
    Error(String),
    Moved(Result<(), String>),
}

enum RunError {
    Stopped,
    Nanonis(NanonisTcpError),
    Other(String),
}
impl From<NanonisTcpError> for RunError {
    fn from(e: NanonisTcpError) -> Self {
        Self::Nanonis(e)
    }
}
impl RunError {
    fn is_io(&self) -> bool {
        matches!(self, RunError::Nanonis(NanonisTcpError::Io(_)))
    }
    fn message(&self) -> String {
        match self {
            RunError::Stopped => "stopped".into(),
            RunError::Nanonis(e) => format!("Nanonis error: {e}"),
            RunError::Other(m) => m.clone(),
        }
    }
}

pub struct LithographyWorker {
    ctx: egui::Context,
    commands: Receiver<LithoCommand>,
    events: Sender<LithoEvent>,
    stop: bool,
}

impl LithographyWorker {
    pub fn new(ctx: &egui::Context, commands: Receiver<LithoCommand>, events: Sender<LithoEvent>) -> Self {
        Self {
            ctx: ctx.clone(),
            commands,
            events,
            stop: false,
        }
    }

    fn emit(&self, event: LithoEvent) {
        let _ = self.events.send(event);
        self.ctx.request_repaint();
    }
    fn status(&self, text: impl Into<String>) {
        self.emit(LithoEvent::Status(text.into()));
    }
    fn check_stop(&mut self) -> Result<(), RunError> {
        while let Ok(cmd) = self.commands.try_recv() {
            match cmd {
                LithoCommand::Stop => self.stop = true,
                other => warn!("ignoring {other:?} while lithography is active"),
            }
        }
        if self.stop {
            Err(RunError::Stopped)
        } else {
            Ok(())
        }
    }

    fn wait_z_status(&mut self, conn: &mut NanonisTcp, wanted: u16) -> Result<(), RunError> {
        let deadline = Instant::now() + STATUS_TIMEOUT;
        loop {
            let status = conn.zctrl_status_get()?;
            if status == wanted {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(RunError::Other(format!(
                    "timed out waiting for Z feedback {}; last status was {status}",
                    if wanted == Z_ON { "ON" } else { "OFF" }
                )));
            }
            std::thread::sleep(POLL);
        }
    }
    fn set_feedback(&mut self, conn: &mut NanonisTcp, on: bool) -> Result<(), RunError> {
        conn.zctrl_onoffset(on)?;
        self.wait_z_status(conn, if on { Z_ON } else { Z_OFF })
    }

    /// Command Z and return the readback once within tolerance, or the last
    /// readback after [`Z_SETTLE_TIMEOUT`].
    fn set_z(&mut self, conn: &mut NanonisTcp, target: f64) -> Result<(f64, bool), RunError> {
        conn.zctrl_zpos_set(target as f32)?;
        let tol = z_tolerance(target);
        let deadline = Instant::now() + Z_SETTLE_TIMEOUT;
        loop {
            let z = conn.zctrl_zpos_get()? as f64;
            if (z - target).abs() <= tol {
                return Ok((z, true));
            }
            if Instant::now() >= deadline {
                return Ok((z, false));
            }
            std::thread::sleep(POLL);
        }
    }

    /// Feedback OFF (confirmed), then lift by `lift_pm` from the current Z.
    fn inactive_jump(&mut self, conn: &mut NanonisTcp, s: &LithoSettings) -> Result<(), RunError> {
        self.set_feedback(conn, false)?;
        // Read only after OFF is confirmed: before that Z may still be moving.
        let before = conn.zctrl_zpos_get()? as f64;
        let target = before + s.lift_pm * 1e-12;
        let (z, ok) = self.set_z(conn, target)?;
        if !ok {
            let msg = format!(
                "Z lift not verified: target {:.4} nm, readback {:.4} nm (lift {} pm)",
                target * 1e9,
                z * 1e9,
                s.lift_pm
            );
            if !s.allow_z_mismatch {
                return Err(RunError::Other(msg + "; refusing XY move"));
            }
            warn!("{msg}; continuing because the mismatch override is enabled");
            self.status(format!("⚠ {msg} (override on)"));
        }
        Ok(())
    }

    /// Put the tip back on the captured surface Z (verified), then feedback ON.
    fn restore_surface(&mut self, conn: &mut NanonisTcp, surface: f64) -> Result<(), RunError> {
        let (z, ok) = self.set_z(conn, surface)?;
        if !ok {
            return Err(RunError::Other(format!(
                "surface Z not verified: target {:.4} nm, readback {:.4} nm; refusing feedback re-entry",
                surface * 1e9,
                z * 1e9
            )));
        }
        self.set_feedback(conn, true)
    }

    /// `steps` linear updates over `ramp_ms`, each preceded by its wait.
    fn ramp(
        &mut self,
        conn: &mut NanonisTcp,
        from: (f64, f64),
        to: (f64, f64),
        s: &LithoSettings,
    ) -> Result<(), RunError> {
        let interval = Duration::from_secs_f64(s.ramp_ms / 1000. / s.ramp_steps as f64);
        for (bias, setpoint) in ramp_values(from, to, s.ramp_steps) {
            self.check_stop()?;
            std::thread::sleep(interval);
            conn.bias_set(bias as f32)?;
            conn.zctrl_setpnt_set(setpoint as f32)?;
        }
        Ok(())
    }

    fn xy_get(conn: &mut NanonisTcp) -> Result<DVec2, RunError> {
        let r = conn.fol_me_xy_pos_get(true)?;
        Ok(DVec2::new(r.x_pos, r.y_pos))
    }

    /// Command an XY target and wait until the tip arrives. The timeout scales
    /// with distance; unlike the Python tool, a timeout is an error.
    fn move_xy(&mut self, conn: &mut NanonisTcp, target: DVec2, speed: f64) -> Result<(), RunError> {
        conn.fol_me_xy_pos_set(target.x, target.y, false)?;
        let distance = Self::xy_get(conn)?.distance(target);
        if distance <= XY_TOLERANCE_M {
            return Ok(());
        }
        let timeout = (5. * distance / speed.max(1e-15) + 1.).max(2.);
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        loop {
            self.check_stop()?;
            let pos = Self::xy_get(conn)?;
            if pos.distance(target) <= XY_TOLERANCE_M {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(RunError::Other(format!(
                    "tip did not reach ({:.3}, {:.3}) nm within {timeout:.1} s; it is at ({:.3}, {:.3}) nm",
                    target.x * 1e9,
                    target.y * 1e9,
                    pos.x * 1e9,
                    pos.y * 1e9
                )));
            }
            std::thread::sleep(POLL);
        }
    }

    /// Cancel the current XY target by re-targeting the live position.
    fn halt_xy(conn: &mut NanonisTcp) -> Result<(), RunError> {
        let pos = Self::xy_get(conn)?;
        conn.fol_me_xy_pos_set(pos.x, pos.y, false)?;
        Ok(())
    }

    fn execute(
        &mut self,
        conn: &mut NanonisTcp,
        job: &LithoJob,
        scan_values: (f64, f64),
        surface: &mut Option<f64>,
    ) -> Result<(), RunError> {
        let s = &job.settings;
        let litho_values = (s.litho_bias_v, s.litho_setpoint_pa * 1e-12);
        let zlift = s.transition == Transition::ZLift;
        let ramp = s.transition == Transition::Ramp;
        let total = job.path.len();

        self.status("Feedback ON, moving to the path start…");
        self.set_feedback(conn, true)?;
        conn.fol_me_speed_set((s.travel_speed_nm_s * 1e-9) as f32, true)?;
        self.move_xy(conn, job.path[0].start, s.travel_speed_nm_s * 1e-9)?;
        self.check_stop()?;

        // Lithography values only once the tip is at the start point.
        conn.bias_set(litho_values.0 as f32)?;
        conn.zctrl_setpnt_set(litho_values.1 as f32)?;
        let speed = s.litho_speed_nm_s * 1e-9;
        conn.fol_me_speed_set(speed as f32, true)?;
        if zlift {
            *surface = Some(conn.zctrl_zpos_get()? as f64);
        }
        self.status("Writing…");

        let mut in_litho = true;
        let mut last_report = Instant::now();
        for (i, seg) in job.path.iter().enumerate() {
            self.check_stop()?;
            if seg.active && !in_litho {
                if let (true, Some(z)) = (zlift, *surface) {
                    self.restore_surface(conn, z)?;
                } else {
                    self.set_feedback(conn, true)?;
                }
                if ramp {
                    self.ramp(conn, scan_values, litho_values, s)?;
                }
                in_litho = true;
            } else if !seg.active {
                if zlift {
                    self.inactive_jump(conn, s)?;
                }
                if in_litho && ramp {
                    self.ramp(conn, litho_values, scan_values, s)?;
                }
                in_litho = false;
            }
            self.move_xy(conn, seg.end, speed)?;
            if last_report.elapsed() > Duration::from_millis(100) || i + 1 == total {
                last_report = Instant::now();
                self.emit(LithoEvent::Progress { done: i + 1, total });
            }
        }
        Ok(())
    }

    /// Restore feedback, Z surface, scan values and Follow Me speed.
    fn finalize(
        &mut self,
        conn: &mut NanonisTcp,
        halt: bool,
        surface: Option<f64>,
        scan_values: (f64, f64),
        folme_speed: (f32, bool),
    ) -> Result<(), RunError> {
        if halt {
            if let Err(e) = Self::halt_xy(conn) {
                warn!("could not halt Follow Me before finalization: {}", e.message());
            }
        }
        match surface {
            Some(z) => {
                self.set_feedback(conn, false)?;
                self.restore_surface(conn, z)?;
            }
            None => self.set_feedback(conn, true)?,
        }
        conn.bias_set(scan_values.0 as f32)?;
        conn.zctrl_setpnt_set(scan_values.1 as f32)?;
        conn.fol_me_speed_set(folme_speed.0, folme_speed.1)?;
        Ok(())
    }

    fn run(&mut self, conn: &mut NanonisTcp, job: LithoJob) -> eyre::Result<()> {
        self.stop = false;
        if let Err(msg) = job.settings.validate() {
            self.emit(LithoEvent::Error(msg));
            return Ok(());
        }
        if job.path.is_empty() {
            self.emit(LithoEvent::Error("the path is empty".into()));
            return Ok(());
        }
        info!("starting lithography: {} segments, {:?}", job.path.len(), job.settings);
        // Preconditions, before anything changes on the instrument.
        let pre = (|| -> Result<_, RunError> {
            if conn.scan_status_get()?.running {
                return Err(RunError::Other("a scan is running; stop it first".into()));
            }
            let scan_values = (conn.bias_get()? as f64, conn.zctrl_setpnt_get()? as f64);
            let folme = conn.fol_me_speed_get()?;
            Ok((scan_values, (folme.speed, folme.custom_speed)))
        })();
        let (scan_values, folme_speed) = match pre {
            Ok(v) => v,
            Err(e) => {
                self.emit(LithoEvent::Error(e.message()));
                return if e.is_io() { Err(eyre::eyre!(e.message())) } else { Ok(()) };
            }
        };

        let mut surface = None;
        let result = self.execute(conn, &job, scan_values, &mut surface);
        if result.as_ref().is_err_and(RunError::is_io) {
            let msg = result.err().unwrap().message();
            error!("lithography lost the connection: {msg}");
            self.emit(LithoEvent::Error(format!(
                "{msg}. Connection lost mid-run: check feedback, Z, bias and setpoint in Nanonis!"
            )));
            return Err(eyre::eyre!(msg));
        }
        self.status("Finalizing: restoring feedback and scan values…");
        let halt = result.is_err();
        let finalized = self.finalize(conn, halt, surface, scan_values, folme_speed);
        self.stop = false;

        match (result, finalized) {
            (_, Err(e)) => {
                error!("lithography finalization failed: {}", e.message());
                self.emit(LithoEvent::Error(format!(
                    "finalization failed ({}): check feedback, Z, bias and setpoint in Nanonis!",
                    e.message()
                )));
                if e.is_io() {
                    return Err(eyre::eyre!(e.message()));
                }
            }
            (Ok(()), Ok(())) => {
                info!("lithography completed");
                let scan = job.settings.scan_after.then(|| {
                    conn.scan_action(ActionType::Start, ScanDir::Up)
                        .map_err(|e| format!("could not start the scan: {e}"))
                });
                self.emit(LithoEvent::Finished { scan });
            }
            (Err(RunError::Stopped), Ok(())) => {
                info!("lithography stopped by the user");
                self.emit(LithoEvent::Stopped);
            }
            (Err(e), Ok(())) => {
                error!("lithography failed: {}", e.message());
                self.emit(LithoEvent::Error(format!("{} (feedback and scan values restored)", e.message())));
            }
        }
        Ok(())
    }

    fn move_to(&mut self, conn: &mut NanonisTcp, point: DVec2, speed: f64) -> eyre::Result<()> {
        self.stop = false;
        let result = (|| -> Result<(), RunError> {
            let saved = conn.fol_me_speed_get()?;
            conn.fol_me_speed_set(speed as f32, true)?;
            let moved = self.move_xy(conn, point, speed);
            if moved.is_err() {
                let _ = Self::halt_xy(conn);
            }
            conn.fol_me_speed_set(saved.speed, saved.custom_speed)?;
            moved
        })();
        self.stop = false;
        let io = result.as_ref().is_err_and(RunError::is_io);
        self.emit(LithoEvent::Moved(result.map_err(|e| e.message())));
        if io {
            return Err(eyre::eyre!("connection lost during move"));
        }
        Ok(())
    }
}

impl Worker for LithographyWorker {
    fn name(&self) -> String {
        "Lithography Worker".to_string()
    }
    fn init(&mut self, _conn: &mut NanonisTcp) -> eyre::Result<()> {
        Ok(())
    }
    fn work(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        match self.commands.recv_timeout(Duration::from_millis(200)) {
            Ok(LithoCommand::Run(job)) => self.run(conn, job),
            Ok(LithoCommand::MoveTo { point, speed }) => self.move_to(conn, point, speed),
            Ok(LithoCommand::Stop) | Err(_) => Ok(()),
        }
    }
}

fn z_tolerance(target: f64) -> f64 {
    Z_TOLERANCE_M.max(target.abs() * 1e-6)
}

/// The `steps` (bias, setpoint) values of a linear ramp, excluding `from`
/// and including `to`.
fn ramp_values(from: (f64, f64), to: (f64, f64), steps: u32) -> impl Iterator<Item = (f64, f64)> {
    (1..=steps).map(move |i| {
        let f = i as f64 / steps as f64;
        (from.0 + (to.0 - from.0) * f, from.1 + (to.1 - from.1) * f)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_matches_python() {
        // 20 steps 4 V / 2000 pA -> 2 V / 50 pA (from the GUI self-test).
        let v: Vec<_> = ramp_values((4.0, 2000e-12), (2.0, 50e-12), 20).collect();
        assert_eq!(v.len(), 20);
        assert!((v[0].0 - 3.9).abs() < 1e-12);
        assert!((v[0].1 - (2000e-12 - 1950e-12 / 20.)).abs() < 1e-24);
        assert!((v[19].0 - 2.0).abs() < 1e-12 && (v[19].1 - 50e-12).abs() < 1e-24);
    }

    #[test]
    fn z_lift_tolerance_and_polarity() {
        // +100 pm moves a negative Z towards zero.
        let z: f64 = -136.459e-9;
        assert!((z + 100e-12 - -136.359e-9).abs() < 1e-21);
        assert_eq!(z_tolerance(z), 2e-12);
        assert!((z_tolerance(1e-5) - 1e-11).abs() < 1e-24);
    }

    #[test]
    fn settings_validation() {
        assert!(LithoSettings::default().validate().is_ok());
        let comb_lift = LithoSettings {
            mode: PathMode::Comb,
            transition: Transition::ZLift,
            ..Default::default()
        };
        assert!(comb_lift.validate().unwrap_err().contains("Z-lift"));
        let ok_lift = LithoSettings {
            transition: Transition::ZLift,
            ..Default::default()
        };
        assert!(ok_lift.validate().is_ok());
        let bad = LithoSettings {
            litho_setpoint_pa: 0.,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }
}
