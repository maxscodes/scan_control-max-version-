//! Runs an I(V) series on its own Nanonis connection.
//!
//! Ported from the "Multiple I(V) experiment" Python tool. Per series:
//! 1. Z controller off (and, if feedback-between is set, on → settle → off),
//!    then read the reference Z.
//! 2. For each curve: BiasSpectr.Start, then (if more curves follow) optionally
//!    re-settle on the setpoint, switch feedback off, wait until Nanonis
//!    *reports* it off, ZPosSet to the absolute target and verify the readback.
//! 3. Normal completion turns Z feedback back on; an error or abort leaves it off.

use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender};
use nanonis_tcp::{blocking::NanonisTcp, error::NanonisTcpError};
use tracing::{error, info, warn};

use crate::connection::nanonis::{
    spectroscopy::model::{extract_iv, split_channels, utc_timestamp, IvCurve, IvSeriesSettings},
    worker::Worker,
};

/// ZCtrl.StatusGet codes.
const Z_OFF: u16 = 1;
const Z_ON: u16 = 2;
const STATUS_TIMEOUT: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(50);
/// Minimum time feedback is left on to settle.
const MIN_SETTLE: Duration = Duration::from_millis(100);

#[derive(Debug)]
pub enum SpecCommand {
    Run(IvSeriesSettings),
    /// Stop after the curve in progress (a sweep can't be interrupted).
    Abort,
}

#[derive(Debug, Clone)]
pub enum SpecEvent {
    Status(String),
    Curve(IvCurve),
    /// All curves taken; Z feedback was switched back on.
    Finished,
    /// Stopped early; Z feedback was left off.
    Aborted,
    /// Failed; Z feedback was left off if the connection allowed it.
    Error(String),
}

enum RunError {
    Aborted,
    Nanonis(NanonisTcpError),
    Other(String),
}
impl From<NanonisTcpError> for RunError {
    fn from(e: NanonisTcpError) -> Self {
        Self::Nanonis(e)
    }
}

pub struct SpectroscopyWorker {
    ctx: egui::Context,
    commands: Receiver<SpecCommand>,
    events: Sender<SpecEvent>,
    abort: bool,
}

impl SpectroscopyWorker {
    pub fn new(
        ctx: &egui::Context,
        commands: Receiver<SpecCommand>,
        events: Sender<SpecEvent>,
    ) -> Self {
        Self {
            ctx: ctx.clone(),
            commands,
            events,
            abort: false,
        }
    }

    fn emit(&self, event: SpecEvent) {
        let _ = self.events.send(event);
        self.ctx.request_repaint();
    }
    fn status(&self, text: impl Into<String>) {
        self.emit(SpecEvent::Status(text.into()));
    }

    /// Note an Abort if one arrived; other commands are ignored mid-run.
    fn poll_abort(&mut self) {
        while let Ok(cmd) = self.commands.try_recv() {
            match cmd {
                SpecCommand::Abort => self.abort = true,
                SpecCommand::Run(_) => warn!("ignoring Run while a series is active"),
            }
        }
    }
    fn check_abort(&mut self) -> Result<(), RunError> {
        self.poll_abort();
        if self.abort {
            Err(RunError::Aborted)
        } else {
            Ok(())
        }
    }
    fn sleep(&mut self, dur: Duration) -> Result<(), RunError> {
        let end = Instant::now() + dur;
        loop {
            self.check_abort()?;
            let now = Instant::now();
            if now >= end {
                return Ok(());
            }
            std::thread::sleep(POLL.min(end - now));
        }
    }

    /// Wait until ZCtrl.StatusGet reports `wanted`. Switching off passes
    /// through "Switching Off"; Z moves must wait until it is really off.
    fn wait_z_status(&mut self, conn: &mut NanonisTcp, wanted: u16) -> Result<(), RunError> {
        let deadline = Instant::now() + STATUS_TIMEOUT;
        loop {
            let status = conn.zctrl_status_get()?;
            if status == wanted {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(RunError::Other(format!(
                    "timed out waiting for Z controller status {wanted}; last status was {status}"
                )));
            }
            std::thread::sleep(POLL);
        }
    }

    fn set_feedback(&mut self, conn: &mut NanonisTcp, on: bool) -> Result<(), RunError> {
        conn.zctrl_onoffset(on)?;
        self.wait_z_status(conn, if on { Z_ON } else { Z_OFF })
    }

    /// Feedback on, let it settle on the setpoint, feedback off.
    fn resettle(&mut self, conn: &mut NanonisTcp, s: &IvSeriesSettings) -> Result<(), RunError> {
        self.set_feedback(conn, true)?;
        self.sleep(Duration::from_secs_f64(s.delay_s).max(MIN_SETTLE))?;
        self.set_feedback(conn, false)
    }

    /// Wait until ZPosGet is within `tol` of `target`; returns the readback.
    fn wait_z_position(
        &mut self,
        conn: &mut NanonisTcp,
        target: f64,
        tol: f64,
    ) -> Result<f64, RunError> {
        let deadline = Instant::now() + STATUS_TIMEOUT;
        loop {
            let actual = conn.zctrl_zpos_get()? as f64;
            if (actual - target).abs() <= tol {
                return Ok(actual);
            }
            if Instant::now() >= deadline {
                return Err(RunError::Other(format!(
                    "Z did not reach the requested position: target {:.3} pm, actual {:.3} pm \
                     (piezo range limit?)",
                    target * 1e12,
                    actual * 1e12
                )));
            }
            std::thread::sleep(POLL);
        }
    }

    fn run_series(&mut self, conn: &mut NanonisTcp, s: &IvSeriesSettings) -> Result<(), RunError> {
        if conn.scan_status_get()?.running {
            return Err(RunError::Other(
                "a scan is running; stop it before starting spectroscopy".into(),
            ));
        }
        let n = s.num_curves as usize;

        // Reference height, taken after the optional settle so the
        // controller's return to its setpoint doesn't shift the zero.
        self.status("Establishing Z reference…");
        self.set_feedback(conn, false)?;
        if s.feedback_between {
            self.resettle(conn, s)?;
        }
        let start_z = conn.zctrl_zpos_get()? as f64;
        // Measured position immediately before each sweep (not after: that
        // would include drift during the sweep).
        let mut curve_z = start_z;

        for index in 0..n {
            self.check_abort()?;
            self.status(format!("Running curve {}/{n}…", index + 1));
            let resp = conn.bias_spectr_start(true, s.nanonis_name(index))?;
            let channels = split_channels(&resp.channel_names, resp.data.size, &resp.data.data)
                .map_err(RunError::Other)?;
            let (bias_v, current_pa, didv_pa) =
                extract_iv(&resp.channel_names, &channels).map_err(RunError::Other)?;
            self.emit(SpecEvent::Curve(IvCurve {
                index,
                z_offset_pm: (curve_z - start_z) * 1e12,
                bias_v,
                current_pa,
                didv_pa,
                timestamp: utc_timestamp(),
            }));

            if index + 1 < n {
                self.check_abort()?;
                self.status("Preparing next curve…");
                if s.feedback_between {
                    self.resettle(conn, s)?;
                } else {
                    self.set_feedback(conn, false)?;
                }
                let target = s.z_target(start_z, index + 1);
                conn.zctrl_zpos_set(target as f32)?;
                curve_z = self.wait_z_position(conn, target, s.z_tolerance())?;
            }
        }
        Ok(())
    }

    fn run(&mut self, conn: &mut NanonisTcp, s: IvSeriesSettings) -> eyre::Result<()> {
        info!("starting I(V) series: {s:?}");
        self.abort = false;
        let result = self.run_series(conn, &s);
        // Only a completed series restores feedback; otherwise leave it off.
        let restore = result.is_ok();
        let io_lost = matches!(result, Err(RunError::Nanonis(NanonisTcpError::Io(_))));
        let mut restore_err = None;
        if !io_lost {
            if let Err(e) = conn.zctrl_onoffset(restore) {
                error!("failed to set Z feedback {} after series: {e}", if restore { "on" } else { "off" });
                restore_err = Some(e);
            }
        }
        self.abort = false;
        match result {
            Ok(()) => match restore_err {
                None => self.emit(SpecEvent::Finished),
                Some(e) => self.emit(SpecEvent::Error(format!(
                    "all curves taken, but turning Z feedback back on failed: {e}"
                ))),
            },
            Err(RunError::Aborted) => {
                info!("I(V) series aborted");
                self.emit(SpecEvent::Aborted);
            }
            Err(RunError::Other(msg)) => {
                error!("I(V) series failed: {msg}");
                self.emit(SpecEvent::Error(msg));
            }
            Err(RunError::Nanonis(e)) => {
                error!("I(V) series failed: {e}");
                self.emit(SpecEvent::Error(format!("Nanonis error: {e}")));
                if io_lost {
                    // Let the worker loop reconnect.
                    return Err(e.into());
                }
            }
        }
        Ok(())
    }
}

impl Worker for SpectroscopyWorker {
    fn name(&self) -> String {
        "Spectroscopy Worker".to_string()
    }

    fn init(&mut self, _conn: &mut NanonisTcp) -> eyre::Result<()> {
        Ok(())
    }

    fn work(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        match self.commands.recv_timeout(Duration::from_millis(200)) {
            Ok(SpecCommand::Run(settings)) => self.run(conn, settings),
            // Nothing running: nothing to abort.
            Ok(SpecCommand::Abort) | Err(_) => Ok(()),
        }
    }
}
