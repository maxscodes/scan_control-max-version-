//! Scan Control panel: status, start/pause/resume/stop/save, bias and
//! setpoint, live current, and scan progress.

use std::time::{Duration, Instant};

use egui::{Button, DragValue, ProgressBar, RichText, Ui, Vec2};
use nanonis_tcp::ScanDir;
use tracing::info;

use crate::{
    connection::nanonis::{command_channel::CommandChannelSender, ScanStatus},
    theme,
};

#[derive(Debug, Clone, Copy)]
pub enum ScanCommand {
    Start(ScanDir),
    Pause,
    Resume,
    Stop,
    /// Have Nanonis save the current scan buffer (Scan.Save).
    Save,
    /// Volts.
    SetBias(f32),
    /// Amps.
    SetSetpoint(f32),
}

/// Values read back from Nanonis by the slow status worker.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Readback {
    /// Volts.
    pub bias: f32,
    /// Amps.
    pub setpoint: f32,
    /// Measured tunnelling current, amps.
    pub current: f32,
}

/// What the panel needs to know each frame.
pub struct PanelInput<'a> {
    pub status: &'a ScanStatus,
    pub sequence_active: bool,
    /// Why bias/setpoint can't be changed right now (another tool drives the tip).
    pub locked: Option<&'a str>,
    pub readback: Option<Readback>,
    /// (rows received, total rows) of the frame being scanned.
    pub rows: Option<(usize, usize)>,
}

pub struct ScanControlState {
    sender: CommandChannelSender<ScanCommand, ()>,
    pending: Option<ScanCommand>,
    paused: bool,
    bias_v: f64,
    setpoint_pa: f64,
    saved_at: Option<Instant>,
    /// Start of the current scan's progress (time, rows) for the time estimate.
    progress_start: Option<(Instant, usize)>,
}

impl ScanControlState {
    pub fn new(sender: &CommandChannelSender<ScanCommand, ()>) -> Self {
        Self {
            sender: sender.clone(),
            pending: None,
            paused: false,
            bias_v: 0.,
            setpoint_pa: 0.,
            saved_at: None,
            progress_start: None,
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Short status text, also used by the status bar.
    pub fn status_text(&self, status: &ScanStatus) -> String {
        let dir = match status.scan_dir {
            ScanDir::Up => "⬆",
            ScanDir::Down => "⬇",
        };
        if self.paused {
            format!("Paused {dir}")
        } else if status.scanning {
            format!("Scanning {dir}")
        } else {
            "Idle".to_string()
        }
    }

    pub fn show_menu(&mut self, ui: &mut Ui, input: PanelInput, compact: &mut bool) {
        if self.sender.poll_complete().is_some() {
            if matches!(self.pending, Some(ScanCommand::Save)) {
                self.saved_at = Some(Instant::now());
            }
            self.pending = None;
        }
        let status = input.status;
        let mut command = None;

        // 1. Status pill and compact toggle.
        ui.horizontal(|ui| {
            let (color, pulse) = if self.paused {
                (theme::WARN, false)
            } else if status.scanning {
                (theme::OK, true)
            } else {
                (theme::IDLE, false)
            };
            theme::status_pill(ui, color, &self.status_text(status), pulse);
            if input.sequence_active {
                ui.label(RichText::new("sequence").small().color(theme::WARN));
            }
            if self.pending.is_some() {
                ui.spinner();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let icon = if *compact { "⏷" } else { "⏶" };
                if ui
                    .small_button(icon)
                    .on_hover_text(if *compact { "Show everything" } else { "Compact view" })
                    .clicked()
                {
                    *compact = !*compact;
                }
            });
        });

        // 2. Scan buttons. Pause is tracked locally (Scan.StatusGet has no
        // paused flag), so all buttons stay enabled to recover from changes
        // made in Nanonis.
        let big = |text: &str| Button::new(RichText::new(text).size(14.)).min_size(Vec2::new(0., 28.));
        ui.add_enabled_ui(self.pending.is_none(), |ui| {
            ui.columns(2, |c| {
                if c[0].add_sized([c[0].available_width(), 28.], big("⬆ Scan up")).clicked() {
                    command = Some(ScanCommand::Start(ScanDir::Up));
                }
                if c[1].add_sized([c[1].available_width(), 28.], big("⬇ Scan down")).clicked() {
                    command = Some(ScanCommand::Start(ScanDir::Down));
                }
            });
            ui.columns(3, |c| {
                let (label, cmd) = if self.paused {
                    ("▶ Resume", ScanCommand::Resume)
                } else {
                    ("⏸ Pause", ScanCommand::Pause)
                };
                if c[0].add_sized([c[0].available_width(), 26.], big(label)).clicked() {
                    command = Some(cmd);
                }
                let stop = Button::new(RichText::new("⏹ Stop").size(14.).color(egui::Color32::WHITE))
                    .fill(theme::ERR);
                if c[1].add_sized([c[1].available_width(), 26.], stop).clicked() {
                    command = Some(ScanCommand::Stop);
                }
                let saved = self.saved_at.is_some_and(|t| t.elapsed() < Duration::from_secs(3));
                let save_label = if saved { "✔ Saved" } else { "💾 Save" };
                if c[2]
                    .add_sized([c[2].available_width(), 26.], big(save_label))
                    .on_hover_text("Save the current scan in Nanonis (.sxm in the session folder)")
                    .clicked()
                {
                    command = Some(ScanCommand::Save);
                }
                if saved {
                    c[2].ctx().request_repaint_after(Duration::from_millis(500));
                }
            });
        });

        // 3. Bias, setpoint and live current.
        ui.add_space(2.);
        egui::Grid::new("scan_control_values").num_columns(2).spacing([8., 4.]).show(ui, |ui| {
            let rb = input.readback;
            let editable = input.locked.is_none() && rb.is_some();
            ui.label("Bias");
            let resp = ui
                .add_enabled(
                    editable,
                    DragValue::new(&mut self.bias_v).speed(0.01).max_decimals(3).suffix(" V"),
                )
                .on_disabled_hover_text(input.locked.unwrap_or("Waiting for Nanonis"));
            if resp.lost_focus() || resp.drag_stopped() {
                if rb.is_some_and(|r| (r.bias as f64 - self.bias_v).abs() > 1e-6) {
                    command = Some(ScanCommand::SetBias(self.bias_v as f32));
                }
            } else if !resp.has_focus() && !resp.dragged() {
                if let Some(r) = rb {
                    self.bias_v = r.bias as f64;
                }
            }
            ui.end_row();

            ui.label("Setpoint");
            let resp = ui
                .add_enabled(
                    editable,
                    DragValue::new(&mut self.setpoint_pa)
                        .speed(0.5)
                        .range(0.0..=1e5)
                        .max_decimals(1)
                        .suffix(" pA"),
                )
                .on_disabled_hover_text(input.locked.unwrap_or("Waiting for Nanonis"));
            if resp.lost_focus() || resp.drag_stopped() {
                if rb.is_some_and(|r| (r.setpoint as f64 * 1e12 - self.setpoint_pa).abs() > 1e-3) {
                    command = Some(ScanCommand::SetSetpoint((self.setpoint_pa * 1e-12) as f32));
                }
            } else if !resp.has_focus() && !resp.dragged() {
                if let Some(r) = rb {
                    self.setpoint_pa = r.setpoint as f64 * 1e12;
                }
            }
            ui.end_row();

            ui.label("Current");
            match rb {
                Some(r) => ui.monospace(format!("{:.1} pA", r.current * 1e12)),
                None => ui.weak("–"),
            };
            ui.end_row();
        });

        // 4. Progress of the running scan.
        self.show_progress(ui, status, input.rows);

        if let Some(cmd) = command {
            info!("Sending scan command {cmd:?}");
            match cmd {
                ScanCommand::Pause => self.paused = true,
                ScanCommand::Resume | ScanCommand::Stop | ScanCommand::Start(_) => {
                    self.paused = false
                }
                _ => {}
            }
            if matches!(cmd, ScanCommand::Start(_)) {
                self.progress_start = None;
            }
            self.pending = Some(cmd);
            self.sender.send(cmd);
        }
    }

    fn show_progress(&mut self, ui: &mut Ui, status: &ScanStatus, rows: Option<(usize, usize)>) {
        let Some((done, total)) = rows.filter(|&(_, t)| t > 0) else {
            return;
        };
        if !status.scanning && !self.paused {
            self.progress_start = None;
            return;
        }
        // Restart the estimate when a new frame begins.
        match self.progress_start {
            Some((_, r0)) if done >= r0 => {}
            _ => self.progress_start = Some((Instant::now(), done)),
        }
        let frac = done as f32 / total as f32;
        let mut text = format!("line {done}/{total}");
        if let Some((t0, r0)) = self.progress_start {
            let dt = t0.elapsed().as_secs_f64();
            if done > r0 + 2 && dt > 2. && !self.paused {
                let left = (total - done) as f64 * dt / (done - r0) as f64;
                text += &format!(" · ~{}:{:02} left", left as u64 / 60, left as u64 % 60);
            }
        }
        ui.add(ProgressBar::new(frac).text(text).desired_height(16.));
    }
}
