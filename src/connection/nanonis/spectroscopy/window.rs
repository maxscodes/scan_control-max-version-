use crossbeam::channel::{Receiver, Sender};
use egui::{ecolor::Hsva, Color32, ComboBox, DragValue, RichText, Ui};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use tracing::info;

use crate::connection::nanonis::spectroscopy::{
    model::{auto_didv_channel, didv_channel_names, save_curves, utc_timestamp, IvCurve, IvSeriesSettings},
    worker::{SpecCommand, SpecEvent},
};

const PLOT_HEIGHT: f32 = 380.;

#[derive(Clone, Copy, PartialEq, Eq)]
enum StatusKind {
    Idle,
    Busy,
    Done,
    Error,
}

pub struct SpectroscopyState {
    window_open: bool,
    settings: IvSeriesSettings,
    cmd_tx: Sender<SpecCommand>,
    events: Receiver<SpecEvent>,
    running: bool,
    aborting: bool,
    status: String,
    status_kind: StatusKind,
    curves: Vec<IvCurve>,
    /// Base name of the series the curves came from (used as export prefix).
    curves_base_name: String,
    show_log: bool,
    show_didv: bool,
    show_didv_log: bool,
    /// `None` = automatic choice.
    didv_channel: Option<String>,
    save_tx: Sender<Result<String, String>>,
    save_rx: Receiver<Result<String, String>>,
}

impl SpectroscopyState {
    pub fn new(cmd_tx: Sender<SpecCommand>, events: Receiver<SpecEvent>) -> Self {
        let (save_tx, save_rx) = crossbeam::channel::unbounded();
        Self {
            window_open: false,
            settings: IvSeriesSettings::default(),
            cmd_tx,
            events,
            running: false,
            aborting: false,
            status: "Ready".into(),
            status_kind: StatusKind::Idle,
            curves: Vec::new(),
            curves_base_name: String::new(),
            show_log: false,
            show_didv: false,
            show_didv_log: false,
            didv_channel: None,
            save_tx,
            save_rx,
        }
    }

    /// A series is in progress. Other tip-moving controls should be locked.
    pub fn is_running(&self) -> bool {
        self.running
    }

    fn set_status(&mut self, kind: StatusKind, text: impl Into<String>) {
        self.status_kind = kind;
        self.status = text.into();
    }

    fn poll_events(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                SpecEvent::Status(text) => {
                    if !self.aborting {
                        self.set_status(StatusKind::Busy, text)
                    }
                }
                SpecEvent::Curve(curve) => self.curves.push(curve),
                SpecEvent::Finished => {
                    self.running = false;
                    let n = self.curves.len();
                    self.set_status(
                        StatusKind::Done,
                        format!("Complete: {n} curves. Z feedback restored ON"),
                    );
                }
                SpecEvent::Aborted => {
                    self.running = false;
                    self.set_status(StatusKind::Error, "Aborted. Z feedback left OFF");
                }
                SpecEvent::Error(msg) => {
                    self.running = false;
                    self.set_status(StatusKind::Error, format!("Error: {msg}. Z feedback left OFF"));
                }
            }
            if !self.running {
                self.aborting = false;
            }
        }
        while let Ok(saved) = self.save_rx.try_recv() {
            match saved {
                Ok(msg) => self.set_status(StatusKind::Done, msg),
                Err(msg) => self.set_status(StatusKind::Error, msg),
            }
        }
    }

    fn start(&mut self) {
        if let Err(msg) = self.settings.validate() {
            self.set_status(StatusKind::Error, msg);
            return;
        }
        self.curves.clear();
        self.curves_base_name = self.settings.base_name.trim().to_string();
        self.running = true;
        self.aborting = false;
        self.set_status(StatusKind::Busy, "Starting…");
        let _ = self.cmd_tx.send(SpecCommand::Run(self.settings.clone()));
    }

    fn save_dialog(&self) {
        let curves = self.curves.clone();
        let prefix = self.curves_base_name.clone();
        let tx = self.save_tx.clone();
        std::thread::spawn(move || {
            let Some(parent) = rfd::FileDialog::new()
                .set_title("Choose a folder for the I(V) curves")
                .pick_folder()
            else {
                return;
            };
            // A fresh subfolder per export so earlier batches are never overwritten.
            let stamp = utc_timestamp().replace(['-', ':', 'Z'], "").replace('T', "_");
            let dir = parent.join(format!("{prefix}_BATCH_{stamp}"));
            let result = std::fs::create_dir_all(&dir)
                .and_then(|_| save_curves(&dir, &prefix, &curves))
                .map(|n| {
                    info!("saved {n} I(V) curves to {}", dir.display());
                    format!("Saved {n} curves to {}", dir.display())
                })
                .map_err(|e| format!("Could not save curves to {}: {e}", dir.display()));
            let _ = tx.send(result);
        });
    }

    /// The "I(V) Spectroscopy" button and window. `blocked` gives the reason
    /// a new series can't start right now (scan or sequence running).
    pub fn show_menu(&mut self, ui: &mut Ui, blocked: Option<&str>) {
        self.poll_events();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.window_open, egui::Button::new("I(V) Spectroscopy"))
                .clicked()
            {
                self.window_open = true;
            }
            if self.running {
                ui.label(&self.status);
                ui.spinner();
            }
        });
        let mut open = self.window_open;
        egui::Window::new("I(V) Spectroscopy")
            .frame(crate::theme::window_frame(ui.ctx()))
            .default_width(1100.)
            .resizable(true)
            .collapsible(true)
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(260.);
                        self.show_controls(ui, blocked);
                    });
                    ui.separator();
                    ui.vertical(|ui| self.show_plots(ui));
                });
            });
        self.window_open = open;
    }

    fn show_controls(&mut self, ui: &mut Ui, blocked: Option<&str>) {
        ui.label(
            RichText::new(
                "Sweep range, points and channels are set in the Nanonis Bias \
                 Spectroscopy module.",
            )
            .weak(),
        );
        ui.add_space(6.);
        ui.add_enabled_ui(!self.running, |ui| {
            let s = &mut self.settings;
            egui::Grid::new("iv settings").num_columns(2).show(ui, |ui| {
                ui.label("Number of curves");
                ui.add(DragValue::new(&mut s.num_curves).range(1..=1000));
                ui.end_row();

                ui.label("File base name");
                ui.add(egui::TextEdit::singleline(&mut s.base_name).desired_width(120.));
                ui.end_row();

                ui.label("Z change per curve");
                ui.add(DragValue::new(&mut s.z_step_pm).speed(1.).suffix(" pm"))
                    .on_hover_text("Relative to the reference height; negative moves the tip closer");
                ui.end_row();

                ui.label("Between curves");
                ComboBox::from_id_salt("iv feedback")
                    .selected_text(if s.feedback_between { "Z ctrl ON" } else { "Z ctrl OFF" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut s.feedback_between, false, "Leave Z controller OFF");
                        ui.selectable_value(&mut s.feedback_between, true, "Turn Z controller ON");
                    })
                    .response
                    .on_hover_text(
                        "ON: re-settle on the setpoint before each Z step.\n\
                         OFF: the Z controller stays off for the whole series.",
                    );
                ui.end_row();

                ui.label("Settle time");
                ui.add_enabled(
                    s.feedback_between,
                    DragValue::new(&mut s.delay_s).speed(0.05).range(0.0..=3600.0).suffix(" s"),
                )
                .on_hover_text("How long the Z controller stays on between curves (min 0.1 s)");
                ui.end_row();
            });
        });
        ui.add_space(8.);
        ui.horizontal(|ui| {
            let can_start = !self.running && blocked.is_none();
            if ui.add_enabled(can_start, egui::Button::new("▶ Start series")).clicked() {
                self.start();
            }
            if ui
                .add_enabled(self.running && !self.aborting, egui::Button::new("■ Abort"))
                .on_hover_text("Stops after the curve in progress. Z feedback is left OFF.")
                .clicked()
            {
                self.aborting = true;
                self.set_status(StatusKind::Busy, "Aborting after this curve…");
                let _ = self.cmd_tx.send(SpecCommand::Abort);
            }
        });
        if let (Some(reason), false) = (blocked, self.running) {
            ui.colored_label(Color32::ORANGE, reason);
        }
        if ui
            .add_enabled(
                !self.running && !self.curves.is_empty(),
                egui::Button::new("Save curves…"),
            )
            .on_hover_text("Writes one .dat file per curve into a new folder")
            .clicked()
        {
            self.save_dialog();
        }
        ui.add_space(6.);
        let color = match self.status_kind {
            StatusKind::Idle => ui.visuals().text_color(),
            StatusKind::Busy => Color32::ORANGE,
            StatusKind::Done => Color32::LIGHT_GREEN,
            StatusKind::Error => Color32::from_rgb(0xff, 0x60, 0x60),
        };
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(color, &self.status);
            if self.running {
                ui.spinner();
            }
        });
    }

    fn show_plots(&mut self, ui: &mut Ui) {
        let channels = didv_channel_names(&self.curves);
        if self.didv_channel.as_ref().is_some_and(|c| !channels.contains(c)) {
            self.didv_channel = None;
        }
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_log, "log |I|");
            ui.checkbox(&mut self.show_didv, "dI/dV");
            ui.checkbox(&mut self.show_didv_log, "log |dI/dV|");
            ui.add_enabled_ui(self.show_didv || self.show_didv_log, |ui| {
                ui.label("Channel:");
                let auto = auto_didv_channel(&channels).cloned();
                let shown = match (&self.didv_channel, &auto) {
                    (Some(c), _) => c.clone(),
                    (None, Some(a)) => format!("Auto ({a})"),
                    (None, None) => "Auto".into(),
                };
                ComboBox::from_id_salt("iv didv channel")
                    .selected_text(shown)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.didv_channel, None, "Auto");
                        for c in &channels {
                            ui.selectable_value(&mut self.didv_channel, Some(c.clone()), c);
                        }
                    });
            });
        });
        let didv_channel = self
            .didv_channel
            .clone()
            .or_else(|| auto_didv_channel(&channels).cloned());

        // (title, y label, log scale, data getter)
        type Getter<'a> = Box<dyn Fn(&IvCurve) -> Option<&[f64]> + 'a>;
        let mut plots: Vec<(String, &str, bool, Getter)> = vec![(
            "I(V)".into(),
            "Current (pA)",
            false,
            Box::new(|c: &IvCurve| Some(c.current_pa.as_slice())),
        )];
        if self.show_log {
            plots.push((
                "log |I|".into(),
                "log₁₀ |I| (pA)",
                true,
                Box::new(|c: &IvCurve| Some(c.current_pa.as_slice())),
            ));
        }
        let ch = didv_channel.clone().unwrap_or_default();
        let didv_title = match &didv_channel {
            Some(c) => format!("dI/dV: {c}"),
            None => "dI/dV unavailable".into(),
        };
        if self.show_didv {
            let ch = ch.clone();
            plots.push((
                didv_title.clone(),
                "Lock-in (pA)",
                false,
                Box::new(move |c: &IvCurve| c.didv(&ch)),
            ));
        }
        if self.show_didv_log {
            let ch = ch.clone();
            plots.push((
                format!("log {didv_title}"),
                "log₁₀ |lock-in| (pA)",
                true,
                Box::new(move |c: &IvCurve| c.didv(&ch)),
            ));
        }

        let n = plots.len();
        let gap = ui.spacing().item_spacing.x;
        let width = ((ui.available_width() - gap * (n as f32 - 1.)) / n as f32).max(150.);
        ui.horizontal_top(|ui| {
            for (i, (title, y_label, log, get)) in plots.into_iter().enumerate() {
                ui.vertical(|ui| {
                    ui.set_width(width);
                    ui.label(RichText::new(&title).strong());
                    let mut plot = Plot::new(("iv plot", i))
                        .width(width)
                        .height(PLOT_HEIGHT)
                        .x_axis_label("Bias (V)")
                        .y_axis_label(y_label)
                        .link_axis("iv bias axis", [true, false]);
                    if i == 0 {
                        plot = plot.legend(Legend::default());
                    }
                    plot.show(ui, |plot_ui| {
                        for curve in &self.curves {
                            let Some(y) = get(curve) else { continue };
                            let points: Vec<[f64; 2]> = curve
                                .bias_v
                                .iter()
                                .zip(y)
                                .filter_map(|(&v, &y)| {
                                    if log {
                                        (y.abs() > 0.).then(|| [v, y.abs().log10()])
                                    } else {
                                        Some([v, y])
                                    }
                                })
                                .collect();
                            if points.is_empty() {
                                continue;
                            }
                            plot_ui.line(
                                Line::new(curve.label(), PlotPoints::from(points))
                                    .color(curve_color(curve.index)),
                            );
                        }
                    });
                    if (title.starts_with("dI/dV") || title.starts_with("log dI/dV"))
                        && channels.is_empty()
                        && !self.curves.is_empty()
                    {
                        ui.weak("Enable a lock-in / dI/dV channel in Nanonis Bias Spectroscopy");
                    }
                });
            }
        });
        if self.curves.is_empty() {
            ui.weak("No curves yet.");
        }
    }
}

/// Same colour for a curve in every plot, well separated between neighbours.
fn curve_color(index: usize) -> Color32 {
    let h = (index as f32 * 0.618_034).fract();
    Hsva::new(h, 0.85, 0.65, 1.0).into()
}
