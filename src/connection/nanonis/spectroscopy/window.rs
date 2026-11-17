use crossbeam::channel::{Receiver, Sender};
use std::collections::HashSet;

use egui::{Color32, ComboBox, DragValue, RichText, Stroke, Ui};
use egui_plot::{AxisHints, GridInput, GridMark, Line, Plot, PlotPoints};
use crate::connection::nanonis::spectroscopy::{
    combined::{curve_color, save_batch, PlotChoice, SharedPlotChoice},
    model::{auto_didv_channel, didv_channel_names, IvCurve, IvSeriesSettings},
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
    /// Curves (by index) hidden from the plots by clicking the legend.
    hidden: HashSet<usize>,
    /// The plots shown here; saved figures (also a sequence's) use the same.
    plot_choice: SharedPlotChoice,
    /// The .dat files Nanonis saved for the current curves.
    nanonis_files: Vec<std::path::PathBuf>,
    save_tx: Sender<Result<String, String>>,
    save_rx: Receiver<Result<String, String>>,
}

impl SpectroscopyState {
    pub fn new(cmd_tx: Sender<SpecCommand>, events: Receiver<SpecEvent>, plot_choice: &SharedPlotChoice) -> Self {
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
            hidden: HashSet::new(),
            plot_choice: plot_choice.clone(),
            nanonis_files: Vec::new(),
            save_tx,
            save_rx,
        }
    }

    /// Whether the tool window is open (toggled from the Scan Control tools).
    pub fn open_mut(&mut self) -> &mut bool {
        &mut self.window_open
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
                SpecEvent::NanonisFiles(files) => self.nanonis_files = files,
                SpecEvent::Finished => {
                    self.running = false;
                    let n = self.curves.len();
                    let saved = self.nanonis_saved_note();
                    self.set_status(
                        StatusKind::Done,
                        format!("Complete: {n} curves. Z feedback restored ON. {saved}"),
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

    /// Whether Nanonis saved its own files for the current curves.
    fn nanonis_saved_note(&self) -> String {
        match self.nanonis_files.len() {
            0 if !self.curves.is_empty() => "⚠ Nanonis didn't save .dat files: turn on saving in its \
                Bias Spectroscopy module so the curves open in the Nanonis viewer."
                .into(),
            0 => String::new(),
            n => format!("Nanonis saved {n} .dat files."),
        }
    }

    fn start(&mut self) {
        if let Err(msg) = self.settings.validate() {
            self.set_status(StatusKind::Error, msg);
            return;
        }
        self.curves.clear();
        self.nanonis_files.clear();
        self.hidden.clear();
        self.curves_base_name = self.settings.base_name.trim().to_string();
        self.running = true;
        self.aborting = false;
        self.set_status(StatusKind::Busy, "Starting…");
        let _ = self.cmd_tx.send(SpecCommand::Run(self.settings.clone()));
    }

    fn save_dialog(&self) {
        let curves = self.curves.clone();
        let prefix = self.curves_base_name.clone();
        // The figure gets the plots shown here.
        let choice = self.choice();
        let files = self.nanonis_files.clone();
        let tx = self.save_tx.clone();
        std::thread::spawn(move || {
            let Some(parent) = crate::components::file_dialog_native::dialog()
                .set_title("Choose a folder for the I(V) curves")
                .pick_folder()
            else {
                return;
            };
            // A fresh subfolder per export so earlier batches are never overwritten.
            let result = save_batch(&parent, &prefix, &curves, &choice, &files).map(|(_, msg)| msg);
            let _ = tx.send(result);
        });
    }

    fn choice(&self) -> PlotChoice {
        PlotChoice {
            log_i: self.show_log,
            didv: self.show_didv,
            log_didv: self.show_didv_log,
            didv_channel: self.didv_channel.clone(),
        }
    }

    /// A sequence started an I(V) series: show its curves here.
    pub fn sequence_series_started(&mut self, base_name: &str) {
        if self.running {
            return;
        }
        self.curves.clear();
        self.hidden.clear();
        self.nanonis_files.clear();
        self.curves_base_name = base_name.to_string();
        self.set_status(StatusKind::Busy, format!("Sequence: I(V) series {base_name}…"));
    }
    /// A curve from a sequence's I(V) series.
    pub fn sequence_curve(&mut self, curve: IvCurve) {
        if !self.running {
            self.curves.push(curve);
        }
    }
    /// How a sequence's I(V) series ended (including where it was saved).
    pub fn sequence_message(&mut self, ok: bool, text: String, files: Vec<std::path::PathBuf>) {
        if !self.running {
            self.nanonis_files = files;
            self.set_status(if ok { StatusKind::Done } else { StatusKind::Error }, text);
        }
    }

    /// The "I(V) Spectroscopy" button and window. `blocked` gives the reason
    /// a new series can't start right now (scan or sequence running).
    pub fn show_menu(&mut self, ui: &mut Ui, blocked: Option<&str>) {
        self.poll_events();
        if self.running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(format!("I(V): {}", self.status)).small());
            });
        }
        let mut open = self.window_open;
        egui::Window::new("I(V) Spectroscopy")
            .vscroll(true)
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
            .on_hover_text(
                "Writes into a new folder: the .dat files Nanonis saved (they open in the \
                 Nanonis viewer), plus every curve together in one plot (..._all_N_curves.jpg \
                 and .svg) and one table (..._all_N_curves.dat)",
            )
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

    /// Legend above the plots (so it never covers a trace), in series order.
    /// Click an entry to hide or show that curve.
    fn show_legend(&mut self, ui: &mut Ui) {
        if self.curves.is_empty() {
            return;
        }
        let mut curves: Vec<(usize, String)> =
            self.curves.iter().map(|c| (c.index, c.label())).collect();
        curves.sort_by_key(|c| c.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 10.;
            for (index, label) in curves {
                let shown = !self.hidden.contains(&index);
                let color = curve_color(index);
                let text_color = if shown {
                    ui.visuals().text_color()
                } else {
                    ui.visuals().weak_text_color()
                };
                let resp = ui
                    .horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.;
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(18., 12.), egui::Sense::hover());
                        let stroke = if shown {
                            Stroke::new(LINE_WIDTH, color)
                        } else {
                            Stroke::new(1., color.gamma_multiply(0.4))
                        };
                        ui.painter()
                            .line_segment([rect.left_center(), rect.right_center()], stroke);
                        ui.add(
                            egui::Label::new(RichText::new(&label).color(text_color))
                                .sense(egui::Sense::click()),
                        )
                    })
                    .inner
                    .on_hover_text("Click to hide or show this curve");
                if resp.clicked() {
                    if shown {
                        self.hidden.insert(index);
                    } else {
                        self.hidden.remove(&index);
                    }
                }
            }
            if !self.hidden.is_empty() && ui.small_button("Show all").clicked() {
                self.hidden.clear();
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
            {
                ui.label("dI/dV channel:");
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
            }
        });
        let didv_channel = self
            .didv_channel
            .clone()
            .or_else(|| auto_didv_channel(&channels).cloned());
        // Shared so a sequence saves the same plots.
        let choice = self.choice();
        if *self.plot_choice.lock() != choice {
            *self.plot_choice.lock() = choice;
        }

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

        self.show_legend(ui);
        let start = self.curves.iter().min_by_key(|c| c.index).map(|c| c.start_conditions());

        let n = plots.len();
        let gap = ui.spacing().item_spacing.x;
        let width = ((ui.available_width() - gap * (n as f32 - 1.)) / n as f32).max(150.);
        // Width of the data area, less a typical y axis.
        let data_w = (width as f64 - 70.).max(60.);
        ui.horizontal_top(|ui| {
            for (i, (title, y_label, log, get)) in plots.into_iter().enumerate() {
                ui.vertical(|ui| {
                    ui.set_width(width);
                    ui.label(RichText::new(&title).strong());
                    let plot = Plot::new(("iv plot", i))
                        .width(width)
                        .height(PLOT_HEIGHT)
                        // Same bias numbers on every plot: the step comes from
                        // the shared range and the common width, not from
                        // each plot's own size.
                        .custom_x_axes(vec![AxisHints::new_x()
                            .label("Bias (V)")
                            .formatter(move |m, r| bias_label(m, r, data_w))
                            // Full strength: no fading, same colour everywhere.
                            .label_spacing(1.0..=2.0)])
                        .x_grid_spacer(move |input| bias_grid(input, data_w))
                        // Full-strength y numbers too: shown or hidden,
                        // never faded.
                        .custom_y_axes(vec![AxisHints::new_y()
                            .label(y_label)
                            .label_spacing(20.0..=21.0)])
                        .link_axis("iv bias axis", [true, false]);
                    let shown = plot.show(ui, |plot_ui| {
                        let mut curves: Vec<&IvCurve> = self.curves.iter().collect();
                        curves.sort_by_key(|c| c.index);
                        for curve in curves {
                            if self.hidden.contains(&curve.index) {
                                continue;
                            }
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
                                    .color(curve_color(curve.index))
                                    .width(LINE_WIDTH),
                            );
                        }
                    });
                    // Starting bias and setpoint, top-left inside the I(V) plot.
                    if let (0, Some(text)) = (i, &start) {
                        let rect = shown.response.rect;
                        let painter = ui.painter_at(rect);
                        let galley = painter.layout_no_wrap(
                            text.clone(),
                            egui::FontId::proportional(13.),
                            ui.visuals().strong_text_color(),
                        );
                        // Clear of the y axis numbers.
                        let pos = rect.left_top() + egui::vec2(62., 6.);
                        let bg = egui::Rect::from_min_size(pos, galley.size()).expand(3.);
                        painter.rect_filled(bg, 3., ui.visuals().extreme_bg_color.gamma_multiply(0.85));
                        painter.galley(pos, galley, Color32::PLACEHOLDER);
                    }
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

/// Plot line width in points.
const LINE_WIDTH: f32 = 3.5;

/// Smallest 1-2-5 step at least `min`.
fn nice_step(min: f64) -> f64 {
    let mag = 10f64.powf(min.log10().floor());
    [1., 2., 5., 10.].iter().map(|m| m * mag).find(|s| *s >= min * (1. - 1e-9)).unwrap_or(10. * mag)
}

/// Bias axis label step: the finest 1-2-5 step (…0.1, 0.2, 0.5, 1, 2…) whose
/// labels fit in `data_w` points across `lo..hi`. Every plot gets the same
/// range and width, so every plot gets the same numbers.
fn label_step(lo: f64, hi: f64, data_w: f64) -> f64 {
    let span = hi - lo;
    if !(span > 0.) || !(data_w > 0.) {
        return 1.;
    }
    let mut step = nice_step(span / 20.);
    for _ in 0..40 {
        let d = decimals(step);
        let chars = [lo, hi].iter().map(|v| format!("{:.*}", d, v).len()).max().unwrap_or(1);
        // ~7.5 pt per character plus a gap; never closer than 26 pt.
        let need = (chars as f64 * 7.5 + 12.).max(26.);
        if step * data_w / span >= need {
            return step;
        }
        step = nice_step(step * 1.001);
    }
    step
}

fn decimals(step: f64) -> usize {
    (-(step.log10() + 1e-9).floor()).max(0.) as usize
}

/// Grid for the bias axis: labelled lines every [`label_step`] and fainter
/// lines between.
fn bias_grid(input: GridInput, data_w: f64) -> Vec<GridMark> {
    let (lo, hi) = input.bounds;
    if !(hi > lo) {
        return Vec::new();
    }
    let major = label_step(lo, hi, data_w);
    let mantissa = major / 10f64.powf(major.log10().floor());
    let minor = major / if (mantissa - 2.).abs() < 1e-6 { 4. } else { 5. };
    let first = (lo / minor).ceil() as i64;
    let last = (hi / minor).floor() as i64;
    if last - first > 10_000 {
        return Vec::new();
    }
    (first..=last)
        .map(|k| {
            let value = k as f64 * minor;
            let on_major = ((value / major).round() * major - value).abs() < minor * 1e-3;
            GridMark { value, step_size: if on_major { major } else { minor } }
        })
        .collect()
}

/// Label for a labelled bias line (empty for the fainter lines between),
/// with just enough decimals for the step.
fn bias_label(mark: GridMark, range: &std::ops::RangeInclusive<f64>, data_w: f64) -> String {
    let major = label_step(*range.start(), *range.end(), data_w);
    if mark.step_size < major * 0.999 {
        return String::new();
    }
    let d = decimals(major);
    let s = format!("{:.*}", d, mark.value);
    if s.trim_start_matches('-').chars().all(|c| c == '0' || c == '.') {
        format!("{:.*}", d, 0.)
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(lo: f64, hi: f64, w: f64) -> Vec<String> {
        bias_grid(GridInput { bounds: (lo, hi), base_step_size: 0.01 }, w)
            .into_iter()
            .map(|m| bias_label(m, &(lo..=hi), w))
            .filter(|l| !l.is_empty())
            .collect()
    }

    #[test]
    fn bias_labels_suit_the_range() {
        // Wide sweep: whole volts.
        assert_eq!(labels(-6.8, 1.8, 270.), vec!["-6", "-5", "-4", "-3", "-2", "-1", "0", "1"]);
        // ±1 V: half volts.
        assert_eq!(labels(-1.05, 1.05, 270.), vec!["-1.0", "-0.5", "0.0", "0.5", "1.0"]);
        // ±0.5 V: 0.2 V.
        assert_eq!(labels(-0.52, 0.52, 270.), vec!["-0.4", "-0.2", "0.0", "0.2", "0.4"]);
        assert_eq!(nice_step(0.3), 0.5);
        assert_eq!(nice_step(0.11), 0.2);
    }
}
