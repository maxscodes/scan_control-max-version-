use core::f64;
use std::path::PathBuf;

use egui::{
    Button, CollapsingHeader, Color32, DragValue, Id, Stroke, Ui, Vec2,
};
use egui_plot::{Line, LineStyle, Plot, PlotPoint, PlotPoints, Points};
use glam::{DAffine2, DMat2, DVec2, IVec2};
use serde::{Deserialize, Serialize};
use itertools::Itertools as _;
use tracing::info;

use crate::{
    app::CONFIG_DIR,
    components::{selectable_list::SelectableList, si_drag::si_drag_value},
    connection::{
        nanonis::{
            coarse_cal::{CalPlan, CalResult},
            coarse_learn::{CoarseLearn, LearnAction},
            pair_view::PairView,
            sequence::ScanStart,
            command_channel::CommandChannelSender,
            sequence::Frame,
        },
        shared_state::SharedState,
        ScanArea,
    },
    scan_view::{
        world_delta_transform, BorderRectangle, ScanViewCtx,
    },
    utils::vec_interop::IntoEgui as _,
    view_object::Object,
};

/// Coarse motor calibration: how far one motor step moves the tip, per volt
/// of drive amplitude, in metres, for each motor direction. Each is a 2-D
/// vector in scan-area (piezo) axes, so cross-coupling between the axes is
/// kept. A move is `column(sign of steps) * steps * volts` summed over the
/// two axes, so stepping forward and back can differ.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoarseCalibration {
    pub x_pos: [f64; 2],
    pub x_neg: [f64; 2],
    pub y_pos: [f64; 2],
    pub y_neg: [f64; 2],
}
impl Default for CoarseCalibration {
    fn default() -> Self {
        Self::from_matrix(DMat2::IDENTITY * 1e-6)
    }
}
impl CoarseCalibration {
    /// The same calibration both ways (the old single-matrix format).
    pub fn from_matrix(m: DMat2) -> Self {
        Self {
            x_pos: m.x_axis.to_array(),
            x_neg: m.x_axis.to_array(),
            y_pos: m.y_axis.to_array(),
            y_neg: m.y_axis.to_array(),
        }
    }
    /// Column for motor axis `axis` (0 = X, 1 = Y) moving in `positive` direction.
    pub fn column(&self, axis: usize, positive: bool) -> DVec2 {
        DVec2::from_array(match (axis, positive) {
            (0, true) => self.x_pos,
            (0, false) => self.x_neg,
            (_, true) => self.y_pos,
            (_, false) => self.y_neg,
        })
    }
    pub fn set_column(&mut self, axis: usize, positive: bool, v: DVec2) {
        let slot = match (axis, positive) {
            (0, true) => &mut self.x_pos,
            (0, false) => &mut self.x_neg,
            (_, true) => &mut self.y_pos,
            (_, false) => &mut self.y_neg,
        };
        *slot = v.to_array();
    }
    /// Real move (nm) for `steps` at drive amplitudes `volts`.
    pub fn steps_to_real(&self, steps: IVec2, volts: DVec2) -> DVec2 {
        let cx = self.column(0, steps.x >= 0);
        let cy = self.column(1, steps.y >= 0);
        (cx * steps.x as f64 * volts.x + cy * steps.y as f64 * volts.y) * 1e9
    }
    /// Nearest whole steps for a real move (nm): solve with each pair of
    /// directions and keep the solution whose signs fit the directions used.
    pub fn real_to_steps(&self, real: DVec2, volts: DVec2) -> IVec2 {
        let mut best: Option<(bool, f64, IVec2)> = None;
        for px in [true, false] {
            for py in [true, false] {
                let m = DMat2::from_cols(
                    self.column(0, px) * volts.x * 1e9,
                    self.column(1, py) * volts.y * 1e9,
                );
                if m.determinant().abs() < 1e-18 {
                    continue;
                }
                let s = m.inverse() * real;
                if !s.is_finite() {
                    continue;
                }
                let fits = (s.x >= -0.5) == px && (s.y >= -0.5) == py
                    || (s.x.abs() < 0.5 && (s.y >= -0.5) == py)
                    || (s.y.abs() < 0.5 && (s.x >= -0.5) == px);
                let steps = s.round().as_ivec2();
                let err = (self.steps_to_real(steps, volts) - real).length();
                let better = match best {
                    None => true,
                    Some((bf, be, _)) => (fits && !bf) || (fits == bf && err < be),
                };
                if better {
                    best = Some((fits, err, steps));
                }
            }
        }
        best.map_or(IVec2::ZERO, |b| b.2)
    }
    /// Every direction has a usable calibration.
    pub fn is_usable(&self) -> bool {
        [(0, true), (0, false), (1, true), (1, false)]
            .iter()
            .all(|&(a, p)| self.column(a, p).length() > 0.)
            && DMat2::from_cols(self.column(0, true), self.column(1, true)).determinant().abs() > 0.
    }
}

fn calibration_path() -> PathBuf {
    CONFIG_DIR.config_local_dir().join("course_calibration.json")
}
/// The single-matrix file older versions wrote.
fn legacy_matrix_path() -> PathBuf {
    CONFIG_DIR.config_local_dir().join("course_matrix.bin")
}
/// When the calibration was last changed, and how.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationStamp {
    /// Unix time, s.
    pub unix: Option<u64>,
    /// e.g. "auto calibration (+X, -X)".
    pub how: String,
}
impl CalibrationStamp {
    pub fn now(how: impl Into<String>) -> Self {
        let unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());
        Self { unix, how: how.into() }
    }
    /// "3 days ago (2026-11-11 15:32 UTC), auto calibration".
    pub fn describe(&self) -> String {
        let Some(t) = self.unix else {
            return "unknown".into();
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(t, |d| d.as_secs());
        let ago = now.saturating_sub(t);
        let rel = match ago {
            0..60 => "just now".to_string(),
            60..3600 => format!("{} min ago", ago / 60),
            3600..172800 => format!("{} h ago", ago / 3600),
            _ => format!("{} days ago", ago / 86400),
        };
        let date = time::OffsetDateTime::from_unix_timestamp(t as i64)
            .map(|d| {
                format!(
                    "{:04}-{:02}-{:02} {:02}:{:02} UTC",
                    d.year(),
                    u8::from(d.month()),
                    d.day(),
                    d.hour(),
                    d.minute()
                )
            })
            .unwrap_or_default();
        if self.how.is_empty() {
            format!("{rel} ({date})")
        } else {
            format!("{rel} ({date}) \u{b7} {}", self.how)
        }
    }
}

/// The calibration file: the calibration plus when it was last changed.
#[derive(Serialize, Deserialize)]
struct SavedCalibration {
    #[serde(flatten)]
    calib: CoarseCalibration,
    #[serde(default)]
    updated: CalibrationStamp,
}

/// Modification time of a file, as a stamp (for files without one).
fn file_stamp(path: &std::path::Path, how: &str) -> CalibrationStamp {
    let unix = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    CalibrationStamp { unix, how: how.into() }
}

fn load_calibration() -> (CoarseCalibration, CalibrationStamp) {
    if let Some(s) = std::fs::read_to_string(calibration_path())
        .ok()
        .and_then(|s| serde_json::from_str::<SavedCalibration>(&s).ok())
    {
        let updated = if s.updated.unix.is_some() {
            s.updated
        } else {
            file_stamp(&calibration_path(), "")
        };
        return (s.calib, updated);
    }
    match std::fs::read(legacy_matrix_path()) {
        Ok(buf) if buf.len() == std::mem::size_of::<DMat2>() => {
            info!("using the old coarse calibration matrix for both directions");
            (
                CoarseCalibration::from_matrix(*bytemuck::from_bytes::<DMat2>(&buf)),
                file_stamp(&legacy_matrix_path(), "older single-matrix calibration"),
            )
        }
        _ => (CoarseCalibration::default(), CalibrationStamp::default()),
    }
}
fn save_calibration(c: &CoarseCalibration, updated: &CalibrationStamp) {
    let saved = SavedCalibration { calib: *c, updated: updated.clone() };
    match serde_json::to_string_pretty(&saved) {
        Ok(json) => {
            if let Err(e) = std::fs::write(calibration_path(), json) {
                tracing::error!("could not save the coarse calibration: {e}");
            }
        }
        Err(e) => tracing::error!("could not encode the coarse calibration: {e}"),
    }
}

// #[derive(Clone)]
pub struct CourseMoveCommand {
    pub steps: IVec2,
    pub group: u32,
    pub auto_withdraw: bool,
    pub auto_approach: bool
}

pub struct CourseMotionState {
    menu_active: bool,
    move_target: DVec2,
    /// The baseline: the result of the last calibration test (or edit).
    calib: CoarseCalibration,
    /// When `calib` was last changed.
    calib_updated: CalibrationStamp,
    /// The calibration learned from corrections, and which one moves use.
    learn: CoarseLearn,
    voltages: SharedState<DVec2>,
    freqs: SharedState<DVec2>,
    move_sender: CommandChannelSender<CourseMoveCommand, ()>,
    currently_moving: bool,
    group: u32,
    last_x_move: Option<(DVec2, IVec2)>,
    last_y_move: Option<(DVec2, IVec2)>,
    auto_withdraw: bool,
    auto_approach: bool,
    /// Auto calibration: steps per move (0 = suggest), settle time, a run
    /// to start, and the last measured result.
    auto_steps: [i32; 2],
    /// Frame size (nm) the step suggestion was made for; a new frame size
    /// gets a new suggestion.
    auto_steps_frame: Option<DVec2>,
    auto_settle_s: f64,
    cal_request: Option<CalPlan>,
    cal_result: Option<CalResult>,
    /// Progress of a running calibration (fraction, text), and whether its
    /// images are being compared. Set by the connection each frame.
    pub cal_progress: Option<(f32, String)>,
    pub cal_measuring: bool,
    /// An aborted run left the calibration frame in Nanonis (set by the
    /// connection), and the user asked to put it back.
    pub cal_can_restore: bool,
    restore_request: bool,
    /// Calibration frame size (µm, 0 = suggest) and lines.
    auto_frame_um: f64,
    auto_lines: u32,
    /// Moves per direction, and how each scan is started.
    auto_repeats: u32,
    /// Number of step counts (small to large), and the smallest.
    auto_sizes: u32,
    auto_min_steps: i32,
    /// Motor axes to calibrate (X, Y).
    auto_axes: [bool; 2],
    auto_scan_start: ScanStart,
    auto_preset: CalPreset,
    /// Nanonis' time per line, forward and backward (s), for the estimate.
    pub line_times: Option<(f64, f64)>,
    /// Forward time per line for the calibration scans; `None` keeps Nanonis'.
    auto_line_time: Option<f64>,
    /// A calibration move being checked by eye.
    cal_align: Option<AlignView>,
    /// The last applied calibration's run, loaded to look at again, with
    /// when it was applied; `align_saved`: the by-eye window shows it.
    saved_cal: Option<(CalResult, String)>,
    align_saved: bool,
    /// Outcome of saving or loading the last calibration's scans.
    saved_note: Option<String>,
}

/// Checking calibration move `k` by eye.
struct AlignView {
    k: usize,
    view: PairView,
}
impl AlignView {
    /// Move `k`'s scans; zoom and processing as in `like`.
    fn new(ctx: &egui::Context, result: &CalResult, k: usize, like: Option<&PairView>) -> Option<Self> {
        let (a, b) = result.pair(k)?;
        let m = result.moves.get(k)?;
        Some(Self { k, view: PairView::new(ctx, a, b, m.shift_px.unwrap_or(DVec2::ZERO), like) })
    }
}

/// Ready-made calibration settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CalPreset {
    /// Small frame, low resolution, one move each way, short waits.
    Fast,
    Standard,
    /// Several moves each way, averaged; higher resolution, longer settle.
    Accurate,
    /// Step counts from small to large, each there and back twice: fits
    /// the step (and cross-coupling) apart from the per-move shift.
    Overnight,
}
impl CalPreset {
    fn label(self) -> &'static str {
        match self {
            CalPreset::Fast => "Fast",
            CalPreset::Standard => "Standard",
            CalPreset::Accurate => "Accurate",
            CalPreset::Overnight => "Overnight",
        }
    }
    fn hint(self) -> &'static str {
        match self {
            CalPreset::Fast => "2 \u{b5}m frame, 64 lines, one move each way, 10 s settle, start once",
            CalPreset::Standard => "Frame for ~10 steps per move, 128 lines, one move each way, 30 s settle",
            CalPreset::Accurate => {
                "Frame for ~10 steps per move, 256 lines, three moves each way (averaged), 60 s settle"
            }
            CalPreset::Overnight => {
                "Step counts from 2 up to ~35 % of the frame in 5 sizes, each +X, \u{2212}X, +Y, \
                 \u{2212}Y twice; 128 lines, 60 s settle. Fits each direction's step (with its \
                 cross-coupling) apart from what every move adds. Takes many hours: see the estimate."
            }
        }
    }
}
impl CourseMotionState {
    pub fn new(
        voltages: &SharedState<DVec2>,
        freqs: &SharedState<DVec2>,
        move_sender: &CommandChannelSender<CourseMoveCommand, ()>,
    ) -> Self {
        let (calib, calib_updated) = load_calibration();

        Self {
            menu_active: false,
            move_target: DVec2::ZERO,
            calib,
            calib_updated,
            learn: CoarseLearn::load(),
            voltages: voltages.clone(),
            freqs: freqs.clone(),
            move_sender: move_sender.clone(),
            currently_moving: false,
            group: 1,
            last_x_move: None,
            last_y_move: None,
            auto_withdraw: true,
            auto_approach: true,
            auto_steps: [0, 0],
            auto_steps_frame: None,
            auto_settle_s: 30.,
            cal_request: None,
            cal_result: None,
            cal_progress: None,
            cal_measuring: false,
            cal_can_restore: false,
            restore_request: false,
            auto_frame_um: 0.,
            auto_lines: 128,
            auto_repeats: 1,
            auto_sizes: 1,
            auto_min_steps: 2,
            auto_axes: [true, true],
            auto_scan_start: ScanStart::default(),
            cal_align: None,
            saved_cal: None,
            align_saved: false,
            saved_note: None,
            auto_preset: CalPreset::Standard,
            line_times: None,
            auto_line_time: None,
        }
    }
    /// Record a finished coarse move: update the calibration bookkeeping, the
    /// move history, and shift the scan area on the map. Used both by this
    /// menu and by the sequence runner.
    pub fn apply_completed_move(&mut self, scan_area: &mut ScanArea, steps: IVec2) {
        if steps.y == 0 {
            let last_pos = scan_area.world_transform.translation;
            self.last_x_move = Some((last_pos, steps));
        } else {
            self.last_x_move = None;
        }
        if steps.x == 0 {
            let last_pos = scan_area.world_transform.translation;
            self.last_y_move = Some((last_pos, steps));
        } else {
            self.last_y_move = None;
        }
        let real_world_move = self.steps_to_real(steps);
        let id = scan_area.coarse_log.next_id();
        scan_area.coarse_log.moves.push(crate::analysis::coarse_fit::MoveRecord {
            id,
            unix: crate::analysis::coarse_fit::unix_now(),
            steps: steps.to_array(),
            volts: self.voltages.peek().to_array(),
            freq: self.freqs.peek().to_array(),
            predicted_nm: real_world_move.to_array(),
            learned: self.learn.active().is_some(),
        });
        scan_area.course_move_history.push(real_world_move);
        scan_area.world_transform =
            scan_area.world_transform * DAffine2::from_translation(real_world_move);
    }
    /// Motor group index (0-based) as sent to Motor.StartMove.
    pub fn group_index(&self) -> u32 {
        self.group - 1
    }
    /// Open or close the Coarse Motion window (closing saves the calibration).
    pub fn set_open(&mut self, open: bool) {
        if open && !self.menu_active && !self.currently_moving {
            // A running move (e.g. from Ring Assist) still needs its target.
            self.move_target = DVec2::ZERO;
        }
        if !open && self.menu_active {
            save_calibration(&self.calib, &self.calib_updated);
        }
        self.menu_active = open;
    }

    /// Calibrate one axis from the last single-axis move: how far the scan
    /// area was dragged on the map since that move.
    fn calibrate_axis(&mut self, axis: usize, object_list: &mut SelectableList<Object>, volts: f64) {
        let last = if axis == 0 { self.last_x_move.take() } else { self.last_y_move.take() };
        let (Some((last_pos, steps)), Some(scan_area)) =
            (last, object_list.iter_mut().find_map(|entry| entry.as_scan_area_mut()))
        else {
            return;
        };
        let n = if axis == 0 { steps.x } else { steps.y };
        let world_motion = scan_area.world_transform.translation - last_pos;
        let real_world_motion = scan_area.world_transform.inverse().transform_vector2(world_motion);
        let old = self.calib;
        self.calib.set_column(axis, n >= 0, real_world_motion / (n as f64 * volts) * 1e-9);
        self.learn.on_new_baseline(old, &self.calib_updated, self.calib);
        let name = match (axis, n >= 0) {
            (0, true) => "X (+X)",
            (0, false) => "X (\u{2212}X)",
            (_, true) => "Y (+Y)",
            (_, false) => "Y (\u{2212}Y)",
        };
        self.calib_updated = CalibrationStamp::now(format!("Calibrate {name}"));
        save_calibration(&self.calib, &self.calib_updated);
    }

    /// The Coarse Motion window (opened from the Scan Control tools).
    /// `locked` disables starting a move (e.g. while a sequence is running).
    pub fn show_menu(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>, locked: bool) {
        if self.menu_active {
            object_list.clear_selected();
        }
        if self.move_sender.poll_complete().is_some() {
            self.currently_moving = false;
            if let Some(scan_area) = object_list
                .iter_mut()
                .find_map(|entry| entry.as_scan_area_mut())
            {
                let steps = self.get_steps();
                self.apply_completed_move(scan_area, steps);
            };
            self.move_target = DVec2::ZERO;
        }
        if self.currently_moving {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Coarse move running\u{2026}");
            });
        }
        if !self.menu_active {
            return;
        }
        // Scan frame size (nm) and scan state, for Auto calibrate.
        let (frame_size, scanning, range) = object_list
            .iter()
            .find_map(|o| o.as_scan_area())
            .map(|a| {
                let size = a.image_transform.map(|t| {
                    let f = Frame::from_transform(t);
                    DVec2::new(f.width, f.height)
                });
                (size, a.scan_status.scanning, a.area_size)
            })
            .unwrap_or((None, false, DVec2::ZERO));
        let mut open = true;
        egui::Window::new("Coarse Motion")
            // Keeps the position saved under the old title.
            .id(Id::new("Course Motion"))
            .vscroll(true)
            .frame(crate::theme::window_frame(ui.ctx()))
            .default_size([300., 360.])
            .resizable(true)
            .collapsible(true)
            .enabled(!self.currently_moving)
            .open(&mut open)
            .show(&ui.ctx(), |ui| {
                ui.set_min_width(260.);
                let mut voltages = *self.voltages.peek();
                let mut steps = self.get_steps();
                egui::Grid::new("coarse_steps").num_columns(4).spacing([8., 4.]).show(ui, |ui| {
                    ui.label("X steps");
                    ui.add_enabled(voltages.x > 0., DragValue::new(&mut steps.x));
                    ui.label("Y steps");
                    ui.add_enabled(voltages.y > 0., DragValue::new(&mut steps.y));
                    ui.end_row();
                });
                self.write_steps(steps);
                ui.weak("Or ctrl+drag on the map to set the move.");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.auto_withdraw, "Auto withdraw");
                    ui.checkbox(&mut self.auto_approach, "Auto approach");
                });
                ui.add_space(4.);
                let steps = self.get_steps();
                let can_move = !locked && (steps.x != 0 || steps.y != 0);
                let execute = ui.add_enabled(
                    can_move,
                    Button::new("Execute Move").min_size(Vec2::new(ui.available_width(), 28.)),
                );
                if execute.clicked() {
                    info!("Executing course move {steps:?} on group {}", self.group);
                    self.currently_moving = true;
                    self.move_sender.send(CourseMoveCommand {
                        steps,
                        group: self.group - 1,
                        auto_withdraw: self.auto_withdraw,
                        auto_approach: self.auto_approach,
                    });
                }
                if locked {
                    ui.colored_label(crate::theme::WARN, "A sequence is running");
                }
                ui.separator();
                CollapsingHeader::new("Config")
                    .default_open(false)
                    .show_unindented(ui, |ui| {
                        egui::Grid::new("coarse_config").num_columns(2).spacing([8., 4.]).show(ui, |ui| {
                            ui.label("Group");
                            ui.add(DragValue::new(&mut self.group).speed(0).range(1..=4));
                            ui.end_row();
                            ui.label("Driver amplitudes");
                            ui.horizontal(|ui| {
                                ui.add_enabled(false, DragValue::new(&mut voltages.x));
                                ui.add_enabled(false, DragValue::new(&mut voltages.y));
                            });
                            ui.end_row();
                        });
                        ui.add_space(4.);
                        ui.label("Calibration (m per step per V, as x, y)")
                            .on_hover_text(
                                "How far one step moves the tip in each motor direction, \
                                 in scan-frame axes. Forward and back are calibrated separately.",
                            );
                        egui::Grid::new("course cal grid").show(ui, |ui| {
                            for (axis, pos, name) in [(0, true, "+X"), (0, false, "\u{2212}X"), (1, true, "+Y"), (1, false, "\u{2212}Y")] {
                                let before = self.calib.column(axis, pos);
                                let mut c = before;
                                ui.label(name);
                                ui.add(si_drag_value(&mut c.x));
                                ui.add(si_drag_value(&mut c.y));
                                ui.end_row();
                                if c != before {
                                    let old = self.calib;
                                    self.calib.set_column(axis, pos, c);
                                    self.learn.on_new_baseline(old, &self.calib_updated, self.calib);
                                    self.calib_updated = CalibrationStamp::now("edited by hand");
                                }
                            }
                        });
                        ui.weak(format!("Last updated: {}", self.calib_updated.describe()));
                        ui.horizontal(|ui| {
                            let hint = "After a single-axis move, drag the scan area on the map to \
                                        where the scan shows it really went, then click.";
                            if ui
                                .add_enabled(self.last_x_move.is_some(), Button::new("Calibrate X"))
                                .on_hover_text(hint)
                                .clicked()
                            {
                                self.calibrate_axis(0, object_list, voltages.x);
                            }
                            if ui
                                .add_enabled(self.last_y_move.is_some(), Button::new("Calibrate Y"))
                                .on_hover_text(hint)
                                .clicked()
                            {
                                self.calibrate_axis(1, object_list, voltages.y);
                            }
                        });
                    });
                CollapsingHeader::new("Learned from corrections")
                    .default_open(false)
                    .show_unindented(ui, |ui| {
                        let volts = *self.voltages.peek();
                        match self.learn.ui(ui, &self.calib, &self.calib_updated, volts) {
                            Some(LearnAction::Promote(c, how)) => self.set_calibration(c, how),
                            Some(LearnAction::Restore(c, st)) => {
                                self.set_calibration(c, format!("restored: {}", st.how));
                            }
                            None => {}
                        }
                    });
                CollapsingHeader::new("Auto calibrate")
                    .default_open(self.cal_result.is_some())
                    .show_unindented(ui, |ui| {
                        self.show_auto_calibration(ui, frame_size, range, scanning, locked);
                    });
            });
        if !open {
            self.set_open(false);
        }
        if self.align_saved {
            self.show_align_window(ui.ctx());
        }
    }

    pub fn show_overlay(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>) {
        let Some(scan_area) = object_list
            .iter_mut()
            .find_map(|entry| entry.as_scan_area_mut())
        else {
            return;
        };
        if self.menu_active {
            if ui.input(|i| i.modifiers.ctrl) && !self.currently_moving {
                let [_, _, translate] = world_delta_transform(ui);
                let world_translate = translate.translation;
                let scan_world_translate = scan_area
                    .world_transform
                    .inverse()
                    .transform_vector2(world_translate);
                self.move_target += scan_world_translate;
            }
            self.show_current_move(ui, scan_area);
        }
    }
    fn show_current_move(&self, ui: &mut Ui, scan_area: &ScanArea) {
        let Some(ctx) = ui.data(|map| map.get_temp::<ScanViewCtx>(Id::new(()))) else {
            return;
        };

        let world2screen = ctx.world2egui();
        let course_world2screen = |p: IVec2| {
            (world2screen * scan_area.world_transform)
                .transform_point2(self.steps_to_real(p))
                .to_egui_pos2()
        };
        let steps = self.get_steps();

        std::iter::once(IVec2::ZERO)
            .chain(course_path_iter(steps))
            .map(course_world2screen)
            .tuple_windows()
            .for_each(|(a, b)| {
                ui.painter()
                    .line_segment([a, b], Stroke::new(1., Color32::ORANGE));
            });
        course_path_iter(steps)
            .map(course_world2screen)
            .for_each(|point| {
                ui.painter().circle_filled(point, 4., Color32::ORANGE);
            });

        let real_course_move = self.steps_to_real(steps);
        let move_transform =
            DAffine2::from_scale_angle_translation(scan_area.area_size, 0., real_course_move);
        BorderRectangle {
            transform: scan_area.world_transform * move_transform,
            color: Color32::from_rgb(0xe3, 0xb6, 0x2d),
            dashed: false,
        }
        .show(ui);
    }
    /// A calibration run the user asked for (taken by the connection, which
    /// runs it as a sequence).
    pub fn take_calibration_request(&mut self) -> Option<CalPlan> {
        self.cal_request.take()
    }
    /// The user asked to put the frame back after an aborted run.
    pub fn take_restore_request(&mut self) -> bool {
        std::mem::take(&mut self.restore_request)
    }
    /// The measured result of a calibration run, shown for Apply.
    pub fn set_calibration_result(&mut self, r: CalResult) {
        self.menu_active = true;
        self.cal_result = Some(r);
    }

    fn apply_preset(&mut self, p: CalPreset, range: DVec2) {
        self.auto_preset = p;
        // New frame size → new step suggestion.
        self.auto_steps = [0, 0];
        self.auto_steps_frame = None;
        let max_um = if range.min_element() > 0. { 0.9 * range.min_element() * 1e-3 } else { f64::INFINITY };
        let (frame_um, lines, repeats, settle_s, starts, sizes) = match p {
            CalPreset::Fast => (2f64.min(max_um), 64, 1, 10., 1, 1),
            // 0 = suggest from the step size.
            CalPreset::Standard => (0., 128, 1, 30., 3, 1),
            CalPreset::Accurate => (0., 256, 3, 60., 3, 1),
            CalPreset::Overnight => (0., 128, 2, 60., 3, 5),
        };
        self.auto_sizes = sizes;
        self.auto_min_steps = 2;
        self.auto_frame_um = frame_um;
        self.auto_lines = lines;
        self.auto_repeats = repeats;
        self.auto_settle_s = settle_s;
        self.auto_scan_start = ScanStart { starts, gap_s: 3., ..Default::default() };
    }

    /// Steps the motor needs to move `nm` along `axis`, sized by the
    /// direction with the larger step: if the calibration is off, too few
    /// steps still overlap, too many don't.
    fn steps_for(&self, axis: usize, nm: f64) -> Option<i32> {
        let mut unit = IVec2::ZERO;
        unit[axis] = 1;
        let per = self.steps_to_real(unit).length().max(self.steps_to_real(-unit).length());
        (per > 0. && per.is_finite()).then(|| (nm / per).round().clamp(1., 5000.) as i32)
    }

    fn show_auto_calibration(
        &mut self,
        ui: &mut Ui,
        frame: Option<DVec2>,
        range: DVec2,
        scanning: bool,
        locked: bool,
    ) {
        ui.set_max_width(440.);
        ui.weak(
            "Sets the frame size and resolution below, scans, steps +X, scans, steps \
             \u{2212}X, scans, then the same for Y, and measures each move from how far the \
             image moved. Your frame size and resolution are put back at the end. Pick an \
             area with clear features.",
        );
        let Some(current) = frame.filter(|s| s.x > 0. && s.y > 0.) else {
            ui.colored_label(Color32::ORANGE, "No scan frame yet: connect to Nanonis first.");
            return;
        };
        if self.cal_can_restore {
            ui.horizontal(|ui| {
                ui.colored_label(Color32::ORANGE, "The last run was aborted with the calibration frame set.");
                if ui.button("Restore frame & resolution").clicked() {
                    self.restore_request = true;
                }
            });
        }
        ui.horizontal(|ui| {
            ui.label("Preset");
            for p in [CalPreset::Fast, CalPreset::Standard, CalPreset::Accurate, CalPreset::Overnight] {
                if ui.selectable_label(self.auto_preset == p, p.label()).on_hover_text(p.hint()).clicked() {
                    self.apply_preset(p, range);
                }
            }
        });
        // Frame: big enough that a third of it is ~10 steps (fewer steps
        // measure the step size less reliably), within the piezo range.
        if self.auto_frame_um <= 0. {
            let ten = (0..2)
                .filter_map(|axis| {
                    let mut unit = IVec2::ZERO;
                    unit[axis] = 1;
                    let per = self.steps_to_real(unit).length();
                    (per > 0. && per.is_finite()).then_some(per * 10. / 0.35)
                })
                .fold(0., f64::max);
            let max_nm = if range.min_element() > 0. { 0.9 * range.min_element() } else { f64::INFINITY };
            let nm = if ten > 0. { ten.min(max_nm) } else { current.max_element() };
            self.auto_frame_um = (nm * 1e-3 * 100.).round() / 100.;
        }
        let size = DVec2::splat(self.auto_frame_um * 1e3);
        ui.weak(format!("Calibration last updated: {}", self.calib_updated.describe()));
        self.show_saved_calibration(ui);
        if let Some((frac, text)) = &self.cal_progress {
            ui.add(egui::ProgressBar::new(*frac).text(text.as_str()).desired_height(16.));
            ui.weak("Pause or abort it in the Sequence window.");
        } else if self.cal_measuring {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Comparing the scans\u{2026}");
            });
        }
        let volts = *self.voltages.peek();
        // Suggest moves of about a third of the frame, again whenever the
        // frame size changes.
        if self
            .auto_steps_frame
            .is_none_or(|f| (f - size).abs().max_element() > 0.01 * size.max_element())
        {
            self.auto_steps = [0, 0];
            self.auto_steps_frame = Some(size);
        }
        for axis in 0..2 {
            if self.auto_steps[axis] <= 0 {
                let along = if axis == 0 { size.x } else { size.y };
                self.auto_steps[axis] = self.steps_for(axis, 0.35 * along).unwrap_or(10);
            }
        }
        let mut too_far = false;
        egui::Grid::new("auto_cal_grid").num_columns(3).show(ui, |ui| {
            ui.label("Frame");
            ui.add(DragValue::new(&mut self.auto_frame_um).range(0.01..=100.).speed(0.01).suffix(" \u{b5}m"));
            ui.weak(format!("now {:.2} \u{b5}m", current.max_element() * 1e-3));
            ui.end_row();
            ui.label("Resolution");
            ui.add(DragValue::new(&mut self.auto_lines).range(32..=1024).suffix(" lines"));
            ui.weak("square pixels");
            ui.end_row();
            ui.label("Moves each way");
            ui.add(DragValue::new(&mut self.auto_repeats).range(1..=5));
            ui.weak(if self.auto_sizes > 1 { "per step size" } else { "averaged; more = more accurate, slower" });
            ui.end_row();
            ui.label("Step sizes");
            ui.horizontal(|ui| {
                ui.add(DragValue::new(&mut self.auto_sizes).range(1..=8));
                if self.auto_sizes > 1 {
                    ui.add(DragValue::new(&mut self.auto_min_steps).range(1..=100).prefix("from "));
                }
            });
            ui.weak(if self.auto_sizes > 1 {
                "small \u{2192} large; fits step and per-move shift"
            } else {
                "one step count"
            })
            .on_hover_text(
                "Several step counts, from small to large: each direction is fitted with a \
                 line, so the step size (and its cross-coupling) is measured apart from what \
                 every move adds whatever its length (approach, drift, short first steps). \
                 Each larger move is searched for near where the smaller ones predict.",
            );
            ui.end_row();
            ui.label("Start each scan");
            ui.add(DragValue::new(&mut self.auto_scan_start.starts).range(1..=5).suffix("\u{d7}"));
            ui.add(
                DragValue::new(&mut self.auto_scan_start.gap_s)
                    .range(0. ..=30.)
                    .speed(0.1)
                    .prefix("every ")
                    .suffix(" s"),
            );
            ui.end_row();
            ui.label("Calibrate");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.auto_axes[0], "X");
                ui.checkbox(&mut self.auto_axes[1], "Y");
            });
            ui.weak("one axis alone redoes just that one");
            ui.end_row();
            let sizes = self.auto_sizes;
            let series = |axis: usize, largest: i32, min: i32| {
                let plan = CalPlan {
                    steps: [largest; 2],
                    sizes,
                    min_steps: [min; 2],
                    repeats: 1,
                    axes: [true; 2],
                    settle_s: 0.,
                    withdraw: false,
                    approach: false,
                    volts: DVec2::ONE,
                    frame_nm: 0.,
                    lines: 0,
                    scan_start: ScanStart::default(),
                    line_time_s: None,
                    restore: None,
                    restore_line_time_s: None,
                };
                plan.counts(axis).iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ")
            };
            for (axis, name) in [(0, "X steps"), (1, "Y steps")] {
                if !self.auto_axes[axis] {
                    continue;
                }
                if sizes > 1 {
                    ui.label(format!("{name} (largest)"));
                } else {
                    ui.label(name);
                }
                ui.add(DragValue::new(&mut self.auto_steps[axis]).range(1..=5000));
                let along = if axis == 0 { size.x } else { size.y };
                let pct = |sign: i32| {
                    let mut s = IVec2::ZERO;
                    s[axis] = sign * self.auto_steps[axis];
                    self.steps_to_real(s).length() / along * 100.
                };
                let (p, n) = (pct(1), pct(-1));
                too_far |= p > 60. || n > 60.;
                let text = if sizes > 1 {
                    format!(
                        "{} \u{b7} largest \u{2248} {p:.0} % / {n:.0} % of the frame",
                        series(axis, self.auto_steps[axis], self.auto_min_steps)
                    )
                } else {
                    format!("\u{2248} {p:.0} % / {n:.0} % of the frame")
                };
                ui.weak(text).on_hover_text("Expected move forward / back with the current calibration");
                ui.end_row();
            }
            ui.label("Settle after approach");
            ui.add(DragValue::new(&mut self.auto_settle_s).range(0. ..=600.).speed(1.).suffix(" s"));
            ui.end_row();
            ui.label("Scan speed");
            ui.horizontal(|ui| {
                let mut set = self.auto_line_time.is_some();
                if ui.checkbox(&mut set, "set").on_hover_text("Off: scan at Nanonis' current speed").changed() {
                    self.auto_line_time = set.then(|| self.line_times.map_or(1., |t| t.0));
                }
                if let Some(t) = &mut self.auto_line_time {
                    ui.add(DragValue::new(t).range(0.02..=60.).speed(0.01).suffix(" s/line"));
                }
            });
            match (self.auto_line_time, self.line_times) {
                (Some(t), _) => ui.weak(format!("forward; = {:.0} nm/s on this frame", size.x / t)),
                (None, Some((f, _))) => ui.weak(format!("Nanonis' now: {f:.2} s/line forward")),
                (None, None) => ui.weak("Nanonis' current speed"),
            };
            ui.end_row();
        });
        if too_far {
            ui.colored_label(
                Color32::ORANGE,
                "Moves over ~60 % of the frame may leave too little overlap to measure.",
            );
        }
        let plan = CalPlan {
            steps: self.auto_steps,
            repeats: self.auto_repeats,
            sizes: self.auto_sizes,
            min_steps: [self.auto_min_steps; 2],
            axes: self.auto_axes,
            settle_s: self.auto_settle_s,
            withdraw: self.auto_withdraw,
            approach: self.auto_approach,
            volts,
            frame_nm: size.x,
            lines: self.auto_lines,
            scan_start: self.auto_scan_start,
            line_time_s: self.auto_line_time,
            // Filled in by the connection from the current Nanonis settings.
            restore: None,
            restore_line_time_s: None,
        };
        // Forward + backward time per line: the speed set here keeps
        // Nanonis' backward/forward ratio.
        let line_s = match (self.auto_line_time, self.line_times) {
            (Some(t), Some((f, b))) => Some(t * (1. + b / f)),
            (Some(t), None) => Some(2. * t),
            (None, Some((f, b))) => Some(f + b),
            (None, None) => None,
        };
        match line_s {
            Some(line_s) => ui.label(format!(
                "\u{2248} {} \u{b7} {} scans, {} coarse moves",
                crate::connection::nanonis::sequence::fmt_duration(plan.estimate_s(line_s, 35.)),
                plan.scans(),
                plan.scans() - 1,
            ))
            .on_hover_text(
                "Scans at Nanonis' current time per line, about 35 s per withdraw, move and \
                 approach, plus the settle time. If Nanonis keeps the tip speed constant, \
                 a smaller frame also scans faster than this.",
            ),
            None => ui.weak(format!("{} scans, {} coarse moves", plan.scans(), plan.scans() - 1)),
        };
        if (0..2).any(|a| self.auto_axes[a] && self.auto_steps[a] < 5) {
            ui.colored_label(
                Color32::ORANGE,
                "Fewer than ~5 steps per move measure the step size poorly; use a bigger frame.",
            );
        }
        if range.min_element() > 0. && size.max_element() > range.min_element() {
            ui.colored_label(Color32::ORANGE, "The frame is larger than the scan range.");
        }
        let why_not = if !self.auto_axes.contains(&true) {
            Some("Tick X or Y")
        } else if locked {
            Some("A sequence is running")
        } else if self.currently_moving {
            Some("A coarse move is running")
        } else if scanning {
            Some("A scan is running; stop it first")
        } else if volts.x <= 0. || volts.y <= 0. {
            Some("Driver amplitudes unknown")
        } else {
            None
        };
        if ui
            .add_enabled(why_not.is_none(), Button::new("\u{25b6} Run auto calibration"))
            .on_disabled_hover_text(why_not.unwrap_or(""))
            .clicked()
        {
            self.cal_result = None;
            self.cal_request = Some(plan);
        }

        let Some(result) = &mut self.cal_result else { return };
        ui.separator();
        ui.strong("Measured");
        if let Some(p) = &result.problem {
            ui.colored_label(Color32::ORANGE, p);
        }
        let v = result.volts;
        let nm_per_step = |col: DVec2, axis: usize| col.length() * if axis == 0 { v.x } else { v.y } * 1e9;
        egui::Grid::new("auto_cal_result").num_columns(9).striped(true).show(ui, |ui| {
            for (h, tip) in [
                ("Move", ""),
                ("Steps", "Step counts used"),
                ("nm/step", "Distance per step (the fitted slope with several step counts)"),
                ("x, y", "Tip move per step along the scan's x and y, nm: both non-zero = cross-coupled"),
                ("Per move", "What every move adds whatever its step count (fit intercept), nm"),
                ("Before", "nm/step of the calibration in use, and the change"),
                ("Match", ""),
                ("Spread", ""),
                ("Use", ""),
            ] {
                let r = ui.strong(h);
                if !tip.is_empty() {
                    r.on_hover_text(tip);
                }
            }
            ui.end_row();
            for row in result.rows.iter_mut() {
                let good = row.score >= crate::analysis::register::GOOD_SCORE;
                ui.label(row.name());
                if row.steps_min != row.steps {
                    ui.label(format!("{}\u{2026}{}", row.steps_min.abs(), row.steps.abs()));
                } else {
                    ui.label(format!("{:+}", row.steps));
                }
                ui.label(format!("{:.2}", row.nm_per_step()));
                let v = row.step_vector_nm();
                ui.label(format!("{:+.2}, {:+.2}", v.x, v.y));
                match row.offset_nm {
                    Some(o) => ui.label(format!("{:+.0}, {:+.0}", o.x, o.y)),
                    None => ui.weak("\u{2013}"),
                };
                let before = nm_per_step(row.old_column, row.axis);
                let change = if before > 0. { (row.nm_per_step() / before - 1.) * 100. } else { f64::NAN };
                ui.label(format!("{before:.2} ({change:+.0} %)"));
                let score = if row.total > 1 {
                    format!("{:.2} ({}/{})", row.score, row.good, row.total)
                } else {
                    format!("{:.2}", row.score)
                };
                if row.manual > 0 && row.manual == row.good {
                    ui.label(if row.total > 1 { format!("by eye ({}/{})", row.good, row.total) } else { "by eye".into() })
                        .on_hover_text("Aligned by eye");
                } else if good {
                    let mut text = score;
                    if row.manual > 0 {
                        text += &format!(", {} by eye", row.manual);
                    }
                    ui.label(text).on_hover_text("Worst match among the moves used (moves that matched / made)");
                } else {
                    ui.colored_label(Color32::ORANGE, score)
                        .on_hover_text("Weak match: the images may not overlap or show few features");
                }
                if row.good > 1 {
                    let tip = if row.offset_nm.is_some() {
                        "Scatter of the moves about the fitted line, % of the average move"
                    } else {
                        "How much nm/step varied between the moves"
                    };
                    ui.label(format!("\u{b1}{:.0} %", row.spread_pct)).on_hover_text(tip);
                } else {
                    ui.weak("\u{2013}");
                }
                ui.checkbox(&mut row.use_it, "");
                ui.end_row();
            }
        });
        for (axis, name) in [(0, "X"), (1, "Y")] {
            let f = result.rows.iter().find(|r| r.axis == axis && r.positive);
            let b = result.rows.iter().find(|r| r.axis == axis && !r.positive);
            if let (Some(f), Some(b)) = (f, b) {
                ui.weak(format!(
                    "{name}: back / forward = {:.2}",
                    b.nm_per_step() / f.nm_per_step().max(1e-12)
                ));
            }
        }
        // Cross-coupling: which way each motor moves the tip, and the angle
        // between them (90° with the motors along the scan axes = none).
        let fwd = |axis: usize| result.rows.iter().find(|r| r.axis == axis && r.positive).map(|r| r.step_vector_nm());
        let angle = |v: DVec2| v.y.atan2(v.x).to_degrees();
        // Sideways share: how far off the nearest scan axis, as tan(angle).
        let sideways = |v: DVec2| {
            let a = angle(v).rem_euclid(90.);
            let off = if a > 45. { a - 90. } else { a };
            off.to_radians().tan() * 100.
        };
        let mut parts = Vec::new();
        for (axis, name) in [(0, "X"), (1, "Y")] {
            if let Some(v) = fwd(axis) {
                parts.push(format!("+{name} along {:.0}\u{b0} ({:+.0} % sideways)", angle(v), sideways(v)));
            }
        }
        if let (Some(x), Some(y)) = (fwd(0), fwd(1)) {
            let between = (angle(y) - angle(x)).rem_euclid(360.);
            let between = if between > 180. { 360. - between } else { between };
            parts.push(format!("{between:.0}\u{b0} apart"));
        }
        if !parts.is_empty() {
            ui.weak(format!("Tip moves: {} (scan x = 0\u{b0}, y = 90\u{b0})", parts.join(" \u{b7} ")))
                .on_hover_text(
                    "Motors exactly along the scan axes would read 0 % sideways and 90\u{b0} apart. \
                     Any coupling is kept in the calibration either way.",
                );
        }
        if result.rows.iter().any(|r| r.offset_nm.is_some()) {
            ui.weak(
                "Per move: what every move adds whatever its step count (approach, drift, short \
                 first steps). It isn't part of the calibration, which uses the slope; grid \
                 tracking corrects for it.",
            );
        }
        // The fits, to check by eye: each move's distance against its step
        // count, with the fitted line. Clicking a point opens that move.
        let mut open_from_plot = None;
        if result.moves.len() > 1 {
            CollapsingHeader::new("Fits")
                .id_salt("cal_fit_plots")
                .default_open(result.rows.iter().any(|r| r.offset_nm.is_some()))
                .show(ui, |ui| {
                    ui.weak(
                        "Blue: distance moved along the step direction; orange: sideways \
                         (both nm), against the step count. Lines: the fit. Filled points are \
                         used, hollow ones (weak match) are not. Click a point to check that \
                         move by eye.",
                    );
                    egui::Grid::new("cal_fit_grid").num_columns(2).show(ui, |ui| {
                        for (i, row) in result.rows.iter().enumerate() {
                            let d = row.step_vector_nm().normalize_or_zero();
                            let perp = d.perp();
                            let pts: Vec<(usize, f64, DVec2, bool)> = result
                                .moves
                                .iter()
                                .enumerate()
                                .filter(|(_, m)| (m.axis, m.positive) == (row.axis, row.positive))
                                .filter_map(|(k, m)| {
                                    let tip = result.tip_for(k, m.shift_px?)?;
                                    let used = m.manual || m.score >= crate::analysis::register::GOOD_SCORE;
                                    Some((k, m.steps.unsigned_abs() as f64, tip, used))
                                })
                                .collect();
                            let off = row.offset_nm.unwrap_or(DVec2::ZERO);
                            let per = row.nm_per_step();
                            let max_n = pts.iter().map(|p| p.1).fold(row.steps.unsigned_abs() as f64, f64::max);
                            ui.vertical(|ui| {
                                ui.strong(format!("{}: {:.2} nm/step", row.name(), per));
                                let blue = Color32::from_rgb(80, 160, 255);
                                let orange = Color32::from_rgb(255, 170, 70);
                                let shown = Plot::new(("cal fit", i))
                                    .width(240.)
                                    .height(150.)
                                    .x_axis_label("steps")
                                    .allow_drag(false)
                                    .allow_zoom(false)
                                    .allow_scroll(false)
                                    .allow_boxed_zoom(false)
                                    .show(ui, |plot_ui| {
                                        let line = |a: f64, b: f64| PlotPoints::from(vec![[0., a], [max_n, a + b * max_n]]);
                                        plot_ui.line(Line::new("fit", line(off.dot(d), per)).color(blue));
                                        plot_ui.line(
                                            Line::new("fit sideways", line(off.dot(perp), 0.))
                                                .color(orange)
                                                .style(LineStyle::dashed_dense()),
                                        );
                                        for &(_, n, tip, used) in &pts {
                                            plot_ui.points(
                                                Points::new("along", vec![[n, tip.dot(d)]]).radius(4.).filled(used).color(blue),
                                            );
                                            plot_ui.points(
                                                Points::new("sideways", vec![[n, tip.dot(perp)]])
                                                    .radius(3.)
                                                    .filled(used)
                                                    .color(orange),
                                            );
                                        }
                                    });
                                if shown.response.clicked() {
                                    if let Some(p) = shown.response.interact_pointer_pos() {
                                        let t = &shown.transform;
                                        let near = pts
                                            .iter()
                                            .flat_map(|&(k, n, tip, _)| {
                                                [tip.dot(d), tip.dot(perp)].map(|y| {
                                                    (k, t.position_from_point(&PlotPoint::new(n, y)).distance(p))
                                                })
                                            })
                                            .min_by(|a, b| a.1.total_cmp(&b.1));
                                        if let Some((k, dist)) = near {
                                            if dist < 14. {
                                                open_from_plot = Some(k);
                                            }
                                        }
                                    }
                                }
                            });
                            if i % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
                });
        }
        // An axis where no move matched: almost always the moves went further
        // than the old calibration predicted, so the scans don't overlap.
        let mut retry = None;
        for (axis, name) in [(0, "X"), (1, "Y")] {
            let rows: Vec<&crate::connection::nanonis::coarse_cal::DirResult> = result.rows.iter().filter(|r| r.axis == axis).collect();
            if rows.is_empty() || rows.iter().any(|r| r.good > 0) {
                continue;
            }
            let steps = rows[0].steps.unsigned_abs() as i32;
            let fewer = (steps / 4).max(1);
            ui.colored_label(
                Color32::ORANGE,
                format!(
                    "{name}: no move matched. The scans before and after each {name} move                      probably don't overlap: the motor stepped further than the old                      calibration says. The {name} numbers above are not usable."
                ),
            );
            if ui
                .button(format!("Set up {name} only with {fewer} steps"))
                .on_hover_text(format!(
                    "A quarter of the {steps} steps used, so a move several times larger than                      expected still overlaps. Then press Run. If the moves are still too                      small to measure, use more steps or a bigger frame."
                ))
                .clicked()
            {
                retry = Some((axis, fewer));
            }
        }
        // Check each move by eye.
        let mut open_k = open_from_plot;
        ui.horizontal_wrapped(|ui| {
            ui.label("Check by eye:").on_hover_text(
                "Lay the scan after a move over the one before it and line them up yourself. \
                 Your alignment replaces the automatic match for that move.",
            );
            for (k, m) in result.moves.iter().enumerate() {
                let same = |o: &&crate::connection::nanonis::coarse_cal::MoveMeasure| {
                    (o.axis, o.positive) == (m.axis, m.positive)
                };
                let n = result.moves.iter().filter(same).count();
                let i = result.moves[..k].iter().filter(same).count() + 1;
                let varied = result.moves.iter().filter(same).any(|o| o.steps != m.steps);
                let mut text = if varied {
                    // Step counts differ: name the move by its count.
                    let same_n = |o: &&crate::connection::nanonis::coarse_cal::MoveMeasure| same(o) && o.steps == m.steps;
                    let n2 = result.moves.iter().filter(same_n).count();
                    let i2 = result.moves[..k].iter().filter(same_n).count() + 1;
                    if n2 > 1 { format!("{} {}\u{b7}{i2}", m.name(), m.steps.abs()) } else { format!("{} {}", m.name(), m.steps.abs()) }
                } else if n > 1 {
                    format!("{} {i}", m.name())
                } else {
                    m.name().to_string()
                };
                if m.manual {
                    text += " \u{2714}";
                }
                let open = self.cal_align.as_ref().is_some_and(|v| v.k == k);
                if ui
                    .add_enabled(result.pair(k).is_some(), Button::new(text).selected(open))
                    .on_disabled_hover_text("Scans missing")
                    .clicked()
                {
                    open_k = Some(k);
                }
            }
        });
        if let Some((axis, fewer)) = retry {
            self.auto_axes = [axis == 0, axis == 1];
            self.auto_steps[axis] = fewer;
        }
        let Some(result) = &mut self.cal_result else { return };
        let any = result.rows.iter().any(|r| r.use_it);
        let mut apply = false;
        let mut discard = false;
        ui.horizontal(|ui| {
            apply = ui.add_enabled(any, Button::new("Apply")).clicked();
            discard = ui.button("Discard").clicked();
        });
        if apply {
            let mut c = self.calib;
            for row in result.rows.iter().filter(|r| r.use_it) {
                c.set_column(row.axis, row.positive, row.column);
            }
            let dirs: Vec<&str> = result.rows.iter().filter(|r| r.use_it).map(|r| r.name()).collect();
            let how = format!("auto calibration ({})", dirs.join(", "));
            info!("applied auto calibration: {c:?}");
            // Keep the scans, to check the moves by eye later.
            self.saved_note = Some(match result.save(&c, &how) {
                Ok(dir) => format!("Scans saved to {}", dir.display()),
                Err(e) => format!("Could not save the scans: {e}"),
            });
            self.saved_cal = None;
            self.set_calibration(c, how);
            self.cal_result = None;
        } else if discard {
            self.cal_result = None;
        }
        if let (Some(k), Some(r)) = (open_k, &self.cal_result) {
            let like = self.cal_align.take().map(|v| v.view);
            self.cal_align = AlignView::new(ui.ctx(), r, k, like.as_ref());
            self.align_saved = false;
        }
        if !self.align_saved {
            self.show_align_window(ui.ctx());
        }
    }

    /// Look again at the moves of the last applied calibration.
    fn show_saved_calibration(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            let exists = super::coarse_cal::saved_dir().join("calibration.json").exists();
            if ui
                .add_enabled(exists, Button::new("Review last applied calibration"))
                .on_hover_text("Check its moves by eye again (the scans saved when it was applied)")
                .clicked()
            {
                match CalResult::load_saved() {
                    Ok((r, when)) => {
                        let like = self.cal_align.take().map(|v| v.view);
                        self.cal_align = AlignView::new(ui.ctx(), &r, 0, like.as_ref());
                        self.saved_note = Some(format!("Applied {when}"));
                        self.saved_cal = Some((r, when));
                        self.align_saved = true;
                    }
                    Err(e) => self.saved_note = Some(format!("Could not load the saved scans: {e}")),
                }
            }
            if ui
                .add_enabled(exists, Button::new("Open folder"))
                .on_hover_text("The saved scans and a before | after PNG for each move")
                .clicked()
            {
                let _ = std::process::Command::new("explorer").arg(super::coarse_cal::saved_dir()).spawn();
            }
        });
        if let Some(n) = &self.saved_note {
            ui.weak(n.as_str());
        }
    }

    /// The scan after a calibration move over the scan before it, to line
    /// up by eye.
    fn show_align_window(&mut self, ctx: &egui::Context) {
        let saved = self.align_saved;
        let source = if saved { self.saved_cal.as_mut().map(|(r, _)| r) } else { self.cal_result.as_mut() };
        let (Some(view), Some(result)) = (&mut self.cal_align, source) else {
            self.cal_align = None;
            self.align_saved = false;
            return;
        };
        let k = view.k;
        let (Some(m), Some((a, b))) = (result.moves.get(k).copied(), result.pair(k)) else {
            self.cal_align = None;
            return;
        };
        let (a, b) = (a.clone(), b.clone());
        let mut open = true;
        let mut close = false;
        let mut goto = None;
        let n_moves = result.moves.len();
        let title = if saved {
            format!("Last applied calibration: move {} ({}/{})", m.name(), k + 1, result.moves.len())
        } else {
            format!("Check move {} by eye", m.name())
        };
        egui::Window::new(title)
            .id(Id::new("cal_align_window"))
            .open(&mut open)
            .resizable(true)
            .default_size([820., 900.])
            .min_size([360., 400.])
            .collapsible(true)
            .show(ctx, |ui| {
                view.view.ui(ui, &a, &b, m.auto_px, 70.);
                let shift = view.view.shift;
                let steps = m.steps;
                if let Some(tip) = result.tip_for(k, shift) {
                    ui.label(format!(
                        "Shift {:+.1}, {:+.1} px \u{2192} tip moved {:.0} nm ({:+.0}, {:+.0}) \u{2192} {:.2} nm/step",
                        shift.x,
                        shift.y,
                        tip.length(),
                        tip.x,
                        tip.y,
                        tip.length() / steps.unsigned_abs().max(1) as f64,
                    ));
                }
                match m.auto_px {
                    Some(auto) => ui.weak(format!(
                        "Blue outline: automatic match ({:+.1}, {:+.1} px, score {:.2})",
                        auto.x, auto.y, m.score
                    )),
                    None => ui.weak("The automatic match found nothing"),
                };
                if saved {
                    ui.weak(format!(
                        "{} move, {} steps{}. Looking only: changes here aren't applied.",
                        m.name(),
                        m.steps,
                        if m.manual { ", aligned by eye" } else { "" }
                    ));
                }
                ui.add_enabled_ui(!saved, |ui| ui.horizontal(|ui| {
                    if ui.button("\u{2714} Use this alignment").clicked() {
                        result.set_manual(k, shift);
                        close = true;
                    }
                    if ui
                        .add_enabled(m.auto_px.is_some(), Button::new("Back to automatic match"))
                        .clicked()
                    {
                        view.view.shift = m.auto_px.unwrap_or(DVec2::ZERO);
                        if m.manual {
                            result.set_automatic(k);
                        }
                    }
                    if ui.button("No shift").on_hover_text("Lay the scans exactly on top of each other").clicked() {
                        view.view.shift = DVec2::ZERO;
                    }
                }));
                ui.horizontal(|ui| {
                    if ui.add_enabled(k > 0, Button::new("\u{25c0} Previous move")).clicked() {
                        goto = Some(k - 1);
                    }
                    if ui.add_enabled(k + 1 < n_moves, Button::new("Next move \u{25b6}")).clicked() {
                        goto = Some(k + 1);
                    }
                });
            });
        if !open || close {
            self.cal_align = None;
            self.align_saved = false;
        }
        let source = if saved { self.saved_cal.as_ref().map(|(r, _)| r) } else { self.cal_result.as_ref() };
        if let (Some(g), Some(r)) = (goto, source) {
            // Keep the processing and view when stepping through the moves.
            let like = self.cal_align.take().map(|v| v.view);
            self.cal_align = AlignView::new(ctx, r, g, like.as_ref());
        }
    }

    /// Real move (nm, scan-area axes) for `steps` at the current amplitudes.
    pub fn steps_to_real(&self, steps: IVec2) -> DVec2 {
        self.active().steps_to_real(steps, *self.voltages.peek())
    }
    /// The calibration moves use: the learned one in "on-the-fly" mode
    /// (once there is one), else the last test.
    pub fn active(&self) -> CoarseCalibration {
        self.learn.active().unwrap_or(self.calib)
    }
    /// The last calibration test (never changed by learning).
    pub fn baseline(&self) -> CoarseCalibration {
        self.calib
    }
    /// Learn from the user's drags of stamps and the scan region (call
    /// every frame).
    pub fn learn_sync(&mut self, ui: &Ui, object_list: &mut SelectableList<Object>) {
        self.learn.sync(ui, object_list, &self.calib, self.calib_updated.unix);
    }
    /// A drag of the latest stamp moves the scan region with it (learning
    /// from drags does that itself).
    pub fn follows_latest_stamp(&self) -> bool {
        self.learn.follows_latest()
    }
    /// The coarse motion can turn distances into steps.
    pub fn is_calibrated(&self) -> bool {
        let v = *self.voltages.peek();
        v.x > 0. && v.y > 0. && self.active().is_usable()
    }
    /// Driver amplitudes now (V).
    pub fn volts(&self) -> DVec2 {
        *self.voltages.peek()
    }
    /// The calibration in use for moves (see [`Self::active`]).
    pub fn calibration(&self) -> CoarseCalibration {
        self.active()
    }
    /// Replace the baseline (a new calibration test) and save it, noting
    /// when and `how`. Learning starts again from it.
    pub fn set_calibration(&mut self, c: CoarseCalibration, how: impl Into<String>) {
        let old = self.calib;
        self.learn.on_new_baseline(old, &self.calib_updated, c);
        self.calib = c;
        self.calib_updated = CalibrationStamp::now(how);
        save_calibration(&self.calib, &self.calib_updated);
    }
    pub fn get_steps(&self) -> IVec2 {
        self.real_to_steps(self.move_target)
    }
    /// Nearest whole number of motor steps for a real-world (nm) move.
    pub fn real_to_steps(&self, real: DVec2) -> IVec2 {
        self.active().real_to_steps(real, *self.voltages.peek())
    }
    /// The Course Motion window is open (it owns ctrl+drag on the map).
    pub fn menu_open(&self) -> bool {
        self.menu_active
    }
    /// A coarse move is running.
    pub fn is_moving(&self) -> bool {
        self.currently_moving
    }
    /// Start a coarse move of `steps` (e.g. from Ring Assist), with this
    /// menu's withdraw/approach settings. When it completes, the scan area is
    /// shifted on the map as for a move made here. Returns false if a move is
    /// already running.
    pub fn start_move(&mut self, steps: IVec2) -> bool {
        if self.currently_moving || steps == IVec2::ZERO {
            return false;
        }
        // Completion applies `get_steps()`, which is derived from the target.
        self.move_target = self.steps_to_real(steps);
        info!("Executing course move {steps:?} on group {} (Ring Assist)", self.group);
        self.currently_moving = true;
        self.move_sender.send(CourseMoveCommand {
            steps,
            group: self.group - 1,
            auto_withdraw: self.auto_withdraw,
            auto_approach: self.auto_approach,
        });
        true
    }
    /// Withdraw / approach settings used for moves.
    pub fn withdraw_approach(&self) -> (bool, bool) {
        (self.auto_withdraw, self.auto_approach)
    }
    fn write_steps(&mut self, steps: IVec2) {
        if steps != self.get_steps() {
            self.move_target = self.steps_to_real(steps);
        }
    }
}

fn course_path_iter(steps: IVec2) -> impl Iterator<Item = IVec2> {
    integer_iter(steps.x)
        .map(move |x| IVec2 { x, y: 0 })
        .chain(integer_iter(steps.y).map(move |y| IVec2 { x: steps.x, y }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_round_trip_with_different_directions() {
        let c = CoarseCalibration {
            x_pos: [100e-9, 5e-9],
            x_neg: [80e-9, 4e-9],
            y_pos: [-3e-9, 120e-9],
            y_neg: [-2e-9, 90e-9],
        };
        let v = DVec2::new(1., 1.);
        for steps in [IVec2::new(10, 0), IVec2::new(-10, 0), IVec2::new(0, 7), IVec2::new(0, -7), IVec2::new(-4, 9), IVec2::new(6, -3)] {
            let real = c.steps_to_real(steps, v);
            assert_eq!(c.real_to_steps(real, v), steps, "{steps:?} -> {real:?}");
        }
        // Back steps are shorter: 10 steps back cover 800 nm in x.
        assert!((c.steps_to_real(IVec2::new(-10, 0), v).x + 800.).abs() < 1e-6);
    }

    #[test]
    fn files_without_a_date_still_load() {
        let json = r#"{"x_pos":[1e-7,0],"x_neg":[1e-7,0],"y_pos":[0,1e-7],"y_neg":[0,1e-7]}"#;
        let s: SavedCalibration = serde_json::from_str(json).unwrap();
        assert_eq!(s.updated, CalibrationStamp::default());
        // And the new format keeps both.
        let saved = SavedCalibration { calib: s.calib, updated: CalibrationStamp { unix: Some(1_700_000_000), how: "auto calibration (+X)".into() } };
        let back: SavedCalibration = serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        assert_eq!(back.calib, s.calib);
        assert_eq!(back.updated, saved.updated);
    }

    #[test]
    fn stamp_reads_naturally() {
        let s = CalibrationStamp { unix: Some(1_700_000_000), how: "edited by hand".into() };
        let text = s.describe();
        assert!(text.contains("days ago") && text.contains("2023-11-14 22:13 UTC") && text.ends_with("edited by hand"), "{text}");
        assert_eq!(CalibrationStamp::default().describe(), "unknown");
        assert!(CalibrationStamp::now("x").describe().starts_with("just now"));
    }

    #[test]
    fn old_matrix_applies_both_ways() {
        let c = CoarseCalibration::from_matrix(DMat2::from_cols(DVec2::new(1e-7, 0.), DVec2::new(0., 2e-7)));
        let v = DVec2::new(2., 2.);
        assert_eq!(c.steps_to_real(IVec2::new(-3, 0), v), -c.steps_to_real(IVec2::new(3, 0), v));
        assert!(c.is_usable());
    }
}
fn integer_iter(end: i32) -> impl Iterator<Item = i32> {
    let del = end.signum();
    (0..end.abs()).map(move |v| (v + 1) * del)
}
