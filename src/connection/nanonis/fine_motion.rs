//! Fine Motion: move within the piezo range without the coarse motors.
//!
//! - **Scan frame**: stamp the current scan, then move the scan frame
//!   (Scan.FrameSet) to a new spot inside the piezo range, and optionally
//!   start the next scan.
//! - **Tip**: send the tip to a point (Follow Me, FolMe.XYPosSet), from the
//!   window or by ctrl+click / ctrl+drag on the map.
//!
//! A running scan is stopped first. Positions are piezo nm, the same
//! coordinates as the scan frame centre and the tip readback.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use egui::{Button, Color32, DragValue, Id, RichText, Stroke, Ui};
use glam::{DAffine2, DMat2, DVec2};
use nanonis_tcp::{LineDir, ScanDir};
use tracing::{info, warn};

use crate::{
    components::selectable_list::SelectableList,
    connection::{
        nanonis::{
            scan_control::{ScanCommand, ScanControlState},
            NanonisConnection,
        },
        queue::OverwriteQueueSender,
        ScanArea,
    },
    scan_view::{world_delta_transform, BorderRectangle, ImageEncoder, ScanViewCtx},
    theme,
    utils::vec_interop::{IntoEgui as _, IntoGlam as _},
    view_object::Object,
};

const TARGET_COLOR: Color32 = Color32::from_rgb(0x2b, 0xc4, 0xe0);
/// How long to wait for a running scan to stop.
const STOP_TIMEOUT: Duration = Duration::from_secs(10);
/// How long to wait for a fresh full frame before stamping what is there.
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);
/// Time for the new frame to reach Nanonis before a scan is started.
const SETTLE: Duration = Duration::from_millis(600);
/// Minimum time between Follow Me targets while ctrl+dragging the tip.
const FOLLOW_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Frame,
    Tip,
}

#[derive(Clone, Copy, PartialEq)]
enum Goal {
    /// New scan frame centre.
    Frame(DVec2),
    /// New tip position.
    Tip(DVec2),
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    /// Waiting for the running scan to stop.
    Stopping,
    /// Waiting for a fresh full frame to stamp.
    Fetching { forward: bool, backward: bool },
    /// The frame was set; waiting before starting the next scan.
    Settling,
}

struct Job {
    goal: Goal,
    phase: Phase,
    since: Instant,
}

pub struct FineMotionState {
    open: bool,
    mode: Mode,
    /// Target frame centre; `None` follows the current frame.
    frame_target: Option<DVec2>,
    /// Target tip position; `None` follows the tip.
    tip_target: Option<DVec2>,
    stamp_first: bool,
    then_scan: Option<ScanDir>,
    /// Overlap between neighbouring frames for the arrow steps, percent.
    overlap_pct: f64,
    /// Tip arrow step, nm.
    tip_step: f64,
    /// Send the tip as soon as the target is set on the map.
    follow: bool,
    /// A tip target set on the map that hasn't been sent yet.
    follow_dirty: bool,
    last_follow: Option<Instant>,
    /// Frame centres before each frame move, for "Back".
    history: Vec<DVec2>,
    job: Option<Job>,
    /// Start this goal on the next update.
    start: Option<Goal>,
    /// Stop the tip on the next update.
    halt: bool,
    /// The frame moved: clear the old live picture.
    clear_live: bool,
    /// Last message: text and whether it is a warning.
    msg: Option<(String, bool)>,
    /// Data of the last stamp made here, so the same scan isn't stamped twice.
    stamped: Option<Arc<Box<[f32]>>>,
    /// Piezo nm → screen points (linear part), from the last map draw.
    screen_lin: DMat2,
}

impl FineMotionState {
    pub fn new() -> Self {
        Self {
            open: false,
            mode: Mode::Frame,
            frame_target: None,
            tip_target: None,
            stamp_first: true,
            then_scan: None,
            overlap_pct: 10.,
            tip_step: 50.,
            follow: true,
            follow_dirty: false,
            last_follow: None,
            history: Vec::new(),
            job: None,
            start: None,
            halt: false,
            clear_live: false,
            msg: None,
            stamped: None,
            screen_lin: DMat2::IDENTITY,
        }
    }

    /// A fine move is in progress.
    pub fn is_running(&self) -> bool {
        self.job.is_some()
    }

    /// The window is open and owns ctrl+click / ctrl+drag on the map.
    fn owns_map(&self) -> bool {
        self.open
    }

    /// Launcher button and window. `locked` says why moves can't start now.
    pub fn is_open(&self) -> bool {
        self.open
    }
    pub fn set_open(&mut self, open: bool) {
        self.open = open;
        if !open {
            self.follow_dirty = false;
        }
    }

    /// The Fine Motion window (opened from the Scan Control tools).
    pub fn show_menu(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>, locked: Option<&str>) {
        if !self.open {
            return;
        }
        // Ctrl+click / drag on the map is ours: don't also move selected layers.
        if ui.input(|i| i.modifiers.ctrl) {
            object_list.clear_selected();
        }
        let Some(scan_area) = object_list.iter().find_map(|o| o.as_scan_area()) else {
            return;
        };
        let mut open = self.open;
        egui::Window::new("Fine Motion")
            .vscroll(true)
            .frame(theme::window_frame(ui.ctx()))
            .resizable(false)
            .collapsible(true)
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                ui.set_max_width(300.);
                self.show_window(ui, scan_area, locked);
            });
        if !open {
            self.set_open(false);
        }
    }

    fn show_window(&mut self, ui: &mut Ui, area: &ScanArea, locked: Option<&str>) {
        let Some(frame) = area.image_transform else {
            ui.weak("Waiting for the scan frame from Nanonis…");
            return;
        };
        let busy = self.job.is_some();
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, Mode::Frame, "Scan frame");
            ui.selectable_value(&mut self.mode, Mode::Tip, "Tip (Follow Me)");
        });
        ui.separator();
        let range = area.area_size;
        match self.mode {
            Mode::Frame => self.show_frame(ui, area, frame, range, busy, locked),
            Mode::Tip => self.show_tip(ui, area, frame, range, busy, locked),
        }
        if let Some(job) = &self.job {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(match job.phase {
                    Phase::Stopping => "Stopping the scan…",
                    Phase::Fetching { .. } => "Fetching the frame to stamp…",
                    Phase::Settling => "Setting the frame…",
                });
            });
        } else if let Some(r) = locked {
            ui.colored_label(theme::WARN, r);
        }
        if let Some((text, warn)) = &self.msg {
            if *warn {
                ui.colored_label(theme::WARN, text);
            } else {
                ui.weak(text);
            }
        }
    }

    fn show_frame(
        &mut self,
        ui: &mut Ui,
        area: &ScanArea,
        frame: DAffine2,
        range: DVec2,
        busy: bool,
        locked: Option<&str>,
    ) {
        let current = frame.translation;
        let half = frame_half_extent(frame.matrix2);
        let mut target = self.frame_target.unwrap_or(current);
        let size = DVec2::new(frame.matrix2.x_axis.length(), frame.matrix2.y_axis.length());
        egui::Grid::new("fine_frame").num_columns(3).spacing([8., 4.]).show(ui, |ui| {
            ui.label("Frame centre");
            ui.monospace(format!("{:.1}", current.x));
            ui.monospace(format!("{:.1} nm", current.y));
            ui.end_row();
            ui.label("Target");
            ui.add_enabled(!busy, DragValue::new(&mut target.x).speed(1.).max_decimals(1));
            ui.add_enabled(!busy, DragValue::new(&mut target.y).speed(1.).max_decimals(1).suffix(" nm"));
            ui.end_row();
            ui.label("Move by");
            let d = target - current;
            ui.monospace(format!("{:+.1}", d.x));
            ui.monospace(format!("{:+.1} nm", d.y));
            ui.end_row();
        });
        ui.weak(format!(
            "Frame {:.0} × {:.0} nm, piezo range {:.2} × {:.2} µm",
            size.x,
            size.y,
            range.x * 1e-3,
            range.y * 1e-3
        ));

        // Arrows step one frame (less the overlap) along the frame's axes,
        // picked to match the arrows' directions on screen.
        ui.add_space(4.);
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!busy, |ui| {
                let keep = 1. - self.overlap_pct / 100.;
                let axes = [frame.matrix2.x_axis * keep, frame.matrix2.y_axis * keep];
                if let Some(v) = arrow_pad(ui, "fine_frame_arrows", |screen| {
                    pick_axis(self.screen_lin, axes, screen)
                }) {
                    target += v;
                }
            });
            ui.vertical(|ui| {
                ui.label("Step one frame");
                ui.add(
                    DragValue::new(&mut self.overlap_pct)
                        .range(0.0..=90.)
                        .speed(1.)
                        .prefix("overlap ")
                        .suffix(" %"),
                );
            });
        });

        ui.horizontal(|ui| {
            ui.add_enabled_ui(!busy, |ui| {
                if ui
                    .add_enabled(!self.history.is_empty(), Button::new("⟲ Back"))
                    .on_hover_text("Target the frame position before the last move")
                    .clicked()
                {
                    target = *self.history.last().unwrap();
                }
                if ui.button("Range centre").on_hover_text("Target the middle of the piezo range").clicked() {
                    target = DVec2::ZERO;
                }
                if let Some(tip) = area.tip_pos {
                    if ui.button("At tip").on_hover_text("Centre the target frame on the tip").clicked() {
                        target = tip;
                    }
                }
                if ui.button("Reset").on_hover_text("Target the current frame").clicked() {
                    target = current;
                }
            });
        });

        let (clamped, was_out) = clamp_centre(target, half, range);
        if was_out {
            ui.colored_label(theme::WARN, "Kept inside the piezo range.");
        }
        target = clamped;
        self.frame_target = (target.distance(current) > 0.05).then_some(target);

        ui.add_space(4.);
        ui.add_enabled_ui(!busy, |ui| {
            ui.checkbox(&mut self.stamp_first, "Stamp the current scan first");
            ui.horizontal(|ui| {
                ui.label("Then");
                ui.selectable_value(&mut self.then_scan, None, "don't scan");
                ui.selectable_value(&mut self.then_scan, Some(ScanDir::Up), "⬆ scan up");
                ui.selectable_value(&mut self.then_scan, Some(ScanDir::Down), "⬇ scan down");
            });
        });
        let label = match (area.scan_status.scanning, self.stamp_first) {
            (true, true) => "Stop scan, stamp & move frame",
            (true, false) => "Stop scan & move frame",
            (false, true) => "Stamp & move frame",
            (false, false) => "Move frame",
        };
        let can = !busy && locked.is_none() && self.frame_target.is_some();
        if ui
            .add_enabled(can, Button::new(RichText::new(label).strong()).min_size([ui.available_width(), 26.].into()))
            .clicked()
        {
            self.start = Some(Goal::Frame(target));
        }
        ui.weak("Ctrl+drag the dashed frame on the map, or ctrl+click to centre it there.");
    }

    fn show_tip(
        &mut self,
        ui: &mut Ui,
        area: &ScanArea,
        frame: DAffine2,
        range: DVec2,
        busy: bool,
        locked: Option<&str>,
    ) {
        let Some(tip) = area.tip_pos else {
            ui.weak("Waiting for the tip position…");
            return;
        };
        let mut target = self.tip_target.unwrap_or(tip);
        let before = target;
        egui::Grid::new("fine_tip").num_columns(3).spacing([8., 4.]).show(ui, |ui| {
            ui.label("Tip at");
            ui.monospace(format!("{:.1}", tip.x));
            ui.monospace(format!("{:.1} nm", tip.y));
            ui.end_row();
            ui.label("Target");
            ui.add_enabled(!busy, DragValue::new(&mut target.x).speed(1.).max_decimals(1));
            ui.add_enabled(!busy, DragValue::new(&mut target.y).speed(1.).max_decimals(1).suffix(" nm"));
            ui.end_row();
            ui.label("Distance");
            ui.monospace(format!("{:.1} nm", target.distance(tip)));
            ui.end_row();
        });

        ui.add_space(4.);
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!busy, |ui| {
                let step = self.tip_step;
                let screen_lin = self.screen_lin;
                if let Some(v) = arrow_pad(ui, "fine_tip_arrows", |screen| {
                    // Step `step` nm in the arrow's direction on screen.
                    let inv = screen_lin.inverse();
                    let v = inv * screen;
                    (v.length() > 0.).then(|| v.normalize() * step)
                }) {
                    target += v;
                }
            });
            ui.vertical(|ui| {
                ui.label("Arrow step");
                ui.add(DragValue::new(&mut self.tip_step).range(0.1..=1e5).speed(1.).suffix(" nm"));
            });
        });
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("Frame centre").on_hover_text("Target the middle of the scan frame").clicked() {
                    target = frame.translation;
                }
                if ui.button("Reset").on_hover_text("Target where the tip is").clicked() {
                    target = tip;
                }
            });
        });

        let (clamped, was_out) = clamp_centre(target, DVec2::ZERO, range);
        if was_out {
            ui.colored_label(theme::WARN, "Kept inside the piezo range.");
        }
        target = clamped;
        if target != before && self.follow {
            // Arrows and typed values also go at once when following.
            self.follow_dirty = true;
        }
        self.tip_target = (target.distance(tip) > 0.05 || self.follow_dirty).then_some(target);

        ui.add_space(4.);
        ui.checkbox(&mut self.follow, "Move at once (follow ctrl+click / drag, arrows)")
            .on_hover_text("Off: set the target, then press Go");
        if area.scan_status.scanning && self.follow {
            ui.colored_label(theme::WARN, "Scanning: the tip only follows once the scan is stopped.");
        }
        ui.horizontal(|ui| {
            let label = if area.scan_status.scanning { "Stop scan & go" } else { "Go" };
            let can = !busy && locked.is_none() && self.tip_target.is_some();
            if ui.add_enabled(can, Button::new(RichText::new(label).strong()).min_size([120., 26.].into())).clicked() {
                self.start = Some(Goal::Tip(target));
            }
            if ui
                .add_enabled(locked.is_none(), Button::new(RichText::new("⏹ Halt").color(Color32::WHITE)).fill(theme::ERR))
                .on_hover_text("Stop the tip where it is")
                .clicked()
            {
                self.start = None;
                self.follow_dirty = false;
                self.tip_target = None;
                self.halt = true;
            }
        });
        ui.weak("Ctrl+click or ctrl+drag on the map to place the tip target.");
    }

    /// Run the move steps. Called every frame with the scan area.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        area: &mut ScanArea,
        encoder: &ImageEncoder,
        scan_control: &mut ScanControlState,
        frame_tx: &OverwriteQueueSender<LineDir>,
        got_forward: bool,
        got_backward: bool,
        locked: Option<&str>,
    ) {
        if std::mem::take(&mut self.halt) {
            scan_control.request(ScanCommand::HaltTip);
            self.msg = Some(("Tip halted.".into(), false));
        }
        if let Some(goal) = self.start.take() {
            if let Some(why) = locked {
                self.msg = Some((format!("Not moved: {why}."), true));
            } else if self.job.is_none() {
                self.begin(goal, area, scan_control);
            }
        }
        if self.job.is_some() {
            if let Some(why) = locked {
                self.job = None;
                self.msg = Some((format!("Stopped: {why}."), true));
            }
        }
        if let Some(job) = &mut self.job {
            let elapsed = job.since.elapsed();
            let goal = job.goal;
            match job.phase {
                Phase::Stopping => {
                    if !area.scan_status.scanning && !scan_control.busy() {
                        self.after_stop(goal, area, scan_control, frame_tx);
                    } else if elapsed > STOP_TIMEOUT {
                        self.job = None;
                        self.msg = Some(("The scan didn't stop; nothing moved.".into(), true));
                    }
                }
                Phase::Fetching { forward, backward } => {
                    let forward = forward || got_forward;
                    let backward = backward || got_backward;
                    job.phase = Phase::Fetching { forward, backward };
                    let fresh = forward && backward;
                    if fresh || elapsed > FETCH_TIMEOUT {
                        if !fresh {
                            warn!("fine motion: fresh frame not received, stamping current data");
                        }
                        self.stamp(area, encoder);
                        if let Goal::Frame(to) = goal {
                            self.set_frame(to, area);
                        }
                    }
                }
                Phase::Settling => {
                    if elapsed > SETTLE {
                        self.job = None;
                        if let Some(dir) = self.then_scan {
                            scan_control.request(ScanCommand::Start(dir));
                        }
                    }
                }
            }
        }

        if std::mem::take(&mut self.clear_live) {
            // The old picture would otherwise be drawn at the new place.
            area.live_image.clear_texture(encoder);
        }

        // Live Follow Me: send the newest target, no faster than the worker
        // takes commands.
        if self.mode == Mode::Tip && self.follow_dirty && self.job.is_none() {
            let ready = !scan_control.busy()
                && self.last_follow.is_none_or(|t| t.elapsed() > FOLLOW_INTERVAL);
            if locked.is_some() || area.scan_status.scanning {
                self.follow_dirty = false;
            } else if ready {
                if let Some(to) = self.tip_target {
                    scan_control.request(ScanCommand::MoveTip { x: to.x * 1e-9, y: to.y * 1e-9 });
                    self.last_follow = Some(Instant::now());
                }
                self.follow_dirty = false;
            }
        }
        // Once the tip arrives, follow it again.
        if let (Some(t), Some(tip)) = (self.tip_target, area.tip_pos) {
            if !self.follow_dirty && self.job.is_none() && t.distance(tip) < 0.5 {
                self.tip_target = None;
            }
        }
    }

    fn begin(&mut self, goal: Goal, area: &ScanArea, scan_control: &mut ScanControlState) {
        self.msg = None;
        self.follow_dirty = false;
        if area.scan_status.scanning {
            info!("fine motion: stopping the scan first");
            scan_control.request(ScanCommand::Stop);
        }
        // Not scanning: this just waits for any earlier command to finish.
        self.job = Some(Job { goal, phase: Phase::Stopping, since: Instant::now() });
    }

    fn after_stop(
        &mut self,
        goal: Goal,
        area: &mut ScanArea,
        scan_control: &mut ScanControlState,
        frame_tx: &OverwriteQueueSender<LineDir>,
    ) {
        match goal {
            Goal::Tip(to) => {
                info!("fine motion: tip to ({:.1}, {:.1}) nm", to.x, to.y);
                scan_control.request(ScanCommand::MoveTip { x: to.x * 1e-9, y: to.y * 1e-9 });
                self.job = None;
                self.msg = Some((format!("Tip sent to ({:.1}, {:.1}) nm.", to.x, to.y), false));
            }
            Goal::Frame(to) => {
                if self.stamp_first && area.channel_selected.is_some() && has_data(area) {
                    NanonisConnection::request_full_frame(frame_tx);
                    self.job = Some(Job {
                        goal,
                        phase: Phase::Fetching { forward: false, backward: false },
                        since: Instant::now(),
                    });
                } else {
                    self.set_frame(to, area);
                    if self.stamp_first {
                        self.msg = Some(("Nothing to stamp; moved without stamping.".into(), true));
                    }
                }
            }
        }
    }

    /// Stamp the live image, unless this exact scan was stamped here already.
    fn stamp(&mut self, area: &mut ScanArea, encoder: &ImageEncoder) {
        let data = area.live_image.forward_data.data.clone();
        if self.stamped.as_ref().is_some_and(|s| same_data(s, &data)) {
            info!("fine motion: scan already stamped");
            return;
        }
        area.stamp.push_front(area.live_image.stamp(encoder));
        self.stamped = Some(data);
    }

    fn set_frame(&mut self, to: DVec2, area: &mut ScanArea) {
        let Some(mut frame) = area.image_transform else {
            self.job = None;
            return;
        };
        let from = frame.translation;
        self.history.push(from);
        frame.translation = to;
        area.image_transform = Some(frame);
        self.clear_live = true;
        let d = to - from;
        info!(
            "fine motion: frame centre ({:.1}, {:.1}) -> ({:.1}, {:.1}) nm",
            from.x, from.y, to.x, to.y
        );
        self.msg = Some((format!("Frame moved by ({:+.1}, {:+.1}) nm.", d.x, d.y), false));
        self.frame_target = None;
        self.job = Some(Job { goal: Goal::Frame(to), phase: Phase::Settling, since: Instant::now() });
    }

    /// Map input (ctrl+click / ctrl+drag) and the target drawing.
    /// `map_free` is false while another tool owns ctrl+drag.
    pub fn show_overlay(&mut self, ui: &mut Ui, area: &ScanArea, map_free: bool) {
        let Some(ctx) = ui.data(|map| map.get_temp::<ScanViewCtx>(Id::new(()))) else {
            return;
        };
        let to_screen = ctx.world2egui() * area.world_transform;
        self.screen_lin = to_screen.matrix2;
        if !self.owns_map() {
            return;
        }
        let Some(frame) = area.image_transform else {
            return;
        };

        // Input.
        let ctrl = ui.input(|i| i.modifiers.ctrl);
        if ctrl && map_free && self.job.is_none() {
            let resp = &ctx.screen_response;
            let pointer = resp.interact_pointer_pos().or(resp.hover_pos());
            let at = pointer.map(|p| to_screen.inverse().transform_point2(p.to_glam()));
            match self.mode {
                Mode::Frame => {
                    let half = frame_half_extent(frame.matrix2);
                    let mut target = self.frame_target.unwrap_or(frame.translation);
                    if resp.dragged_by(egui::PointerButton::Primary) {
                        let [_, _, translate] = world_delta_transform(ui);
                        target += area
                            .world_transform
                            .inverse()
                            .transform_vector2(translate.translation);
                    } else if resp.clicked() {
                        if let Some(at) = at {
                            target = at;
                        }
                    }
                    let target = clamp_centre(target, half, area.area_size).0;
                    self.frame_target =
                        (target.distance(frame.translation) > 0.05).then_some(target);
                }
                Mode::Tip => {
                    if resp.dragged_by(egui::PointerButton::Primary) || resp.clicked() {
                        if let Some(at) = at {
                            let at = clamp_centre(at, DVec2::ZERO, area.area_size).0;
                            self.tip_target = Some(at);
                            if self.follow {
                                self.follow_dirty = true;
                            }
                        }
                    }
                }
            }
        }

        // Drawing.
        let stroke = Stroke::new(1.5, TARGET_COLOR);
        match self.mode {
            Mode::Frame => {
                if let Some(target) = self.frame_target {
                    let mut t = frame;
                    t.translation = target;
                    BorderRectangle {
                        transform: area.world_transform * t,
                        color: TARGET_COLOR,
                        dashed: true,
                    }
                    .show(ui);
                    let a = to_screen.transform_point2(frame.translation).to_egui_pos2();
                    let b = to_screen.transform_point2(target).to_egui_pos2();
                    let painter = ui.painter();
                    painter.arrow(a, b - a, stroke);
                    let d = target - frame.translation;
                    painter.text(
                        b + egui::vec2(6., -6.),
                        egui::Align2::LEFT_BOTTOM,
                        format!("{:+.1}, {:+.1} nm", d.x, d.y),
                        egui::FontId::proportional(12.),
                        TARGET_COLOR,
                    );
                }
            }
            Mode::Tip => {
                if let (Some(target), Some(tip)) = (self.tip_target, area.tip_pos) {
                    let painter = ui.painter();
                    let a = to_screen.transform_point2(tip).to_egui_pos2();
                    let b = to_screen.transform_point2(target).to_egui_pos2();
                    if a.distance(b) > 2. {
                        painter.line_segment([a, b], Stroke::new(1., TARGET_COLOR));
                    }
                    let r = 7.;
                    painter.circle_stroke(b, r, stroke);
                    painter.line_segment([b - egui::vec2(r + 4., 0.), b + egui::vec2(r + 4., 0.)], stroke);
                    painter.line_segment([b - egui::vec2(0., r + 4.), b + egui::vec2(0., r + 4.)], stroke);
                }
            }
        }
    }
}

/// Half the size of the frame's axis-aligned bounding box (piezo nm).
fn frame_half_extent(m: DMat2) -> DVec2 {
    (m.x_axis.abs() + m.y_axis.abs()) * 0.5
}

/// Keep a centre so that a box of `half` extent stays inside the piezo range
/// (centred on 0). Returns the centre and whether it had to move. An unknown
/// range (zero) isn't enforced.
fn clamp_centre(c: DVec2, half: DVec2, range: DVec2) -> (DVec2, bool) {
    if range.x <= 0. || range.y <= 0. {
        return (c, false);
    }
    let lim = (range * 0.5 - half).max(DVec2::ZERO);
    let out = c.clamp(-lim, lim);
    (out, out.distance(c) > 1e-6)
}

/// Of ± each axis, the one pointing most along `screen` once drawn.
fn pick_axis(screen_lin: DMat2, axes: [DVec2; 2], screen: DVec2) -> Option<DVec2> {
    axes.into_iter()
        .flat_map(|a| [a, -a])
        .filter(|a| a.length() > 0.)
        .max_by(|a, b| {
            let da = (screen_lin * *a).normalize_or_zero().dot(screen);
            let db = (screen_lin * *b).normalize_or_zero().dot(screen);
            da.total_cmp(&db)
        })
}

/// Four arrow buttons. `step` turns an on-screen direction (y down) into a
/// piezo move; returns the move of the arrow clicked.
fn arrow_pad(ui: &mut Ui, id: &str, step: impl Fn(DVec2) -> Option<DVec2>) -> Option<DVec2> {
    let mut out = None;
    let size = egui::vec2(26., 22.);
    egui::Grid::new(id).spacing([2., 2.]).show(ui, |ui| {
        let mut b = |ui: &mut Ui, text: &str, dir: DVec2| {
            if ui.add_sized(size, Button::new(text)).clicked() {
                out = step(dir);
            }
        };
        ui.label("");
        b(ui, "⬆", DVec2::new(0., -1.));
        ui.label("");
        ui.end_row();
        b(ui, "⬅", DVec2::new(-1., 0.));
        ui.label("");
        b(ui, "➡", DVec2::new(1., 0.));
        ui.end_row();
        ui.label("");
        b(ui, "⬇", DVec2::new(0., 1.));
        ui.label("");
        ui.end_row();
    });
    out
}

/// The live image holds some real data.
fn has_data(area: &ScanArea) -> bool {
    area.live_image.forward_data.data.iter().any(|v| v.is_finite())
}

fn same_data(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_keeps_the_frame_inside_the_range() {
        let range = DVec2::new(1000., 1000.);
        let half = DVec2::new(100., 50.);
        let (c, out) = clamp_centre(DVec2::new(450., -10.), half, range);
        assert!(out);
        assert_eq!(c, DVec2::new(400., -10.));
        let (c, out) = clamp_centre(DVec2::new(0., -460.), half, range);
        assert!(out);
        assert_eq!(c, DVec2::new(0., -450.));
        let (c, out) = clamp_centre(DVec2::new(10., 10.), half, range);
        assert!(!out);
        assert_eq!(c, DVec2::new(10., 10.));
        // Unknown range: not enforced.
        assert!(!clamp_centre(DVec2::splat(1e9), half, DVec2::ZERO).1);
    }

    #[test]
    fn rotated_frame_extent_covers_its_corners() {
        let t = DAffine2::from_scale_angle_translation(DVec2::new(200., -100.), 0.5, DVec2::ZERO);
        let half = frame_half_extent(t.matrix2);
        for c in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
            let p = t.transform_point2(DVec2::from(c));
            assert!(p.x.abs() <= half.x + 1e-9 && p.y.abs() <= half.y + 1e-9);
        }
    }

    #[test]
    fn arrows_follow_the_screen() {
        // Piezo y up on screen (screen y is down).
        let flip = DMat2::from_diagonal(DVec2::new(1., -1.));
        let axes = [DVec2::new(100., 0.), DVec2::new(0., -80.)];
        assert_eq!(pick_axis(flip, axes, DVec2::new(0., -1.)), Some(DVec2::new(0., 80.)));
        assert_eq!(pick_axis(flip, axes, DVec2::new(-1., 0.)), Some(DVec2::new(-100., 0.)));
        // Rotated 90°: screen right is piezo -y.
        let rot = DMat2::from_angle(std::f64::consts::FRAC_PI_2);
        assert_eq!(pick_axis(rot, axes, DVec2::new(1., 0.)), Some(DVec2::new(0., -80.)));
    }
}
