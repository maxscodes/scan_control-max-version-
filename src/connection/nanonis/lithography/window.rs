use std::{collections::BTreeSet, path::PathBuf};

use crossbeam::channel::{Receiver, Sender};
use egui::{Color32, DragValue, Id, RichText, Stroke, Ui};
use glam::{DAffine2, DVec2};
use uuid::Uuid;

use crate::{
    components::{file_dialog_native::dialog, selectable_list::SelectableList},
    connection::{
        nanonis::{
            lithography::{
                path::{append_strokes, plan, PathMode, PathStats, Polygon, Segment},
                shapes::{Geometry, Handle, Shape, ShapeKind},
                worker::{JobSegment, LithoCommand, LithoEvent, LithoJob, LithoSettings, Transition},
            },
            sequence::Frame,
        },
        ScanArea,
    },
    scan_view::{GDSImage, ScanViewCtx},
    utils::vec_interop::IntoEgui as _,
    view_object::Object,
};

const ACTIVE_COLOR: Color32 = Color32::from_rgb(0xff, 0x4d, 0x6d);
const TRAVEL_COLOR: Color32 = Color32::from_rgb(0xf4, 0xd3, 0x5e);
const DONE_COLOR: Color32 = Color32::from_rgb(0x6c, 0x75, 0x7d);
const SHAPE_COLOR: Color32 = Color32::from_rgb(0x4d, 0xd0, 0x9a);
const SHAPE_SELECTED: Color32 = Color32::from_rgb(0x9c, 0xff, 0xd2);
/// Handle pick radius on the map, screen points.
const PICK: f64 = 8.;

/// Everything a planned path depends on; a change makes the plan stale.
#[derive(Debug, Clone, PartialEq)]
struct PlanKey {
    gds: Option<GdsKey>,
    /// The enabled drawn shapes.
    shapes: Vec<Shape>,
    area_transform: [f64; 6],
    frame_transform: [f64; 6],
    mode: PathMode,
    pitch_nm: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct GdsKey {
    id: Uuid,
    transform: [f64; 6],
    layers: Vec<u16>,
}

struct Plan {
    key: PlanKey,
    /// Segments in Nanonis piezo nm.
    path: Vec<Segment>,
    stats: PathStats,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StatusKind {
    Idle,
    Busy,
    Done,
    Error,
}

pub struct LithographyState {
    window_open: bool,
    settings: LithoSettings,
    gds: Option<Uuid>,
    excluded_layers: BTreeSet<u16>,
    plan: Option<Plan>,
    show_path: bool,
    confirming: bool,
    running: bool,
    moving: bool,
    stopping: bool,
    progress: Option<(usize, usize)>,
    status: String,
    status_kind: StatusKind,
    cmd_tx: Sender<LithoCommand>,
    events: Receiver<LithoEvent>,
    /// Write the GDS polygons (drawn shapes are written either way).
    use_gds: bool,
    /// Shapes drawn in the window, map nm.
    shapes: Vec<Shape>,
    selected_shape: Option<usize>,
    /// Ctrl+drag / ctrl+click on the map edits the shapes.
    edit_on_map: bool,
    /// Shape and handle being dragged on the map.
    grab: Option<(usize, Handle)>,
    shapes_tx: Sender<Result<Vec<Shape>, String>>,
    shapes_rx: Receiver<Result<Vec<Shape>, String>>,
}

impl LithographyState {
    pub fn new(cmd_tx: Sender<LithoCommand>, events: Receiver<LithoEvent>) -> Self {
        let (shapes_tx, shapes_rx) = crossbeam::channel::unbounded();
        Self {
            window_open: false,
            settings: LithoSettings::default(),
            gds: None,
            excluded_layers: BTreeSet::new(),
            plan: None,
            show_path: true,
            confirming: false,
            running: false,
            moving: false,
            stopping: false,
            progress: None,
            status: "Ready".into(),
            status_kind: StatusKind::Idle,
            cmd_tx,
            events,
            use_gds: true,
            shapes: Vec::new(),
            selected_shape: None,
            edit_on_map: false,
            grab: None,
            shapes_tx,
            shapes_rx,
        }
    }

    /// Ctrl+drag / ctrl+click on the map edits drawn shapes.
    pub fn map_edit_active(&self) -> bool {
        self.window_open && self.edit_on_map && !self.is_running()
    }

    /// Lithography or a tip move is in progress: other tip controls lock.
    pub fn is_running(&self) -> bool {
        self.running || self.moving
    }

    fn set_status(&mut self, kind: StatusKind, text: impl Into<String>) {
        self.status_kind = kind;
        self.status = text.into();
    }

    fn poll_events(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                LithoEvent::Status(text) => {
                    if !self.stopping {
                        self.set_status(StatusKind::Busy, text);
                    }
                }
                LithoEvent::Progress { done, total } => self.progress = Some((done, total)),
                LithoEvent::Finished { scan } => {
                    self.running = false;
                    let msg = match scan {
                        None => "Lithography complete; feedback ON, scan values restored".to_string(),
                        Some(Ok(())) => "Lithography complete; scan of the frame started".to_string(),
                        Some(Err(e)) => format!("Lithography complete, but {e}"),
                    };
                    self.set_status(StatusKind::Done, msg);
                }
                LithoEvent::Stopped => {
                    self.running = false;
                    self.set_status(
                        StatusKind::Error,
                        "Stopped; feedback ON and scan values restored. Tip left where it stopped.",
                    );
                }
                LithoEvent::Error(msg) => {
                    self.running = false;
                    self.set_status(StatusKind::Error, format!("Error: {msg}"));
                }
                LithoEvent::Moved(result) => {
                    self.moving = false;
                    match result {
                        Ok(()) => self.set_status(StatusKind::Done, "Tip parked at the path start (no lithography)"),
                        Err(e) => self.set_status(StatusKind::Error, format!("Move failed: {e}")),
                    }
                }
            }
            if !self.is_running() {
                self.stopping = false;
            }
        }
        while let Ok(loaded) = self.shapes_rx.try_recv() {
            match loaded {
                Ok(shapes) => {
                    self.set_status(StatusKind::Idle, format!("Loaded {} shapes", shapes.len()));
                    self.shapes = shapes;
                    self.selected_shape = None;
                }
                Err(e) => self.set_status(StatusKind::Error, e),
            }
        }
    }

    pub fn show_menu(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>, blocked: Option<&str>) {
        self.poll_events();
        if self.map_edit_active() && ui.input(|i| i.modifiers.ctrl) {
            // Ctrl+drag edits shapes: don't also move selected layers.
            object_list.clear_selected();
        }
        let object_list = &*object_list;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.window_open, egui::Button::new("Lithography"))
                .clicked()
            {
                self.window_open = true;
            }
            if self.running {
                if let Some((done, total)) = self.progress {
                    ui.label(format!("litho {done}/{total}"));
                }
                ui.spinner();
            }
        });
        let gds_list: Vec<(Uuid, String, &GDSImage)> = object_list
            .iter()
            .filter_map(|o| match &**o {
                Object::Gds { image, .. } => Some((image.uuid(), o.name().to_string(), image)),
                _ => None,
            })
            .collect();
        let scan_area = object_list.iter().find_map(|o| o.as_scan_area());
        if self.gds.is_none_or(|id| !gds_list.iter().any(|g| g.0 == id)) {
            self.gds = gds_list.first().map(|g| g.0);
        }
        let mut open = self.window_open;
        egui::Window::new("Lithography")
            .frame(crate::theme::window_frame(ui.ctx()))
            .default_width(380.)
            .resizable(true)
            .collapsible(true)
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.show_window(ui, &gds_list, scan_area, blocked);
                });
            });
        self.window_open = open;
    }

    fn current_key(&self, gds: Option<&GDSImage>, area: &ScanArea) -> Option<PlanKey> {
        let gds = gds.map(|gds| GdsKey {
            id: gds.uuid(),
            transform: gds.transform.to_cols_array(),
            layers: gds
                .polygons()
                .keys()
                .copied()
                .filter(|l| !self.excluded_layers.contains(l))
                .collect(),
        });
        let shapes: Vec<Shape> = self.shapes.iter().filter(|s| s.enabled).cloned().collect();
        if gds.is_none() && shapes.is_empty() {
            return None;
        }
        Some(PlanKey {
            gds,
            shapes,
            area_transform: area.world_transform.to_cols_array(),
            frame_transform: area.image_transform?.to_cols_array(),
            mode: self.settings.mode,
            pitch_nm: self.settings.pitch_nm,
        })
    }

    fn show_window(
        &mut self,
        ui: &mut Ui,
        gds_list: &[(Uuid, String, &GDSImage)],
        area: Option<&ScanArea>,
        blocked: Option<&str>,
    ) {
        ui.colored_label(
            Color32::from_rgb(0xff, 0x70, 0x70),
            "REAL HARDWARE: Start writes lithography bias/current and moves the tip. \
             Check tip/sample safety, current limits and Z polarity first.",
        );
        ui.add_space(4.);
        let editable = !self.is_running();
        let selected_gds = self.gds.and_then(|id| gds_list.iter().find(|g| g.0 == id)).map(|g| g.2);

        ui.add_enabled_ui(editable, |ui| {
            ui.label(RichText::new("Geometry").strong());
            if gds_list.is_empty() {
                ui.weak("No GDS on the map (File \u{2192} Import Files), or draw shapes below.");
            } else {
                ui.checkbox(&mut self.use_gds, "Write a GDS");
            }
            if !gds_list.is_empty() && self.use_gds {
                egui::ComboBox::from_label("GDS layer")
                    .selected_text(
                        gds_list
                            .iter()
                            .find(|g| Some(g.0) == self.gds)
                            .map_or("–", |g| g.1.as_str()),
                    )
                    .show_ui(ui, |ui| {
                        for (id, name, _) in gds_list {
                            ui.selectable_value(&mut self.gds, Some(*id), name);
                        }
                    });
                if let Some(gds) = selected_gds {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("GDS layers:");
                        for &layer in gds.polygons().keys() {
                            let mut on = !self.excluded_layers.contains(&layer);
                            if ui.checkbox(&mut on, layer.to_string()).changed() {
                                if on {
                                    self.excluded_layers.remove(&layer);
                                } else {
                                    self.excluded_layers.insert(layer);
                                }
                            }
                        }
                    });
                }
                ui.weak("The GDS is used where it sits on the map: drag it to align before previewing.");
            }
            ui.add_space(4.);
            self.show_shapes(ui, area);
            ui.add_space(4.);
            match area.and_then(|a| a.image_transform).map(Frame::from_transform) {
                Some(f) => ui.label(format!(
                    "Selection = scan frame: {:.1}×{:.1} nm at ({:.1}, {:.1}) nm, {:.1}°",
                    f.width, f.height, f.center_x, f.center_y, f.angle
                )),
                None => ui.weak("Waiting for the scan frame from Nanonis…"),
            };

            ui.separator();
            ui.label(RichText::new("Path").strong());
            for mode in PathMode::ALL {
                ui.radio_value(&mut self.settings.mode, mode, mode.label())
                    .on_hover_text(mode.description());
            }
            if self.settings.mode.requires_ramp() {
                self.settings.transition = Transition::Ramp;
            }
            ui.horizontal(|ui| {
                ui.label("Raster pitch");
                ui.add(DragValue::new(&mut self.settings.pitch_nm).speed(0.01).range(0.01..=1000.0).suffix(" nm"));
            });
        });

        let gds = selected_gds.filter(|_| self.use_gds);
        let key = area.and_then(|a| self.current_key(gds, a));
        ui.horizontal(|ui| {
            if ui
                .add_enabled(editable && key.is_some(), egui::Button::new("Preview path"))
                .clicked()
            {
                self.preview(gds, area.unwrap(), key.clone().unwrap());
            }
            ui.checkbox(&mut self.show_path, "show on map");
        });
        let stale = self.plan.as_ref().is_some_and(|p| Some(&p.key) != key.as_ref());
        if let Some(p) = &self.plan {
            let s = p.stats;
            let est = s.estimate_s(self.settings.litho_speed_nm_s, self.settings.ramp_s());
            ui.label(format!(
                "{} segments · {:.1} nm written · {:.1} nm travel · {} transitions",
                s.segments, s.active_nm, s.travel_nm, s.transitions
            ));
            ui.label(format!(
                "Estimated time ≈ {:.1} min ({est:.0} s) at {} nm/s, excluding the move to the start and settling",
                est / 60.,
                self.settings.litho_speed_nm_s
            ));
            if stale && !self.running {
                ui.colored_label(
                    Color32::ORANGE,
                    "⚠ The GDS, shapes, frame, layers or path settings changed: preview again",
                );
            }
        }

        ui.separator();
        ui.add_enabled_ui(editable, |ui| {
            let s = &mut self.settings;
            ui.label(RichText::new("Writing").strong());
            egui::Grid::new("litho params").num_columns(2).show(ui, |ui| {
                ui.label("Litho bias");
                ui.add(DragValue::new(&mut s.litho_bias_v).speed(0.01).suffix(" V"));
                ui.end_row();
                ui.label("Litho setpoint");
                ui.add(DragValue::new(&mut s.litho_setpoint_pa).speed(10.).range(0.001..=1e6).suffix(" pA"));
                ui.end_row();
                ui.label("Litho speed");
                ui.add(DragValue::new(&mut s.litho_speed_nm_s).speed(0.1).range(0.001..=1e5).suffix(" nm/s"));
                ui.end_row();
                ui.label("Travel speed to start");
                ui.add(DragValue::new(&mut s.travel_speed_nm_s).speed(1.).range(0.001..=1e5).suffix(" nm/s"));
                ui.end_row();
            });
            ui.weak("Scan values (bias/setpoint) are read from Nanonis at start and restored at the end.");

            ui.label(RichText::new("Between written areas").strong());
            let comb = s.mode.requires_ramp();
            ui.radio_value(&mut s.transition, Transition::Ramp, "Bias/current ramp (Z feedback stays ON)");
            ui.add_enabled_ui(!comb, |ui| {
                ui.radio_value(&mut s.transition, Transition::ZLift, "Z-lift (feedback OFF, no ramp)")
                    .on_disabled_hover_text("Comb modes need the ramp: their returns are scanned at scan values");
            });
            match s.transition {
                Transition::Ramp => {
                    ui.horizontal(|ui| {
                        ui.add(DragValue::new(&mut s.ramp_steps).range(1..=1000).suffix(" steps"));
                        ui.label("over");
                        ui.add(DragValue::new(&mut s.ramp_ms).speed(1.).range(0.1..=60000.0).suffix(" ms"));
                    });
                }
                Transition::ZLift => {
                    ui.horizontal(|ui| {
                        ui.label("Lift (+Z)");
                        ui.add(DragValue::new(&mut s.lift_pm).speed(1.).range(0.0..=100000.0).suffix(" pm"));
                    });
                    ui.checkbox(&mut s.allow_z_mismatch, "Allow Z readback mismatch (warn only)");
                    if s.allow_z_mismatch {
                        ui.colored_label(Color32::ORANGE, "⚠ An unverified lift will NOT stop the tip moving");
                    }
                }
            }
            ui.checkbox(&mut s.scan_after, "Scan the frame (up) after a complete run");
        });

        ui.separator();
        let ready = self.plan.is_some() && !stale && self.settings.validate().is_ok();
        if let Err(e) = self.settings.validate() {
            ui.colored_label(Color32::ORANGE, e);
        }
        if let (Some(reason), false) = (blocked, self.is_running()) {
            ui.colored_label(Color32::ORANGE, reason);
        }
        let can_act = ready && blocked.is_none() && !self.is_running();
        if self.confirming && can_act {
            self.show_confirm(ui);
        } else {
            self.confirming = false;
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_act, egui::Button::new("Move tip to path start"))
                    .on_hover_text("Follow Me move only: no lithography values are written")
                    .clicked()
                {
                    self.move_to_start();
                }
                if ui
                    .add_enabled(can_act, egui::Button::new("Start lithography…"))
                    .clicked()
                {
                    self.confirming = true;
                }
                if ui
                    .add_enabled(
                        self.is_running() && !self.stopping,
                        egui::Button::new(RichText::new("■ STOP").strong().color(Color32::WHITE))
                            .fill(Color32::from_rgb(0xb0, 0x20, 0x20)),
                    )
                    .on_hover_text("Halts the tip, then restores feedback and scan values")
                    .clicked()
                {
                    self.stopping = true;
                    self.set_status(StatusKind::Busy, "Stopping safely…");
                    let _ = self.cmd_tx.send(LithoCommand::Stop);
                }
            });
        }
        if let (true, Some((done, total))) = (self.running, self.progress) {
            ui.add(
                egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                    .text(format!("segment {done}/{total}")),
            );
        }
        let color = match self.status_kind {
            StatusKind::Idle => ui.visuals().text_color(),
            StatusKind::Busy => Color32::ORANGE,
            StatusKind::Done => Color32::LIGHT_GREEN,
            StatusKind::Error => Color32::from_rgb(0xff, 0x60, 0x60),
        };
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(color, &self.status);
            if self.is_running() {
                ui.spinner();
            }
        });
    }

    fn show_confirm(&mut self, ui: &mut Ui) {
        let s = self.settings.clone();
        egui::Frame::group(ui.style())
            .stroke(Stroke::new(1.5, Color32::from_rgb(0xff, 0x70, 0x70)))
            .show(ui, |ui| {
                ui.label(RichText::new("Confirm REAL lithography").strong());
                ui.label(format!(
                    "{} · bias {} V · setpoint {} pA · {} nm/s",
                    s.mode.label(),
                    s.litho_bias_v,
                    s.litho_setpoint_pa,
                    s.litho_speed_nm_s
                ));
                match s.transition {
                    Transition::Ramp => ui.label(format!(
                        "Between areas: ramp {} steps / {} ms, Z feedback stays ON",
                        s.ramp_steps, s.ramp_ms
                    )),
                    Transition::ZLift => ui.label(format!(
                        "Between areas: feedback OFF, +{} pm lift; readback mismatch {}",
                        s.lift_pm,
                        if s.allow_z_mismatch { "only WARNS" } else { "stops before XY" }
                    )),
                };
                ui.label("Confirm feedback polarity, Z direction, travel limits and the lab procedure.");
                ui.horizontal(|ui| {
                    if ui
                        .button(RichText::new("Confirm: start lithography").color(Color32::from_rgb(0xff, 0x90, 0x90)))
                        .clicked()
                    {
                        self.confirming = false;
                        self.start();
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirming = false;
                    }
                });
            });
    }

    fn preview(&mut self, gds: Option<&GDSImage>, area: &ScanArea, key: PlanKey) {
        match build_path(gds, area, &key) {
            Ok(path) => {
                let stats = PathStats::of(&path);
                self.set_status(StatusKind::Idle, format!("Path ready: {} segments", stats.segments));
                self.plan = Some(Plan { key, path, stats });
            }
            Err(e) => {
                self.plan = None;
                self.set_status(StatusKind::Error, format!("Path failed: {e}"));
            }
        }
    }

    fn move_to_start(&mut self) {
        let Some(first) = self.plan.as_ref().and_then(|p| p.path.first()).copied() else {
            return;
        };
        self.moving = true;
        self.set_status(StatusKind::Busy, "Moving tip to the path start…");
        let _ = self.cmd_tx.send(LithoCommand::MoveTo {
            point: first.start * 1e-9,
            speed: self.settings.travel_speed_nm_s * 1e-9,
        });
    }

    fn start(&mut self) {
        let Some(plan) = &self.plan else { return };
        let path = plan
            .path
            .iter()
            .map(|s| JobSegment {
                start: s.start * 1e-9,
                end: s.end * 1e-9,
                active: s.active,
            })
            .collect();
        self.running = true;
        self.stopping = false;
        self.progress = Some((0, plan.path.len()));
        self.set_status(StatusKind::Busy, "Starting lithography…");
        let _ = self.cmd_tx.send(LithoCommand::Run(LithoJob {
            settings: self.settings.clone(),
            path,
        }));
    }

    /// Draw the drawn shapes (editable with ctrl on the map when
    /// `map_free`) and the planned path (active solid red, travel yellow).
    pub fn show_overlay(&mut self, ui: &mut Ui, area: &ScanArea, map_free: bool) {
        if self.window_open && !self.shapes.is_empty() {
            if let Some(ctx) = ui.data(|map| map.get_temp::<ScanViewCtx>(Id::new(()))) {
                if map_free && self.map_edit_active() {
                    self.map_input(ui, &ctx);
                } else {
                    self.grab = None;
                }
                self.draw_shapes(ui, &ctx);
            }
        }
        let Some(plan) = &self.plan else { return };
        if !self.show_path || !(self.window_open || self.running) {
            return;
        }
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let tf = ctx.world2egui() * area.world_transform;
        let done = if self.running { self.progress.map_or(0, |p| p.0) } else { 0 };
        let painter = ui.painter_at(ctx.rect);
        for (i, s) in plan.path.iter().enumerate() {
            let a = tf.transform_point2(s.start).to_egui_pos2();
            let b = tf.transform_point2(s.end).to_egui_pos2();
            let stroke = if i < done {
                Stroke::new(1., DONE_COLOR)
            } else if s.active {
                Stroke::new(1.5, ACTIVE_COLOR)
            } else {
                Stroke::new(1., TRAVEL_COLOR)
            };
            painter.line_segment([a, b], stroke);
        }
        if let Some(first) = plan.path.first() {
            painter.circle_stroke(tf.transform_point2(first.start).to_egui_pos2(), 4., Stroke::new(1.5, ACTIVE_COLOR));
        }
    }
}

/// Plan the path for `key`: GDS polygons and drawn shapes (map world) →
/// piezo nm → frame-local nm, plan, and convert the segments back to piezo
/// nm. Filled shapes are planned with the GDS polygons; drawn lines are
/// written along their length after them.
fn build_path(gds: Option<&GDSImage>, area: &ScanArea, key: &PlanKey) -> Result<Vec<Segment>, String> {
    let frame_tf = DAffine2::from_cols_array(&key.frame_transform);
    let frame = Frame::from_transform(frame_tf);
    // Frame-local axes follow the frame as drawn on the map (Nanonis angle,
    // positive clockwise).
    let rigid = DAffine2::from_angle_translation(-frame.angle.to_radians(), DVec2::new(frame.center_x, frame.center_y));
    let world_to_local = rigid.inverse() * area.world_transform.inverse();
    let mut polygons: Vec<Polygon> = Vec::new();
    if let (Some(gds), Some(gk)) = (gds, &key.gds) {
        let to_local = world_to_local * gds.transform;
        polygons.extend(
            gds.polygons()
                .iter()
                .filter(|(layer, _)| gk.layers.contains(layer))
                .flat_map(|(_, polys)| polys)
                .filter(|poly| poly.len() >= 3)
                .map(|poly| poly.iter().map(|v| to_local.transform_point2(v.as_dvec2())).collect::<Polygon>()),
        );
    }
    let size = DVec2::new(frame.width, frame.height);
    let mut strokes: Vec<Polygon> = Vec::new();
    for shape in &key.shapes {
        let local = |pts: Vec<DVec2>| -> Polygon { pts.into_iter().map(|p| world_to_local.transform_point2(p)).collect() };
        match shape.geometry() {
            Some(Geometry::Fill(poly)) => polygons.push(local(poly)),
            Some(Geometry::Stroke(line)) => {
                let line = local(line);
                let half = size / 2. + DVec2::splat(1e-6);
                if line.iter().any(|p| p.x.abs() > half.x || p.y.abs() > half.y) {
                    return Err(format!("line \"{}\" leaves the scan frame; move it inside or shrink it", shape.name));
                }
                strokes.push(line);
            }
            None => {}
        }
    }
    if polygons.is_empty() && strokes.is_empty() {
        return Err("no polygons on the selected GDS layers and no usable drawn shapes".into());
    }
    let mut local = if polygons.is_empty() {
        Vec::new()
    } else {
        plan(key.mode, &polygons, size, key.pitch_nm)?
    };
    append_strokes(&mut local, &strokes);
    Ok(local
        .into_iter()
        .map(|s| Segment {
            start: rigid.transform_point2(s.start),
            end: rigid.transform_point2(s.end),
            ..s
        })
        .collect())
}

impl LithographyState {
    /// The drawn-shape list, add buttons and the selected shape's editor.
    fn show_shapes(&mut self, ui: &mut Ui, area: Option<&ScanArea>) {
        ui.label(RichText::new("Drawn shapes").strong());
        // New shapes go in the middle of the scan frame, sized to it.
        let (c, size) = area
            .and_then(|a| Some(a.world_transform * a.image_transform?))
            .map_or((DVec2::ZERO, 100.), |t| (t.translation, t.matrix2.x_axis.length().max(1e-3)));
        let mut add = None;
        ui.horizontal_wrapped(|ui| {
            for (label, make) in [
                ("+ Rectangle", Shape::new_rect as fn(DVec2, f64) -> Shape),
                ("+ Ellipse", Shape::new_ellipse),
                ("+ Arc", Shape::new_arc),
                ("+ Line", Shape::new_line),
                ("+ Curve", Shape::new_curve),
                ("+ Polygon", Shape::new_polygon),
            ] {
                if ui.button(label).clicked() {
                    add = Some(make(c, size));
                }
            }
        });
        if let Some(mut s) = add {
            s.name = format!("{} {}", s.kind.label(), self.shapes.len() + 1);
            self.shapes.push(s);
            self.selected_shape = Some(self.shapes.len() - 1);
            self.edit_on_map = true;
        }

        let (mut remove, mut dup) = (None, None);
        if self.shapes.is_empty() {
            ui.weak("No shapes. Add one above, then ctrl+drag it on the map.");
        } else {
            egui::Grid::new("litho shapes").num_columns(4).striped(true).show(ui, |ui| {
                for (i, s) in self.shapes.iter_mut().enumerate() {
                    ui.checkbox(&mut s.enabled, "").on_hover_text("Write this shape");
                    let name = if s.name.is_empty() { s.kind.label() } else { s.name.as_str() };
                    if ui.selectable_label(self.selected_shape == Some(i), format!("{}. {name}", i + 1)).clicked() {
                        self.selected_shape = Some(i);
                    }
                    ui.weak(s.summary());
                    ui.horizontal(|ui| {
                        if ui.small_button("\u{29c9}").on_hover_text("Duplicate").clicked() {
                            dup = Some(i);
                        }
                        if ui.small_button("\u{1f5d1}").on_hover_text("Delete").clicked() {
                            remove = Some(i);
                        }
                    });
                    ui.end_row();
                }
            });
        }
        if let Some(i) = dup {
            let mut s = self.shapes[i].clone();
            s.name.push_str(" copy");
            s.translate(DVec2::splat(0.05 * size));
            self.shapes.insert(i + 1, s);
            self.selected_shape = Some(i + 1);
        }
        if let Some(i) = remove {
            self.shapes.remove(i);
            self.grab = None;
            self.selected_shape = match self.selected_shape {
                Some(j) if j > i => Some(j - 1),
                Some(j) if j == i => None,
                other => other,
            };
        }

        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.edit_on_map, "\u{2725} Edit on map")
                .on_hover_text("Hold Ctrl on the map to drag shapes and their handles");
            if ui.button("Save shapes\u{2026}").clicked() {
                self.save_shapes();
            }
            if ui.button("Load shapes\u{2026}").clicked() {
                self.load_shapes();
            }
            if ui.add_enabled(!self.shapes.is_empty(), egui::Button::new("Clear")).clicked() {
                self.shapes.clear();
                self.selected_shape = None;
                self.grab = None;
            }
        });
        if self.edit_on_map {
            ui.weak(
                "Ctrl+drag a shape to move it, or its handles to reshape it. Ctrl+click selects a \
                 shape; with a line, curve or polygon selected it adds a point (on an edge: inserts \
                 it there), and ctrl+click on a point removes it.",
            );
        }
        if let Some(s) = self.selected_shape.and_then(|i| self.shapes.get_mut(i)) {
            egui::Frame::group(ui.style()).show(ui, |ui| shape_editor(ui, s));
        }
    }

    fn save_shapes(&mut self) {
        let json = match serde_json::to_string_pretty(&self.shapes) {
            Ok(json) => json,
            Err(e) => return self.set_status(StatusKind::Error, format!("Could not save shapes: {e}")),
        };
        let tx = self.shapes_tx.clone();
        std::thread::spawn(move || {
            let Some(path) = dialog()
                .add_filter("Lithography shapes", &["json"])
                .set_title("Save Shapes")
                .set_file_name("shapes.json")
                .save_file()
            else {
                return;
            };
            if let Err(e) = std::fs::write(&path, json) {
                let _ = tx.send(Err(format!("Could not save {}: {e}", path.display())));
            } else {
                tracing::info!("saved lithography shapes to {}", path.display());
            }
        });
    }

    fn load_shapes(&mut self) {
        let tx = self.shapes_tx.clone();
        std::thread::spawn(move || {
            let Some(path): Option<PathBuf> = dialog()
                .add_filter("Lithography shapes", &["json"])
                .set_title("Load Shapes")
                .pick_file()
            else {
                return;
            };
            let result = std::fs::read_to_string(&path)
                .map_err(|e| format!("Could not read {}: {e}", path.display()))
                .and_then(|s| {
                    serde_json::from_str::<Vec<Shape>>(&s)
                        .map_err(|e| format!("Could not parse {}: {e}", path.display()))
                });
            let _ = tx.send(result);
        });
    }

    /// Ctrl+drag moves a shape or one of its handles; ctrl+click selects,
    /// adds or removes points.
    fn map_input(&mut self, ui: &Ui, ctx: &ScanViewCtx) {
        let resp = &ctx.screen_response;
        if !ui.input(|i| i.modifiers.ctrl) {
            self.grab = None;
            return;
        }
        let to_screen = ctx.world2egui();
        let tol = PICK / to_screen.matrix2.x_axis.length().max(1e-12);
        let Some(pointer) = resp.interact_pointer_pos().or(resp.hover_pos()) else {
            return;
        };
        let ps = DVec2::new(pointer.x as f64, pointer.y as f64);
        let at = to_screen.inverse().transform_point2(ps);
        let sel = self.selected_shape.filter(|&i| i < self.shapes.len());
        let sel_handle = sel.and_then(|i| {
            self.shapes[i]
                .handles()
                .into_iter()
                .map(|(h, p)| (h, to_screen.transform_point2(p).distance(ps)))
                .filter(|(_, d)| *d <= PICK)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(h, _)| h)
        });
        // Topmost (last drawn) shape under the pointer.
        let under = (0..self.shapes.len()).rev().find(|&i| self.shapes[i].hit(at, tol));

        if resp.drag_started_by(egui::PointerButton::Primary) {
            self.grab = match (sel, sel_handle) {
                (Some(i), Some(h)) => Some((i, h)),
                _ => {
                    let body = sel.filter(|&i| self.shapes[i].hit(at, tol)).or(under);
                    self.selected_shape = body.or(self.selected_shape);
                    body.map(|i| (i, Handle::Body))
                }
            };
        }
        if resp.dragged_by(egui::PointerButton::Primary) {
            if let Some((i, h)) = self.grab.filter(|(i, _)| *i < self.shapes.len()) {
                let delta = to_screen.inverse().transform_vector2(DVec2::new(
                    resp.drag_delta().x as f64,
                    resp.drag_delta().y as f64,
                ));
                self.shapes[i].drag(h, at, delta);
            }
        } else {
            self.grab = None;
        }
        if resp.clicked() {
            let polyline = sel.filter(|&i| {
                matches!(self.shapes[i].kind, ShapeKind::Polygon { .. } | ShapeKind::Line { .. })
            });
            match (sel, sel_handle) {
                (Some(i), Some(Handle::Vertex(k))) => {
                    self.shapes[i].remove_point(k);
                }
                _ => match (under.filter(|&j| Some(j) != sel), polyline) {
                    (Some(j), _) => self.selected_shape = Some(j),
                    (None, Some(i)) => self.shapes[i].add_point(at, tol),
                    (None, None) => self.selected_shape = under,
                },
            }
        }
    }

    fn draw_shapes(&self, ui: &Ui, ctx: &ScanViewCtx) {
        let to_screen = ctx.world2egui();
        let pos = |p: DVec2| to_screen.transform_point2(p).to_egui_pos2();
        let painter = ui.painter_at(ctx.rect);
        for (i, s) in self.shapes.iter().enumerate() {
            let selected = self.selected_shape == Some(i);
            let color = match (s.enabled, selected) {
                (_, true) => SHAPE_SELECTED,
                (true, false) => SHAPE_COLOR,
                (false, false) => Color32::GRAY,
            };
            let stroke = Stroke::new(if selected { 2. } else { 1.2 }, color);
            match s.geometry() {
                Some(Geometry::Fill(poly)) => {
                    painter.add(egui::Shape::closed_line(poly.into_iter().map(pos).collect(), stroke));
                }
                Some(Geometry::Stroke(line)) => {
                    painter.add(egui::Shape::line(line.into_iter().map(pos).collect(), stroke));
                }
                None => {}
            }
            // Centre line of a wide line or arc.
            let mut centre = s.clone();
            if let ShapeKind::Line { width, .. } | ShapeKind::Arc { width, .. } = &mut centre.kind {
                if *width > 0. {
                    *width = 0.;
                    if let Some(Geometry::Stroke(line)) = centre.geometry() {
                        let pts: Vec<_> = line.into_iter().map(pos).collect();
                        painter.extend(egui::Shape::dashed_line(&pts, Stroke::new(1., color.gamma_multiply(0.6)), 4., 3.));
                    }
                }
            }
            let handles = s.handles();
            if let Some((_, first)) = handles.first() {
                painter.text(
                    pos(*first) + egui::vec2(6., -6.),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{}", i + 1),
                    egui::FontId::proportional(12.),
                    color,
                );
            }
            if selected && self.map_edit_active() {
                for (h, p) in handles {
                    let at = pos(p);
                    if matches!(h, Handle::Center) {
                        painter.circle_stroke(at, 4., Stroke::new(1.5, color));
                    } else {
                        painter.rect_filled(egui::Rect::from_center_size(at, egui::vec2(6., 6.)), 1., color);
                    }
                }
            }
        }
    }
}

/// Numeric editor for one shape (map nm, degrees counter-clockwise).
fn shape_editor(ui: &mut Ui, s: &mut Shape) {
    fn nm(v: &mut f64) -> DragValue<'_> {
        DragValue::new(v).speed(0.5).max_decimals(2).suffix(" nm")
    }
    fn deg(v: &mut f64) -> DragValue<'_> {
        DragValue::new(v).speed(1.).max_decimals(1).suffix("\u{b0}")
    }
    let xy = |ui: &mut Ui, label: &str, p: &mut [f64; 2]| {
        ui.label(label);
        ui.horizontal(|ui| {
            ui.add(nm(&mut p[0]).prefix("x "));
            ui.add(nm(&mut p[1]).prefix("y "));
        });
        ui.end_row();
    };
    let width_hint = "0 = a single line written along it; more = a filled band raster-written with the pitch";
    egui::Grid::new("litho shape editor").num_columns(2).show(ui, |ui| {
        ui.label("Name");
        ui.text_edit_singleline(&mut s.name);
        ui.end_row();
        match &mut s.kind {
            ShapeKind::Rect { center, size, angle } => {
                xy(ui, "Centre", center);
                ui.label("Size");
                ui.horizontal(|ui| {
                    ui.add(nm(&mut size[0]).range(0.01..=1e7).prefix("w "));
                    ui.add(nm(&mut size[1]).range(0.01..=1e7).prefix("h "));
                });
                ui.end_row();
                ui.label("Angle");
                ui.add(deg(angle));
                ui.end_row();
            }
            ShapeKind::Ellipse { center, radii, angle } => {
                xy(ui, "Centre", center);
                ui.label("Radii");
                ui.horizontal(|ui| {
                    ui.add(nm(&mut radii[0]).range(0.01..=1e7).prefix("x "));
                    ui.add(nm(&mut radii[1]).range(0.01..=1e7).prefix("y "));
                    if ui.small_button("circle").on_hover_text("y radius = x radius").clicked() {
                        radii[1] = radii[0];
                    }
                });
                ui.end_row();
                ui.label("Angle");
                ui.add(deg(angle));
                ui.end_row();
            }
            ShapeKind::Arc { center, radius, start, end, width } => {
                xy(ui, "Centre", center);
                ui.label("Radius");
                ui.add(nm(radius).range(0.01..=1e7));
                ui.end_row();
                ui.label("From / to");
                ui.horizontal(|ui| {
                    ui.add(deg(start));
                    ui.add(deg(end));
                });
                ui.end_row();
                ui.label("Width");
                ui.add(nm(width).range(0.0..=1e7)).on_hover_text(width_hint);
                ui.end_row();
            }
            ShapeKind::Polygon { .. } => {}
            ShapeKind::Line { smooth, width, .. } => {
                ui.label("Shape");
                ui.horizontal(|ui| {
                    ui.radio_value(smooth, false, "straight segments");
                    ui.radio_value(smooth, true, "smooth curve");
                });
                ui.end_row();
                ui.label("Width");
                ui.add(nm(width).range(0.0..=1e7)).on_hover_text(width_hint);
                ui.end_row();
            }
        }
    });
    let mut remove = None;
    if let Some(points) = s.points_mut() {
        egui::Grid::new("litho shape points").num_columns(3).show(ui, |ui| {
            for (k, p) in points.iter_mut().enumerate() {
                ui.label(format!("P{}", k + 1));
                ui.horizontal(|ui| {
                    ui.add(nm(&mut p[0]).prefix("x "));
                    ui.add(nm(&mut p[1]).prefix("y "));
                });
                if ui.small_button("\u{1f5d1}").on_hover_text("Remove this point").clicked() {
                    remove = Some(k);
                }
                ui.end_row();
            }
        });
        if ui.small_button("+ Point").on_hover_text("Add a point past the last one").clicked() {
            let n = points.len();
            let next = match n {
                0 => [0., 0.],
                1 => [points[0][0] + 10., points[0][1]],
                _ => [2. * points[n - 1][0] - points[n - 2][0], 2. * points[n - 1][1] - points[n - 2][1]],
            };
            points.push(next);
        }
    }
    if let Some(k) = remove {
        s.remove_point(k);
    }
}
