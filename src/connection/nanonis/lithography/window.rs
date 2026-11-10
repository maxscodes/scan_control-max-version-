use std::collections::BTreeSet;

use crossbeam::channel::{Receiver, Sender};
use egui::{Color32, DragValue, Id, RichText, Stroke, Ui};
use glam::{DAffine2, DVec2};
use uuid::Uuid;

use crate::{
    components::selectable_list::SelectableList,
    connection::{
        nanonis::{
            lithography::{
                path::{plan, PathMode, PathStats, Polygon, Segment},
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

/// Everything a planned path depends on; a change makes the plan stale.
#[derive(Debug, Clone, PartialEq)]
struct PlanKey {
    gds: Uuid,
    gds_transform: [f64; 6],
    area_transform: [f64; 6],
    frame_transform: [f64; 6],
    layers: Vec<u16>,
    mode: PathMode,
    pitch_nm: f64,
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
}

impl LithographyState {
    pub fn new(cmd_tx: Sender<LithoCommand>, events: Receiver<LithoEvent>) -> Self {
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
        }
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
    }

    pub fn show_menu(&mut self, ui: &mut Ui, object_list: &SelectableList<Object>, blocked: Option<&str>) {
        self.poll_events();
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

    fn current_key(&self, gds: &GDSImage, area: &ScanArea) -> Option<PlanKey> {
        Some(PlanKey {
            gds: gds.uuid(),
            gds_transform: gds.transform.to_cols_array(),
            area_transform: area.world_transform.to_cols_array(),
            frame_transform: area.image_transform?.to_cols_array(),
            layers: gds
                .polygons()
                .keys()
                .copied()
                .filter(|l| !self.excluded_layers.contains(l))
                .collect(),
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
        let gds = self.gds.and_then(|id| gds_list.iter().find(|g| g.0 == id)).map(|g| g.2);

        ui.add_enabled_ui(editable, |ui| {
            ui.label(RichText::new("Geometry").strong());
            if gds_list.is_empty() {
                ui.weak("Import a .gds file onto the map, then drag it into alignment.");
            } else {
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
                if let Some(gds) = gds {
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

        let key = gds.zip(area).and_then(|(g, a)| self.current_key(g, a));
        ui.horizontal(|ui| {
            if ui
                .add_enabled(editable && key.is_some(), egui::Button::new("Preview path"))
                .clicked()
            {
                self.preview(gds.unwrap(), area.unwrap(), key.clone().unwrap());
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
                    "⚠ The GDS, frame, layers or path settings changed: preview again",
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

    fn preview(&mut self, gds: &GDSImage, area: &ScanArea, key: PlanKey) {
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

    /// Draw the planned path on the map (active solid red, travel yellow).
    pub fn show_overlay(&self, ui: &mut Ui, area: &ScanArea) {
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

/// Plan the path for `key`: GDS polygons (map world) → piezo nm → frame-local
/// nm, plan, and convert the segments back to piezo nm.
fn build_path(gds: &GDSImage, area: &ScanArea, key: &PlanKey) -> Result<Vec<Segment>, String> {
    let frame_tf = DAffine2::from_cols_array(&key.frame_transform);
    let frame = Frame::from_transform(frame_tf);
    // Frame-local axes follow the frame as drawn on the map (Nanonis angle,
    // positive clockwise).
    let rigid = DAffine2::from_angle_translation(-frame.angle.to_radians(), DVec2::new(frame.center_x, frame.center_y));
    let to_local = rigid.inverse() * area.world_transform.inverse() * gds.transform;
    let polygons: Vec<Polygon> = gds
        .polygons()
        .iter()
        .filter(|(layer, _)| key.layers.contains(layer))
        .flat_map(|(_, polys)| polys)
        .filter(|poly| poly.len() >= 3)
        .map(|poly| poly.iter().map(|v| to_local.transform_point2(v.as_dvec2())).collect())
        .collect();
    if polygons.is_empty() {
        return Err("no polygons on the selected GDS layers".into());
    }
    let local = plan(key.mode, &polygons, DVec2::new(frame.width, frame.height), key.pitch_nm)?;
    Ok(local
        .into_iter()
        .map(|s| Segment {
            start: rigid.transform_point2(s.start),
            end: rigid.transform_point2(s.end),
            ..s
        })
        .collect())
}
