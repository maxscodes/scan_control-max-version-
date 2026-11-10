//! Ring Assist: runs the ring-centre estimator on live frames and draws what
//! it would do on the map. Display only. It never moves the tip.
//!
//! It also keeps the last frame that gave a confident verdict. When a frame is
//! unsure, the planned fallback (for when movement is added) is to move half
//! way back toward that frame and try again. The overlay draws that target.

use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use egui::{Align2, Color32, FontId, Id, RichText, Stroke, Ui};
use glam::{DAffine2, DVec2};

use crate::{
    analysis::{
        ring_center::{self, RingParams, Verdict},
        ring_stamp::{self, RingStamp, ARC_COLOR, ARM_COLOR, UNSURE_COLOR},
    },
    connection::ScanArea,
    scan_view::{BorderRectangle, ScanViewCtx},
    utils::vec_interop::IntoEgui as _,
};

/// Don't analyse more often than this while a scan is streaming in.
const MIN_INTERVAL: Duration = Duration::from_millis(1000);
/// A frame counts as complete when at least this fraction of pixels is valid.
const COMPLETE_FRACTION: f64 = 0.98;
/// Below this fraction of valid pixels there's too little to analyse.
const MIN_VALID_FRACTION: f64 = 0.3;
/// Completed frames kept for the trail on the map.
const HISTORY_LEN: usize = 12;
const DEFAULT_MIN_RADIUS_UM: f64 = 1.5;

const GOOD_COLOR: Color32 = Color32::from_rgb(0x9a, 0xe6, 0xa8);
const FEATURE_COLOR: Color32 = Color32::from_rgb(0x00, 0xc8, 0xff);

#[derive(Clone)]
pub struct Reading {
    at: Instant,
    /// Quad (−0.5..0.5) to world (nm) transform of the analysed frame.
    frame: DAffine2,
    valid_fraction: f64,
    verdict: Verdict,
    inlier_fraction: f64,
    radius_um: f64,
    /// Fitted ring centre (world), for arc verdicts.
    center_world: Option<DVec2>,
    /// Unit move direction (world), if the verdict has one.
    move_dir_world: Option<DVec2>,
    /// Peeled local features: centre and radius (world, nm).
    features_world: Vec<(DVec2, f64)>,
    /// Edge pixels of the final fit (world), for the debug view.
    edges_world: Vec<DVec2>,
}

impl Reading {
    fn complete(&self) -> bool {
        self.valid_fraction >= COMPLETE_FRACTION
    }
    fn good(&self) -> bool {
        !matches!(self.verdict, Verdict::Unsure(_))
    }
    fn frame_center(&self) -> DVec2 {
        self.frame.translation
    }
    fn frame_width(&self) -> f64 {
        self.frame.matrix2.x_axis.length()
    }
    fn summary(&self) -> String {
        match self.verdict {
            Verdict::Arc { .. } => "On ring arcs: follow curvature".into(),
            Verdict::ArmEnd { dir } => format!(
                "At arm ends: move sideways ({})",
                if dir.x < 0. { "left" } else { "right" }
            ),
            Verdict::Unsure(r) => format!("Unsure: {}", r.describe()),
        }
    }
    fn color(&self) -> Color32 {
        match self.verdict {
            Verdict::Arc { .. } => ARC_COLOR,
            Verdict::ArmEnd { .. } => ARM_COLOR,
            Verdict::Unsure(_) => UNSURE_COLOR,
        }
    }
}

pub struct RingAssistState {
    window_open: bool,
    enabled: bool,
    show_debug: bool,
    /// Analyse each stamped image and save the result with it.
    save_with_stamps: bool,
    params: RingParams,
    min_radius_um: f64,
    last_run: Option<Instant>,
    /// New frame data has arrived but hasn't been analysed yet.
    pending: bool,
    /// Data of the last analysed frame, to skip re-analysing identical frames.
    last_data: Option<Arc<Box<[f32]>>>,
    channel_warning: Option<String>,
    current: Option<Reading>,
    last_good: Option<Reading>,
    history: VecDeque<Reading>,
}

impl RingAssistState {
    pub fn new() -> Self {
        Self {
            window_open: false,
            enabled: false,
            show_debug: false,
            params: RingParams::default(),
            min_radius_um: DEFAULT_MIN_RADIUS_UM,
            last_run: None,
            pending: false,
            last_data: None,
            channel_warning: None,
            current: None,
            last_good: None,
            history: VecDeque::new(),
            save_with_stamps: true,
        }
    }

    /// Ring result to save with a stamped image, if Ring Assist is on, set to
    /// save with stamps, and the stamped channel is Z.
    pub fn stamp_result(
        &self,
        channel: Option<&str>,
        data: &[f32],
        size: [u32; 2],
        frame: DAffine2,
    ) -> Option<RingStamp> {
        if !self.enabled || !self.save_with_stamps {
            return None;
        }
        if !channel.is_some_and(|c| c.trim_start().starts_with('Z')) {
            return None;
        }
        ring_stamp::analyse_frame(data, size, frame, &self.params, self.min_radius_um)
    }

    /// Call once per UI update after the live image data was refreshed.
    pub fn update(&mut self, scan_area: &ScanArea, got_forward: bool) {
        if !self.enabled {
            return;
        }
        self.pending |= got_forward;
        if !self.pending || self.last_run.is_some_and(|t| t.elapsed() < MIN_INTERVAL) {
            return;
        }
        self.pending = false;
        self.last_run = Some(Instant::now());

        // The live image holds whichever channel is displayed; we need height.
        match &scan_area.channel_selected {
            Some(ch) if ch.trim_start().starts_with('Z') => self.channel_warning = None,
            Some(ch) => {
                self.channel_warning = Some(format!("Displayed channel is \"{ch}\"; select Z."));
                return;
            }
            None => {
                self.channel_warning = Some("No channel displayed; select Z.".into());
                return;
            }
        }
        let Some(image_transform) = scan_area.image_transform else {
            return;
        };
        let data = &scan_area.live_image.forward_data;
        if self
            .last_data
            .as_ref()
            .is_some_and(|d| Arc::ptr_eq(d, &data.data) || d[..] == data.data[..])
        {
            return;
        }
        let frame = scan_area.world_transform * image_transform;
        if let Some(reading) = self.analyse(&data.data, data.size, frame) {
            self.last_data = Some(data.data.clone());
            if reading.complete() && reading.good() {
                self.last_good = Some(reading.clone());
            }
            if reading.complete() {
                self.history.push_front(reading.clone());
                self.history.truncate(HISTORY_LEN);
            }
            self.current = Some(reading);
        }
    }

    fn analyse(&self, data: &[f32], size: [u32; 2], frame: DAffine2) -> Option<Reading> {
        let (w, h) = (size[0] as usize, size[1] as usize);
        if w < 8 || h < 8 || data.len() != w * h {
            return None;
        }
        let valid_fraction = data.iter().filter(|v| v.is_finite()).count() as f64 / data.len() as f64;
        if valid_fraction < MIN_VALID_FRACTION {
            return None;
        }
        let nm_per_px = frame.matrix2.x_axis.length() / w as f64;
        let mut params = self.params;
        params.min_ring_radius_px = Some(self.min_radius_um * 1e3 / nm_per_px);

        let pre = ring_center::preprocess(data, w, h, &params);
        let res = ring_center::estimate_preprocessed(&pre.img, pre.width, pre.height, pre.scale, &params);
        let verdict = ring_center::decide(&res, pre.scale, &params);

        // Analysed pixel -> input pixel -> quad -> world.
        let s = pre.scale;
        let to_world = |p: DVec2| {
            let input = (p + 0.5) * s - 0.5;
            let quad = DVec2::new((input.x + 0.5) / w as f64 - 0.5, (input.y + 0.5) / h as f64 - 0.5);
            frame.transform_point2(quad)
        };
        let dir_to_world = |d: DVec2| {
            frame
                .transform_vector2(DVec2::new(d.x / w as f64, d.y / h as f64))
                .normalize_or_zero()
        };
        let nm_per_analysed_px = nm_per_px * s;

        let (inlier_fraction, radius_um, center_world, features_world, edges_world) = match &res {
            Ok(a) => (
                a.est.inlier_fraction,
                a.est.radius_px * nm_per_analysed_px * 1e-3,
                matches!(verdict, Verdict::Arc { .. }).then(|| to_world(a.est.center_px)),
                a.features
                    .iter()
                    .map(|f| (to_world(f.center_px), f.radius_px * nm_per_analysed_px))
                    .collect(),
                a.edges.iter().map(|e| to_world(e.0)).collect(),
            ),
            Err(_) => (0., 0., None, vec![], vec![]),
        };
        let move_dir_world = match verdict {
            Verdict::Arc { dir } | Verdict::ArmEnd { dir } => Some(dir_to_world(dir)),
            Verdict::Unsure(_) => None,
        };
        Some(Reading {
            at: Instant::now(),
            frame,
            valid_fraction,
            verdict,
            inlier_fraction,
            radius_um,
            center_world,
            move_dir_world,
            features_world,
            edges_world,
        })
    }

    /// Where the fallback would move: half way from the current frame back
    /// toward the last good one. Only when the current complete frame is unsure.
    fn fallback_target(&self) -> Option<(DVec2, DVec2)> {
        let cur = self.current.as_ref()?;
        let good = self.last_good.as_ref()?;
        if cur.good() || !cur.complete() {
            return None;
        }
        let from = cur.frame_center();
        let to = (from + good.frame_center()) / 2.;
        ((to - from).length() > 1.).then_some((from, to))
    }

    pub fn show_menu(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.window_open, egui::Button::new("Ring Assist"))
                .clicked()
            {
                self.window_open = true;
            }
            if self.enabled {
                if let Some(cur) = &self.current {
                    ui.label(RichText::new(cur.summary()).color(cur.color()));
                }
            }
        });
        if self.enabled && self.pending {
            ui.ctx().request_repaint_after(MIN_INTERVAL);
        }
        let mut open = self.window_open;
        egui::Window::new("Ring Assist")
            .default_width(320.)
            .resizable(true)
            .collapsible(true)
            .open(&mut open)
            .show(ui.ctx(), |ui| self.show_window(ui));
        self.window_open = open;
    }

    fn show_window(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new(
                "Analyses each Z frame and shows where it would move. Display only: it never \
                 moves the tip.",
            )
            .weak(),
        );
        if ui.checkbox(&mut self.enabled, "Analyse live frames").changed() && self.enabled {
            self.rerun();
        }
        if let Some(w) = &self.channel_warning {
            ui.label(RichText::new(w).color(UNSURE_COLOR));
        }
        ui.separator();

        match &self.current {
            None => {
                ui.label(RichText::new("No frame analysed yet.").weak());
            }
            Some(cur) => {
                ui.label(RichText::new(cur.summary()).color(cur.color()).strong());
                egui::Grid::new("ring_assist_current").num_columns(2).show(ui, |ui| {
                    ui.label("Frame");
                    ui.label(if cur.complete() {
                        "complete".to_string()
                    } else {
                        format!("partial ({:.0}% scanned)", cur.valid_fraction * 100.)
                    });
                    ui.end_row();
                    ui.label("Ring-likeness (inliers)");
                    ui.label(format!(
                        "{:.2}  (unsure below {:.2})",
                        cur.inlier_fraction, self.params.min_inlier_fraction
                    ));
                    ui.end_row();
                    ui.label("Fitted radius");
                    ui.label(format!("{:.1} µm", cur.radius_um));
                    ui.end_row();
                    ui.label("Local features removed");
                    ui.label(cur.features_world.len().to_string());
                    ui.end_row();
                    ui.label("Analysed");
                    ui.label(format!("{:.0} s ago", cur.at.elapsed().as_secs_f64()));
                    ui.end_row();
                });
            }
        }
        ui.separator();
        match &self.last_good {
            None => {
                ui.label(RichText::new("No confident complete frame yet.").weak());
            }
            Some(good) => {
                ui.label(format!(
                    "Last good frame: {} ({:.0} s ago)",
                    good.summary(),
                    good.at.elapsed().as_secs_f64()
                ));
            }
        }
        if let Some((from, to)) = self.fallback_target() {
            ui.label(
                RichText::new(format!(
                    "Fallback: move {:.2} µm half way back toward the last good frame.",
                    (to - from).length() * 1e-3
                ))
                .color(GOOD_COLOR),
            );
        }
        ui.separator();
        ui.checkbox(&mut self.show_debug, "Show edges and removed features");
        ui.checkbox(&mut self.save_with_stamps, "Save result with stamps")
            .on_hover_text("Each stamped image keeps its ring result in the project. Hide them with \"Show ring results on stamps\" in the Scan Region menu.");
        if ui.button("Clear trail").clicked() {
            self.history.clear();
            self.last_good = None;
        }

        let mut changed = false;
        egui::CollapsingHeader::new("Settings").show(ui, |ui| {
            egui::Grid::new("ring_assist_settings").num_columns(2).show(ui, |ui| {
                ui.label("Min ring radius (µm)");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.min_radius_um).speed(0.05).range(0.1..=20.))
                    .changed();
                ui.end_row();
                ui.label("Unsure below inliers");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.params.min_inlier_fraction).speed(0.01).range(0.0..=1.))
                    .changed();
                ui.end_row();
                ui.label("Downsample to (px)");
                let mut target = self.params.target_px.unwrap_or(0);
                if ui.add(egui::DragValue::new(&mut target).range(0..=512)).changed() {
                    self.params.target_px = (target > 0).then_some(target);
                    changed = true;
                }
                ui.end_row();
                ui.label("Blur (px)");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.params.blur_sigma).speed(0.1).range(0.0..=10.))
                    .changed();
                ui.end_row();
                ui.label("Edge fraction");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.params.edge_fraction).speed(0.01).range(0.01..=1.))
                    .changed();
                ui.end_row();
            });
            if ui.button("Reset to defaults").clicked() {
                self.params = RingParams::default();
                self.min_radius_um = DEFAULT_MIN_RADIUS_UM;
                changed = true;
            }
        });
        if changed {
            self.rerun();
        }
    }

    /// Re-analyse the current frame on the next update.
    fn rerun(&mut self) {
        self.last_data = None;
        self.last_run = None;
        self.pending = true;
    }

    pub fn show_overlay(&self, ui: &mut Ui, _scan_area: &ScanArea) {
        if !self.enabled {
            return;
        }
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let w2s = ctx.world2egui();
        let to_screen = |p: DVec2| w2s.transform_point2(p).to_egui_pos2();
        let painter = ui.painter().clone();
        let painter = &painter;

        let arrow = |from: DVec2, dir: DVec2, len: f64, color: Color32, width: f32| {
            let a = to_screen(from);
            let b = to_screen(from + dir * len);
            painter.arrow(a, b - a, Stroke::new(width, color));
        };

        // Trail of earlier complete frames.
        for r in self.history.iter().skip(1) {
            if let Some(dir) = r.move_dir_world {
                arrow(r.frame_center(), dir, r.frame_width() * 0.2, r.color().gamma_multiply(0.4), 1.5);
            } else {
                painter.circle_stroke(to_screen(r.frame_center()), 3., Stroke::new(1., UNSURE_COLOR.gamma_multiply(0.4)));
            }
        }

        // Last good frame outline, when it isn't the current one.
        if let Some(good) = &self.last_good {
            let is_current = self.current.as_ref().is_some_and(|c| c.at == good.at);
            if !is_current {
                BorderRectangle { transform: good.frame, color: GOOD_COLOR, dashed: true }.show(ui);
                painter.text(
                    to_screen(good.frame.transform_point2(DVec2::new(-0.5, -0.5))),
                    Align2::LEFT_BOTTOM,
                    "last good",
                    FontId::proportional(12.),
                    GOOD_COLOR,
                );
            }
        }

        let Some(cur) = &self.current else { return };
        let color = if cur.complete() { cur.color() } else { cur.color().gamma_multiply(0.6) };
        let c = cur.frame_center();
        match cur.move_dir_world {
            Some(dir) => arrow(c, dir, cur.frame_width() * 0.4, color, 3.),
            None => {
                let r = 8.;
                let p = to_screen(c);
                let stroke = Stroke::new(3., color);
                painter.line_segment([p + egui::vec2(-r, -r), p + egui::vec2(r, r)], stroke);
                painter.line_segment([p + egui::vec2(-r, r), p + egui::vec2(r, -r)], stroke);
            }
        }
        painter.text(
            to_screen(c) + egui::vec2(10., -10.),
            Align2::LEFT_BOTTOM,
            cur.summary(),
            FontId::proportional(13.),
            color,
        );
        // Ring centre, when it's near enough to be worth drawing.
        if let Some(center) = cur.center_world {
            if (center - c).length() < 3. * cur.frame_width() {
                painter.circle_stroke(to_screen(center), 6., Stroke::new(2., color));
            }
        }
        if let Some((from, to)) = self.fallback_target() {
            let (a, b) = (to_screen(from), to_screen(to));
            painter.extend(egui::Shape::dashed_line(&[a, b], Stroke::new(2., GOOD_COLOR), 6., 4.));
            painter.circle_stroke(b, 5., Stroke::new(2., GOOD_COLOR));
            painter.text(b + egui::vec2(8., 0.), Align2::LEFT_CENTER, "½ back", FontId::proportional(12.), GOOD_COLOR);
        }
        if self.show_debug {
            for &e in &cur.edges_world {
                painter.circle_filled(to_screen(e), 1.5, Color32::from_rgb(0xff, 0x40, 0x40));
            }
            for &(fc, r) in &cur.features_world {
                let sc = to_screen(fc);
                let sr = (to_screen(fc + DVec2::new(r, 0.)) - sc).length();
                painter.circle_stroke(sc, sr, Stroke::new(1.5, FEATURE_COLOR));
            }
        }
    }
}
