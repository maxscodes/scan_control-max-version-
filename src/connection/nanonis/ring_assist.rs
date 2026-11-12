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
        gds_prior::{self, BandCheck, Bands, GdsPrediction, GdsRings},
        ring_center::{self, RingParams, Verdict},
        ring_stamp::{
            self, quad_to_px_dir, RingLabel, RingStamp, StampVerdict, ARC_COLOR, ARM_COLOR,
            LABEL_COLOR, UNSURE_COLOR,
        },
        ring_training::{self, Dataset, GdsInfo, LabelKind, Record, Report, TuneParams},
    },
    components::selectable_list::SelectableList,
    connection::{LiveImage, ScanArea},
    scan_view::{BorderRectangle, ScanViewCtx},
    utils::vec_interop::{IntoEgui as _, IntoGlam as _},
    view_object::Object,
};

/// What a map click sets while picking.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pick {
    /// Corrected direction (the stamp's arrow was wrong).
    Correct,
    /// Optional direction from the smallest ring into the middle.
    Middle,
}

/// Retrain automatically after this many new labels...
const RETRAIN_EVERY: usize = 5;
/// ...once the dataset has at least this many labels.
const RETRAIN_MIN_LABELS: usize = 10;

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
use crate::analysis::gds_prior::GDS_COLOR;
const WARN_COLOR: Color32 = Color32::from_rgb(0xff, 0xb3, 0x47);
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
    size: [u32; 2],
    /// Broad band direction measured in the frame.
    bands: Option<Bands>,
    /// What the GDS predicts here (refreshed when the GDS moves).
    gds: Option<GdsPrediction>,
    band_check: Option<BandCheck>,
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
    fn refresh_gds(&mut self, gds: &[GdsRings]) {
        self.gds = gds_prior::predict_any(gds, self.frame).map(|(_, p)| p);
        self.band_check = self.gds.map(|p| gds_prior::check(self.bands, self.size, self.frame, &p));
    }
}

/// One line on how the GDS prediction and the image analysis compare.
fn image_vs_gds(move_dir_world: Option<DVec2>, gds: &GdsPrediction) -> Option<(String, Color32)> {
    let d = move_dir_world?;
    let a = d.dot(gds.dir_world).clamp(-1., 1.).acos().to_degrees();
    Some(if a <= 45. {
        (format!("Image analysis agrees ({a:.0}\u{b0})"), ARC_COLOR)
    } else {
        (format!("Image analysis points {a:.0}\u{b0} away from the GDS"), WARN_COLOR)
    })
}

fn band_check_color(c: &BandCheck) -> Color32 {
    match c {
        BandCheck::Agree(_) => ARC_COLOR,
        BandCheck::Disagree(_) => WARN_COLOR,
        BandCheck::NoBands => Color32::GRAY,
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
    /// Labelled frames shared by all projects.
    dataset: Dataset,
    /// Stamp waiting for a map click: a corrected direction, or the
    /// direction to the middle for a smallest-ring label.
    picking: Option<(uuid::Uuid, Pick)>,
    labels_since_train: usize,
    training: Option<crossbeam::channel::Receiver<Report>>,
    report: Option<Report>,
    label_error: Option<String>,
    /// Analysis of the selected stamp, so the Teach panel doesn't re-run it
    /// on every repaint. Cleared when the settings change.
    teach_cache: Option<(uuid::Uuid, Option<RingStamp>)>,
    /// GDS placement and labelled stamps the dataset's GDS predictions were
    /// last brought up to date for.
    gds_sync_key: Vec<(uuid::Uuid, [f64; 6])>,
    show_gds: bool,
    /// Live frame being analysed on a background thread.
    job: Option<crossbeam::channel::Receiver<(Arc<Box<[f32]>>, Option<Reading>)>>,
}

impl RingAssistState {
    pub fn new() -> Self {
        // Settings accepted from earlier retraining, or the defaults.
        let tune = TuneParams::load();
        Self {
            window_open: false,
            enabled: false,
            show_debug: false,
            params: tune.params,
            min_radius_um: tune.min_radius_um,
            dataset: Dataset::load(Dataset::default_dir()),
            picking: None,
            labels_since_train: 0,
            training: None,
            report: None,
            label_error: None,
            teach_cache: None,
            last_run: None,
            pending: false,
            last_data: None,
            channel_warning: None,
            current: None,
            last_good: None,
            history: VecDeque::new(),
            save_with_stamps: true,
            gds_sync_key: Vec::new(),
            show_gds: true,
            job: None,
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
    pub fn update(&mut self, scan_area: &ScanArea, got_forward: bool, gds: &[GdsRings]) {
        if !self.enabled {
            return;
        }
        // GDS predictions are cheap: keep them current if the GDS is moved.
        for r in self
            .current
            .iter_mut()
            .chain(self.last_good.iter_mut())
            .chain(self.history.iter_mut())
        {
            r.refresh_gds(gds);
        }
        self.poll_job(gds);
        self.pending |= got_forward;
        if self.job.is_some()
            || !self.pending
            || self.last_run.is_some_and(|t| t.elapsed() < MIN_INTERVAL)
        {
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
        // Analyse off the UI thread; the result is picked up by `poll_job`.
        let (tx, rx) = crossbeam::channel::bounded(1);
        let (data, size) = (data.data.clone(), data.size);
        let (params, min_radius_um) = (self.params, self.min_radius_um);
        std::thread::spawn(move || {
            let reading = Self::analyse(&data, size, frame, params, min_radius_um);
            let _ = tx.send((data, reading));
            crate::app::request_repaint();
        });
        self.job = Some(rx);
    }

    /// Take a finished background analysis, if any.
    fn poll_job(&mut self, gds: &[GdsRings]) {
        let Some(rx) = &self.job else { return };
        let (data, reading) = match rx.try_recv() {
            Ok(r) => r,
            Err(crossbeam::channel::TryRecvError::Empty) => return,
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                self.job = None;
                return;
            }
        };
        self.job = None;
        let Some(mut reading) = reading else { return };
        reading.refresh_gds(gds);
        self.last_data = Some(data);
        if reading.complete() && reading.good() {
            self.last_good = Some(reading.clone());
        }
        if reading.complete() {
            self.history.push_front(reading.clone());
            self.history.truncate(HISTORY_LEN);
        }
        self.current = Some(reading);
    }

    fn analyse(
        data: &[f32],
        size: [u32; 2],
        frame: DAffine2,
        params: RingParams,
        min_radius_um: f64,
    ) -> Option<Reading> {
        let (w, h) = (size[0] as usize, size[1] as usize);
        if w < 8 || h < 8 || data.len() != w * h {
            return None;
        }
        let valid_fraction = data.iter().filter(|v| v.is_finite()).count() as f64 / data.len() as f64;
        if valid_fraction < MIN_VALID_FRACTION {
            return None;
        }
        let nm_per_px = frame.matrix2.x_axis.length() / w as f64;
        let mut params = params;
        params.min_ring_radius_px = Some(min_radius_um * 1e3 / nm_per_px);

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
        let bands = gds_prior::band_orientation(data, size, nm_per_px);
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
            size,
            bands,
            gds: None,
            band_check: None,
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

    pub fn show_menu(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>) {
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
            .frame(crate::theme::window_frame(ui.ctx()))
            .default_width(320.)
            .resizable(true)
            .collapsible(true)
            .open(&mut open)
            .show(ui.ctx(), |ui| self.show_window(ui, object_list));
        self.window_open = open;
    }

    fn show_window(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>) {
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
        self.show_gds_section(ui);
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
        ui.checkbox(&mut self.show_gds, "Show GDS prediction on the map");
        ui.checkbox(&mut self.save_with_stamps, "Save result with stamps")
            .on_hover_text("Each stamped image keeps its ring result in the project. Hide them with \"Show ring results on stamps\" in the Scan Region menu.");
        if ui.button("Clear trail").clicked() {
            self.history.clear();
            self.last_good = None;
        }

        let gds = gds_prior::collect(object_list);
        self.sync_dataset_gds(object_list, &gds);
        self.show_teach(ui, object_list, &gds);
        self.show_training(ui);

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
            self.save_tune();
            self.rerun();
        }
    }

    /// The GDS prediction for the current frame, next to the image analysis.
    fn show_gds_section(&self, ui: &mut Ui) {
        let Some(cur) = &self.current else { return };
        ui.separator();
        ui.label(RichText::new("GDS").strong());
        let Some(p) = &cur.gds else {
            ui.label(
                RichText::new("Not on a GDS ring pattern (load and line up the GDS to use it).")
                    .weak(),
            );
            return;
        };
        ui.label(
            RichText::new(format!(
                "Ring centre {:.1} \u{b5}m away (estimate from where the GDS sits on the map)",
                p.dist_nm * 1e-3
            ))
            .color(GDS_COLOR),
        );
        if p.smallest_in_frame {
            ui.label(RichText::new("The smallest ring should be in this frame.").color(GDS_COLOR));
        }
        if let Some(c) = &cur.band_check {
            let mut text = c.describe();
            if matches!(c, BandCheck::Disagree(_)) {
                text += ": check the GDS alignment";
            }
            ui.label(RichText::new(text).color(band_check_color(c)));
        }
        if let Some((t, c)) = image_vs_gds(cur.move_dir_world, p) {
            ui.label(RichText::new(t).color(c));
        }
    }

    /// Bring the dataset's stored GDS predictions up to date for the labelled
    /// stamps in this project, when the GDS or the labels change.
    fn sync_dataset_gds(&mut self, object_list: &SelectableList<Object>, gds: &[GdsRings]) {
        let mut key: Vec<(uuid::Uuid, [f64; 6])> =
            gds.iter().map(|g| (g.uuid, g.transform.to_cols_array())).collect();
        let mut labelled = Vec::new();
        for o in object_list.iter() {
            if let Object::ScanImage { image, .. } = &**o {
                if image.ring_label.is_some() {
                    labelled.push(image);
                    key.push((image.uuid(), image.transform.to_cols_array()));
                }
            }
        }
        if key == self.gds_sync_key {
            return;
        }
        self.gds_sync_key = key;
        if gds.is_empty() {
            return;
        }
        let updates: Vec<(String, Option<GdsInfo>)> = labelled
            .into_iter()
            .filter(|image| self.dataset.gds_of(&image.uuid().to_string()).is_some())
            .map(|image| (image.uuid().to_string(), gds_info(gds, image)))
            .collect();
        if let Err(e) = self.dataset.update_gds(&updates) {
            self.label_error = Some(format!("Could not update GDS predictions: {e}"));
        }
    }

    fn tune(&self) -> TuneParams {
        TuneParams { params: self.params, min_radius_um: self.min_radius_um }
    }

    fn save_tune(&self) {
        if let Err(e) = self.tune().save() {
            tracing::error!("could not save Ring Assist settings: {e}");
        }
    }

    /// Label the selected stamp: Good / Correct (click on map) / Should stop.
    fn show_teach(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<Object>,
        gds: &[GdsRings],
    ) {
        ui.separator();
        ui.label(RichText::new("Teach").strong());
        let (g, c, s, m) = self.dataset.counts();
        ui.label(format!(
            "{} labels ({g} good, {c} corrected, {s} stop, {m} smallest ring)",
            self.dataset.len()
        ));
        let selected = object_list
            .iter_selected_indexes()
            .find(|&i| matches!(&*object_list[i], Object::ScanImage { .. }));
        let Some(i) = selected else {
            ui.label(RichText::new("Select a stamped Z image to label it.").weak());
            return;
        };
        let Object::ScanImage { image, name, .. } = &mut *object_list[i] else {
            return;
        };
        if image.unit != "m" {
            ui.label(RichText::new("Only Z images can be labelled.").weak());
            return;
        }
        let name = name.clone();
        // The stamp's saved result, or analyse it now with the current settings.
        let model = match image.ring {
            Some(r) => Some(r),
            None => match self.teach_cache {
                Some((id, r)) if id == image.uuid() => r,
                _ => {
                    let r = self.analyse_stamp(image);
                    self.teach_cache = Some((image.uuid(), r));
                    r
                }
            },
        };
        match &model {
            Some(m) => ui.label(
                RichText::new(format!("{name}: {}", m.summary())).color(m.color()),
            ),
            None => ui.label(format!("{name}: no result")),
        };
        if let Some(p) = gds_prior::stamp_prior(gds, image) {
            ui.label(
                RichText::new(format!(
                    "GDS: ring centre {:.1} \u{b5}m away, {}",
                    p.pred.dist_nm * 1e-3,
                    p.check.describe()
                ))
                .color(GDS_COLOR),
            );
        }
        if let Some(l) = &image.ring_label {
            ui.label(format!("Your label: {}", l.describe()));
        }
        let good_dir = model
            .filter(|m| !matches!(m.verdict, StampVerdict::Unsure(_)))
            .map(|m| m.dir_quad);
        let picking = self.picking == Some((image.uuid(), Pick::Correct));
        let mut label = None;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(good_dir.is_some(), egui::Button::new("\u{2714} Good"))
                .on_hover_text("The arrow points the right way")
                .on_disabled_hover_text("The model was unsure here: use Correct or Should stop")
                .clicked()
            {
                label = good_dir.map(RingLabel::Good);
            }
            if ui
                .selectable_label(picking, "Correct\u{2026}")
                .on_hover_text("Then click on the map where the tip should go")
                .clicked()
            {
                self.picking = if picking { None } else { Some((image.uuid(), Pick::Correct)) };
            }
            if ui
                .button("\u{25A0} Should stop")
                .on_hover_text("No move wanted here (centre, electrodes)")
                .clicked()
            {
                label = Some(RingLabel::Stop);
            }
        });
        let mut smallest = false;
        if ui
            .button("◎ At smallest ring")
            .on_hover_text(
                "The smallest middle ring is in this frame (stage 1 reached). \
                 Then optionally click toward the middle.",
            )
            .clicked()
        {
            smallest = true;
        }
        match self.picking {
            Some((id, Pick::Correct)) if id == image.uuid() => {
                ui.label(
                    RichText::new("Click on the map where the tip should go (Esc cancels).")
                        .color(LABEL_COLOR),
                );
            }
            Some((id, Pick::Middle)) if id == image.uuid() => {
                ui.label(
                    RichText::new("Click toward the middle (optional, Esc to skip).")
                        .color(LABEL_COLOR),
                );
            }
            _ => {}
        }
        let g = gds_info(gds, image);
        if let Some(l) = label {
            self.picking = None;
            self.apply_label(image, &name, l, g, ui.ctx());
        }
        if smallest {
            // Saved now without a direction; a map click can add one.
            self.apply_label(image, &name, RingLabel::SmallestRing(None), g, ui.ctx());
            self.picking = Some((image.uuid(), Pick::Middle));
        }
        if let Some(e) = &self.label_error {
            ui.label(RichText::new(e).color(UNSURE_COLOR));
        }
    }

    fn analyse_stamp(&self, image: &LiveImage) -> Option<RingStamp> {
        ring_stamp::analyse_frame(
            &image.forward_data.data,
            image.forward_data.size,
            image.transform,
            &self.params,
            self.min_radius_um,
        )
    }

    fn apply_label(
        &mut self,
        image: &mut LiveImage,
        name: &str,
        label: RingLabel,
        gds: Option<GdsInfo>,
        ctx: &egui::Context,
    ) {
        let size = image.forward_data.size;
        if size[0] == 0 || size[1] == 0 {
            self.label_error = Some("This image has no data.".into());
            return;
        }
        let nm_per_px = image.transform.matrix2.x_axis.length() / size[0] as f64;
        let (kind, dir) = match label {
            RingLabel::Good(d) => (LabelKind::Good, quad_to_px_dir(d, size)),
            RingLabel::Direction(d) => (LabelKind::Direction, quad_to_px_dir(d, size)),
            RingLabel::Stop => (LabelKind::Stop, DVec2::ZERO),
            RingLabel::SmallestRing(d) => (
                LabelKind::SmallestRing,
                d.map_or(DVec2::ZERO, |d| quad_to_px_dir(d, size)),
            ),
        };
        // Adding the direction to an existing smallest-ring label is the same
        // label, not a new one, for the retrain counter.
        let refinement = matches!(
            (image.ring_label, label),
            (Some(RingLabel::SmallestRing(_)), RingLabel::SmallestRing(Some(_)))
        );
        let record = Record {
            id: image.uuid().to_string(),
            kind,
            dir: dir.to_array(),
            w: size[0],
            h: size[1],
            nm_per_px,
            source: name.to_string(),
            time: format!("{:?}", time::UtcDateTime::now()),
            gds,
        };
        match self.dataset.add(record, &image.forward_data.data) {
            Ok(()) => {
                image.ring_label = Some(label);
                self.label_error = None;
                if !refinement {
                    self.labels_since_train += 1;
                }
                if self.training.is_none()
                    && self.labels_since_train >= RETRAIN_EVERY
                    && self.dataset.len() >= RETRAIN_MIN_LABELS
                {
                    self.start_training(ctx);
                }
            }
            Err(e) => self.label_error = Some(format!("Could not save label: {e}")),
        }
    }

    fn start_training(&mut self, ctx: &egui::Context) {
        let dir = self.dataset.dir().to_path_buf();
        let current = self.tune();
        let (tx, rx) = crossbeam::channel::bounded(1);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let samples = Dataset::load(dir).samples();
            let report = ring_training::evaluate(&samples, current);
            let _ = tx.send(report);
            ctx.request_repaint();
        });
        self.training = Some(rx);
        self.labels_since_train = 0;
        self.report = None;
    }

    /// Retraining status, and the before/after report with Accept/Dismiss.
    fn show_training(&mut self, ui: &mut Ui) {
        if let Some(rx) = &self.training {
            if let Ok(report) = rx.try_recv() {
                self.report = Some(report);
                self.training = None;
            }
        }
        ui.horizontal(|ui| {
            if self.training.is_some() {
                ui.spinner();
                ui.label("Retraining\u{2026}");
            } else if ui
                .add_enabled(self.dataset.len() >= 3, egui::Button::new("Retrain now"))
                .on_hover_text(format!(
                    "Also runs by itself after every {RETRAIN_EVERY} new labels \
                     (once there are {RETRAIN_MIN_LABELS})"
                ))
                .clicked()
            {
                self.start_training(ui.ctx());
            }
        });
        let Some(r) = &self.report else { return };
        let pct = |n: usize| if r.n > 0 { 100. * n as f64 / r.n as f64 } else { 0. };
        ui.label(format!(
            "Current settings: {}/{} right ({:.0}%)",
            r.current_correct,
            r.n,
            pct(r.current_correct)
        ));
        ui.label(format!(
            "New settings: {}/{} on held-out scans ({:.0}%), {}/{} on all",
            r.cv_correct,
            r.n,
            pct(r.cv_correct),
            r.new_correct,
            r.n
        ));
        if r.gds_n > 0 {
            ui.label(
                RichText::new(format!(
                    "GDS prediction: {}/{} right ({:.0}%) on labels with a GDS",
                    r.gds_correct,
                    r.gds_n,
                    100. * r.gds_correct as f64 / r.gds_n as f64
                ))
                .color(GDS_COLOR),
            );
            ui.label(format!(
                "GDS where available, else current settings: {}/{} ({:.0}%)",
                r.combined_correct,
                r.n,
                pct(r.combined_correct)
            ));
        }
        let mut accept = false;
        let mut dismiss = false;
        if r.is_improvement() {
            for c in &r.changes {
                ui.label(format!("  \u{2022} {c}"));
            }
            ui.horizontal(|ui| {
                accept = ui.button("Accept").clicked();
                dismiss = ui.button("Dismiss").clicked();
            });
        } else {
            ui.horizontal(|ui| {
                ui.label(RichText::new("No better settings found.").weak());
                dismiss = ui.button("OK").clicked();
            });
        }
        if accept {
            let new = r.new;
            self.params = new.params;
            self.min_radius_um = new.min_radius_um;
            self.save_tune();
            self.rerun();
            self.report = None;
        } else if dismiss {
            self.report = None;
        }
    }

    /// While picking a corrected direction: draw a guide line from the stamp
    /// to the pointer and take the next click on the map as the label.
    pub fn show_label_overlay(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>) {
        let Some((id, pick)) = self.picking else { return };
        let gds = gds_prior::collect(object_list);
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.picking = None;
            return;
        }
        let Some(i) = object_list.iter().position(
            |o| matches!(&**o, Object::ScanImage { image, .. } if image.uuid() == id),
        ) else {
            self.picking = None;
            return;
        };
        let Some(ctx) = ui.data(|map| map.get_temp::<ScanViewCtx>(Id::new(()))) else {
            return;
        };
        let w2s = ctx.world2egui();
        let Object::ScanImage { image, name, .. } = &mut *object_list[i] else {
            return;
        };
        let p0 = w2s.transform_point2(image.transform.translation).to_egui_pos2();
        let stroke = Stroke::new(2., LABEL_COLOR);
        if let Some(hover) = ctx.screen_response.hover_pos() {
            ui.painter().extend(egui::Shape::dashed_line(&[p0, hover], stroke, 6., 4.));
            ui.painter().text(
                hover + egui::vec2(10., 10.),
                Align2::LEFT_TOP,
                match pick {
                    Pick::Correct => "click where the tip should go",
                    Pick::Middle => "click toward the middle (Esc to skip)",
                },
                FontId::proportional(12.),
                LABEL_COLOR,
            );
        }
        if ctx.screen_response.clicked() {
            if let Some(pos) = ctx.screen_response.interact_pointer_pos() {
                let world = w2s.inverse().transform_point2(pos.to_glam());
                let q = image.transform.inverse().transform_point2(world);
                if q.length() > 1e-6 {
                    let name = name.clone();
                    let g = gds_info(&gds, image);
                    self.picking = None;
                    let label = match pick {
                        Pick::Correct => RingLabel::Direction(q.normalize()),
                        Pick::Middle => RingLabel::SmallestRing(Some(q.normalize())),
                    };
                    self.apply_label(image, &name, label, g, ui.ctx());
                }
            }
        }
    }

    /// Re-analyse the current frame on the next update.
    fn rerun(&mut self) {
        // A result from the old settings is no longer wanted.
        self.job = None;
        self.teach_cache = None;
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
        if self.show_gds {
            if let Some(p) = &cur.gds {
                let a = to_screen(c);
                let b = to_screen(c + p.dir_world * cur.frame_width() * 0.55);
                let stroke = Stroke::new(2., GDS_COLOR);
                painter.extend(egui::Shape::dashed_line(&[a, b], stroke, 8., 4.));
                let tip = (b - a).normalized() * 11.;
                painter.arrow(b - tip, tip, stroke);
                painter.text(
                    b + egui::vec2(6., 0.),
                    Align2::LEFT_CENTER,
                    "GDS",
                    FontId::proportional(12.),
                    GDS_COLOR,
                );
                if p.dist_nm < 6. * cur.frame_width() {
                    let pc = to_screen(p.center_world);
                    painter.circle_stroke(pc, 7., stroke);
                    painter.line_segment([pc + egui::vec2(-4., 0.), pc + egui::vec2(4., 0.)], stroke);
                    painter.line_segment([pc + egui::vec2(0., -4.), pc + egui::vec2(0., 4.)], stroke);
                }
            }
        }
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

/// The GDS prediction for a stamp, as stored with its label.
fn gds_info(gds: &[GdsRings], image: &LiveImage) -> Option<GdsInfo> {
    let (_, p) = gds_prior::predict_any(gds, image.transform)?;
    let size = image.forward_data.size;
    Some(GdsInfo {
        dir: quad_to_px_dir(p.dir_quad(image.transform), size).to_array(),
        smallest: p.smallest_in_frame,
    })
}
