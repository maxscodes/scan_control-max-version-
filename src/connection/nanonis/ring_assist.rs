//! Ring Assist: runs the ring-centre estimator on live frames and draws what
//! it would do on the map. It only moves the tip in semi-auto centring, after
//! the user confirms each coarse move.
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
use glam::{DAffine2, DVec2, IVec2};

use crate::{
    analysis::{
        gds_prior::{self, BandCheck, Bands, GdsPrediction, GdsRings},
        landing::{self, Track, View as LandingView},
        ring_center::{self, RingParams, Verdict},
        ring_stamp::{
            self, quad_to_px_dir, RingLabel, RingStamp, StampVerdict, ARC_COLOR, ARM_COLOR,
            LABEL_COLOR, UNSURE_COLOR,
        },
        ring_live::{self, JointCheck, LiveMatch, MoveLimits, MovePlan, Shot},
        ring_match::{MatchMode, MatchParams},
        ring_training::{self, Dataset, GdsInfo, LabelKind, Record, Report, TuneParams},
    },
    components::selectable_list::SelectableList,
    connection::{LiveImage, ScanArea},
    scan_view::{BorderRectangle, ScanViewCtx},
    utils::vec_interop::{IntoEgui as _, IntoGlam as _},
    view_object::Object,
};

use super::course_motion::CourseMotionState;

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
/// Ring match direction: sure / unsure.
const SURE_COLOR: Color32 = Color32::from_rgb(0x4c, 0xe0, 0x6a);
const UNSURE_MATCH_COLOR: Color32 = Color32::from_rgb(0xff, 0xd8, 0x3a);
const PLAN_COLOR: Color32 = Color32::from_rgb(0xff, 0x8c, 0x1a);
/// Landing guesses on the map.
const LANDING_COLOR: Color32 = Color32::from_rgb(0xff, 0x6e, 0xc7);
/// Landing counts as sure when this share of the odds agrees on the way.
const LANDING_SURE: f64 = 0.8;
/// A frame this far (world nm) from the track's last one starts a new track.
const LANDING_RESET_NM: f64 = 150_000.;
/// Guesses listed in the window.
const LANDING_LIST: usize = 5;

/// The landing track and the GDS it is on.
struct LandingTrack {
    gds: uuid::Uuid,
    track: Track,
}

/// What a background analysis sends back.
struct JobResult {
    data: Arc<Box<[f32]>>,
    reading: Option<Reading>,
    /// The landing track after this frame, and the GDS it is on.
    landing: Option<(uuid::Uuid, LandingView)>,
    /// The job started the landing track over.
    reset: bool,
}

/// A coarse move waiting for the user's go-ahead.
struct Proposal {
    /// The reading it was made from.
    reading_at: Instant,
    plan: MovePlan,
    steps: IVec2,
    /// Frame centre (world) the move starts from.
    from: DVec2,
    sure: bool,
    /// Why the move can't be made, if so.
    problem: Option<String>,
}

/// Where semi-auto centring is.
enum Centring {
    Idle,
    /// A move this module started is running.
    Moving,
    /// Moved; waiting for a frame scanned after the move (seen partial,
    /// then complete).
    AwaitingFresh { moved_at: Instant, seen_partial: bool },
}

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
    /// The analysed frame (forward), for checking it against later frames.
    shot: Shot,
    /// Ring match (no GDS placement needed), on complete frames.
    live: Option<LiveMatch>,
    /// This frame checked together with the frame before the last move.
    joint: Option<JointCheck>,
}

impl Reading {
    /// Where the ring match puts the centre, and whether the direction is
    /// sure: from the check across frames when there is one, else from this
    /// frame alone.
    fn ring_target(&self, p: &MatchParams) -> Option<(DVec2, bool)> {
        match (&self.joint, &self.live) {
            (Some(j), _) => Some((j.center_world, j.agrees || j.gap >= p.sure_gap)),
            (None, Some(m)) => Some((m.center_world, m.sure(p))),
            _ => None,
        }
    }
    /// One line on the ring match.
    fn match_summary(&self, p: &MatchParams) -> Option<(String, Color32)> {
        let m = self.live.as_ref()?;
        let (_, sure) = self.ring_target(p)?;
        let side = if m.backward { " (backward scan)" } else { "" };
        let mut t = format!(
            "Ring match {:.2}{side}, direction {} (gap {:.3})",
            m.score,
            if sure { "sure" } else { "unsure" },
            m.gap
        );
        if let Some(j) = &self.joint {
            t += if j.overrides_current {
                "; frame before the move says the other way"
            } else if j.agrees {
                "; agrees with the frame before the move"
            } else {
                "; checked with the frame before the move"
            };
        }
        Some((t, if sure { SURE_COLOR } else { UNSURE_MATCH_COLOR }))
    }
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
    /// GDS matching settings (only changed by retraining for now).
    matching: MatchParams,
    /// Ring kinds (band radii, world nm) of the GDS on the map, for blind
    /// matching in retraining.
    ring_kinds_nm: Vec<Vec<(f64, f64)>>,
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
    job: Option<crossbeam::channel::Receiver<JobResult>>,
    /// GDS placement the ring kinds were last taken from.
    kinds_key: Vec<(uuid::Uuid, [f64; 6])>,
    /// Also match the backward scan and keep the better fit. Off by default:
    /// on the labelled frames it fixed 3 but broke 5 (forward 26/40, better
    /// fit 24/40).
    use_backward: bool,
    /// Semi-auto centring: propose a coarse move after each frame, and make
    /// it when the user agrees.
    semi_auto: bool,
    limits: MoveLimits,
    proposal: Option<Proposal>,
    /// Reading a proposal was already made (or skipped) for.
    proposed_for: Option<Instant>,
    centring: Centring,
    /// Recent centring actions, newest first.
    centring_log: VecDeque<String>,
    /// Guess where complete frames are on the GDS.
    landing_on: bool,
    /// Updated by the analysis jobs (one at a time).
    landing: Arc<parking_lot::Mutex<Option<LandingTrack>>>,
    /// Start the landing track over with the next complete frame.
    landing_reset: bool,
    landing_view: Option<(uuid::Uuid, LandingView)>,
    /// Guess picked in the list, for the map and Shift GDS.
    landing_pick: usize,
    /// GDS position before the last landing shift, for Undo.
    landing_undo: Option<(uuid::Uuid, DAffine2)>,
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
            matching: tune.matching,
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
            ring_kinds_nm: load_kinds(),
            kinds_key: Vec::new(),
            use_backward: false,
            semi_auto: false,
            limits: MoveLimits::default(),
            proposal: None,
            proposed_for: None,
            centring: Centring::Idle,
            centring_log: VecDeque::new(),
            landing_on: true,
            landing: Default::default(),
            landing_reset: false,
            landing_view: None,
            landing_pick: 0,
            landing_undo: None,
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
        self.sync_kinds(gds);
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
        let bwd = &scan_area.live_image.backward_data;
        let bwd = (self.use_backward && bwd.size == size).then(|| Shot { data: bwd.data.clone(), size, frame });
        let (params, min_radius_um) = (self.params, self.min_radius_um);
        let (kinds, matching) = (self.ring_kinds_nm.clone(), self.matching);
        // The latest matched frame somewhere else (before a move), to check
        // this one against.
        let prev = self
            .history
            .iter()
            .find(|r| r.live.is_some() && (r.frame_center() - frame.translation).length() > 0.2 * r.frame_width())
            .map(|r| (r.shot.clone(), r.live.clone().unwrap()));
        // Landing on the GDS that covers the frame, else the first with rings.
        let land = if self.landing_on {
            let near = gds_prior::predict_any(gds, frame).and_then(|(i, _)| gds.get(i));
            near.filter(|g| g.landing.is_some())
                .or_else(|| gds.iter().find(|g| g.landing.is_some()))
                .map(|g| (g.uuid, g.transform, g.landing.clone().unwrap()))
        } else {
            None
        };
        let (track, reset, landing_p) = (self.landing.clone(), self.landing_reset, landing::params(&matching));
        std::thread::spawn(move || {
            let mut reading = Self::analyse(&data, size, frame, params, min_radius_um);
            if let Some(r) = reading.as_mut().filter(|r| r.complete()) {
                r.live = ring_live::match_shot(&r.shot, bwd.as_ref(), &kinds, &matching);
                if let (Some(m), Some((ps, pm))) = (&r.live, &prev) {
                    r.joint = ring_live::joint_check(ps, pm, &r.shot, m, &kinds, &matching);
                }
            }
            let landing = match (reading.as_ref().filter(|r| r.complete()), land) {
                (Some(r), Some((uuid, transform, source))) => {
                    let mut guard = track.lock();
                    let fresh = reset
                        || guard.as_ref().is_none_or(|t| {
                            t.gds != uuid
                                || !t.track.same_gds(transform)
                                || t.track.latest().is_some_and(|l| (l.center() - r.shot.center()).length() > LANDING_RESET_NM)
                        });
                    if fresh {
                        *guard = Some(LandingTrack { gds: uuid, track: Track::new(source.library(), transform) });
                    }
                    let t = &mut guard.as_mut().unwrap().track;
                    if !t.has(&r.shot) {
                        t.add(r.shot.clone(), &landing_p);
                    }
                    t.view().map(|v| (uuid, v))
                }
                _ => None,
            };
            let reset = reset && landing.is_some();
            let _ = tx.send(JobResult { data, reading, landing, reset });
            crate::app::request_repaint();
        });
        self.job = Some(rx);
    }

    /// Take a finished background analysis, if any.
    fn poll_job(&mut self, gds: &[GdsRings]) {
        let Some(rx) = &self.job else { return };
        let JobResult { data, reading, landing, reset } = match rx.try_recv() {
            Ok(r) => r,
            Err(crossbeam::channel::TryRecvError::Empty) => return,
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                self.job = None;
                return;
            }
        };
        self.job = None;
        if reset {
            self.landing_reset = false;
        }
        if let Some(v) = landing {
            self.landing_view = Some(v);
            self.landing_pick = 0;
        }
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
        data: &Arc<Box<[f32]>>,
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
            shot: Shot { data: data.clone(), size, frame },
            live: None,
            joint: None,
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

    /// `locked` (another tool drives the tip) blocks centring moves.
    pub fn show_menu(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<Object>,
        course: &mut CourseMotionState,
        locked: Option<&str>,
    ) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.window_open, egui::Button::new("Ring Assist"))
                .clicked()
            {
                self.window_open = true;
            }
            if ui
                .selectable_label(self.semi_auto, "Semi-auto centring")
                .on_hover_text(
                    "After each complete frame, propose the coarse move toward the ring centre \
                     and make it when you click Proceed.",
                )
                .clicked()
            {
                self.set_semi_auto(!self.semi_auto);
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
        self.update_centring(ui, object_list, course, locked);
    }

    fn set_semi_auto(&mut self, on: bool) {
        self.semi_auto = on;
        self.proposal = None;
        self.centring = Centring::Idle;
        if on {
            // Don't act on a frame from before semi-auto was turned on.
            self.proposed_for = self.current.as_ref().map(|c| c.at);
            if !self.enabled {
                self.enabled = true;
                self.rerun();
            }
        }
    }

    fn log(&mut self, text: String) {
        tracing::info!("ring centring: {text}");
        self.centring_log.push_front(text);
        self.centring_log.truncate(6);
    }

    /// Semi-auto centring: follow the move, propose the next one, and show
    /// the proposal window.
    fn update_centring(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<Object>,
        course: &mut CourseMotionState,
        locked: Option<&str>,
    ) {
        if !self.semi_auto {
            return;
        }
        // Follow a running move and wait for a fresh frame after it.
        match &mut self.centring {
            Centring::Moving if !course.is_moving() => {
                self.centring = Centring::AwaitingFresh { moved_at: Instant::now(), seen_partial: false };
                self.log("Move done; waiting for a frame scanned after the move.".into());
            }
            Centring::AwaitingFresh { moved_at, seen_partial } => {
                if let Some(cur) = self.current.as_ref().filter(|c| c.at > *moved_at) {
                    if !cur.complete() {
                        *seen_partial = true;
                    } else if *seen_partial {
                        self.centring = Centring::Idle;
                    }
                }
            }
            _ => {}
        }
        if matches!(self.centring, Centring::Idle) && self.proposal.is_none() {
            self.make_proposal(object_list, course);
        }
        let mut open = true;
        let mut proceed = false;
        let mut decline = false;
        let mut use_anyway = false;
        let waiting = matches!(self.centring, Centring::AwaitingFresh { .. });
        let screen = ui.data(|m| m.get_temp::<ScanViewCtx>(Id::new(()))).map(|c| c.world2egui());
        egui::Window::new("Ring centring")
            .frame(crate::theme::window_frame(ui.ctx()))
            .default_width(340.)
            .collapsible(true)
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                match (&self.proposal, &self.centring) {
                    (_, Centring::Moving) => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Moving...");
                        });
                    }
                    (_, Centring::AwaitingFresh { .. }) => {
                        ui.label("Waiting for a new frame scanned after the move.");
                        ui.label(
                            RichText::new("Start a scan if it didn't restart after the approach.").weak(),
                        );
                    }
                    (None, _) => {
                        ui.label(RichText::new("Waiting for a complete frame with a ring match.").weak());
                        if let Some(cur) = &self.current {
                            if cur.complete() && cur.live.is_none() {
                                ui.label(
                                    RichText::new(if self.ring_kinds_nm.is_empty() {
                                        "No ring kinds known yet: load the device GDS on the map once."
                                    } else {
                                        "The last frame didn't match the rings well enough."
                                    })
                                    .color(UNSURE_COLOR),
                                );
                            }
                        }
                    }
                    (Some(pr), _) => {
                        let reading = self.current.as_ref().filter(|c| c.at == pr.reading_at);
                        show_proposal(ui, pr, reading, &self.matching, screen, course.withdraw_approach());
                        ui.add_space(6.);
                        ui.horizontal(|ui| {
                            let can = pr.problem.is_none() && locked.is_none() && !course.is_moving();
                            proceed = ui
                                .add_enabled(can, egui::Button::new(RichText::new("Proceed").strong()))
                                .clicked();
                            decline = ui.button("Do not proceed").clicked();
                        });
                        if let Some(l) = locked {
                            ui.label(RichText::new(l).color(WARN_COLOR));
                        }
                    }
                }
                if waiting {
                    use_anyway = ui
                        .small_button("Use the current frame anyway")
                        .on_hover_text("Skip waiting for a fresh scan (only if this frame was taken after the move).")
                        .clicked();
                }
                egui::CollapsingHeader::new("Move limits").show(ui, |ui| {
                    egui::Grid::new("centring_limits").num_columns(2).show(ui, |ui| {
                        ui.label("Largest move (frame widths)");
                        ui.add(egui::DragValue::new(&mut self.limits.max_frames).speed(0.05).range(0.1..=10.));
                        ui.end_row();
                        ui.label("Test move when unsure");
                        ui.add(egui::DragValue::new(&mut self.limits.test_frames).speed(0.05).range(0.1..=5.));
                        ui.end_row();
                    });
                });
                if !self.centring_log.is_empty() {
                    ui.separator();
                    for l in &self.centring_log {
                        ui.label(RichText::new(l).small().weak());
                    }
                }
            });
        if proceed {
            if let Some(pr) = self.proposal.take() {
                self.proposed_for = Some(pr.reading_at);
                if course.start_move(pr.steps) {
                    self.centring = Centring::Moving;
                    self.log(format!(
                        "Moving {:+} x, {:+} y steps ({:.2} \u{b5}m{}).",
                        pr.steps.x,
                        pr.steps.y,
                        pr.plan.world.length() * 1e-3,
                        if pr.plan.test { ", test move" } else { "" }
                    ));
                }
            }
        }
        if decline {
            if let Some(pr) = self.proposal.take() {
                self.proposed_for = Some(pr.reading_at);
                self.log("Move declined; waiting for the next frame.".into());
            }
        }
        if use_anyway {
            self.centring = Centring::Idle;
        }
        if !open {
            self.set_semi_auto(false);
        }
    }

    /// Plan a move from the current frame, if it is complete, new and matched.
    fn make_proposal(&mut self, object_list: &SelectableList<Object>, course: &CourseMotionState) {
        let Some(cur) = &self.current else { return };
        if !cur.complete() || self.proposed_for == Some(cur.at) {
            return;
        }
        let Some((center, sure)) = cur.ring_target(&self.matching) else { return };
        let Some(scan_area) = object_list.iter().find_map(|o| match &**o {
            Object::ScanArea { image, .. } => Some(image),
            _ => None,
        }) else {
            return;
        };
        let plan = ring_live::plan_move(&cur.shot, center, sure, self.limits);
        // The coarse move is in the scan area's own axes.
        let local = scan_area.world_transform.matrix2.inverse() * plan.world;
        let per_step = course.steps2real_world();
        let steps = course.real_to_steps(local);
        let problem = if per_step.determinant().abs() < 1e-12 {
            Some("Coarse motion isn't calibrated or the driver amplitudes are unknown.".to_string())
        } else if steps == IVec2::ZERO {
            Some(if plan.at_center {
                "The ring centre is in this frame. Done.".to_string()
            } else {
                format!(
                    "The move ({:.2} \u{b5}m) is less than one coarse step; move the scan frame instead.",
                    plan.world.length() * 1e-3
                )
            })
        } else {
            None
        };
        self.proposal = Some(Proposal { reading_at: cur.at, plan, steps, from: cur.frame_center(), sure, problem });
    }

    /// Keep the ring kinds of the GDS on the map, and remember them for when
    /// no GDS is loaded.
    fn sync_kinds(&mut self, gds: &[GdsRings]) {
        if gds.is_empty() {
            return;
        }
        let key: Vec<(uuid::Uuid, [f64; 6])> = gds.iter().map(|g| (g.uuid, g.transform.to_cols_array())).collect();
        if key == self.kinds_key {
            return;
        }
        self.kinds_key = key;
        let kinds = gds_prior::ring_kinds_on_map(gds);
        if !kinds.is_empty() && kinds != self.ring_kinds_nm {
            self.ring_kinds_nm = kinds;
            save_kinds(&self.ring_kinds_nm);
        }
    }

    fn show_window(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>) {
        ui.label(
            RichText::new(
                "Analyses each Z frame and shows where it would move. It only moves the tip in \
                 semi-auto centring, after you click Proceed.",
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
                if let Some((t, c)) = cur.match_summary(&self.matching) {
                    ui.label(RichText::new(t).color(c).strong());
                }
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
        self.show_landing(ui, object_list);
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
        if ui
            .checkbox(&mut self.use_backward, "Also match the backward scan")
            .on_hover_text("Keep whichever scan fits the rings better. On the labelled frames this didn't help overall.")
            .changed()
        {
            self.rerun();
        }
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

    /// Where the frames are on the GDS: the way to the centre, how sure,
    /// and the best places, with Shift GDS to line the GDS up with one.
    fn show_landing(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>) {
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Landing").strong());
            if ui
                .checkbox(&mut self.landing_on, "Guess where frames are on the GDS")
                .on_hover_text(
                    "Compares each complete frame with every ring structure in the GDS. \
                     Where the GDS sits on the map isn't used, only its scale and rotation.",
                )
                .changed()
                && self.landing_on
            {
                self.rerun();
            }
        });
        if !self.landing_on {
            return;
        }
        let gds = gds_prior::collect(object_list);
        let screen = ui.data(|m| m.get_temp::<ScanViewCtx>(Id::new(()))).map(|c| c.world2egui());
        let words = |v: DVec2| {
            screen
                .map(|s| s.transform_vector2(v))
                .filter(|v| v.length() > 1e-9)
                .map(|v| screen_words(egui::vec2(v.x as f32, v.y as f32)))
        };
        let busy = self.job.is_some();
        let Some((gid, v)) = &self.landing_view else {
            ui.horizontal(|ui| {
                if busy {
                    ui.spinner();
                }
                ui.label(
                    RichText::new(if !gds.iter().any(|g| g.landing.is_some()) {
                        "Load the device GDS on the map (its placement needn't be right)."
                    } else if busy {
                        "Working on the frame\u{2026}"
                    } else {
                        "Waiting for a complete Z frame."
                    })
                    .weak(),
                );
            });
            return;
        };
        let gid = *gid;
        let Some(g) = gds.iter().find(|g| g.uuid == gid) else {
            ui.label(RichText::new("The GDS these guesses are on is hidden or gone.").color(WARN_COLOR));
            return;
        };
        if let (Some((dir, a, b)), Some(c)) = (v.way_world(), v.consensus) {
            let range = if b - a < 1000. {
                format!("about {:.1} \u{b5}m", (a + b) / 2. * 1e-3)
            } else {
                format!("{:.0}\u{2013}{:.0} \u{b5}m", a * 1e-3, b * 1e-3)
            };
            ui.label(
                RichText::new(format!(
                    "Ring centre {}{range} away.",
                    words(dir).map_or(String::new(), |w| format!("{w}, "))
                ))
                .color(LANDING_COLOR)
                .strong(),
            );
            let (t, col) = if c.agree >= LANDING_SURE {
                (format!("{:.0}% of the odds agree on the way: sure.", c.agree * 100.), SURE_COLOR)
            } else {
                (
                    format!(
                        "Only {:.0}% of the odds agree on the way: not sure. A move along the bands \
                         (sideways, not toward the centre) tells the places apart.",
                        c.agree * 100.
                    ),
                    UNSURE_MATCH_COLOR,
                )
            };
            ui.label(RichText::new(t).color(col));
        }
        ui.label(
            RichText::new(format!(
                "From {} frame{}. Which ring and which structure are rough: one band fits a ring \
                 further in or out about as well.",
                v.frames,
                if v.frames == 1 { "" } else { "s" }
            ))
            .weak(),
        );
        let mut pick = self.landing_pick.min(v.hyps.len().saturating_sub(1));
        egui::Grid::new("landing_hyps").num_columns(5).striped(true).show(ui, |ui| {
            ui.label("");
            ui.label("odds");
            ui.label("where");
            ui.label("fit");
            ui.label("GDS off by");
            ui.end_row();
            for (i, (h, o)) in v.hyps.iter().zip(&v.odds).enumerate().take(LANDING_LIST) {
                let Some(loc) = v.locate(h, g.transform) else { continue };
                ui.radio_value(&mut pick, i, format!("#{}", i + 1));
                ui.label(format!("{:.0}%", o * 100.));
                let kind = &v.library.kinds[h.kind];
                let side = if kind.systems.len() > 1 && !loc.place.between {
                    words(loc.target_world - loc.structure_world).map_or(String::new(), |w| format!(", {w} half"))
                } else {
                    String::new()
                };
                let place = if loc.place.between {
                    "between the halves".to_string()
                } else {
                    format!("ring {}{side}", loc.place.ring)
                };
                ui.label(format!("{}: {place}", kind.name));
                ui.label(format!("{:.2}", h.score()));
                ui.label(format!("{:.1} \u{b5}m", loc.gds_shift.length() * 1e-3));
                ui.end_row();
            }
        });
        self.landing_pick = pick;
        let shift = v.hyps.get(pick).and_then(|h| v.locate(h, g.transform)).map(|l| l.gds_shift);
        let mut do_shift = None;
        let mut undo = None;
        let mut reset = false;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(shift.is_some(), egui::Button::new(format!("Shift GDS to #{}", pick + 1)))
                .on_hover_text("Move the GDS on the map so its nearest copy of this structure lines up with the guess.")
                .clicked()
            {
                do_shift = shift;
            }
            if self.landing_undo.is_some_and(|(u, _)| u == gid) && ui.button("Undo shift").clicked() {
                undo = self.landing_undo.take().map(|(_, t)| t);
            }
            reset = ui
                .button("Start over")
                .on_hover_text("Forget the earlier frames (e.g. after moving to another structure).")
                .clicked();
        });
        if reset {
            self.landing_reset = true;
            self.landing_view = None;
            self.rerun();
        }
        if do_shift.is_some() || undo.is_some() {
            self.move_gds(object_list, gid, do_shift, undo);
        }
    }

    /// Shift the GDS (or put it back), updating its stamps' reference so
    /// the "GDS moved" warning stays quiet for a deliberate move.
    fn move_gds(
        &mut self,
        object_list: &mut SelectableList<Object>,
        id: uuid::Uuid,
        shift: Option<DVec2>,
        restore: Option<DAffine2>,
    ) {
        let mut new = None;
        for o in object_list.iter_mut() {
            if let Object::Gds { image, .. } = &mut **o {
                if image.uuid() == id {
                    if let Some(s) = shift {
                        self.landing_undo = Some((id, image.transform));
                        image.transform.translation += s;
                    }
                    if let Some(t) = restore {
                        image.transform = t;
                    }
                    new = Some(image.transform);
                }
            }
        }
        let Some(t) = new else { return };
        for o in object_list.iter_mut() {
            if let Object::ScanImage { image, .. } = &mut **o {
                if image.gds_ref.is_some_and(|(u, _)| u == id) {
                    image.gds_ref = Some((id, t));
                }
            }
        }
        if let Some(s) = shift {
            self.log(format!("Shifted the GDS by ({:+.2}, {:+.2}) \u{b5}m to the landing guess.", s.x * 1e-3, s.y * 1e-3));
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
        let kinds = gds_prior::ring_kinds_on_map(gds);
        if !kinds.is_empty() {
            self.ring_kinds_nm = kinds;
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

    /// A stamp got its Nanonis file name: use it for its training label.
    pub fn rename_label(&mut self, id: uuid::Uuid, name: &str) {
        if let Err(e) = self.dataset.set_sources(&[(id.to_string(), name.to_string())]) {
            tracing::warn!("could not rename ring label: {e}");
        }
    }

    fn tune(&self) -> TuneParams {
        TuneParams { params: self.params, min_radius_um: self.min_radius_um, matching: self.matching }
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
            self.apply_label(image, &name, l, g.clone(), ui.ctx());
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
        let kinds = self.ring_kinds_nm.clone();
        let (tx, rx) = crossbeam::channel::bounded(1);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let samples = Dataset::load(dir).samples();
            let report = ring_training::evaluate(&samples, current, &kinds);
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
        if r.match_n > 0 {
            ui.label(
                RichText::new(format!(
                    "Matching scans to the GDS rings ({} labels with ring geometry):",
                    r.match_n
                ))
                .color(GDS_COLOR),
            );
            for (mode, n) in &r.mode_correct {
                ui.label(format!(
                    "  {}: {}/{} ({:.0}%)",
                    mode.name(),
                    n,
                    r.match_n,
                    100. * *n as f64 / r.match_n as f64
                ));
            }
            ui.label(format!(
                "  New match settings: {}/{} on held-out scans, {}/{} on all",
                r.match_cv_correct, r.match_n, r.match_new_correct, r.match_n
            ));
        }
        if !r.blind.is_empty() {
            ui.label(
                RichText::new(format!("Blind matching (no GDS placement), all {} labels:", r.n))
                    .color(GDS_COLOR),
            );
            for (mode, n, conf) in &r.blind {
                let name = if *mode == MatchMode::Off { "detector only" } else { mode.name() };
                ui.label(format!("  {name}: {n}/{} ({:.0}%), {conf} matched", r.n, pct(*n)));
            }
        }
        if r.is_improvement() {
            let detector: &[String] = if r.detector_improves() { &r.changes } else { &[] };
            let matching: &[String] = if r.match_improves() { &r.match_changes } else { &[] };
            for c in detector.iter().chain(matching) {
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
            if r.detector_improves() {
                self.params = new.params;
                self.min_radius_um = new.min_radius_um;
            }
            if r.match_improves() {
                self.matching = new.matching;
            }
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

        self.draw_landing(painter, &to_screen);
        let Some(cur) = &self.current else { return };
        let mut color = if cur.complete() { cur.color() } else { cur.color().gamma_multiply(0.6) };
        let c = cur.frame_center();
        // The ring match leads; the old detector is drawn faded beside it.
        if let Some((target, sure)) = cur.ring_target(&self.matching) {
            color = color.gamma_multiply(0.35);
            let mc = if sure { SURE_COLOR } else { UNSURE_MATCH_COLOR };
            let d = target - c;
            if d.length() > 1e-6 {
                arrow(c, d.normalize(), (cur.frame_width() * 0.45).min(d.length()), mc, 4.);
            }
            if d.length() < 3. * cur.frame_width() {
                painter.circle_stroke(to_screen(target), 7., Stroke::new(2.5, mc));
            }
            if let Some((t, _)) = cur.match_summary(&self.matching) {
                painter.text(to_screen(c) + egui::vec2(10., 8.), Align2::LEFT_TOP, t, FontId::proportional(13.), mc);
            }
        }
        if let Some(pr) = &self.proposal {
            let (a, b) = (to_screen(pr.from), to_screen(pr.from + pr.plan.world));
            painter.extend(egui::Shape::dashed_line(&[a, b], Stroke::new(2.5, PLAN_COLOR), 8., 5.));
            BorderRectangle {
                transform: DAffine2::from_translation(pr.plan.world) * cur.frame,
                color: PLAN_COLOR,
                dashed: true,
            }
            .show(ui);
            painter.text(
                b + egui::vec2(8., -8.),
                Align2::LEFT_BOTTOM,
                format!("planned move {:+}, {:+}", pr.steps.x, pr.steps.y),
                FontId::proportional(13.),
                PLAN_COLOR,
            );
        }
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

impl RingAssistState {
    /// The best landing guesses on the map: the centre each heads for, the
    /// rings of the picked one, and the agreed way from the frame.
    fn draw_landing(&self, painter: &egui::Painter, to_screen: &dyn Fn(DVec2) -> egui::Pos2) {
        if !self.landing_on {
            return;
        }
        let Some((_, v)) = &self.landing_view else { return };
        let font = FontId::proportional(12.);
        for (i, (h, o)) in v.hyps.iter().zip(&v.odds).enumerate().take(LANDING_LIST) {
            let picked = i == self.landing_pick;
            if !picked && (i >= 3 || *o < 0.02) {
                continue;
            }
            let Some((place, _, target)) = v.place_world(h) else { continue };
            let col = if picked { LANDING_COLOR } else { LANDING_COLOR.gamma_multiply(0.5) };
            let p = to_screen(target);
            let r = 6.;
            let stroke = Stroke::new(if picked { 2.5 } else { 1.5 }, col);
            painter.line_segment([p + egui::vec2(-r, -r), p + egui::vec2(r, r)], stroke);
            painter.line_segment([p + egui::vec2(-r, r), p + egui::vec2(r, -r)], stroke);
            painter.text(p + egui::vec2(8., -8.), Align2::LEFT_BOTTOM, format!("#{} {:.0}%", i + 1, o * 100.), font.clone(), col);
            if picked && !place.between {
                // The rings of the system the guess puts the frame on.
                let ring_stroke = Stroke::new(1., LANDING_COLOR.gamma_multiply(0.45));
                if let Some(s) = v.library.kinds[h.kind].systems.get(place.system) {
                    for b in &s.bands {
                        for rad in [b.0, b.1] {
                            let sr = (to_screen(target + DVec2::new(rad * v.scale(), 0.)) - p).length();
                            painter.circle_stroke(p, sr, ring_stroke);
                        }
                    }
                }
            }
        }
        if let (Some((dir, _, _)), Some(c)) = (v.way_world(), v.consensus) {
            let col = if c.agree >= LANDING_SURE { LANDING_COLOR } else { LANDING_COLOR.gamma_multiply(0.6) };
            let a = to_screen(v.frame_center);
            let b = to_screen(v.frame_center + dir * v.frame_width * 0.5);
            painter.arrow(a, b - a, Stroke::new(3., col));
            painter.text(b + egui::vec2(6., 6.), Align2::LEFT_TOP, format!("landing {:.0}%", c.agree * 100.), font, col);
        }
    }
}

/// The GDS prediction for a stamp, as stored with its label.
fn gds_info(gds: &[GdsRings], image: &LiveImage) -> Option<GdsInfo> {
    let (_, p) = gds_prior::predict_any(gds, image.transform)?;
    let size = image.forward_data.size;
    let exp = gds_prior::expected(gds, image.transform, size);
    Some(GdsInfo {
        dir: quad_to_px_dir(p.dir_quad(image.transform), size).to_array(),
        smallest: p.smallest_in_frame,
        center_px: exp.as_ref().map(|e| e.center_px.to_array()),
        bands_px: exp.map_or(vec![], |e| e.bands_px.iter().map(|b| [b.0, b.1]).collect()),
    })
}

/// Where the ring kinds are remembered between sessions.
fn kinds_path() -> std::path::PathBuf {
    crate::app::CONFIG_DIR.config_local_dir().join("ring_kinds.json")
}

fn load_kinds() -> Vec<Vec<(f64, f64)>> {
    std::fs::read_to_string(kinds_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn save_kinds(kinds: &[Vec<(f64, f64)>]) {
    if let Ok(t) = serde_json::to_string(kinds) {
        if let Err(e) = std::fs::write(kinds_path(), t) {
            tracing::warn!("could not save ring kinds: {e}");
        }
    }
}

/// "up-left" etc. for a direction on screen (y down).
fn screen_words(v: egui::Vec2) -> &'static str {
    let a = (-v.y).atan2(v.x).to_degrees();
    const NAMES: [&str; 8] = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"];
    NAMES[(((a + 360. + 22.5) % 360.) / 45.) as usize % 8]
}

/// The body of the proposal window.
fn show_proposal(
    ui: &mut Ui,
    pr: &Proposal,
    reading: Option<&Reading>,
    p: &MatchParams,
    screen: Option<DAffine2>,
    (withdraw, approach): (bool, bool),
) {
    let words = screen
        .map(|s| s.transform_vector2(pr.plan.world))
        .filter(|v| v.length() > 1e-9)
        .map(|v| screen_words(egui::vec2(v.x as f32, v.y as f32)));
    let dist = pr.plan.dist_nm * 1e-3;
    let headline = if pr.plan.at_center {
        format!("The ring centre is in this frame, {dist:.2} \u{b5}m from its middle.")
    } else {
        format!(
            "I think the ring centre is {}about {dist:.1} \u{b5}m away.",
            words.map_or(String::new(), |w| format!("{w}, "))
        )
    };
    ui.label(RichText::new(headline).strong());
    if let Some(r) = reading {
        if let Some(m) = &r.live {
            let (text, color) = if pr.sure {
                ("Direction: sure.".to_string(), SURE_COLOR)
            } else {
                (
                    format!(
                        "Direction: not sure. The other side fits almost as well (gap {:.3}, sure at {:.2}).",
                        m.gap, p.sure_gap
                    ),
                    UNSURE_MATCH_COLOR,
                )
            };
            ui.label(RichText::new(text).color(color));
            ui.label(
                RichText::new(format!(
                    "Match {:.2} on the {} scan.",
                    m.score,
                    if m.backward { "backward" } else { "forward" }
                ))
                .weak(),
            );
        }
        if let Some(j) = &r.joint {
            let (t, c) = if j.overrides_current {
                ("Checked against the frame before the move: this frame alone points the other way, but the centre that fits both frames is the one shown. The last move may have gone the wrong way.", WARN_COLOR)
            } else if j.agrees {
                ("Checked against the frame before the move: both agree.", SURE_COLOR)
            } else {
                ("Checked against the frame before the move: the centre shown fits both frames best.", UNSURE_MATCH_COLOR)
            };
            ui.label(RichText::new(t).color(c));
        }
    }
    ui.separator();
    if let Some(problem) = &pr.problem {
        ui.label(RichText::new(problem).color(if pr.plan.at_center { SURE_COLOR } else { WARN_COLOR }));
        return;
    }
    ui.label(format!(
        "Coarse steps: X {:+}, Y {:+}  (\u{2248} {:.2} \u{b5}m{})",
        pr.steps.x,
        pr.steps.y,
        pr.plan.world.length() * 1e-3,
        if pr.plan.at_center { ", to centre it" } else { "" }
    ));
    if pr.plan.test {
        ui.label(
            RichText::new("Test move: the next frame will confirm the direction or turn back.")
                .color(UNSURE_MATCH_COLOR),
        );
    }
    ui.label(
        RichText::new(format!(
            "Auto withdraw {}, auto approach {} (set in Course Motion).",
            if withdraw { "on" } else { "off" },
            if approach { "on" } else { "off" }
        ))
        .weak(),
    );
}
