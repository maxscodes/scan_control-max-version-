use std::path::PathBuf;

use crossbeam::channel::{Receiver, Sender};
use egui::{Color32, DragValue, RichText, ScrollArea, Ui};
use glam::{DVec2, IVec2};
use tracing::{error, info};

use crate::{
    components::selectable_list::SelectableList,
    connection::{
        nanonis::{
            coarse_cal::{self, CalFrame, CalPlan, CalResult},
            course_motion::{CoarseCalibration, CourseMotionState},
            sequence::{
                eta::{self, RunClock, ScanState, StepTimes},
                grid_track::{GridTrack, Placement, StampMove, Unmatched},
                model::{
                    compile, expand, with_current_scan, CoarseSpec, CurrentScan, Dir, ScanStart,
                    Frame as ScanFrame,
                    FrameSpec, Position, ScanEntry, Sequence, Speed, Step, CURRENT_SCAN,
                },
                runner::{SequenceCommand, SequenceEvent},
            },
            spectroscopy::IvCurve,
        },
        ScanArea,
    },
    view_object::Object,
};

/// Things the connection must do in response to runner events, because they
/// touch state the editor doesn't own (scan area, coarse motion, live image).
pub enum SequenceAction {
    CoarseMoved(IVec2),
    StampRequest,
    /// An I(V) series started (base name): show it in the I(V) window.
    SpecStarted(String),
    SpecCurve(IvCurve),
    /// The series ended: (ok, message, the .dat files Nanonis saved).
    SpecDone(bool, String, Vec<std::path::PathBuf>),
}

/// Snapshot of what the runner is executing.
pub(super) struct RunView {
    pub steps: Vec<(usize, Step)>,
    /// Global index into `steps` of the step in progress (or next to run when paused).
    pub current: usize,
    pub paused: bool,
    pub wait_remaining: Option<f64>,
    /// Progress text of the step in progress (I(V) series).
    pub detail: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SweepParam {
    Bias,
    Current,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SweepMode {
    /// `count` evenly spaced values from start to stop.
    Count,
    /// From start towards stop in steps of `step`.
    Step,
    /// Any values typed as a list.
    List,
}

struct SweepSettings {
    param: SweepParam,
    mode: SweepMode,
    /// V for bias, pA for current.
    start: f64,
    stop: f64,
    count: u32,
    step: f64,
    list: String,
}

/// A change a tracked grid wants to make, held for the user to accept.
struct GridApproval {
    /// Tile (index in the grid) just placed.
    tile: usize,
    /// The next tile's entry, its coarse move step, and the corrected
    /// steps, when the move would change.
    next: Option<(usize, usize, [i32; 2])>,
}

/// Survey grid: tiles of the scan frame side by side, joined by coarse moves.
pub(super) struct SurveySettings {
    cols: u32,
    rows: u32,
    /// Fraction of the frame shared with the neighbouring tile, in %.
    overlap_pct: f64,
    /// Grow towards +X (else -X) and +Y (else -Y), in scan-frame axes.
    plus_x: bool,
    plus_y: bool,
    /// Wait after each approach before scanning, in s (drift settles).
    settle_s: f64,
    /// Place each tile by matching it with the one before, take dragged
    /// tiles as where they really are, and aim the next move from that.
    pub(super) track: bool,
    /// Hold the run and ask before a tracked grid changes a move or places
    /// a tile well away from its plan.
    approve: bool,
    /// Slow-scan direction of every tile: alternating, or all the same way
    /// (drift then distorts every tile alike, so the seams fit better).
    dir: Dir,
    /// Scan each tile after a move twice, keeping only the second (creep).
    warm_up: bool,
}

/// One tile of the grid: column, row (0-based) in visiting order.
fn survey_order(cols: u32, rows: u32) -> Vec<(u32, u32)> {
    // Serpentine: every move is one tile, never a long jump back.
    (0..rows)
        .flat_map(|r| {
            let cs: Vec<u32> = if r % 2 == 0 { (0..cols).collect() } else { (0..cols).rev().collect() };
            cs.into_iter().map(move |c| (c, r))
        })
        .collect()
}

/// What ctrl+drag on the map edits for the selected entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DragTarget {
    Frame,
    Coarse,
    /// Tip positions: ctrl+click adds or removes one, ctrl+drag moves one.
    Tip,
}

pub struct SequenceState {
    pub(super) seq: Sequence,
    pub(super) selected: Option<usize>,
    /// Ctrl+drag on the map edits this part of the selected entry.
    pub(super) drag: Option<DragTarget>,
    /// Real-space (nm) coarse offset during a drag, so sub-step moves add up.
    pub(super) coarse_drag_real: Option<DVec2>,
    /// Tip position being dragged on the map (index into the entry's points).
    pub(super) tip_drag: Option<usize>,
    pub(super) window_open: bool,
    cmd_tx: Sender<SequenceCommand>,
    events: Receiver<SequenceEvent>,
    pub(super) run: Option<RunView>,
    status: String,
    last_error: Option<String>,
    /// Why the sequence is held: the instrument was used outside it.
    interrupted: Option<String>,
    sweep: SweepSettings,
    pub(super) survey: SurveySettings,
    load_tx: Sender<Result<Sequence, String>>,
    load_rx: Receiver<Result<Sequence, String>>,
    /// After a sequence finishes its plan is hidden from the map, until the
    /// sequence or the selected entry changes (or "Show plan" is clicked).
    pub(super) plan_hidden: Option<(Sequence, Option<usize>)>,
    /// The user hid the plan outlines from the map (to see the scan), until
    /// they show it again. Running is not affected.
    pub(super) plan_off: bool,
    /// What Run does with a scan that is already running in Nanonis.
    current_scan: CurrentScan,
    /// Nanonis' forward and backward time per line, s (set each frame).
    pub line_times: Option<(f64, f64)>,
    /// Lines per frame set in Nanonis now (set each frame).
    pub lines: Option<u32>,
    /// Scan set-up in Nanonis now, for the time estimate.
    scan_now: Option<ScanState>,
    /// What is left of the scan running in Nanonis now, s.
    scan_left: Option<f64>,
    /// Window shrunk to the progress bar and run buttons.
    minimized: bool,
    /// A coarse calibration run: its plan, the calibration before it, and
    /// the scans captured so far.
    cal: Option<(CalPlan, CoarseCalibration, Vec<CalFrame>)>,
    cal_tx: Sender<CalResult>,
    cal_rx: Receiver<CalResult>,
    /// Calibration scans done; the images are being compared.
    cal_measuring: bool,
    /// The file Nanonis saved the last scan of this run to.
    last_scan_file: Option<String>,
    /// Stamps of survey tiles taken in this run, and those of a finished
    /// run waiting to be lined up.
    survey_tiles: Vec<uuid::Uuid>,
    finished_survey: Option<Vec<uuid::Uuid>>,
    /// Layer group of this run's survey tiles, and tiles still to be put in it.
    survey_group: Option<String>,
    tile_groups: Vec<(uuid::Uuid, String)>,
    /// Puts the frame size and resolution back if a calibration run was
    /// aborted before doing so itself.
    cal_restore: Option<(Sequence, u32)>,
    /// Lengths of coarse moves, approaches, … learned over runs.
    times: StepTimes,
    /// Timing of the run in progress, and the scan set-up it started with.
    clock: Option<(RunClock, ScanState)>,
    /// A survey grid steering its moves as it runs (kept after the run,
    /// so tiles can still be dragged into place and the calibration saved).
    pub(super) grid_track: Option<GridTrack>,
    grid_calib_save: Option<(CoarseCalibration, usize)>,
    /// Each tracked tile entry's coarse move as planned, before any correction.
    grid_planned: std::collections::HashMap<usize, [i32; 2]>,
    /// Entries whose move is kept as planned (a correction was turned down).
    grid_pinned: std::collections::HashSet<usize>,
    /// A change the grid wants to make, waiting for the user (run held).
    grid_approval: Option<GridApproval>,
    /// Stamps to move on the map (undo or redo of an automatic placement).
    stamp_moves: Vec<StampMove>,
    /// The last line-up of survey tiles can be undone, and the user asked to.
    lineup_undo_available: bool,
    lineup_undo_request: bool,
    /// Pause once the scan in progress (or the next) is done and stamped.
    pause_after_scan: bool,
    /// "Rebuild & restart" was clicked: waiting for the runner to save the
    /// run at a safe point.
    restart_pending: bool,
}

impl SequenceState {
    pub fn new(cmd_tx: Sender<SequenceCommand>, events: Receiver<SequenceEvent>) -> Self {
        let (load_tx, load_rx) = crossbeam::channel::unbounded();
        let (cal_tx, cal_rx) = crossbeam::channel::unbounded();
        Self {
            seq: Sequence::default(),
            selected: None,
            drag: None,
            coarse_drag_real: None,
            tip_drag: None,
            window_open: false,
            cmd_tx,
            events,
            run: None,
            status: "Idle".into(),
            last_error: None,
            interrupted: None,
            sweep: SweepSettings {
                param: SweepParam::Bias,
                mode: SweepMode::Count,
                start: -1.,
                stop: 1.,
                count: 5,
                step: 0.5,
                list: "-1, -0.5, 0.2, 1.5".into(),
            },
            survey: SurveySettings {
                cols: 3,
                rows: 3,
                overlap_pct: 20.,
                plus_x: true,
                plus_y: false,
                settle_s: 30.,
                track: true,
                approve: true,
                dir: Dir::Alternate,
                warm_up: false,
            },
            load_tx,
            load_rx,
            plan_hidden: None,
            plan_off: false,
            current_scan: CurrentScan::After,
            line_times: None,
            lines: None,
            scan_now: None,
            scan_left: None,
            times: StepTimes::default(),
            clock: None,
            minimized: false,
            cal: None,
            cal_tx,
            cal_rx,
            cal_measuring: false,
            last_scan_file: None,
            cal_restore: None,
            survey_tiles: Vec::new(),
            finished_survey: None,
            survey_group: None,
            tile_groups: Vec::new(),
            grid_track: None,
            grid_calib_save: None,
            grid_planned: Default::default(),
            grid_pinned: Default::default(),
            grid_approval: None,
            stamp_moves: Vec::new(),
            lineup_undo_available: false,
            lineup_undo_request: false,
            pause_after_scan: false,
            restart_pending: false,
        }
    }

    /// Whether the tool window is open (toggled from the Scan Control tools).
    pub fn open_mut(&mut self) -> &mut bool {
        &mut self.window_open
    }

    /// A sequence is executing (not paused). Manual controls are locked.
    pub fn is_running(&self) -> bool {
        self.run.as_ref().is_some_and(|r| !r.paused)
    }
    fn is_active(&self) -> bool {
        self.run.is_some()
    }
    /// Ctrl+drag on the map is editing a sequence entry.
    pub fn map_drag_active(&self) -> bool {
        self.window_open && self.drag.is_some() && !self.plan_off
    }

    fn send(&self, cmd: SequenceCommand) {
        let _ = self.cmd_tx.send(cmd);
    }

    /// The instrument was used from the app while the sequence runs: hold
    /// it at once, leaving Nanonis as the user set it.
    pub fn interrupt(&mut self, reason: &str) {
        if self.is_running() {
            self.status = "Holding\u{2026}".into();
            self.send(SequenceCommand::Interrupt(reason.to_string()));
        }
    }

    pub fn stamp_done(&self) {
        self.send(SequenceCommand::StampDone);
    }

    /// Drain runner events, updating the run view. Returns actions the
    /// connection has to carry out.
    pub fn poll_events(&mut self) -> Vec<SequenceAction> {
        let mut actions = Vec::new();
        while let Ok(event) = self.events.try_recv() {
            match event {
                SequenceEvent::Started => {
                    self.status = "Running".into();
                    if let Some(run) = &mut self.run {
                        run.paused = false;
                    }
                    if let Some((clock, _)) = &mut self.clock {
                        clock.resume();
                    }
                }
                SequenceEvent::StepStarted { index } => {
                    if let Some(run) = &mut self.run {
                        if let Some((clock, _)) = &mut self.clock {
                            clock.step_started(index, &run.steps);
                        }
                        run.current = index;
                        run.wait_remaining = None;
                        run.detail = None;
                    }
                }
                SequenceEvent::SpecStarted { base } => actions.push(SequenceAction::SpecStarted(base)),
                SequenceEvent::SpecStatus(text) => {
                    if let Some(run) = &mut self.run {
                        run.detail = Some(text);
                    }
                }
                SequenceEvent::SpecCurve(curve) => actions.push(SequenceAction::SpecCurve(curve)),
                SequenceEvent::SpecDone { ok, message, files } => {
                    if ok {
                        info!("{message}");
                    } else {
                        error!("{message}");
                    }
                    if let Some(run) = &mut self.run {
                        run.detail = None;
                    }
                    actions.push(SequenceAction::SpecDone(ok, message, files));
                }
                SequenceEvent::WaitProgress { remaining } => {
                    if let Some(run) = &mut self.run {
                        run.wait_remaining = Some(remaining);
                    }
                }
                SequenceEvent::CoarseMoved { steps } => {
                    actions.push(SequenceAction::CoarseMoved(steps))
                }
                SequenceEvent::StampRequest => actions.push(SequenceAction::StampRequest),
                SequenceEvent::ScanDone { file_path } => {
                    info!("sequence scan saved to {file_path:?}");
                    self.last_scan_file = (!file_path.trim().is_empty()).then(|| file_path.trim().to_string());
                }
                SequenceEvent::Paused { index } => {
                    self.status = "Paused".into();
                    if let Some((clock, _)) = &mut self.clock {
                        clock.pause();
                    }
                    if let Some(run) = &mut self.run {
                        run.paused = true;
                        run.current = index;
                        run.wait_remaining = None;
                    }
                }
                SequenceEvent::Interrupted { index, reason } => {
                    self.status = "Held".into();
                    self.interrupted = Some(reason);
                    if let Some(run) = &mut self.run {
                        run.paused = true;
                        run.current = index;
                    }
                }
                SequenceEvent::Resumed => {
                    self.status = "Running".into();
                    // Carried on without an answer: the move stays as planned.
                    self.grid_approval = None;
                    self.interrupted = None;
                    if let Some((clock, _)) = &mut self.clock {
                        clock.resume();
                    }
                    self.last_error = None;
                    if let Some(run) = &mut self.run {
                        run.paused = false;
                    }
                }
                SequenceEvent::Finished => {
                    self.status = "Finished".into();
                    self.grid_approval = None;
                    // A tracked grid is already placed (and hand-corrected).
                    if self.survey_tiles.len() >= 2 && self.grid_track.is_none() {
                        self.finished_survey = Some(std::mem::take(&mut self.survey_tiles));
                    }
                    self.interrupted = None;
                    self.run = None;
                    if let Some((plan, old, frames)) = self.cal.take() {
                        // The run put the frame back itself.
                        self.cal_restore = None;
                        self.status = "Calibration scans done; measuring\u{2026}".into();
                        self.cal_measuring = true;
                        let tx = self.cal_tx.clone();
                        std::thread::spawn(move || {
                            let _ = tx.send(coarse_cal::compute(&frames, &plan, &old));
                            crate::app::request_repaint();
                        });
                    }
                    self.end_clock();
                    self.plan_hidden = Some((self.seq.clone(), self.selected));
                }
                SequenceEvent::Aborted => {
                    self.status = "Aborted".into();
                    self.grid_approval = None;
                    if self.survey_tiles.len() >= 2 && self.grid_track.is_none() {
                        self.finished_survey = Some(std::mem::take(&mut self.survey_tiles));
                    }
                    self.interrupted = None;
                    self.run = None;
                    if let Some((plan, old, frames)) = self.cal.take() {
                        // Stopped early: measure the moves whose scans were
                        // both taken (the frame is put back with "Restore").
                        if frames.len() >= 2 {
                            self.status = format!(
                                "Calibration stopped; measuring the {} moves done\u{2026}",
                                frames.len() - 1
                            );
                            self.cal_measuring = true;
                            let tx = self.cal_tx.clone();
                            std::thread::spawn(move || {
                                let _ = tx.send(coarse_cal::compute(&frames, &plan, &old));
                                crate::app::request_repaint();
                            });
                        } else {
                            self.status = "Calibration aborted".into();
                        }
                    }
                    self.end_clock();
                }
                SequenceEvent::Error { index, message } => {
                    error!("sequence error at step {index}: {message}");
                    self.last_error = Some(message);
                }
                SequenceEvent::Restored { steps, index, plan, auto_resume, note, survey_group, survey_tiles } => {
                    info!("sequence {note}");
                    self.seq = plan;
                    // Tiles taken from here on join the same survey.
                    self.survey_group = survey_group;
                    self.survey_tiles = survey_tiles;
                    self.selected = None;
                    self.plan_hidden = None;
                    self.run = Some(RunView {
                        steps,
                        current: index,
                        paused: true,
                        wait_remaining: None,
                        detail: None,
                    });
                    self.clock = None;
                    if auto_resume {
                        self.status = "Restarted: continues once Nanonis is connected".into();
                    } else {
                        self.status = "Paused".into();
                        self.interrupted = Some(note);
                    }
                    self.window_open = true;
                }
                SequenceEvent::PauseAfterScan(on) => self.pause_after_scan = on,
                SequenceEvent::Detached => {
                    if self.restart_pending {
                        self.restart_pending = false;
                        self.status = "Saved; restarting to rebuild…".into();
                        crate::restart::request();
                    }
                }
            }
        }
        while let Ok(loaded) = self.load_rx.try_recv() {
            match loaded {
                Ok(seq) => {
                    self.seq = seq;
                    self.selected = None;
                }
                Err(e) => self.last_error = Some(e),
            }
        }
        actions
    }

    /// Keep what the run learned about step lengths for the next estimate.
    fn end_clock(&mut self) {
        if let Some((clock, _)) = self.clock.take() {
            self.times = clock.times;
        }
    }

    /// Seconds left in the run, and the fraction done.
    fn run_eta(&self) -> Option<(f64, f32)> {
        let run = self.run.as_ref()?;
        let (clock, scan) = self.clock.as_ref()?;
        let left = eta::remaining(&run.steps, run.current, *scan, clock, run.wait_remaining, self.scan_left);
        let done = clock.elapsed();
        let frac = if done + left > 0. { (done / (done + left)) as f32 } else { 0. };
        Some((left, frac))
    }

    /// `current`: a scan is running in Nanonis (in direction `last_dir`) and
    /// the sequence waits for it / includes it.
    fn start(&mut self, from_entry: usize, group: u32, last_dir: Dir, current: Option<CurrentScan>) {
        self.plan_hidden = None;
        let mut steps = compile(&self.seq, from_entry, last_dir);
        if steps.is_empty() {
            self.last_error = Some("nothing to run (no enabled entries)".into());
            return;
        }
        if let Some(mode) = current {
            steps = with_current_scan(steps, mode, last_dir, true);
        }
        self.cal = None;
        let scan_start = self.seq.scan_start;
        let plan = Some(self.seq.clone());
        self.run_steps(steps, group, scan_start, plan);
    }

    /// Run the coarse calibration scans (the user's entries stay as they are).
    pub fn start_calibration(&mut self, plan: CalPlan, group: u32, old: CoarseCalibration) {
        if self.is_active() {
            self.last_error = Some("finish or abort the running sequence before calibrating".into());
            return;
        }
        let seq = coarse_cal::build_sequence(&plan);
        let steps = compile(&seq, 0, Dir::Down);
        let restore = coarse_cal::restore_sequence(&plan);
        self.cal_restore = (!restore.entries.is_empty()).then_some((restore, group));
        self.cal = Some((plan, old, Vec::new()));
        // Show the progress, small.
        self.window_open = true;
        self.minimized = true;
        self.run_steps(steps, group, seq.scan_start, None);
        self.status = "Calibrating".into();
    }

    /// An aborted calibration left the calibration frame size and
    /// resolution in Nanonis.
    pub fn can_restore_calibration_frame(&self) -> bool {
        self.cal_restore.is_some() && !self.is_active()
    }
    /// Put the frame size and resolution from before the calibration back.
    pub fn restore_calibration_frame(&mut self) {
        if self.is_active() {
            return;
        }
        if let Some((seq, group)) = self.cal_restore.take() {
            let steps = compile(&seq, 0, Dir::Down);
            self.run_steps(steps, group, seq.scan_start, None);
            self.status = "Restoring frame and resolution".into();
        }
    }

    /// The step running now belongs to a survey tile (so its stamp is one).
    pub fn current_entry_is_tile(&self) -> bool {
        if self.cal.is_some() {
            return false;
        }
        let Some(run) = &self.run else { return false };
        run.steps
            .get(run.current)
            .and_then(|(entry, _)| self.seq.entries.get(*entry))
            .is_some_and(|e| e.grid.is_some())
    }
    /// A survey tile was scanned (its image): measure the move that got
    /// here by matching it with the previous tile, and correct the move to
    /// the next tile (or, when asking first, hold the run for the user to
    /// accept the change). Returns how far (piezo nm) the tip is from where
    /// the scan area is: move the scan area by it before stamping, so the
    /// tile lands in the right place on the map.
    pub fn grid_tile_scanned(
        &mut self,
        frame: CalFrame,
        calib: CoarseCalibration,
        volts: DVec2,
        world_m2: glam::DMat2,
    ) -> Option<DVec2> {
        if !self.survey.track || self.cal.is_some() {
            return None;
        }
        let run = self.run.as_ref()?;
        let &(here, _) = run.steps.get(run.current)?;
        let entry = self.seq.entries.get(here)?;
        let target = DVec2::from_array(entry.grid_pos?);
        let moved = entry.coarse.map(|c| IVec2::from_array(c.steps));
        let track = self.grid_track.get_or_insert_with(|| GridTrack::new(calib, volts, world_m2));
        let off = track.tile_scanned(frame, here, target, moved, &calib);
        info!("{}", track.summary());
        let info = *track.tiles_info().last()?;
        let plan = self.next_move_plan();
        let changes_move = plan.is_some_and(|(_, _, planned, steps)| steps != planned);
        // Placed well away from where the moves put it (a tenth of the spacing).
        let spacing = self.grid_spacing(here).unwrap_or(f64::INFINITY);
        let big_shift = matches!(info.placement, Placement::Matched { .. }) && info.auto_shift_nm > 0.1 * spacing;
        let running = self.run.as_ref().is_some_and(|r| !r.paused);
        if self.survey.approve && (changes_move || big_shift) && running {
            self.grid_approval = Some(GridApproval {
                tile: info.index,
                next: plan.filter(|_| changes_move).map(|(next, at, _, steps)| (next, at, steps)),
            });
            // Held once the stamp is done, before anything moves the tip.
            self.send(SequenceCommand::Pause);
            self.status = "Waiting for you to accept or reject the grid's correction".into();
        } else if let Some((next, at, _, steps)) = plan {
            self.apply_move(next, at, steps);
        }
        Some(off)
    }

    /// The scan area was moved by a coarse move of the run: when a tracked
    /// grid is going, how much further (piezo nm) to move it so it sits
    /// where the grid's learned calibration puts the tip.
    pub fn grid_moved(&mut self, steps: IVec2, map_calib: &CoarseCalibration) -> Option<DVec2> {
        if !self.survey.track || self.cal.is_some() || !self.current_entry_is_tile() {
            return None;
        }
        Some(self.grid_track.as_mut()?.moved(steps, map_calib))
    }

    /// Distance between tile `entry`'s plan and the tile before it, nm.
    fn grid_spacing(&self, entry: usize) -> Option<f64> {
        let here = DVec2::from_array(self.seq.entries.get(entry)?.grid_pos?);
        let prev = self.seq.entries[..entry].iter().rev().find_map(|e| e.grid_pos)?;
        Some((here - DVec2::from_array(prev)).length())
    }

    /// The move to the tile after the latest one, unless it has started:
    /// (entry, its coarse move step, planned steps, steps to use).
    fn next_move_plan(&mut self) -> Option<(usize, usize, [i32; 2], [i32; 2])> {
        let (track, run) = (self.grid_track.as_ref()?, self.run.as_ref()?);
        let here = track.latest_entry()?;
        let last_here = run.steps.iter().rposition(|(e, _)| *e == here)?;
        let &(next, _) = run.steps.get(last_here + 1)?;
        let entry = self.seq.entries.get(next)?;
        let target = DVec2::from_array(entry.grid_pos?);
        let current = entry.coarse?.steps;
        let move_at =
            run.steps.iter().position(|(e, s)| *e == next && matches!(s, Step::CoarseMove { .. }))?;
        if move_at <= run.current {
            return None;
        }
        let planned = *self.grid_planned.entry(next).or_insert(current);
        let steps = if self.grid_pinned.contains(&next) {
            planned
        } else {
            self.grid_track.as_ref()?.next_steps(target, IVec2::from_array(planned))?.to_array()
        };
        Some((next, move_at, planned, steps))
    }

    /// Use `steps` for entry `next`'s coarse move (step `move_at`).
    fn apply_move(&mut self, next: usize, move_at: usize, steps: [i32; 2]) {
        let Some(c) = self.seq.entries.get_mut(next).and_then(|e| e.coarse.as_mut()) else { return };
        if c.steps == steps {
            return;
        }
        c.steps = steps;
        self.send(SequenceCommand::AdjustCoarse { entry: next, steps });
        if let Some(run) = &mut self.run {
            if let Some((_, Step::CoarseMove { steps: s })) = run.steps.get_mut(move_at) {
                *s = steps;
            }
        }
    }

    /// Re-aim the move to the tile after the latest one (after tiles were
    /// dragged, a placement undone, or learning switched), unless it has
    /// started. While the user is being asked, only the question changes.
    fn correct_next_move(&mut self) {
        let Some((next, at, planned, steps)) = self.next_move_plan() else { return };
        if let Some(a) = &mut self.grid_approval {
            a.next = (steps != planned).then_some((next, at, steps));
            return;
        }
        self.apply_move(next, at, steps);
    }

    /// The user accepted the grid's correction: use it and carry on.
    fn approve_grid(&mut self) {
        let Some(a) = self.grid_approval.take() else { return };
        if let Some((next, at, steps)) = a.next {
            self.apply_move(next, at, steps);
        }
        self.send(SequenceCommand::Resume);
    }

    /// The user turned the correction down: the tile goes back to where the
    /// moves put it, the next move stays as planned, and the run carries on.
    fn reject_grid(&mut self) {
        let Some(a) = self.grid_approval.take() else { return };
        self.set_tile_undone(a.tile, true);
        if let Some((next, at, _)) = a.next {
            self.pin_move(next, at, true);
        }
        self.send(SequenceCommand::Resume);
    }

    /// Keep entry `next`'s move as planned (or let the grid correct it again).
    fn pin_move(&mut self, next: usize, move_at: usize, pinned: bool) {
        if pinned {
            self.grid_pinned.insert(next);
            if let Some(&planned) = self.grid_planned.get(&next) {
                self.apply_move(next, move_at, planned);
            }
        } else {
            self.grid_pinned.remove(&next);
            self.correct_next_move();
        }
    }

    /// Undo (or redo) the automatic placement of tile `k`.
    fn set_tile_undone(&mut self, k: usize, undone: bool) {
        let Some(m) = self.grid_track.as_mut().and_then(|t| t.set_undone(k, undone)) else { return };
        info!("grid tile {}: automatic placement {}", k + 1, if undone { "undone" } else { "restored" });
        self.stamp_moves.push(m);
        self.correct_next_move();
    }

    /// Stamps to move on the map (undo or redo of automatic placements).
    pub fn take_stamp_moves(&mut self) -> Vec<StampMove> {
        std::mem::take(&mut self.stamp_moves)
    }
    /// Whether the last line-up of survey tiles can be undone.
    pub fn set_lineup_undo_available(&mut self, on: bool) {
        self.lineup_undo_available = on;
    }
    /// The user asked to undo the last line-up (once).
    pub fn take_lineup_undo_request(&mut self) -> bool {
        std::mem::take(&mut self.lineup_undo_request)
    }

    /// Stamps of the tracked grid's tiles.
    pub fn grid_tile_ids(&self) -> Vec<uuid::Uuid> {
        self.grid_track.as_ref().map(|t| t.ids()).unwrap_or_default()
    }
    /// Where the tiles' stamps are on the map now. Returns the map
    /// displacement of the latest tile when tiles were moved by hand (the
    /// tip, i.e. the scan area, moves with it).
    pub fn grid_sync(&mut self, world: impl Fn(uuid::Uuid) -> Option<DVec2>) -> Option<DVec2> {
        let delta = self.grid_track.as_mut()?.sync(world)?;
        if let Some(t) = &self.grid_track {
            info!("tiles moved by hand: {}", t.summary());
        }
        self.correct_next_move();
        Some(delta)
    }
    /// The grid's calibration, when the user asked to save it.
    pub fn take_grid_calibration(&mut self) -> Option<(CoarseCalibration, usize)> {
        self.grid_calib_save.take()
    }
    pub fn record_tile(&mut self, id: uuid::Uuid) {
        self.survey_tiles.push(id);
        if let Some(t) = &mut self.grid_track {
            t.set_latest_id(id);
        }
        let group = self.survey_group.get_or_insert_with(|| {
            let t = time::OffsetDateTime::now_local().unwrap_or_else(|_| time::OffsetDateTime::now_utc());
            format!(
                "Survey {}-{:02}-{:02} {:02}:{:02}",
                t.year(),
                t.month() as u8,
                t.day(),
                t.hour(),
                t.minute()
            )
        });
        self.tile_groups.push((id, group.clone()));
        let (group, tiles) = (self.survey_group.clone(), self.survey_tiles.clone());
        self.send(SequenceCommand::SurveyTiles { group, tiles });
    }
    /// Survey tiles to put in their layer group.
    pub fn take_tile_groups(&mut self) -> Vec<(uuid::Uuid, String)> {
        std::mem::take(&mut self.tile_groups)
    }
    /// Tiles whose stamp is not on the list yet: try again later.
    pub fn keep_tile_groups(&mut self, waiting: Vec<(uuid::Uuid, String)>) {
        self.tile_groups.extend(waiting);
    }
    /// Tile stamps of a survey that just ended (finished or aborted), to line up.
    pub fn take_finished_survey(&mut self) -> Option<Vec<uuid::Uuid>> {
        self.finished_survey.take()
    }
    pub fn set_status(&mut self, text: impl Into<String>) {
        self.status = text.into();
    }

    /// A calibration run is going (its stamps are captured).
    pub fn calibration_active(&self) -> bool {
        self.cal.is_some() && self.run.is_some()
    }
    /// The file the last scan was saved to (once).
    pub fn take_scan_file(&mut self) -> Option<String> {
        self.last_scan_file.take()
    }
    pub fn capture_calibration_frame(&mut self, frame: CalFrame) {
        if let Some((_, _, frames)) = &mut self.cal {
            frames.push(frame);
        }
    }
    /// The measured calibration, once a run has been analysed.
    pub fn take_calibration_result(&mut self) -> Option<CalResult> {
        let r = self.cal_rx.try_recv().ok()?;
        self.cal_measuring = false;
        self.status = "Calibration measured: see Coarse Motion".into();
        Some(r)
    }

    /// `plan`: the run is saved as it goes, to survive an app restart.
    fn run_steps(&mut self, steps: Vec<(usize, Step)>, group: u32, scan_start: ScanStart, plan: Option<Sequence>) {
        self.survey_tiles.clear();
        self.survey_group = None;
        self.grid_track = None;
        self.grid_planned.clear();
        self.grid_pinned.clear();
        self.grid_approval = None;
        self.plan_hidden = None;
        self.last_error = None;
        self.status = "Starting".into();
        self.run = Some(RunView {
            steps: steps.clone(),
            current: 0,
            paused: false,
            wait_remaining: None,
            detail: None,
        });
        self.clock = self
            .scan_now
            .map(|scan| (RunClock::new(self.times.clone()), scan));
        self.send(SequenceCommand::Run { steps, group, scan_start, plan });
    }

    /// Refresh what the time estimate needs from the scan area (each frame,
    /// also while the window is closed).
    fn update_scan_info(&mut self, scan_area: Option<&ScanArea>) {
        let current_frame = scan_area
            .and_then(|a| a.image_transform)
            .map(ScanFrame::from_transform);
        let scanning = scan_area.is_some_and(|a| a.scan_status.scanning);
        self.scan_now = match (scan_area, self.line_times) {
            (Some(a), Some((fwd_s, bwd_s))) => Some(ScanState {
                // Nanonis' setting, which changes before a new image arrives.
                lines: self.lines.unwrap_or(a.live_image.forward_data.size[1]),
                fwd_s,
                bwd_s,
                width_nm: current_frame.map_or(0., |f| f.width),
                restart_s: self.seq.scan_start.extra_s(),
            }),
            _ => None,
        };
        // Rows still to scan in the running frame (unscanned rows are NaN).
        self.scan_left = scan_area.filter(|_| scanning).zip(self.line_times).map(|(a, (f, b))| {
            let d = &a.live_image.forward_data;
            let [w, h] = d.size.map(|v| v as usize);
            let done = if d.data.len() == w * h {
                (0..h).filter(|&r| d.data[r * w + w / 2].is_finite()).count()
            } else {
                0
            };
            (h - done) as f64 * (f + b)
        });
        // A run restored after a restart gets its clock once the scan set-up is known.
        if let (Some(run), None, Some(scan)) = (&self.run, &self.clock, self.scan_now) {
            let mut clock = RunClock::new(self.times.clone()).at(run.current);
            if run.paused {
                clock.pause();
            }
            self.clock = Some((clock, scan));
        }
    }

    /// The "Sequence" button and window. Shown from the Scan Region menu.
    pub fn show_menu(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<Object>,
        course: &CourseMotionState,
        blocked_by: Option<&str>,
    ) {
        // Status and progress stay visible in the panel even with the
        // window closed.
        if self.is_active() {
            ui.label(RichText::new(format!("Sequence: {}", self.status)).small());
            self.progress_bar(ui);
        }
        let scan_area = object_list
            .iter_mut()
            .find_map(|entry| entry.as_scan_area_mut());
        self.update_scan_info(scan_area.as_deref());
        let mut open = self.window_open;
        if self.minimized {
            // Just the progress bar and run buttons.
            egui::Window::new("Sequence")
                .vscroll(true)
                .id(egui::Id::new("sequence_window_minimized"))
                .frame(crate::theme::window_frame(ui.ctx()))
                .default_size([380., 200.])
                .min_width(320.)
                .resizable(true)
                .collapsible(true)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    self.show_window(ui, scan_area.as_deref(), course, blocked_by);
                });
        } else {
            egui::Window::new("Sequence")
                .vscroll(true)
                .frame(crate::theme::window_frame(ui.ctx()))
                .default_size([720., 600.])
                .resizable(true)
                .collapsible(true)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    self.show_window(ui, scan_area.as_deref(), course, blocked_by);
                });
        }
        self.window_open = open;
        if !self.window_open || self.is_active() {
            self.drag = None;
        }
        if self.drag.is_some() {
            // Ctrl+drag would otherwise also move selected layers.
            object_list.clear_selected();
        }
    }

    fn show_window(
        &mut self,
        ui: &mut Ui,
        scan_area: Option<&ScanArea>,
        course: &CourseMotionState,
        blocked_by: Option<&str>,
    ) {
        let editable = !self.is_active();
        // Direction of the last scan in Nanonis: the first alternating
        // entry goes the other way.
        let last_dir: Dir = scan_area.map_or(Dir::Down, |a| a.scan_status.scan_dir.into());
        let current_frame = scan_area
            .and_then(|a| a.image_transform)
            .map(ScanFrame::from_transform);
        let tip_pos = scan_area.and_then(|a| a.tip_pos);
        let scanning = scan_area.is_some_and(|a| a.scan_status.scanning);
        ui.horizontal(|ui| {
            let (icon, tip) = if self.minimized {
                ("\u{1f5d6} Expand", "Show the whole sequence editor")
            } else {
                ("\u{1f5d5} Minimize", "Shrink to the progress bar and run buttons")
            };
            if ui.small_button(icon).on_hover_text(tip).clicked() {
                self.minimized = !self.minimized;
            }
            let (label, tip) = if self.plan_off {
                ("\u{1f441} Show plan on map", "Draw the sequence's frames, moves and tip points on the map again")
            } else {
                ("\u{1f441} Hide plan on map", "Hide the sequence's outlines from the map to see the scan (the sequence keeps running)")
            };
            if ui.small_button(label).on_hover_text(tip).clicked() {
                self.plan_off = !self.plan_off;
            }
        });
        if self.minimized {
            self.show_run_controls(ui, course, last_dir, scanning, blocked_by);
            return;
        }
        ui.add_enabled_ui(editable, |ui| self.show_toolbar(ui));
        ui.separator();
        self.show_entry_table(ui, editable, last_dir);
        ui.separator();
        // Entry details, steps and the bias sweep can be folded away so the
        // window only shows the scan list and the run controls.
        let mut details = crate::theme::prefs().sequence_details;
        let label = if details { "⏶ Hide entry details" } else { "⏷ Show entry details" };
        if ui.small_button(label).clicked() {
            details = !details;
            crate::theme::update_prefs(|p| p.sequence_details = details);
        }
        if details {
            ui.columns(2, |cols| {
                cols[0].add_enabled_ui(editable, |ui| {
                    self.show_details(ui, current_frame, tip_pos, course);
                });
                self.show_steps(&mut cols[1], last_dir);
            });
            ui.separator();
            ui.add_enabled_ui(editable, |ui| self.show_sweep(ui));
            ui.add_enabled_ui(editable, |ui| self.show_survey(ui, current_frame, course));
        }
        ui.separator();
        ui.add_enabled_ui(editable, |ui| {
            ui.horizontal(|ui| {
                let st = &mut self.seq.scan_start;
                ui.label("Wait at the scan start");
                ui.add(DragValue::new(&mut st.park_s).range(0. ..=600.).speed(1.).suffix(" s"));
            })
            .response
            .on_hover_text(
                "Before each scan the tip is moved to where the scan begins and waits \
                 there this long, so the creep after that jump is over before the first \
                 line is recorded. 0 = don't (the scan jumps there when it starts).",
            );
            ui.horizontal(|ui| {
                let st = &mut self.seq.scan_start;
                ui.label("Start each scan");
                ui.add(DragValue::new(&mut st.starts).range(1..=10).suffix("\u{d7}"));
                ui.label("with");
                ui.add(DragValue::new(&mut st.gap_s).range(0. ..=60.).speed(0.1).suffix(" s"));
                ui.label("between starts");
            })
            .response
            .on_hover_text(
                "Start is pressed this many times before the scan is kept: the scanner \
                 settles after jumping to the frame start, so the first lines come out \
                 cleaner. 1 = start once.",
            );
        });
        self.show_run_controls(ui, course, last_dir, scanning, blocked_by);
    }

    fn show_toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if ui.button("+ Entry").clicked() {
                let at = self.selected.map_or(self.seq.entries.len(), |i| i + 1);
                let entry = ScanEntry {
                    name: format!("scan {}", self.seq.entries.len() + 1),
                    ..Default::default()
                };
                self.seq.entries.insert(at, entry);
                self.selected = Some(at);
            }
            let sel = self.selected.filter(|&i| i < self.seq.entries.len());
            if ui.add_enabled(sel.is_some(), egui::Button::new("Dup")).clicked() {
                let i = sel.unwrap();
                let mut e = self.seq.entries[i].clone();
                e.name.push_str(" copy");
                self.seq.entries.insert(i + 1, e);
                self.selected = Some(i + 1);
            }
            if ui
                .add_enabled(sel.is_some_and(|i| i > 0), egui::Button::new("↑"))
                .clicked()
            {
                let i = sel.unwrap();
                self.seq.entries.swap(i, i - 1);
                self.selected = Some(i - 1);
            }
            if ui
                .add_enabled(
                    sel.is_some_and(|i| i + 1 < self.seq.entries.len()),
                    egui::Button::new("↓"),
                )
                .clicked()
            {
                let i = sel.unwrap();
                self.seq.entries.swap(i, i + 1);
                self.selected = Some(i + 1);
            }
            if ui.add_enabled(sel.is_some(), egui::Button::new("Del")).clicked() {
                let i = sel.unwrap();
                self.seq.entries.remove(i);
                self.selected = if self.seq.entries.is_empty() {
                    None
                } else {
                    Some(i.min(self.seq.entries.len() - 1))
                };
            }
            if ui
                .add_enabled(!self.seq.entries.is_empty(), egui::Button::new("Clear"))
                .clicked()
            {
                self.seq.entries.clear();
                self.selected = None;
            }
            ui.separator();
            if ui.button("Save…").clicked() {
                self.save_dialog();
            }
            if ui.button("Load…").clicked() {
                self.load_dialog();
            }
        });
    }

    fn save_dialog(&mut self) {
        let json = match serde_json::to_string_pretty(&self.seq) {
            Ok(json) => json,
            Err(e) => {
                self.last_error = Some(format!("failed to serialize sequence: {e}"));
                return;
            }
        };
        let tx = self.load_tx.clone();
        std::thread::spawn(move || {
            let Some(path) = crate::components::file_dialog_native::dialog()
                .add_filter("Scan sequence", &["json"])
                .set_title("Save Sequence")
                .set_file_name("sequence.json")
                .save_file()
            else {
                return;
            };
            if let Err(e) = std::fs::write(&path, json) {
                let _ = tx.send(Err(format!("failed to save {}: {e}", path.display())));
            } else {
                info!("saved sequence to {}", path.display());
            }
        });
    }

    fn load_dialog(&mut self) {
        let tx = self.load_tx.clone();
        std::thread::spawn(move || {
            let Some(path): Option<PathBuf> = crate::components::file_dialog_native::dialog()
                .add_filter("Scan sequence", &["json"])
                .set_title("Load Sequence")
                .pick_file()
            else {
                return;
            };
            let result = std::fs::read_to_string(&path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))
                .and_then(|s| {
                    serde_json::from_str::<Sequence>(&s)
                        .map_err(|e| format!("failed to parse {}: {e}", path.display()))
                });
            let _ = tx.send(result);
        });
    }

    fn running_entry(&self) -> Option<usize> {
        let run = self.run.as_ref()?;
        run.steps.get(run.current).map(|(i, _)| *i)
    }

    fn show_entry_table(&mut self, ui: &mut Ui, editable: bool, last_dir: Dir) {
        let running = self.running_entry();
        // Resolved direction of each entry's scan (for alternating entries).
        let mut resolved = vec![None; self.seq.entries.len()];
        let compiled = match &self.run {
            Some(run) => run.steps.clone(),
            None => compile(&self.seq, 0, last_dir),
        };
        for (i, step) in compiled {
            if let (Step::Scan(d), Some(slot)) = (&step, resolved.get_mut(i)) {
                *slot = Some(*d);
            }
        }
        ScrollArea::vertical()
            .id_salt("sequence entries")
            .max_height(200.)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                egui::Grid::new("sequence entry grid")
                    .striped(true)
                    .num_columns(12)
                    .show(ui, |ui| {
                        for h in [
                            "#", "On", "Name", "Bias", "Current", "Frame", "Pixels", "Speed",
                            "Coarse", "Dir", "Waits (s)", "Tip",
                        ] {
                            ui.label(RichText::new(h).strong());
                        }
                        ui.end_row();
                        for (i, e) in self.seq.entries.iter_mut().enumerate() {
                            let marker = if running == Some(i) { "▶" } else { "" };
                            ui.label(format!("{marker}{}", i + 1));
                            ui.add_enabled(editable, egui::Checkbox::without_text(&mut e.enabled));
                            let name = if e.name.is_empty() { "(unnamed)" } else { &e.name };
                            if ui
                                .selectable_label(self.selected == Some(i), name)
                                .clicked()
                            {
                                if self.selected != Some(i) {
                                    self.drag = None;
                                }
                                self.selected = Some(i);
                            }
                            ui.label(e.bias.map_or("keep".into(), |v| format!("{v} V")));
                            ui.label(
                                e.current
                                    .map_or("keep".into(), |v| format!("{} pA", v * 1e12)),
                            );
                            ui.label(e.frame.to_string());
                            ui.label(
                                e.resolution
                                    .map_or("keep".into(), |[px, l]| format!("{px}×{l}")),
                            );
                            ui.label(e.speed.map_or("keep".into(), |s| s.to_string()));
                            ui.label(e.coarse.map_or("–".into(), |c| {
                                format!("{:+}, {:+}", c.steps[0], c.steps[1])
                            }));
                            if !e.scan {
                                ui.weak("no scan");
                            } else {
                                match (e.dir, resolved[i]) {
                                    (Dir::Alternate, Some(d)) => {
                                        ui.label(format!("⇅ {}", d.arrow()))
                                            .on_hover_text("alternates with the previous scan")
                                    }
                                    (d, _) => ui.label(d.arrow()),
                                };
                            }
                            ui.label(format!(
                                "{}/{}/{}",
                                e.settle_s, e.wait_before_s, e.wait_after_s
                            ));
                            let n = e.tip.points.len();
                            ui.label(match (n, e.tip.spectroscopy) {
                                (0, _) => "–".to_string(),
                                (n, true) => format!("{n} × I(V)"),
                                (n, false) => format!("{n} moves"),
                            });
                            ui.end_row();
                        }
                    });
                if self.seq.entries.is_empty() {
                    ui.weak("No entries. Use \"+ Entry\" or the sweep generator below.");
                }
            });
    }

    fn show_details(
        &mut self,
        ui: &mut Ui,
        current_frame: Option<ScanFrame>,
        tip_pos: Option<DVec2>,
        course: &CourseMotionState,
    ) {
        let Some(i) = self.selected.filter(|&i| i < self.seq.entries.len()) else {
            ui.weak("Select an entry to edit it.");
            return;
        };
        let mut drag = self.drag;
        let e = &mut self.seq.entries[i];
        ui.label(RichText::new(format!("Entry {}", i + 1)).strong());
        if let Some(target) = drag {
            let what = match target {
                DragTarget::Frame => "Ctrl+drag on the map to move this entry's frame",
                DragTarget::Coarse => "Ctrl+drag on the map to move this entry's coarse move",
                DragTarget::Tip => {
                    "Ctrl+click on the map to add a tip position (or remove one), ctrl+drag to move one"
                }
            };
            ui.colored_label(Color32::from_rgb(0xff, 0x9c, 0xff), what);
        }
        egui::Grid::new("sequence entry details")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Name");
                ui.text_edit_singleline(&mut e.name);
                ui.end_row();

                ui.label("Bias");
                ui.horizontal(|ui| {
                    let mut set = e.bias.is_some();
                    ui.checkbox(&mut set, "set");
                    let mut v = e.bias.unwrap_or(1.0);
                    ui.add_enabled(set, DragValue::new(&mut v).speed(0.01).suffix(" V"));
                    e.bias = set.then_some(v);
                });
                ui.end_row();

                ui.label("Current");
                ui.horizontal(|ui| {
                    let mut set = e.current.is_some();
                    ui.checkbox(&mut set, "set");
                    let mut pa = e.current.map_or(100., |v| v as f64 * 1e12);
                    ui.add_enabled(set, DragValue::new(&mut pa).speed(1.).suffix(" pA"));
                    e.current = set.then_some((pa * 1e-12) as f32);
                });
                ui.end_row();

                ui.label("Coarse move");
                ui.vertical(|ui| {
                    let mut set = e.coarse.is_some();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut set, "before this scan");
                        if drag_toggle(ui, &mut drag, DragTarget::Coarse) {
                            set = true;
                        }
                    });
                    let mut c = e.coarse.unwrap_or_default();
                    ui.add_enabled_ui(set, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("X");
                            ui.add(DragValue::new(&mut c.steps[0]));
                            ui.label("Y");
                            ui.add(DragValue::new(&mut c.steps[1]));
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut c.withdraw, "withdraw");
                            ui.checkbox(&mut c.approach, "approach");
                        });
                        if ui
                            .button("Use Coarse Motion target")
                            .on_hover_text("Copy the steps currently planned in the Coarse Motion window")
                            .clicked()
                        {
                            c.steps = course.get_steps().to_array();
                        }
                    });
                    e.coarse = set.then_some(c);
                });
                ui.end_row();

                ui.label("Settle after move");
                ui.add_enabled(
                    e.coarse.is_some(),
                    DragValue::new(&mut e.settle_s).range(0.0..=f64::MAX).suffix(" s"),
                );
                ui.end_row();

                // Fallback when no frame has been read from Nanonis yet.
                let cur = current_frame.unwrap_or(ScanFrame {
                    center_x: 0.,
                    center_y: 0.,
                    width: 100.,
                    height: 100.,
                    angle: 0.,
                });

                ui.label("Frame position");
                ui.vertical(|ui| {
                    let f = &mut e.frame;
                    drag_toggle(ui, &mut drag, DragTarget::Frame);
                    ui.horizontal(|ui| {
                        let is_keep = f.position == Position::Keep;
                        let is_at = matches!(f.position, Position::At(_));
                        let is_shift = matches!(f.position, Position::Shift(_));
                        if ui
                            .radio(is_keep, "keep")
                            .on_hover_text("Scan wherever the frame is when this entry starts (including moves you make by hand)")
                            .clicked()
                        {
                            f.position = Position::Keep;
                        }
                        if ui
                            .radio(is_at, "move to")
                            .on_hover_text("Absolute frame centre")
                            .clicked()
                            && !is_at
                        {
                            f.position = Position::At([cur.center_x, cur.center_y]);
                        }
                        if ui
                            .radio(is_shift, "shift by")
                            .on_hover_text("Move the frame relative to where it is when this entry starts")
                            .clicked()
                            && !is_shift
                        {
                            f.position = Position::Shift([0., 0.]);
                        }
                    });
                    let w = f.size.map_or(cur.width, |s| s[0]);
                    let h = f.size.map_or(cur.height, |s| s[1]);
                    match &mut f.position {
                        Position::Keep => {}
                        Position::At(c) => {
                            ui.horizontal(|ui| {
                                ui.label("centre");
                                ui.add(DragValue::new(&mut c[0]).prefix("x ").suffix(" nm"));
                                ui.add(DragValue::new(&mut c[1]).prefix("y ").suffix(" nm"));
                            });
                        }
                        Position::Shift(d) => {
                            ui.horizontal(|ui| {
                                ui.label("by");
                                ui.add(DragValue::new(&mut d[0]).prefix("x ").suffix(" nm"));
                                ui.add(DragValue::new(&mut d[1]).prefix("y ").suffix(" nm"));
                            });
                            ui.horizontal(|ui| {
                                ui.weak("one frame:");
                                for (label, delta) in [
                                    ("←", [-w, 0.]),
                                    ("→", [w, 0.]),
                                    ("↑", [0., h]),
                                    ("↓", [0., -h]),
                                ] {
                                    if ui.small_button(label).clicked() {
                                        *d = delta;
                                    }
                                }
                            });
                        }
                    }
                });
                ui.end_row();

                ui.label("Scan size");
                ui.horizontal(|ui| {
                    let mut set = e.frame.size.is_some();
                    ui.checkbox(&mut set, "set");
                    let mut size = e.frame.size.unwrap_or([cur.width, cur.height]);
                    ui.add_enabled_ui(set, |ui| {
                        ui.add(DragValue::new(&mut size[0]).range(0.0..=f64::MAX).prefix("w ").suffix(" nm"));
                        ui.add(DragValue::new(&mut size[1]).range(0.0..=f64::MAX).prefix("h ").suffix(" nm"));
                    });
                    e.frame.size = set.then_some(size);
                });
                ui.end_row();

                ui.label("Frame angle");
                ui.horizontal(|ui| {
                    let mut set = e.frame.angle.is_some();
                    ui.checkbox(&mut set, "set");
                    let mut a = e.frame.angle.unwrap_or(cur.angle);
                    ui.add_enabled(set, DragValue::new(&mut a).suffix("°"));
                    e.frame.angle = set.then_some(a);
                });
                ui.end_row();

                ui.label("");
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(current_frame.is_some(), egui::Button::new("Capture current frame"))
                        .on_hover_text("Set position, size and angle from the frame shown on the map (drag it there first)")
                        .clicked()
                    {
                        e.frame = FrameSpec::capture(cur);
                    }
                    if ui
                        .add_enabled(!e.frame.is_keep(), egui::Button::new("Keep all"))
                        .on_hover_text("Don't change the frame for this entry")
                        .clicked()
                    {
                        e.frame = FrameSpec::default();
                    }
                });
                ui.end_row();

                ui.label("Pixels");
                ui.horizontal(|ui| {
                    let mut set = e.resolution.is_some();
                    ui.checkbox(&mut set, "set");
                    let mut r = e.resolution.unwrap_or([256, 256]);
                    ui.add_enabled_ui(set, |ui| {
                        ui.add(DragValue::new(&mut r[0]).range(1..=16384).suffix(" px/line"));
                        ui.label("×");
                        ui.add(DragValue::new(&mut r[1]).range(1..=16384).suffix(" lines"));
                        if ui.small_button("square").on_hover_text("lines = pixels").clicked() {
                            r[1] = r[0];
                        }
                    });
                    e.resolution = set.then_some(r);
                });
                ui.end_row();

                ui.label("Speed");
                ui.horizontal(|ui| {
                    let mut set = e.speed.is_some();
                    ui.checkbox(&mut set, "set");
                    let mut sp = e.speed.unwrap_or(Speed::LineTime(1.));
                    // Convert using this entry's frame width so the value
                    // means the same speed after switching units.
                    let w = e.frame.size.map_or(cur.width, |s| s[0]).max(1e-9);
                    ui.add_enabled_ui(set, |ui| {
                        let is_time = matches!(sp, Speed::LineTime(_));
                        match &mut sp {
                            Speed::LineTime(t) => {
                                ui.add(DragValue::new(t).speed(0.01).range(0.001..=1000.0).suffix(" s/line"));
                            }
                            Speed::Linear(v) => {
                                ui.add(DragValue::new(v).speed(1.).range(0.001..=f64::MAX).suffix(" nm/s"));
                            }
                        }
                        if ui.radio(is_time, "s/line").clicked() && !is_time {
                            if let Speed::Linear(v) = sp {
                                sp = Speed::LineTime(w / v.max(1e-9));
                            }
                        }
                        if ui.radio(!is_time, "nm/s").clicked() && is_time {
                            if let Speed::LineTime(t) = sp {
                                sp = Speed::Linear(w / t.max(1e-9));
                            }
                        }
                    })
                    .response
                    .on_hover_text("Forward speed; Nanonis's backward/forward speed ratio is kept");
                    e.speed = set.then_some(sp);
                });
                ui.end_row();

                ui.label("Wait before scan");
                ui.add(DragValue::new(&mut e.wait_before_s).range(0.0..=f64::MAX).suffix(" s"));
                ui.end_row();

                ui.label("Wait after scan");
                ui.add(DragValue::new(&mut e.wait_after_s).range(0.0..=f64::MAX).suffix(" s"));
                ui.end_row();

                ui.label("Direction");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut e.dir, Dir::Alternate, "⇅ Alternate")
                        .on_hover_text("Opposite direction to the previous scan");
                    ui.radio_value(&mut e.dir, Dir::Down, "↓ Down");
                    ui.radio_value(&mut e.dir, Dir::Up, "↑ Up");
                });
                ui.end_row();

                ui.label("Scan");
                ui.checkbox(&mut e.scan, "take the scan")
                    .on_hover_text("Off: this entry only sets things up, moves the tip and runs spectroscopy");
                ui.end_row();

                ui.label("Warm-up");
                ui.add_enabled(e.scan, egui::Checkbox::new(&mut e.warm_up_scan, "scan once first, discard it"))
                    .on_hover_text(
                        "Scan twice and keep only the second image. The piezo creeps most \
                         right after a coarse move and approach; the first scan uses up that \
                         time, so the kept image is less distorted. Doubles the scan time.",
                    );
                ui.end_row();

                ui.label("Stamp");
                ui.add_enabled(e.scan, egui::Checkbox::new(&mut e.stamp, "add image to layers when done"));
                ui.end_row();
            });
        ui.add_space(4.);
        egui::CollapsingHeader::new(RichText::new("Tip moves & spectroscopy (after the scan)").strong())
            .id_salt(("seq tip", i))
            .default_open(!e.tip.points.is_empty())
            .show(ui, |ui| show_tip_plan(ui, e, tip_pos, &mut drag));
        self.drag = drag;
        if e.bias == Some(0.) {
            ui.colored_label(Color32::ORANGE, "⚠ Bias is 0 V");
        }
        if e.current.is_some_and(|c| c <= 0.) {
            ui.colored_label(Color32::ORANGE, "⚠ Current setpoint is ≤ 0");
        }
        if e.coarse.is_some_and(|c| c.withdraw && !c.approach) {
            ui.colored_label(
                Color32::ORANGE,
                "⚠ Withdraw without approach: the scan will run withdrawn",
            );
        }
    }

    fn show_steps(&self, ui: &mut Ui, last_dir: Dir) {
        ui.label(RichText::new("Steps").strong());
        let Some(i) = self.selected.filter(|&i| i < self.seq.entries.len()) else {
            return;
        };
        // While running, show the snapshot actually being executed.
        if let Some(run) = &self.run {
            let steps: Vec<_> = run
                .steps
                .iter()
                .enumerate()
                .filter(|(_, (entry, _))| *entry == i)
                .collect();
            if !steps.is_empty() {
                for (n, (global, (_, step))) in steps.into_iter().enumerate() {
                    let text = format!("{} {step}", n + 1);
                    if global == run.current {
                        let extra = match (step, run.wait_remaining) {
                            (Step::Wait(_), Some(rem)) => format!("   ◀ {rem:.0} s"),
                            (Step::Spectroscopy(_), _) if run.detail.is_some() => {
                                format!("   ◀ {}", run.detail.as_deref().unwrap_or_default())
                            }
                            _ if run.paused => "   ◀ paused".into(),
                            _ => "   ◀".into(),
                        };
                        ui.label(
                            RichText::new(format!("{text}{extra}"))
                                .strong()
                                .color(Color32::LIGHT_GREEN),
                        );
                    } else if global < run.current {
                        ui.weak(format!("{text} ✓"));
                    } else {
                        ui.label(text);
                    }
                }
                return;
            }
        }
        let e = &self.seq.entries[i];
        if !e.enabled {
            ui.weak("(disabled — will be skipped)");
            for (n, step) in expand(e).into_iter().enumerate() {
                ui.weak(format!("{} {step}", n + 1));
            }
            return;
        }
        // Compile the whole sequence so alternating directions are resolved.
        let steps = compile(&self.seq, 0, last_dir)
            .into_iter()
            .filter(|(entry, _)| *entry == i);
        for (n, (_, step)) in steps.enumerate() {
            ui.label(format!("{} {step}", n + 1));
        }
    }

    fn show_sweep(&mut self, ui: &mut Ui) {
        egui::CollapsingHeader::new("Image Bias Sequence")
            .default_open(false)
            .show(ui, |ui| {
                let sw = &mut self.sweep;
                ui.horizontal(|ui| {
                    ui.label("Vary");
                    let prev = sw.param;
                    ui.radio_value(&mut sw.param, SweepParam::Bias, "Bias");
                    ui.radio_value(&mut sw.param, SweepParam::Current, "Current");
                    if prev != sw.param {
                        (sw.start, sw.stop, sw.step, sw.list) = match sw.param {
                            SweepParam::Bias => (-1., 1., 0.5, "-1, -0.5, 0.2, 1.5".into()),
                            SweepParam::Current => (50., 200., 50., "20, 50, 100".into()),
                        };
                    }
                });
                let unit = match sw.param {
                    SweepParam::Bias => " V",
                    SweepParam::Current => " pA",
                };
                let speed = match sw.param {
                    SweepParam::Bias => 0.01,
                    SweepParam::Current => 1.,
                };
                ui.horizontal(|ui| {
                    ui.radio_value(&mut sw.mode, SweepMode::Count, "even spacing");
                    ui.radio_value(&mut sw.mode, SweepMode::Step, "step size");
                    ui.radio_value(&mut sw.mode, SweepMode::List, "list");
                });
                let values = match sw.mode {
                    SweepMode::Count => {
                        ui.horizontal(|ui| {
                            ui.label("from");
                            ui.add(DragValue::new(&mut sw.start).speed(speed).suffix(unit));
                            ui.label("to");
                            ui.add(DragValue::new(&mut sw.stop).speed(speed).suffix(unit));
                            ui.label("in");
                            ui.add(DragValue::new(&mut sw.count).range(1..=1000));
                            ui.label("scans");
                        });
                        Ok(sweep_values(sw.start, sw.stop, sw.count))
                    }
                    SweepMode::Step => {
                        ui.horizontal(|ui| {
                            ui.label("from");
                            ui.add(DragValue::new(&mut sw.start).speed(speed).suffix(unit));
                            ui.label("to");
                            ui.add(DragValue::new(&mut sw.stop).speed(speed).suffix(unit));
                            ui.label("step");
                            ui.add(DragValue::new(&mut sw.step).speed(speed / 10.).suffix(unit));
                        });
                        step_values(sw.start, sw.stop, sw.step)
                    }
                    SweepMode::List => {
                        ui.horizontal(|ui| {
                            ui.label(format!("values ({})", unit.trim()));
                            ui.add(
                                egui::TextEdit::singleline(&mut sw.list)
                                    .desired_width(260.)
                                    .hint_text("e.g. -1, -0.5, 0.2, 1.5"),
                            )
                            .on_hover_text("Separate with commas, semicolons or spaces. Scans run in this order.");
                        });
                        list_values(&sw.list)
                    }
                };
                let values = match values {
                    Ok(v) => {
                        let preview = v
                            .iter()
                            .map(|v| format!("{v:.3}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        ui.weak(format!("Values: {preview}{unit}"));
                        v
                    }
                    Err(msg) => {
                        ui.colored_label(Color32::RED, msg);
                        Vec::new()
                    }
                };
                let template_label = match self.selected.filter(|&i| i < self.seq.entries.len()) {
                    Some(i) => format!("Template: entry {} (waits, dir, frame, other settings)", i + 1),
                    None => "Template: default entry (select one to use it as template)".into(),
                };
                ui.weak(template_label);
                if ui
                    .add_enabled(
                        !values.is_empty(),
                        egui::Button::new(format!("Append {} entries", values.len())),
                    )
                    .on_hover_text(
                        "Copies the selected entry for each value. Only the first copy keeps the \
                         template's coarse move and frame shift, so the rest image the same spot.",
                    )
                    .clicked()
                {
                    let template = self
                        .selected
                        .and_then(|i| self.seq.entries.get(i).cloned())
                        .unwrap_or_default();
                    let param = self.sweep.param;
                    for (n, v) in values.into_iter().enumerate() {
                        let mut e = template.clone();
                        e.enabled = true;
                        if n > 0 {
                            e.coarse = None;
                            if matches!(e.frame.position, Position::Shift(_)) {
                                e.frame.position = Position::Keep;
                            }
                        }
                        match param {
                            SweepParam::Bias => {
                                e.bias = Some(v as f32);
                                e.name = format!("bias {v:.3} V");
                            }
                            SweepParam::Current => {
                                e.current = Some((v * 1e-12) as f32);
                                e.name = format!("I {v:.1} pA");
                            }
                        }
                        self.seq.entries.push(e);
                    }
                }
            });
    }

    fn show_survey(&mut self, ui: &mut Ui, current_frame: Option<ScanFrame>, course: &CourseMotionState) {
        egui::CollapsingHeader::new("Survey Grid")
            .default_open(false)
            .show(ui, |ui| {
                let sv = &mut self.survey;
                ui.weak(
                    "Scans a grid of tiles: scan and stamp, coarse move by one frame \
                     (minus the overlap), scan again. Tiles are visited in a snake, \
                     so every move is one tile.",
                );
                ui.horizontal(|ui| {
                    ui.label("Tiles");
                    ui.add(DragValue::new(&mut sv.cols).range(1..=50));
                    ui.label("across ×");
                    ui.add(DragValue::new(&mut sv.rows).range(1..=50));
                    ui.label("down");
                    ui.separator();
                    ui.label("Overlap");
                    ui.add(DragValue::new(&mut sv.overlap_pct).range(0. ..=90.).speed(0.5).suffix(" %"))
                        .on_hover_text(
                            "Share of each frame that also appears in the next tile. Coarse \
                             steps are not exact, so keep some margin or tiles may leave gaps. \
                             When the survey ends the tiles are lined up from these overlaps, \
                             which works best at 15\u{2013}25 %.",
                        );
                });
                ui.horizontal(|ui| {
                    ui.label("Grow towards");
                    ui.selectable_value(&mut sv.plus_x, true, "+X");
                    ui.selectable_value(&mut sv.plus_x, false, "−X");
                    ui.selectable_value(&mut sv.plus_y, true, "+Y");
                    ui.selectable_value(&mut sv.plus_y, false, "−Y");
                    ui.separator();
                    ui.label("Settle after approach");
                    ui.add(DragValue::new(&mut sv.settle_s).range(0. ..=3600.).speed(1.).suffix(" s"));
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut sv.track, "Correct moves as it runs").on_hover_text(
                        "Each tile is placed on the map by matching it with the previous one. \
                         If one is in the wrong place, drag it to the right spot: the step \
                         size and the next moves are worked out from where the tiles are. \
                         Keep 30 % overlap or more.",
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Scan direction");
                    ui.selectable_value(&mut sv.dir, Dir::Alternate, "⇅ Alternate").on_hover_text(
                        "Each tile scans the opposite way to the one before (no time lost \
                         going back to the start line). Drift then stretches one tile and \
                         squeezes the next, so neighbours may not quite meet.",
                    );
                    ui.selectable_value(&mut sv.dir, Dir::Down, "↓ All down").on_hover_text(
                        "Every tile scans downwards: drift distorts every tile the same way, \
                         so the seams fit better.",
                    );
                    ui.selectable_value(&mut sv.dir, Dir::Up, "↑ All up").on_hover_text(
                        "Every tile scans upwards: drift distorts every tile the same way, \
                         so the seams fit better.",
                    );
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut sv.warm_up, "Warm-up scan after each move").on_hover_text(
                        "Scan each tile twice after its coarse move and keep only the second \
                         image. The piezo creeps most right after a move and approach; the \
                         first scan uses up that time, so kept tiles are less distorted and \
                         fit together better. Doubles the scan time of every tile but the \
                         first. Raising \u{201c}Settle after approach\u{201d} helps the same way \
                         without scanning.",
                    );
                    if sv.warm_up && sv.dir == Dir::Alternate {
                        ui.weak("(warm-up and kept scans go opposite ways: every kept tile scans the same way)");
                    }
                });

                // Tile size: the template's frame if it sets one, else Nanonis' frame.
                let template = self
                    .selected
                    .and_then(|i| self.seq.entries.get(i).cloned())
                    .unwrap_or_default();
                let frame = current_frame.map(|f| template.frame.apply(f));
                let Some(frame) = frame.filter(|f| f.width > 0. && f.height > 0.) else {
                    ui.colored_label(Color32::RED, "No scan frame yet: connect to Nanonis first.");
                    return;
                };
                // Frame axes in piezo nm: one tile right and one tile along +Y.
                let axes = frame.to_transform().matrix2;
                let keep = 1. - sv.overlap_pct / 100.;
                let sx = if sv.plus_x { 1. } else { -1. };
                let sy = if sv.plus_y { 1. } else { -1. };
                let step_x = axes * DVec2::new(sx * keep, 0.);
                // The frame transform flips y (scale -height): -1 is +Y.
                let step_y = axes * DVec2::new(0., -sy * keep);
                let steps_x = course.real_to_steps(step_x);
                // The snake goes back the other way on alternate rows; the
                // motor may step differently backwards.
                let steps_x_back = course.real_to_steps(-step_x);
                let steps_y = course.real_to_steps(step_y);
                // What the rounded steps really give, as overlap.
                let real_overlap = |steps: IVec2, axis: DVec2, size: f64| {
                    let moved = course.steps_to_real(steps).dot(axis.normalize_or_zero()).abs();
                    (1. - moved / size) * 100.
                };
                let ov_x = real_overlap(steps_x, step_x, frame.width)
                    .min(real_overlap(steps_x_back, step_x, frame.width));
                let ov_y = real_overlap(steps_y, step_y, frame.height);
                let ov_min = match (sv.cols > 1, sv.rows > 1) {
                    (true, true) => ov_x.min(ov_y),
                    (true, false) => ov_x,
                    (false, true) => ov_y,
                    (false, false) => 100.,
                };
                if sv.track && ov_min < 30. {
                    ui.colored_label(
                        Color32::ORANGE,
                        format!(
                            "Only {ov_min:.0} % overlap: tiles may not share enough to be matched \
                             (look-alikes win). Use 30 % or more for \u{201c}Correct moves as it runs\u{201d}."
                        ),
                    );
                }
                let tiles = sv.cols * sv.rows;
                let scans = if sv.warm_up { 2 * tiles - 1 } else { tiles };
                ui.weak(format!(
                    "Tile {:.2} × {:.2} µm · area ≈ {:.1} × {:.1} µm · {} scans, {} coarse moves",
                    frame.width * 1e-3,
                    frame.height * 1e-3,
                    (frame.width * (1. + (sv.cols - 1) as f64 * keep)) * 1e-3,
                    (frame.height * (1. + (sv.rows - 1) as f64 * keep)) * 1e-3,
                    scans,
                    tiles - 1,
                ));
                ui.weak(format!(
                    "Move per tile: {:?} / back {:?} steps across ({ov_x:.0} % overlap), {:?} steps down ({ov_y:.0} % overlap)",
                    steps_x.to_array(),
                    steps_x_back.to_array(),
                    steps_y.to_array(),
                ));
                let bad = (sv.cols > 1 && (steps_x == IVec2::ZERO || steps_x_back == IVec2::ZERO || ov_x < 0.))
                    || (sv.rows > 1 && (steps_y == IVec2::ZERO || ov_y < 0.));
                if bad {
                    ui.colored_label(
                        Color32::RED,
                        "With the current coarse calibration the moves would not tile the frame. \
                         Check the Coarse Motion calibration or the overlap.",
                    );
                }
                let template_label = match self.selected.filter(|&i| i < self.seq.entries.len()) {
                    Some(i) => format!("Template: entry {} (bias, current, speed, waits)", i + 1),
                    None => "Template: default entry (select one to use it as template)".into(),
                };
                ui.weak(template_label);
                if ui
                    .add_enabled(!bad, egui::Button::new(format!("Append {tiles} grid entries")))
                    .on_hover_text(
                        "The first tile is scanned where the tip is now. Every tile is \
                         stamped. Check the plan on the map before running.",
                    )
                    .clicked()
                {
                    let order = survey_order(sv.cols, sv.rows);
                    let mut prev: Option<(u32, u32)> = None;
                    for (c, r) in order {
                        let mut e = template.clone();
                        e.enabled = true;
                        e.scan = true;
                        e.stamp = true;
                        e.dir = sv.dir;
                        e.tip = Default::default();
                        // Same frame for every tile; the stage does the moving.
                        if matches!(e.frame.position, Position::Shift(_)) {
                            e.frame.position = Position::Keep;
                        }
                        e.name = format!("grid r{} c{}", r + 1, c + 1);
                        e.grid = Some([r, c]);
                        e.grid_pos = Some((step_x * c as f64 + step_y * r as f64).to_array());
                        e.coarse = prev.map(|(pc, pr)| {
                            let steps = if pr != r { steps_y } else if c > pc { steps_x } else { steps_x_back };
                            let base = template.coarse.unwrap_or_default();
                            CoarseSpec { steps: steps.to_array(), withdraw: base.withdraw, approach: base.approach }
                        });
                        if prev.is_some() {
                            e.settle_s = e.settle_s.max(sv.settle_s);
                            e.warm_up_scan = sv.warm_up;
                        }
                        self.seq.entries.push(e);
                        prev = Some((c, r));
                    }
                }
            });
    }

    fn show_run_controls(
        &mut self,
        ui: &mut Ui,
        course: &CourseMotionState,
        last_dir: Dir,
        scanning: bool,
        blocked_by: Option<&str>,
    ) {
        if let (Some(reason), false) = (blocked_by, self.is_active()) {
            ui.colored_label(Color32::ORANGE, reason);
        }
        // A scan started by hand is running: queue behind it or adopt it.
        let current = (scanning && !self.is_active()).then_some(self.current_scan);
        if current.is_some() {
            ui.horizontal(|ui| {
                ui.label(format!("A scan {} is running:", last_dir.arrow()));
                ui.radio_value(&mut self.current_scan, CurrentScan::After, "start the sequence when it ends")
                    .on_hover_text(
                        "The sequence waits for the scan to finish, then runs.                          Abort while waiting leaves the scan running.",
                    );
                ui.radio_value(&mut self.current_scan, CurrentScan::Include, "add it to the sequence")
                    .on_hover_text(
                        "The running scan becomes the first scan of the sequence: it is stamped                          when done, Pause/Abort act on it, and alternating entries follow on from                          its direction.",
                    );
            });
        }
        ui.horizontal(|ui| {
            let active = self.is_active();
            // Starting is blocked while another tool uses the tip.
            let has_entries =
                self.seq.entries.iter().any(|e| e.enabled) && blocked_by.is_none();
            if ui
                .add_enabled(!active && has_entries, egui::Button::new("▶ Run all"))
                .clicked()
            {
                self.start(0, course.group_index(), last_dir, current);
            }
            let sel = self.selected.filter(|&i| i < self.seq.entries.len());
            if ui
                .add_enabled(
                    !active && sel.is_some() && has_entries,
                    egui::Button::new("▶ Run from selected"),
                )
                .clicked()
            {
                self.start(sel.unwrap(), course.group_index(), last_dir, current);
            }
            let paused = self.run.as_ref().is_some_and(|r| r.paused);
            if paused {
                let asked = self.grid_approval.is_some();
                if ui
                    .add_enabled(!asked, egui::Button::new("\u{23f5} Continue"))
                    .on_disabled_hover_text("Accept or reject the grid's correction first (below)")
                    .on_hover_text(
                        "Carry on from where it stopped, keeping any changes you made. \
                         A scan that was paused is resumed; one that was stopped is scanned again.",
                    )
                    .clicked()
                {
                    self.send(SequenceCommand::Resume);
                }
                if ui
                    .button("\u{21bb} Restart entry")
                    .on_hover_text(
                        "Run the current entry again from its start: its bias, current, frame \
                         and speed are set again and its scan starts over. A coarse move that \
                         already finished is not repeated.",
                    )
                    .clicked()
                {
                    self.send(SequenceCommand::RestartEntry);
                }
            } else if ui
                .add_enabled(active, egui::Button::new("⏸ Pause"))
                .on_hover_text(
                    "Pause now. A running scan is paused in Nanonis and continues on Resume; \
                     a wait is cut short and restarts on Resume.",
                )
                .clicked()
            {
                self.status = "Pausing…".into();
                self.send(SequenceCommand::Pause);
            }
            if ui
                .add_enabled(active, egui::Button::new("■ Abort"))
                .on_hover_text("Stop now. A scan in progress is stopped.")
                .clicked()
            {
                self.status = "Aborting…".into();
                self.send(SequenceCommand::Abort);
            }
            if ui
                .toggle_value(&mut self.pause_after_scan, "⏸ After scan")
                .on_hover_text(
                    "Pause once the scan in progress (or the next one, if none is running) \
                     is done and stamped. Turns itself off when it pauses.",
                )
                .changed()
            {
                self.send(SequenceCommand::PauseAfterScan(self.pause_after_scan));
            }
            ui.separator();
            let progress = match &self.run {
                Some(run) => {
                    let entry = match run.steps.get(run.current) {
                        Some((CURRENT_SCAN, _)) => "current scan".to_string(),
                        Some((i, _)) => format!("entry {}", i + 1),
                        None => "entry 0".to_string(),
                    };
                    format!("{} · {entry} · {}", self.status, Self::run_position(run))
                }
                None => self.status.clone(),
            };
            ui.label(progress);
            if self.is_running() {
                ui.spinner();
            }
            if self.plan_hidden.is_some()
                && ui
                    .small_button("Show plan")
                    .on_hover_text("The plan was hidden from the map when the sequence finished")
                    .clicked()
            {
                self.plan_hidden = None;
            }
        });
        self.show_eta(ui, last_dir, current);
        if let (Some(reason), true) = (&self.interrupted, self.run.as_ref().is_some_and(|r| r.paused)) {
            ui.colored_label(
                Color32::ORANGE,
                format!(
                    "Held: {reason}. Nothing more is sent to Nanonis until you choose \
                     Continue, Restart entry or Abort."
                ),
            );
        }
        if let Some(err) = &self.last_error {
            ui.colored_label(Color32::RED, format!("Error: {err}"));
        }
        self.show_rebuild(ui);
        self.show_grid_track(ui);
    }

    /// Name of sequence entry `i` (for the grid's lists).
    fn entry_name(&self, i: usize) -> String {
        self.seq.entries.get(i).map(|e| e.name.clone()).unwrap_or_else(|| format!("entry {}", i + 1))
    }

    /// A tracked grid: the question waiting for the user, what it fixed by
    /// itself (each undoable), and its settings.
    fn show_grid_track(&mut self, ui: &mut Ui) {
        if self.lineup_undo_available
            && ui
                .button("Undo line-up of survey tiles")
                .on_hover_text("Put the tiles back where they were before the survey's tiles were lined up.")
                .clicked()
        {
            self.lineup_undo_request = true;
        }
        let Some(t) = &self.grid_track else { return };
        ui.weak(t.summary());
        let tiles = t.tiles_info();
        let (n, calib, learning) = (t.n_moves(), t.calib, t.learning());
        let pending = self.next_move_plan();

        // The question, while the run is held for it.
        let mut answer = None;
        if let Some(a) = &self.grid_approval {
            egui::Frame::group(ui.style()).stroke(egui::Stroke::new(1.5, Color32::ORANGE)).show(ui, |ui| {
                ui.colored_label(Color32::ORANGE, "The grid wants to correct the survey. Held until you choose.");
                if let Some(info) = tiles.get(a.tile) {
                    ui.label(format!(
                        "{}: {} \u{2014} placed {:.0} nm from where the coarse calibration put it, {:.0} nm from its plan.",
                        self.entry_name(info.entry),
                        placement_text(info.placement),
                        info.auto_shift_nm,
                        info.error_nm.unwrap_or(0.)
                    ));
                }
                match a.next {
                    Some((next, _, steps)) => {
                        let planned = self.grid_planned.get(&next).copied().unwrap_or(steps);
                        ui.label(format!(
                            "Move to {}: {:?} steps instead of the planned {:?}.",
                            self.entry_name(next),
                            steps,
                            planned
                        ));
                    }
                    None => {
                        ui.label("The next move stays as planned.");
                    }
                }
                ui.weak(
                    "Check the new tile against its neighbours on the map. Accept if it sits right; \
                     reject to put it back where the coarse calibration put it and keep the planned \
                     move. You can also drag the tile into place first.",
                );
                ui.horizontal(|ui| {
                    if ui.button("\u{2714} Accept and continue").clicked() {
                        answer = Some(true);
                    }
                    if ui.button("\u{2716} Reject and continue").clicked() {
                        answer = Some(false);
                    }
                });
            });
        }
        match answer {
            Some(true) => self.approve_grid(),
            Some(false) => self.reject_grid(),
            None => {}
        }

        // Automatic fixes so far, each undoable.
        let mut undo = None;
        let mut pin = None;
        let mut learn = None;
        egui::CollapsingHeader::new(format!("Automatic fixes ({} tiles)", tiles.len()))
            .id_salt("grid_fixes")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("grid_fix_list").num_columns(4).striped(true).show(ui, |ui| {
                    for info in &tiles {
                        ui.label(self.entry_name(info.entry));
                        ui.label(placement_text(info.placement));
                        match info.error_nm {
                            Some(e) => ui.weak(format!("{e:.0} nm from plan")),
                            None => ui.weak("start"),
                        };
                        if info.placement == Placement::First {
                            ui.label("");
                        } else if info.undone {
                            if ui
                                .small_button("Redo placement")
                                .on_hover_text("Put the tile back where the automatic placement put it.")
                                .clicked()
                            {
                                undo = Some((info.index, false));
                            }
                        } else if ui
                            .small_button("Undo placement")
                            .on_hover_text(format!(
                                "Move the tile {:.0} nm back to where the coarse calibration put it, and \
                                 don't learn the step size from it.",
                                info.auto_shift_nm
                            ))
                            .clicked()
                        {
                            undo = Some((info.index, true));
                        }
                        ui.end_row();
                    }
                });
                if let Some((next, at, planned, steps)) = pending {
                    let pinned = self.grid_pinned.contains(&next);
                    ui.horizontal(|ui| {
                        if pinned {
                            ui.label(format!("Next move to {}: kept as planned {:?}", self.entry_name(next), planned));
                            if ui.small_button("Let the grid correct it").clicked() {
                                pin = Some((next, at, false));
                            }
                        } else if steps != planned {
                            ui.label(format!(
                                "Next move to {}: {:?} (planned {:?})",
                                self.entry_name(next),
                                steps,
                                planned
                            ));
                            if ui.small_button("Use the planned move").clicked() {
                                pin = Some((next, at, true));
                            }
                        } else {
                            ui.label(format!("Next move to {}: as planned {:?}", self.entry_name(next), planned));
                        }
                    });
                }
                let mut on = learning;
                if ui
                    .checkbox(&mut on, "Learn the step size from the tiles")
                    .on_hover_text(
                        "Off: moves use the coarse calibration the run started with, corrected only \
                         for where the latest tile is.",
                    )
                    .changed()
                {
                    learn = Some(on);
                }
            });
        if let Some((k, on)) = undo {
            self.set_tile_undone(k, on);
        }
        if let Some((next, at, on)) = pin {
            self.pin_move(next, at, on);
        }
        if let Some(on) = learn {
            if let Some(t) = &mut self.grid_track {
                t.set_learning(on);
            }
            self.correct_next_move();
        }
        ui.checkbox(&mut self.survey.approve, "Ask before correcting a move or placing a tile off its plan")
            .on_hover_text(
                "Hold the run after each tile whose next move would change, or that was placed more \
                 than a tenth of the tile spacing away from where the coarse calibration put it.",
            );
        if n > 0
            && ui
                .button(format!("Save grid calibration ({n} moves)"))
                .on_hover_text(
                    "Use the step sizes learned from where the tiles are on the map as the \
                     coarse calibration. Drag any misplaced tiles into place first.",
                )
                .clicked()
        {
            self.grid_calib_save = Some((calib, n));
        }
    }

    /// "Rebuild & restart": save the run at a safe point, rebuild the app
    /// from source and pick the run up again in the new build.
    fn show_rebuild(&mut self, ui: &mut Ui) {
        if crate::restart::source_dir().is_none() {
            return;
        }
        ui.horizontal(|ui| {
            if self.restart_pending {
                ui.spinner();
                ui.label("Restart waits for a safe point (a coarse move, tip move or I(V) series finishes first)");
                if ui.small_button("Cancel").clicked() {
                    self.restart_pending = false;
                    self.send(SequenceCommand::CancelDetach);
                }
                return;
            }
            let hover = if self.is_active() {
                "Close the app, rebuild it from source and start the new build. The sequence \
                 is saved first: a running scan keeps going in Nanonis, and the new app \
                 reopens this project and carries on where the sequence was (a paused \
                 sequence stays paused). If the build fails, its window shows why; run it \
                 again once fixed."
            } else {
                "Close the app, rebuild it from source and start the new build, which reopens \
                 this project. If the build fails, its window shows why; run it again once fixed."
            };
            if ui.button("🔨 Rebuild & restart").on_hover_text(hover).clicked() {
                if self.is_active() {
                    self.restart_pending = true;
                    self.send(SequenceCommand::Detach);
                } else {
                    crate::restart::request();
                }
            }
        });
    }

    /// Progress bar with the time left while running; the estimated length
    /// of the whole sequence before it starts.
    fn show_eta(&self, ui: &mut Ui, last_dir: Dir, current: Option<CurrentScan>) {
        if self.run.is_none() {
            let Some(scan) = self.scan_now else {
                ui.weak("Estimated time: connect to Nanonis to estimate.");
                return;
            };
            let mut steps = compile(&self.seq, 0, last_dir);
            if steps.is_empty() {
                return;
            }
            if let Some(mode) = current {
                steps = with_current_scan(steps, mode, last_dir, true);
            }
            let clock = RunClock::new(self.times.clone());
            let total = eta::remaining(&steps, 0, scan, &clock, None, self.scan_left);
            ui.weak(format!("Estimated time for Run all: ~{}", eta::fmt_duration(total)))
                .on_hover_text(
                    "Scans: lines \u{d7} Nanonis' time per line. Coarse moves, approaches and \
                     I(V) series use rough defaults until a run has timed them.",
                );
            return;
        }
        self.progress_bar(ui);
    }

    /// Progress of the running sequence with the time left.
    fn progress_bar(&self, ui: &mut Ui) {
        if let Some((frac, text)) = self.progress() {
            ui.add(egui::ProgressBar::new(frac).text(text).desired_height(16.));
        }
    }

    /// Calibration run progress: `Some((fraction, text))` while its scans
    /// run, and whether the scans are being compared.
    pub fn calibration_progress(&self) -> (Option<(f32, String)>, bool) {
        let running = self.cal.is_some().then(|| self.progress()).flatten();
        (running, self.cal_measuring)
    }

    /// "scan 3/4": the scan running (or next, between scans) out of the
    /// run's scans; "step k/n" for a run without scans.
    fn run_position(run: &RunView) -> String {
        let is_scan = |s: &Step| matches!(s, Step::Scan(_) | Step::AwaitScan { include: true, .. });
        let total = run.steps.iter().filter(|(_, s)| is_scan(s)).count();
        if total == 0 {
            return format!("step {}/{}", (run.current + 1).min(run.steps.len()), run.steps.len());
        }
        let done = run.steps.iter().take(run.current).filter(|(_, s)| is_scan(s)).count();
        format!("scan {}/{total}", (done + 1).min(total))
    }

    /// Fraction done and the bar's text, while a run is active.
    fn progress(&self) -> Option<(f32, String)> {
        let run = self.run.as_ref()?;
        let step = Self::run_position(run);
        let (frac, text) = match self.run_eta() {
            Some((left, frac)) => {
                let text = if run.paused {
                    format!("{step} \u{b7} paused \u{b7} ~{} left", eta::fmt_duration(left))
                } else {
                    format!("{step} \u{b7} ~{} left", eta::fmt_duration(left))
                };
                (frac, text)
            }
            None => (run.current as f32 / run.steps.len().max(1) as f32, step),
        };
        Some((frac, text))
    }
}

fn sweep_values(start: f64, stop: f64, count: u32) -> Vec<f64> {
    match count {
        0 => vec![],
        1 => vec![start],
        n => (0..n)
            .map(|k| start + (stop - start) * k as f64 / (n - 1) as f64)
            .collect(),
    }
}

/// From `start` towards `stop` in increments of `|step|`, including `stop`
/// when it lands on a step.
fn step_values(start: f64, stop: f64, step: f64) -> Result<Vec<f64>, String> {
    const MAX: usize = 1000;
    if start == stop {
        return Ok(vec![start]);
    }
    if step == 0. || !step.is_finite() {
        return Err("Step must not be 0".into());
    }
    let step = step.abs() * (stop - start).signum();
    let n = ((stop - start) / step + 1e-9).floor() as usize;
    if n >= MAX {
        return Err(format!("Too many values ({}); use a bigger step", n + 1));
    }
    Ok((0..=n).map(|k| start + step * k as f64).collect())
}

/// Parse a list like "-1, -0.5 0.2;1.5".
fn list_values(text: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    for tok in text
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|t| !t.is_empty())
    {
        match tok.parse::<f64>() {
            Ok(v) if v.is_finite() => values.push(v),
            _ => return Err(format!("Not a number: \"{tok}\"")),
        }
    }
    if values.is_empty() {
        return Err("Enter at least one value".into());
    }
    Ok(values)
}

/// The tip positions of an entry and the I(V) settings used at each.
fn show_tip_plan(ui: &mut Ui, e: &mut ScanEntry, tip_pos: Option<DVec2>, drag: &mut Option<DragTarget>) {
    let t = &mut e.tip;
    ui.horizontal(|ui| {
        drag_toggle(ui, drag, DragTarget::Tip);
        if ui
            .add_enabled(tip_pos.is_some(), egui::Button::new("+ Where the tip is"))
            .on_hover_text("Add the tip's current position")
            .clicked()
        {
            if let Some(p) = tip_pos {
                t.points.push(p.to_array());
            }
        }
        if ui.add_enabled(!t.points.is_empty(), egui::Button::new("Clear")).clicked() {
            t.points.clear();
        }
    });
    if t.points.is_empty() {
        ui.weak("No tip positions: the entry just scans.");
    } else {
        let mut remove = None;
        egui::Grid::new(("seq tip points", e.name.as_str())).num_columns(3).show(ui, |ui| {
            for (k, p) in t.points.iter_mut().enumerate() {
                ui.label(format!("P{:02}", k + 1));
                ui.horizontal(|ui| {
                    ui.add(DragValue::new(&mut p[0]).speed(0.5).max_decimals(1).prefix("x ").suffix(" nm"));
                    ui.add(DragValue::new(&mut p[1]).speed(0.5).max_decimals(1).prefix("y ").suffix(" nm"));
                });
                if ui.small_button("🗑").on_hover_text("Remove this position").clicked() {
                    remove = Some(k);
                }
                ui.end_row();
            }
        });
        if let Some(k) = remove {
            t.points.remove(k);
        }
    }
    ui.horizontal(|ui| {
        ui.label("Settle at each position");
        ui.add(DragValue::new(&mut t.settle_s).range(0.0..=f64::MAX).speed(0.1).suffix(" s"));
    });
    ui.checkbox(&mut t.spectroscopy, "Run an I(V) series at each position");
    ui.add_enabled_ui(t.spectroscopy, |ui| {
        let iv = &mut t.iv;
        egui::Grid::new(("seq tip iv", e.name.as_str())).num_columns(2).show(ui, |ui| {
            ui.label("Curves per position");
            ui.add(DragValue::new(&mut iv.num_curves).range(1..=1000));
            ui.end_row();
            ui.label("Base name");
            ui.text_edit_singleline(&mut iv.base_name)
                .on_hover_text("Each position is saved as <base>_<entry>_P01, _P02, …");
            ui.end_row();
            ui.label("Z change per curve");
            ui.add(DragValue::new(&mut iv.z_step_pm).speed(1.).suffix(" pm"))
                .on_hover_text("Relative to the reference height; negative moves the tip closer");
            ui.end_row();
            ui.label("Between curves");
            ui.horizontal(|ui| {
                ui.radio_value(&mut iv.feedback_between, false, "Z ctrl OFF");
                ui.radio_value(&mut iv.feedback_between, true, "Z ctrl ON");
            });
            ui.end_row();
            ui.label("Settle time");
            ui.add_enabled(
                iv.feedback_between,
                DragValue::new(&mut iv.delay_s).speed(0.05).range(0.0..=3600.0).suffix(" s"),
            );
            ui.end_row();
        });
        ui.weak(
            "Sweep range, points and channels come from the Nanonis Bias Spectroscopy module. \
             Each series is saved to the Nanonis session folder (curves, one combined plot \
             .jpg/.svg and one table) and shown in the I(V) window.",
        );
        if let Err(msg) = iv.validate() {
            ui.colored_label(Color32::ORANGE, format!("⚠ {msg}"));
        }
    });
}

/// A small "drag on map" toggle button. Returns true when switched on.
fn drag_toggle(ui: &mut Ui, drag: &mut Option<DragTarget>, target: DragTarget) -> bool {
    let on = *drag == Some(target);
    let clicked = ui
        .selectable_label(on, "✥ drag on map")
        .on_hover_text("Then hold Ctrl and drag on the map to move it")
        .clicked();
    if clicked {
        *drag = if on { None } else { Some(target) };
    }
    clicked && !on
}

/// How a grid tile was placed, in words.
fn placement_text(p: Placement) -> String {
    match p {
        Placement::First => "first tile".into(),
        Placement::Matched { score } => format!("matched (score {score:.2})"),
        Placement::Predicted(Unmatched::NoMatch) => "no clear match: placed as the moves predict".into(),
        Placement::Predicted(Unmatched::Ambiguous) => "two spots matched alike: placed as the moves predict".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_values_go_either_way() {
        assert_eq!(step_values(-1., 1., 0.5).unwrap(), vec![-1., -0.5, 0., 0.5, 1.]);
        assert_eq!(step_values(1., 0., 0.3).unwrap().len(), 4);
        assert_eq!(step_values(0.2, 0.2, 0.).unwrap(), vec![0.2]);
        assert!(step_values(0., 1., 0.).is_err());
    }

    #[test]
    fn survey_is_a_snake() {
        assert_eq!(
            super::survey_order(3, 2),
            vec![(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]
        );
        assert_eq!(super::survey_order(1, 1), vec![(0, 0)]);
    }

    #[test]
    fn list_values_parse() {
        assert_eq!(list_values("-1, -0.5 0.2;1.5").unwrap(), vec![-1., -0.5, 0.2, 1.5]);
        assert!(list_values("1, x").is_err());
        assert!(list_values("  ").is_err());
    }
}
