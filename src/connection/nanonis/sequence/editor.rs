use std::path::PathBuf;

use crossbeam::channel::{Receiver, Sender};
use egui::{Color32, DragValue, RichText, ScrollArea, Ui};
use glam::{DVec2, IVec2};
use tracing::{error, info};

use crate::{
    components::selectable_list::SelectableList,
    connection::{
        nanonis::{
            course_motion::CourseMotionState,
            sequence::{
                model::{
                    compile, expand, with_current_scan, CurrentScan, Dir, Frame as ScanFrame,
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
    sweep: SweepSettings,
    load_tx: Sender<Result<Sequence, String>>,
    load_rx: Receiver<Result<Sequence, String>>,
    /// After a sequence finishes its plan is hidden from the map, until the
    /// sequence or the selected entry changes (or "Show plan" is clicked).
    pub(super) plan_hidden: Option<(Sequence, Option<usize>)>,
    /// What Run does with a scan that is already running in Nanonis.
    current_scan: CurrentScan,
}

impl SequenceState {
    pub fn new(cmd_tx: Sender<SequenceCommand>, events: Receiver<SequenceEvent>) -> Self {
        let (load_tx, load_rx) = crossbeam::channel::unbounded();
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
            sweep: SweepSettings {
                param: SweepParam::Bias,
                mode: SweepMode::Count,
                start: -1.,
                stop: 1.,
                count: 5,
                step: 0.5,
                list: "-1, -0.5, 0.2, 1.5".into(),
            },
            load_tx,
            load_rx,
            plan_hidden: None,
            current_scan: CurrentScan::After,
        }
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
        self.window_open && self.drag.is_some()
    }

    fn send(&self, cmd: SequenceCommand) {
        let _ = self.cmd_tx.send(cmd);
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
                }
                SequenceEvent::StepStarted { index } => {
                    if let Some(run) = &mut self.run {
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
                }
                SequenceEvent::Paused { index } => {
                    self.status = "Paused".into();
                    if let Some(run) = &mut self.run {
                        run.paused = true;
                        run.current = index;
                        run.wait_remaining = None;
                    }
                }
                SequenceEvent::Resumed => {
                    self.status = "Running".into();
                    self.last_error = None;
                    if let Some(run) = &mut self.run {
                        run.paused = false;
                    }
                }
                SequenceEvent::Finished => {
                    self.status = "Finished".into();
                    self.run = None;
                    self.plan_hidden = Some((self.seq.clone(), self.selected));
                }
                SequenceEvent::Aborted => {
                    self.status = "Aborted".into();
                    self.run = None;
                }
                SequenceEvent::Error { index, message } => {
                    error!("sequence error at step {index}: {message}");
                    self.last_error = Some(message);
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
        self.last_error = None;
        self.status = "Starting".into();
        self.run = Some(RunView {
            steps: steps.clone(),
            current: 0,
            paused: false,
            wait_remaining: None,
            detail: None,
        });
        self.send(SequenceCommand::Run { steps, group });
    }

    /// The "Sequence" button and window. Shown from the Scan Region menu.
    pub fn show_menu(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<Object>,
        course: &CourseMotionState,
        blocked_by: Option<&str>,
    ) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.window_open, egui::Button::new("Sequence"))
                .clicked()
            {
                self.window_open = true;
            }
            if self.is_active() {
                ui.label(&self.status);
            }
        });
        let scan_area = object_list
            .iter_mut()
            .find_map(|entry| entry.as_scan_area_mut());
        let mut open = self.window_open;
        egui::Window::new("Sequence")
            .frame(crate::theme::window_frame(ui.ctx()))
            .default_width(720.)
            // Width is adjustable; height follows the content, so the window
            // shrinks when entry details are hidden.
            .resizable([true, false])
            .collapsible(true)
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                self.show_window(ui, scan_area.as_deref(), course, blocked_by);
            });
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
        }
        ui.separator();
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
                            .on_hover_text("Copy the steps currently planned in the Course Motion window")
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
                if ui.button("⏵ Resume").clicked() {
                    self.send(SequenceCommand::Resume);
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
            ui.separator();
            let progress = match &self.run {
                Some(run) => {
                    let entry = match run.steps.get(run.current) {
                        Some((CURRENT_SCAN, _)) => "current scan".to_string(),
                        Some((i, _)) => format!("entry {}", i + 1),
                        None => "entry 0".to_string(),
                    };
                    format!(
                        "{} · {entry} · step {}/{}",
                        self.status,
                        (run.current + 1).min(run.steps.len()),
                        run.steps.len()
                    )
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
        if let Some(err) = &self.last_error {
            ui.colored_label(Color32::RED, format!("Error: {err}"));
        }
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
    fn list_values_parse() {
        assert_eq!(list_values("-1, -0.5 0.2;1.5").unwrap(), vec![-1., -0.5, 0.2, 1.5]);
        assert!(list_values("1, x").is_err());
        assert!(list_values("  ").is_err());
    }
}
