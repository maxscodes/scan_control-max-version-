//! Learning the coarse calibration from the user's drag corrections.
//!
//! Two layers, so the last calibration test is never lost:
//! - the **baseline**: the result of the last test (Auto calibrate,
//!   Calibrate X/Y, a hand edit or a survey grid), kept by
//!   `CourseMotionState` in `course_calibration.json`;
//! - the **learned** calibration, fitted here from corrections and saved
//!   apart in `course_calibration_learned.json`.
//!
//! The mode picks which one moves use. The corrections themselves are kept
//! in `coarse_corrections.jsonl`, across projects.

use std::{collections::HashMap, path::PathBuf, time::Instant};

use egui::{Button, RichText, Ui};
use glam::DVec2;
use serde::{Deserialize, Serialize};
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    analysis::coarse_fit::{self, unix_now, Anchor, DragAnchor, FitParams, FitResult, Observation, DIRS},
    app::CONFIG_DIR,
    components::selectable_list::SelectableList,
    connection::nanonis::course_motion::{CalibrationStamp, CoarseCalibration},
    theme,
    view_object::Object,
};

/// Which calibration coarse moves use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CalMode {
    /// The result of the last calibration test only.
    #[default]
    LastTest,
    /// The last test, updated on the fly from corrections.
    OnTheFly,
}

/// When a new fit becomes the learned calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum UpdatePolicy {
    /// Shown as a proposal; the user accepts it.
    #[default]
    Manual,
    /// Taken as soon as it is certain enough.
    Automatic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedLearn {
    #[serde(default)]
    mode: CalMode,
    #[serde(default)]
    policy: UpdatePolicy,
    #[serde(default = "default_auto_max_pct")]
    auto_max_pct: f64,
    #[serde(default = "yes")]
    only_after_test: bool,
    #[serde(default = "yes")]
    follow_latest: bool,
    #[serde(default)]
    params: FitParams,
    #[serde(default)]
    learned: Option<CoarseCalibration>,
    #[serde(default)]
    learned_updated: CalibrationStamp,
    /// Corrections the learned calibration was fitted from.
    #[serde(default)]
    learned_n: usize,
    /// Earlier baselines (newest last), to go back to.
    #[serde(default)]
    baseline_history: Vec<(CoarseCalibration, CalibrationStamp)>,
}
fn default_auto_max_pct() -> f64 {
    5.
}
fn yes() -> bool {
    true
}
impl Default for SavedLearn {
    fn default() -> Self {
        Self {
            mode: CalMode::LastTest,
            policy: UpdatePolicy::Manual,
            auto_max_pct: default_auto_max_pct(),
            only_after_test: true,
            follow_latest: true,
            params: FitParams::default(),
            learned: None,
            learned_updated: CalibrationStamp::default(),
            learned_n: 0,
            baseline_history: Vec::new(),
        }
    }
}

fn learn_path() -> PathBuf {
    CONFIG_DIR.config_local_dir().join("course_calibration_learned.json")
}
fn obs_path() -> PathBuf {
    CONFIG_DIR.config_local_dir().join("coarse_corrections.jsonl")
}
const HISTORY_LEN: usize = 5;
/// At most this many corrections are kept (newest).
const MAX_OBS: usize = 2000;

/// Something the Coarse Motion window must do with the baseline.
pub enum LearnAction {
    /// Make this the new baseline (the learned calibration, promoted).
    Promote(CoarseCalibration, String),
    /// Go back to an earlier baseline.
    Restore(CoarseCalibration, CalibrationStamp),
}

pub struct CoarseLearn {
    s: SavedLearn,
    obs: Vec<Observation>,
    /// The fit over the corrections in use (the proposal).
    fit: Option<FitResult>,
    /// Indices into `obs` that `fit` was made from (same order).
    fit_idx: Vec<usize>,
    /// For "follow": the drag of each stamp the scan region already took.
    follow_applied: HashMap<Uuid, DVec2>,
    follow_init: bool,
    last_sync: Option<Instant>,
    /// The baseline the fit was made against (to refit when it changes).
    fit_for: Option<(CoarseCalibration, Option<u64>)>,
}

impl CoarseLearn {
    pub fn load() -> Self {
        let s = std::fs::read_to_string(learn_path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let obs = std::fs::read_to_string(obs_path())
            .map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
            .unwrap_or_default();
        Self {
            s,
            obs,
            fit: None,
            fit_idx: Vec::new(),
            follow_applied: HashMap::new(),
            follow_init: false,
            last_sync: None,
            fit_for: None,
        }
    }
    fn save(&self) {
        match serde_json::to_string_pretty(&self.s) {
            Ok(j) => {
                if let Err(e) = std::fs::write(learn_path(), j) {
                    error!("could not save the learned coarse calibration: {e}");
                }
            }
            Err(e) => error!("could not encode the learned coarse calibration: {e}"),
        }
    }
    fn save_obs(&self) {
        let text: String =
            self.obs.iter().filter_map(|o| serde_json::to_string(o).ok()).map(|l| l + "\n").collect();
        if let Err(e) = std::fs::write(obs_path(), text) {
            error!("could not save the coarse corrections: {e}");
        }
    }

    /// A drag of the latest stamp moves the scan region with it.
    pub fn follows_latest(&self) -> bool {
        self.s.follow_latest
    }

    /// The calibration moves should use instead of the baseline, if any.
    pub fn active(&self) -> Option<CoarseCalibration> {
        (self.s.mode == CalMode::OnTheFly).then_some(self.s.learned?).filter(|c| c.is_usable())
    }

    /// A new calibration test (or edit) replaced the baseline `old`: keep
    /// `old` to go back to, and start learning again from the new one.
    pub fn on_new_baseline(&mut self, old: CoarseCalibration, old_stamp: &CalibrationStamp, new: CoarseCalibration) {
        if old != new {
            let same_as_last = self.s.baseline_history.last().is_some_and(|(c, _)| *c == old);
            // A hand edit changes the value every frame while dragging:
            // keep one entry per edit session.
            let editing = old_stamp.how == "edited by hand"
                && self.s.baseline_history.last().is_some_and(|(_, st)| st.how == "edited by hand")
                && old_stamp.unix.zip(self.s.baseline_history.last().and_then(|(_, st)| st.unix)).is_some_and(
                    |(a, b)| a.abs_diff(b) < 10,
                );
            if !same_as_last && !editing {
                self.s.baseline_history.push((old, old_stamp.clone()));
                if self.s.baseline_history.len() > HISTORY_LEN {
                    self.s.baseline_history.remove(0);
                }
            }
        }
        if self.s.learned.take().is_some() {
            info!("new coarse calibration test: the learned calibration starts again from it");
        }
        self.s.learned_n = 0;
        self.s.learned_updated = CalibrationStamp::default();
        self.fit_for = None;
        self.save();
    }

    /// Keep the corrections in step with the map: called every frame with
    /// the baseline; does the work at most a few times a second, and not
    /// while the mouse is held (mid-drag).
    pub fn sync(
        &mut self,
        ui: &Ui,
        object_list: &mut SelectableList<Object>,
        baseline: &CoarseCalibration,
        baseline_unix: Option<u64>,
    ) {
        let busy = ui.input(|i| i.pointer.any_down());
        if busy || self.last_sync.is_some_and(|t| t.elapsed().as_secs_f32() < 0.3) {
            return;
        }
        self.last_sync = Some(Instant::now());

        // A finished drag of the scan region marks where the tip really is.
        let Some(area) = object_list.iter_mut().find_map(|o| o.as_scan_area_mut()) else { return };
        if std::mem::take(&mut area.manual_drag) {
            let w = area.world_transform;
            area.coarse_log.drags.push(DragAnchor {
                key: Uuid::new_v4().to_string(),
                last_move: area.coarse_log.last_move(),
                unix: unix_now(),
                true_world: w.translation.to_array(),
                world_m2: w.matrix2.to_cols_array(),
            });
        }
        let log = area.coarse_log.clone();
        let last_drag_unix = log.drags.iter().map(|d| d.unix).max();

        // Every known position in this project.
        let mut anchors: Vec<Anchor> = log.drags.iter().filter_map(|d| d.anchor()).collect();
        let mut stamps: Vec<(Uuid, DVec2, Anchor)> = Vec::new();
        for o in object_list.iter() {
            if let Object::ScanImage { image, .. } = &**o {
                if let Some(a) = image.coarse_ref.and_then(|c| c.anchor(image.uuid(), image.transform)) {
                    let at = glam::DAffine2::from_cols_array(&image.coarse_ref.unwrap().stamp_at);
                    stamps.push((image.uuid(), image.transform.translation - at.translation, a));
                    anchors.push(a);
                }
            }
        }

        // Follow: the scan region moves with a drag of the latest stamp, if
        // nothing has moved it since.
        if !self.follow_init {
            for (k, drag, _) in &stamps {
                self.follow_applied.insert(*k, *drag);
            }
            self.follow_init = true;
        }
        let latest = stamps
            .iter()
            .filter(|(_, _, a)| a.last_move == log.last_move())
            .max_by_key(|(_, _, a)| a.unix)
            .copied();
        for (k, drag, _) in &stamps {
            let applied = *self.follow_applied.entry(*k).or_insert(*drag);
            if latest.is_some_and(|l| l.0 == *k)
                && self.s.follow_latest
                && last_drag_unix.is_none_or(|t| t <= latest.unwrap().2.unix)
            {
                let delta = *drag - applied;
                if delta != DVec2::ZERO {
                    if let Some(area) = object_list.iter_mut().find_map(|o| o.as_scan_area_mut()) {
                        area.world_transform.translation += delta;
                    }
                }
            }
            self.follow_applied.insert(*k, *drag);
        }

        // Corrections from this project replace their earlier versions;
        // ones whose stamp is here but no longer dragged (Undo) go.
        let here: std::collections::HashSet<String> = anchors.iter().map(|a| a.key.to_string()).collect();
        let fresh = coarse_fit::observations(&log.moves, &anchors);
        let mut changed = false;
        let before = self.obs.len();
        self.obs.retain(|o| !here.contains(&o.key) || fresh.iter().any(|f| f.key == o.key));
        changed |= self.obs.len() != before;
        for mut f in fresh {
            match self.obs.iter_mut().find(|o| o.key == f.key) {
                Some(o) => {
                    f.ignored = o.ignored;
                    // Keep the time it was first measured.
                    f.unix = o.unix;
                    if *o != f {
                        *o = f;
                        changed = true;
                    }
                }
                None => {
                    info!("coarse correction: {} moved {:.0} nm (predicted {:.0} nm)", f.moves_text(), f.actual().length(), f.predicted().length());
                    self.obs.push(f);
                    changed = true;
                }
            }
        }
        if self.obs.len() > MAX_OBS {
            self.obs.sort_by_key(|o| o.unix);
            let extra = self.obs.len() - MAX_OBS;
            self.obs.drain(..extra);
            changed = true;
        }
        if changed {
            self.save_obs();
        }
        if changed || self.fit_for != Some((*baseline, baseline_unix)) {
            self.refit(baseline, baseline_unix);
        }
    }

    /// Fit again from the corrections in use.
    fn refit(&mut self, baseline: &CoarseCalibration, baseline_unix: Option<u64>) {
        self.fit_for = Some((*baseline, baseline_unix));
        let after = if self.s.only_after_test { baseline_unix.unwrap_or(0) } else { 0 };
        self.fit_idx = (0..self.obs.len()).filter(|&i| self.obs[i].unix >= after).collect();
        let sel: Vec<Observation> = self.fit_idx.iter().map(|&i| self.obs[i].clone()).collect();
        let fit = coarse_fit::fit(baseline, &sel, &self.s.params);
        if self.s.policy == UpdatePolicy::Automatic
            && fit.n_used > 0
            && fit.worst_sigma_pct().is_some_and(|s| s <= self.s.auto_max_pct)
            && self.s.learned != Some(fit.calib)
        {
            self.s.learned = Some(fit.calib);
            self.s.learned_n = fit.n_used;
            self.s.learned_updated = CalibrationStamp::now(format!("automatic, {} corrections", fit.n_used));
            info!("learned coarse calibration updated from {} corrections", fit.n_used);
            self.save();
        }
        self.fit = Some(fit);
    }

    /// The "Learned from corrections" section of the Coarse Motion window.
    pub fn ui(
        &mut self,
        ui: &mut Ui,
        baseline: &CoarseCalibration,
        baseline_stamp: &CalibrationStamp,
        volts: DVec2,
    ) -> Option<LearnAction> {
        let mut action = None;
        let mut save = false;
        let mut refit = false;
        ui.set_max_width(460.);
        ui.weak(
            "Ctrl+drag a stamp (or the scan region) to where it really is after a coarse move: \
             each drag measures how far the moves since the last known position really went.",
        );
        ui.add_space(4.);
        ui.horizontal(|ui| {
            ui.label("Moves use");
            save |= ui
                .selectable_value(&mut self.s.mode, CalMode::LastTest, "Last test only")
                .on_hover_text("The calibration from the last test, as it was measured")
                .changed();
            save |= ui
                .selectable_value(&mut self.s.mode, CalMode::OnTheFly, "Last test + on-the-fly")
                .on_hover_text(
                    "The learned calibration: the last test updated from your corrections. \
                     Switch back at any time; the last test is never changed.",
                )
                .changed();
        });
        let using_learned = self.active().is_some();
        if self.s.mode == CalMode::OnTheFly && !using_learned {
            ui.colored_label(theme::WARN, "Nothing learned yet: moves use the last test.");
        }
        ui.weak(format!("Last test: {}", baseline_stamp.describe()));
        if self.s.learned.is_some() {
            ui.weak(format!("Learned: {}", self.s.learned_updated.describe()));
        }

        // Per-direction table, in nm per step at the current amplitudes.
        let fit = self.fit.clone();
        egui::Grid::new("coarse_learn_grid").num_columns(6).spacing([10., 2.]).striped(true).show(ui, |ui| {
            ui.weak("nm/step");
            ui.weak("Last test");
            ui.weak("Learned");
            ui.weak("New fit");
            ui.weak("\u{b1}");
            ui.weak("n");
            ui.end_row();
            for (d, (axis, pos, name)) in DIRS.iter().enumerate() {
                let v = if *axis == 0 { volts.x } else { volts.y };
                let per = |c: &CoarseCalibration| c.column(*axis, *pos).length() * v * 1e9;
                ui.label(*name);
                ui.monospace(format!("{:.2}", per(baseline)));
                match self.s.learned {
                    Some(l) => {
                        let ch = 100. * (per(&l) / per(baseline).max(1e-12) - 1.);
                        ui.monospace(format!("{:.2} ({ch:+.0}%)", per(&l)))
                    }
                    None => ui.weak("\u{2014}"),
                };
                match &fit {
                    Some(f) if f.dirs[d].n > 0 => {
                        ui.monospace(format!("{:.2} ({:+.0}%)", per(&f.calib), f.dirs[d].change_pct));
                        match f.dirs[d].sigma_pct {
                            Some(s) => ui.monospace(format!("{s:.1}%")),
                            None => ui.weak("?"),
                        };
                        ui.monospace(format!("{}", f.dirs[d].n));
                    }
                    _ => {
                        ui.weak("\u{2014}");
                        ui.weak("");
                        ui.weak("0");
                    }
                }
                ui.end_row();
            }
        });
        if volts.x <= 0. || volts.y <= 0. {
            ui.weak("(No drive amplitude read yet: nm/step shows 0.)");
        }

        ui.add_space(4.);
        ui.horizontal_wrapped(|ui| {
            ui.label("Update learned");
            save |= ui
                .selectable_value(&mut self.s.policy, UpdatePolicy::Manual, "When I accept")
                .changed();
            let auto = ui
                .selectable_value(&mut self.s.policy, UpdatePolicy::Automatic, "Automatically")
                .on_hover_text("After each correction, once every direction with data is known to within the limit")
                .changed();
            if auto {
                save = true;
                refit = true;
            }
            if self.s.policy == UpdatePolicy::Automatic {
                ui.label("if \u{b1} \u{2264}");
                if ui
                    .add(egui::DragValue::new(&mut self.s.auto_max_pct).range(0.5..=50.).speed(0.1).suffix(" %"))
                    .changed()
                {
                    save = true;
                    refit = true;
                }
            }
        });
        let can_accept = fit.as_ref().is_some_and(|f| f.n_used > 0 && Some(f.calib) != self.s.learned);
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(can_accept, Button::new("Accept new fit"))
                .on_hover_text("Make the new fit the learned calibration (the last test stays as it is)")
                .clicked()
            {
                if let Some(f) = &fit {
                    self.s.learned = Some(f.calib);
                    self.s.learned_n = f.n_used;
                    self.s.learned_updated = CalibrationStamp::now(format!("accepted, {} corrections", f.n_used));
                    save = true;
                }
            }
            if ui
                .add_enabled(self.s.learned.is_some(), Button::new("Reset learned to last test"))
                .on_hover_text("Forget the learned calibration (the corrections are kept)")
                .clicked()
            {
                self.s.learned = None;
                self.s.learned_n = 0;
                self.s.learned_updated = CalibrationStamp::default();
                save = true;
            }
            if ui
                .add_enabled(self.s.learned.is_some(), Button::new("Make learned the new last test"))
                .on_hover_text("Replace the last test with the learned calibration. The old one is kept below to go back to.")
                .clicked()
            {
                if let Some(l) = self.s.learned {
                    action = Some(LearnAction::Promote(l, format!("learned from {} corrections", self.s.learned_n)));
                }
            }
        });
        if !self.s.baseline_history.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.label("Earlier tests:");
                for (c, st) in self.s.baseline_history.iter().rev() {
                    let label = if st.how.is_empty() { "earlier".to_string() } else { st.how.clone() };
                    if ui
                        .small_button(format!("\u{21b6} {label}"))
                        .on_hover_text(format!("Go back to this calibration ({})", st.describe()))
                        .clicked()
                    {
                        action = Some(LearnAction::Restore(*c, st.clone()));
                    }
                }
            });
        }

        egui::CollapsingHeader::new("Settings").id_salt("coarse_learn_settings").show(ui, |ui| {
            if ui
                .checkbox(&mut self.s.only_after_test, "Only learn from corrections made after the last test")
                .changed()
            {
                save = true;
                refit = true;
            }
            save |= ui
                .checkbox(&mut self.s.follow_latest, "Scan region follows a drag of the latest stamp")
                .on_hover_text("So the next move starts from the corrected position")
                .changed();
            egui::Grid::new("coarse_learn_params").num_columns(2).show(ui, |ui| {
                let p = &mut self.s.params;
                ui.label("Weight of the last test").on_hover_text("In corrections' worth");
                refit |= ui.add(egui::DragValue::new(&mut p.prior_weight).range(0.1..=50.).speed(0.05)).changed();
                ui.end_row();
                ui.label("Half-life (corrections)").on_hover_text("Newer corrections count more");
                refit |= ui.add(egui::DragValue::new(&mut p.half_life).range(1.0..=1000.).speed(0.5)).changed();
                ui.end_row();
                ui.label("Leave out moves off by more than \u{d7}");
                refit |= ui.add(egui::DragValue::new(&mut p.max_ratio).range(1.1..=10.).speed(0.01)).changed();
                ui.end_row();
            });
            save |= refit;
        });

        // The corrections, newest first.
        let after = if self.s.only_after_test { baseline_stamp.unix.unwrap_or(0) } else { 0 };
        let mut order: Vec<usize> = (0..self.obs.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.obs[i].unix));
        let mut delete = None;
        let mut obs_changed = false;
        egui::CollapsingHeader::new(format!("Corrections ({})", self.obs.len()))
            .id_salt("coarse_learn_obs")
            .show(ui, |ui| {
                if self.obs.is_empty() {
                    ui.weak("None yet.");
                    return;
                }
                egui::Grid::new("coarse_learn_obs_grid").num_columns(6).striped(true).spacing([8., 2.]).show(ui, |ui| {
                    ui.weak("When");
                    ui.weak("Moves");
                    ui.weak("Real / predicted");
                    ui.weak("Fit off by");
                    ui.weak("Use");
                    ui.weak("");
                    ui.end_row();
                    for &i in order.iter().take(30) {
                        let o = &mut self.obs[i];
                        ui.label(ago(o.unix));
                        ui.label(o.moves_text());
                        let (a, p) = (o.actual().length(), o.predicted().length());
                        ui.monospace(format!("{a:.0} / {p:.0} nm ({:+.0}%)", 100. * (a / p.max(1e-9) - 1.)));
                        let pos = self.fit_idx.iter().position(|&k| k == i);
                        match (pos, &fit) {
                            (Some(k), Some(f)) => {
                                let r = f.residual_nm[k];
                                let col = if f.used[k] { ui.visuals().text_color() } else { theme::ERR };
                                ui.label(RichText::new(format!("{r:.0} nm{}", if f.used[k] || o.ignored { "" } else { " (left out)" })).color(col));
                            }
                            _ if o.unix < after => {
                                ui.weak("before last test");
                            }
                            _ => {
                                ui.weak("");
                            }
                        }
                        let mut use_it = !o.ignored;
                        if ui.checkbox(&mut use_it, "").changed() {
                            o.ignored = !use_it;
                            obs_changed = true;
                        }
                        if ui.small_button("\u{2715}").on_hover_text("Delete this correction").clicked() {
                            delete = Some(i);
                        }
                        ui.end_row();
                    }
                });
                if order.len() > 30 {
                    ui.weak(format!("\u{2026} and {} older", order.len() - 30));
                }
            });
        if let Some(i) = delete {
            self.obs.remove(i);
            obs_changed = true;
        }
        if obs_changed {
            self.save_obs();
            refit = true;
        }
        if refit {
            self.refit(baseline, baseline_stamp.unix);
        }
        if save {
            self.save();
        }
        action
    }
}

fn ago(unix: u64) -> String {
    let s = unix_now().saturating_sub(unix);
    match s {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", s / 60),
        3600..172800 => format!("{} h ago", s / 3600),
        _ => format!("{} days ago", s / 86400),
    }
}
