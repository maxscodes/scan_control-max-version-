//! Tip Check: judges tip quality from the scans (multi-tip ghosts, tip
//! changes, forward/backward disagreement, bluntness), live and on stamps,
//! and labels archive scans to check and tune the detector.
//!
//! Display only: it never moves the tip or changes a setting, so it doesn't
//! take part in "one tool drives the tip at a time".

use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use crossbeam::channel::Receiver;
use egui::{Align2, Color32, FontId, Id, RichText, Stroke, Ui};
use glam::{DAffine2, DVec2};
use uuid::Uuid;

use crate::{
    analysis::{
        sxm_match::{load_frame, SxmFrame},
        tip_quality::{analyse, TipParams, TipReport, Verdict},
        tip_training::{self, Evaluation, Record, TipLabel},
    },
    components::selectable_list::SelectableList,
    connection::{LiveImage, ScanArea},
    scan_view::ScanViewCtx,
    theme,
    utils::vec_interop::IntoEgui as _,
    view_object::Object,
};

const MIN_INTERVAL: Duration = Duration::from_millis(1500);

pub fn verdict_color(v: &Verdict) -> Color32 {
    match v {
        Verdict::Good => theme::OK,
        Verdict::Suspect => theme::WARN,
        Verdict::Poor => theme::ERR,
        Verdict::Unsure(_) => theme::IDLE,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Live,
    Label,
    Evaluate,
}

/// The live result and the frame (world) it was measured on.
struct Live {
    report: TipReport,
    frame: DAffine2,
    size: [u32; 2],
    complete: bool,
}

/// An archive scan shown for labelling.
struct Loaded {
    path: PathBuf,
    frame: SxmFrame,
    report: TipReport,
    tex: egui::TextureHandle,
}

pub struct TipCheckState {
    window_open: bool,
    /// Check the live scan.
    enabled: bool,
    save_with_stamps: bool,
    show_on_stamps: bool,
    params: TipParams,
    tab: Tab,

    live: Option<Live>,
    job: Option<Receiver<Live>>,
    last_run: Option<Instant>,
    last_data: Option<Arc<Box<[f32]>>>,
    pending: bool,
    /// Stamps waiting for their result.
    stamp_jobs: Vec<(Uuid, Receiver<TipReport>)>,

    // Labelling.
    folder: Option<PathBuf>,
    files: Vec<PathBuf>,
    index: usize,
    skip_labelled: bool,
    save_data: bool,
    loaded: Option<Loaded>,
    load_job: Option<Receiver<Result<(PathBuf, SxmFrame, TipReport), String>>>,
    records: Vec<Record>,
    label_note: Option<String>,

    // Evaluation.
    eval: Option<Evaluation>,
    tuned: Option<(TipParams, Evaluation)>,
    reanalyse_job: Option<Receiver<Vec<Record>>>,
}

impl TipCheckState {
    pub fn new() -> Self {
        Self {
            window_open: false,
            enabled: false,
            save_with_stamps: true,
            show_on_stamps: true,
            params: TipParams::load(),
            tab: Tab::Live,
            live: None,
            job: None,
            last_run: None,
            last_data: None,
            pending: false,
            stamp_jobs: Vec::new(),
            folder: None,
            files: Vec::new(),
            index: 0,
            skip_labelled: true,
            save_data: true,
            loaded: None,
            load_job: None,
            records: tip_training::load(),
            label_note: None,
            eval: None,
            tuned: None,
            reanalyse_job: None,
        }
    }
    pub fn open_mut(&mut self) -> &mut bool {
        &mut self.window_open
    }

    /// Call once per UI update after the live image data was refreshed.
    pub fn update(&mut self, scan_area: &ScanArea, got_forward: bool) {
        if let Some(rx) = &self.job {
            match rx.try_recv() {
                Ok(l) => {
                    self.live = Some(l);
                    self.job = None;
                }
                Err(crossbeam::channel::TryRecvError::Disconnected) => self.job = None,
                Err(_) => {}
            }
        }
        if !self.enabled {
            return;
        }
        self.pending |= got_forward;
        if self.job.is_some() || !self.pending || self.last_run.is_some_and(|t| t.elapsed() < MIN_INTERVAL) {
            return;
        }
        let Some(image_transform) = scan_area.image_transform else { return };
        let img = &scan_area.live_image;
        let data = img.forward_data.data.clone();
        if self.last_data.as_ref().is_some_and(|d| Arc::ptr_eq(d, &data) || d[..] == data[..]) {
            self.pending = false;
            return;
        }
        self.pending = false;
        self.last_run = Some(Instant::now());
        self.last_data = Some(data.clone());
        let size = img.forward_data.size;
        let bwd = (img.backward_data.size == size).then(|| img.backward_data.data.clone());
        let world = scan_area.world_transform * image_transform;
        let params = self.params;
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.job = Some(rx);
        std::thread::spawn(move || {
            let complete = data.iter().all(|v| v.is_finite());
            let report = analyse(&data, bwd.as_deref().map(|b| &b[..]), size, image_transform, &params);
            let _ = tx.send(Live { report, frame: world, size, complete });
            crate::app::request_repaint();
        });
    }

    /// Judge a new stamp in the background (if set to); the result is put
    /// on the stamp by [`Self::poll_stamps`].
    pub fn queue_stamp(&mut self, stamp: &LiveImage, image_transform: Option<DAffine2>) {
        if !self.save_with_stamps {
            return;
        }
        let size = stamp.forward_data.size;
        let data = stamp.forward_data.data.clone();
        let bwd = (stamp.backward_data.size == size).then(|| stamp.backward_data.data.clone());
        // The piezo frame (for the pixel size); the stamp's world transform
        // has the same scale when the map is in nm.
        let frame = image_transform.unwrap_or(stamp.transform);
        let params = self.params;
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.stamp_jobs.push((stamp.uuid(), rx));
        std::thread::spawn(move || {
            let r = analyse(&data, bwd.as_deref().map(|b| &b[..]), size, frame, &params);
            let _ = tx.send(r);
            crate::app::request_repaint();
        });
    }
    pub fn poll_stamps(&mut self, object_list: &mut SelectableList<Object>) {
        self.stamp_jobs.retain(|(id, rx)| match rx.try_recv() {
            Ok(r) => {
                for o in object_list.iter_mut() {
                    if let Object::ScanImage { image, .. } = &mut **o {
                        if image.uuid() == *id {
                            image.tip = Some(r.clone());
                        }
                    }
                }
                false
            }
            Err(crossbeam::channel::TryRecvError::Empty) => true,
            Err(_) => false,
        });
    }

    pub fn show_overlay(&self, ui: &mut Ui, object_list: &SelectableList<Object>) {
        let Some(ctx) = ui.data(|map| map.get_temp::<ScanViewCtx>(Id::new(()))) else { return };
        let w2s = ctx.world2egui();
        let to_screen = |p: DVec2| w2s.transform_point2(p).to_egui_pos2();
        let painter = ui.painter().clone();
        if self.show_on_stamps {
            for o in object_list.iter() {
                if let Object::ScanImage { image, hidden: false, .. } = &**o {
                    if let Some(r) = &image.tip {
                        let corner = to_screen(image.transform.transform_point2(DVec2::new(-0.5, -0.5)));
                        painter.text(
                            corner + egui::vec2(3., 3.),
                            Align2::LEFT_TOP,
                            format!("tip: {}", r.verdict.label()),
                            FontId::proportional(11.),
                            verdict_color(&r.verdict),
                        );
                    }
                }
            }
        }
        if !self.enabled {
            return;
        }
        let Some(live) = &self.live else { return };
        let color = verdict_color(&live.report.verdict);
        let c = live.frame.translation;
        for g in live.report.ghosts.iter().take(3).filter(|g| g.strength >= 0.1) {
            // Pixels → unit quad → world.
            let q = DVec2::new(g.offset_px[0] / live.size[0] as f64, g.offset_px[1] / live.size[1] as f64);
            let v = live.frame.matrix2 * q;
            let a = to_screen(c);
            for s in [1., -1.] {
                let b = to_screen(c + v * s);
                painter.arrow(a, b - a, Stroke::new(2., color.gamma_multiply(g.strength.clamp(0.3, 1.) as f32)));
            }
        }
        let corner = to_screen(live.frame.transform_point2(DVec2::new(-0.5, 0.5)));
        painter.text(
            corner + egui::vec2(3., 3.),
            Align2::LEFT_TOP,
            format!("Tip check: {}{}", live.report.summary(), if live.complete { "" } else { " (partial)" }),
            FontId::proportional(13.),
            color,
        );
    }

    pub fn show_menu(&mut self, ui: &mut Ui) {
        self.poll_background(ui.ctx());
        if !self.window_open {
            return;
        }
        let mut open = true;
        egui::Window::new("Tip Check")
            .frame(theme::window_frame(ui.ctx()))
            .open(&mut open)
            .default_size([460., 520.])
            .resizable(true)
            .vscroll(true)
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tab, Tab::Live, "Live");
                    ui.selectable_value(&mut self.tab, Tab::Label, "Label archive");
                    ui.selectable_value(&mut self.tab, Tab::Evaluate, "Evaluate");
                });
                ui.separator();
                match self.tab {
                    Tab::Live => self.live_ui(ui),
                    Tab::Label => self.label_ui(ui),
                    Tab::Evaluate => self.evaluate_ui(ui),
                }
            });
        if !open {
            self.window_open = false;
        }
    }

    fn poll_background(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.load_job {
            match rx.try_recv() {
                Ok(Ok((path, frame, report))) => {
                    let img = gray_image(&frame.forward, frame.size);
                    let tex = ctx.load_texture("tip_label_image", img, egui::TextureOptions::NEAREST);
                    self.loaded = Some(Loaded { path, frame, report, tex });
                    self.load_job = None;
                }
                Ok(Err(e)) => {
                    self.label_note = Some(e);
                    self.load_job = None;
                }
                Err(crossbeam::channel::TryRecvError::Disconnected) => self.load_job = None,
                Err(_) => {}
            }
        }
        if let Some(rx) = &self.reanalyse_job {
            if let Ok(recs) = rx.try_recv() {
                self.records = recs;
                let _ = tip_training::save_all(&self.records);
                self.eval = Some(tip_training::evaluate(&self.records, &self.params));
                self.reanalyse_job = None;
            }
        }
    }

    fn live_ui(&mut self, ui: &mut Ui) {
        ui.checkbox(&mut self.enabled, "Check the live scan").on_hover_text(
            "Judge the tip from the live image while scanning (best on small and medium frames, Z channel)",
        );
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.save_with_stamps, "Judge stamps");
            ui.checkbox(&mut self.show_on_stamps, "Show on stamps");
        });
        ui.add_space(4.);
        match &self.live {
            Some(l) if self.enabled => report_ui(ui, &l.report, !l.complete),
            _ if self.enabled => {
                ui.weak("Waiting for scan data\u{2026}");
            }
            _ => {
                ui.weak("Off.");
            }
        }
        ui.add_space(6.);
        egui::CollapsingHeader::new("Thresholds").id_salt("tip_thresholds").show(ui, |ui| {
            if params_ui(ui, &mut self.params) {
                self.params.save();
                self.rejudge();
            }
            if ui.button("Defaults").clicked() {
                self.params = TipParams::default();
                self.params.save();
                self.rejudge();
            }
        });
        ui.weak("Display only: nothing here moves the tip.");
    }

    /// Apply new thresholds to the results already shown.
    fn rejudge(&mut self) {
        if let Some(l) = &mut self.live {
            let (v, n) = self.params.judge(&l.report);
            l.report.verdict = v;
            l.report.notes = n;
        }
        if let Some(l) = &mut self.loaded {
            let (v, n) = self.params.judge(&l.report);
            l.report.verdict = v;
            l.report.notes = n;
        }
        self.eval = None;
    }

    fn label_ui(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Choose folder\u{2026}").clicked() {
                if let Some(dir) = crate::components::file_dialog_native::dialog()
                    .set_title("Folder of .sxm scans to label")
                    .pick_folder()
                {
                    self.files = tip_training::find_sxm(&dir);
                    self.folder = Some(dir);
                    self.index = 0;
                    self.loaded = None;
                    self.label_note = Some(format!("{} .sxm files", self.files.len()));
                    self.go_to(0, 1);
                }
            }
            if let Some(f) = &self.folder {
                ui.weak(f.display().to_string());
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.skip_labelled, "Skip labelled");
            ui.checkbox(&mut self.save_data, "Keep image with label").on_hover_text(
                "Save the image next to the label (for training later, even if the archive moves)",
            );
        });
        ui.weak(format!("{} labelled in total", self.records.len()));
        if let Some(n) = &self.label_note {
            ui.weak(n.as_str());
        }
        if self.files.is_empty() {
            return;
        }
        let n = self.files.len();
        ui.horizontal(|ui| {
            if ui.button("\u{25c0}").on_hover_text("Previous (\u{2190})").clicked() {
                self.go_to(self.index.saturating_sub(1), -1);
            }
            ui.label(format!("{} / {n}", self.index + 1));
            if ui.button("\u{25b6}").on_hover_text("Next (\u{2192}, or S to skip)").clicked() {
                self.go_to(self.index + 1, 1);
            }
            if self.load_job.is_some() {
                ui.spinner();
            }
        });
        let Some(l) = &self.loaded else { return };
        let current = self.records.iter().find(|r| r.path == l.path.to_string_lossy()).map(|r| r.label);
        ui.label(RichText::new(l.path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()).strong());
        ui.weak(format!(
            "{} \u{b7} {}\u{d7}{} px \u{b7} {:.3} nm/px{}",
            l.frame.channel,
            l.frame.size[0],
            l.frame.size[1],
            l.report.px_nm,
            current.map_or(String::new(), |c| format!(" \u{b7} labelled {}", c.name()))
        ));
        // The image, with the ghost offsets drawn from its centre.
        let side = ui.available_width().min(440.);
        let [w, h] = l.frame.size;
        let size = egui::vec2(side, side * h as f32 / w.max(1) as f32);
        let resp = ui.add(egui::Image::new((l.tex.id(), size)));
        let r = resp.rect;
        let painter = ui.painter_at(r);
        for g in l.report.ghosts.iter().take(3).filter(|g| g.strength >= 0.1) {
            let v = egui::vec2(
                (g.offset_px[0] / w as f64) as f32 * r.width(),
                (g.offset_px[1] / h as f64) as f32 * r.height(),
            );
            let col = verdict_color(&l.report.verdict);
            painter.arrow(r.center(), v, Stroke::new(2., col));
            painter.arrow(r.center(), -v, Stroke::new(2., col));
        }
        report_ui(ui, &l.report, false);
        ui.add_space(4.);
        let mut chosen = None;
        ui.horizontal_wrapped(|ui| {
            for lab in TipLabel::ALL {
                let key = format!("{:?}", lab.key());
                let text = format!("{} ({key})", lab.name());
                if ui.selectable_label(current == Some(lab), text).clicked() {
                    chosen = Some(lab);
                }
            }
        });
        // Keys, when no text field has focus.
        if !ui.ctx().wants_keyboard_input() {
            ui.input(|i| {
                for lab in TipLabel::ALL {
                    if i.key_pressed(lab.key()) {
                        chosen = Some(lab);
                    }
                }
            });
            let (next, prev) = ui.input(|i| {
                (i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::S), i.key_pressed(egui::Key::ArrowLeft))
            });
            if next {
                self.go_to(self.index + 1, 1);
            } else if prev {
                self.go_to(self.index.saturating_sub(1), -1);
            }
        }
        if let Some(lab) = chosen {
            self.label_current(lab);
            self.go_to(self.index + 1, 1);
        }
    }

    fn label_current(&mut self, label: TipLabel) {
        let Some(l) = &self.loaded else { return };
        let path = l.path.to_string_lossy().into_owned();
        let id = self
            .records
            .iter()
            .find(|r| r.path == path)
            .map(|r| r.id.clone())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let has_data = self.save_data && tip_training::save_data(&id, &l.frame.forward).is_ok();
        let rec = Record {
            id,
            path: path.clone(),
            channel: l.frame.channel.clone(),
            label,
            unix: crate::analysis::coarse_fit::unix_now(),
            size: l.frame.size,
            frame: l.frame.frame.to_cols_array(),
            has_data,
            report: l.report.clone(),
        };
        match self.records.iter_mut().find(|r| r.path == path) {
            Some(r) => *r = rec,
            None => self.records.push(rec),
        }
        if let Err(e) = tip_training::save_all(&self.records) {
            self.label_note = Some(format!("Could not save the label: {e}"));
        }
        self.eval = None;
    }

    /// Load file `i` (or the next unlabelled one in direction `dir`).
    fn go_to(&mut self, mut i: usize, dir: i32) {
        if self.files.is_empty() {
            return;
        }
        if self.skip_labelled {
            while i < self.files.len()
                && self.records.iter().any(|r| r.path == self.files[i].to_string_lossy())
                && !(dir < 0 && i == 0)
            {
                if dir < 0 {
                    i -= 1;
                } else {
                    i += 1;
                }
            }
        }
        if i >= self.files.len() {
            self.label_note = Some("End of the folder.".into());
            return;
        }
        self.index = i;
        let path = self.files[i].clone();
        let params = self.params;
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.load_job = Some(rx);
        std::thread::spawn(move || {
            let r = load_frame(&path, Some("Z")).map(|f| {
                let rep = analyse(&f.forward, Some(&f.backward), f.size, f.frame, &params);
                (path.clone(), f, rep)
            });
            let _ = tx.send(r.map_err(|e| format!("{}: {e}", path.display())));
            crate::app::request_repaint();
        });
    }

    fn evaluate_ui(&mut self, ui: &mut Ui) {
        if self.records.is_empty() {
            ui.weak("No labels yet: label some archive scans first.");
            return;
        }
        let eval = self.eval.get_or_insert_with(|| tip_training::evaluate(&self.records, &self.params)).clone();
        ui.label(format!("{} labelled scans", self.records.len()));
        if let Some(n) = &self.label_note {
            ui.weak(n.as_str());
        }
        eval_table(ui, &eval, "tip_eval_grid");
        ui.add_space(6.);
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Tune thresholds")
                .on_hover_text("Search the multi-tip and instability thresholds for the best balanced accuracy on the labels")
                .clicked()
            {
                self.tuned = Some(tip_training::tune(&self.records, &self.params));
            }
            let busy = self.reanalyse_job.is_some();
            if ui
                .add_enabled(!busy, egui::Button::new("Re-analyse all"))
                .on_hover_text("Run the detector again on every labelled scan (after the detector changed)")
                .clicked()
            {
                let (tx, rx) = crossbeam::channel::bounded(1);
                self.reanalyse_job = Some(rx);
                let mut recs = self.records.clone();
                let params = self.params;
                std::thread::spawn(move || {
                    for r in recs.iter_mut() {
                        if let Some(rep) = tip_training::reanalyse(r, &params) {
                            r.report = rep;
                        }
                    }
                    let _ = tx.send(recs);
                    crate::app::request_repaint();
                });
            }
            if busy {
                ui.spinner();
            }
            if ui.button("Export features CSV").on_hover_text("features.csv: label and detector scores per scan, for training").clicked() {
                self.label_note = Some(match tip_training::export_csv(&self.records) {
                    Ok(p) => format!("Wrote {}", p.display()),
                    Err(e) => format!("Could not export: {e}"),
                });
            }
            if ui.button("Open folder").on_hover_text("labels.jsonl and the saved images, for training").clicked() {
                let _ = std::fs::create_dir_all(tip_training::dir());
                let _ = std::process::Command::new("explorer").arg(tip_training::dir()).spawn();
            }
        });
        if let Some((p, e)) = self.tuned.clone() {
            ui.add_space(4.);
            ui.label(RichText::new("Tuned").strong());
            ui.weak(format!(
                "multi-tip suspect {:.3} / poor {:.3}, instability suspect {:.3} / poor {:.3}",
                p.suspect_multi, p.poor_multi, p.suspect_instability, p.poor_instability
            ));
            eval_table(ui, &e, "tip_tuned_grid");
            ui.horizontal(|ui| {
                if ui.button("Use these thresholds").clicked() {
                    self.params = p;
                    self.params.save();
                    self.tuned = None;
                    self.rejudge();
                }
                if ui.button("Discard").clicked() {
                    self.tuned = None;
                }
            });
            ui.weak("Tuned on the same labels it is scored on: expect a little worse on new scans.");
        }
    }
}

fn eval_table(ui: &mut Ui, e: &Evaluation, id: &str) {
    egui::Grid::new(id).num_columns(5).striped(true).spacing([10., 2.]).show(ui, |ui| {
        ui.weak("Label \\ verdict");
        for v in ["Good", "Suspect", "Poor", "Unsure"] {
            ui.weak(v);
        }
        ui.end_row();
        for (li, lab) in TipLabel::ALL.iter().enumerate() {
            ui.label(lab.name());
            for vi in 0..4 {
                let n = e.counts[li][vi];
                let good = (vi == 0) == !lab.is_bad() && vi != 3;
                let text = RichText::new(n.to_string());
                ui.label(if n > 0 && good { text.color(theme::OK) } else { text });
            }
            ui.end_row();
        }
    });
    let pct = |(a, b): (usize, usize)| if b == 0 { "\u{2014}".to_string() } else { format!("{:.0}% ({a}/{b})", 100. * a as f64 / b as f64) };
    ui.label(format!(
        "Bad tips caught: {} \u{b7} good tips passed: {} \u{b7} balanced accuracy {:.0}%",
        pct(e.caught_bad),
        pct(e.passed_good),
        100. * e.balanced_accuracy()
    ));
    let rows: Vec<String> = e
        .by_scale
        .iter()
        .filter(|s| s.2 + s.4 > 0)
        .map(|(ub, c, b, p, g)| {
            let ub = if ub.is_finite() { format!("\u{2264}{ub} nm/px") } else { "larger".into() };
            format!("{ub}: caught {c}/{b}, passed {p}/{g}")
        })
        .collect();
    if !rows.is_empty() {
        ui.weak(rows.join(" \u{b7} "));
    }
}

/// The scores of a report.
fn report_ui(ui: &mut Ui, r: &TipReport, partial: bool) {
    ui.horizontal(|ui| {
        theme::status_pill(ui, verdict_color(&r.verdict), r.verdict.label(), partial);
        if let Verdict::Unsure(why) = &r.verdict {
            ui.weak(why.as_str());
        }
        if partial {
            ui.weak("(scan in progress)");
        }
    });
    for n in &r.notes {
        ui.label(format!("\u{2022} {n}"));
    }
    let bar = |ui: &mut Ui, name: &str, v: f32, hover: &str| {
        ui.label(name).on_hover_text(hover);
        ui.add(egui::ProgressBar::new(v.clamp(0., 1.)).desired_width(160.).text(format!("{v:.2}")));
        ui.end_row();
    };
    egui::Grid::new(("tip_scores", r.px_nm.to_bits())).num_columns(2).spacing([8., 3.]).show(ui, |ui| {
        bar(ui, "Multi-tip", r.multi_tip, "How clearly the image shows ghost copies of its features");
        bar(ui, "Instability", r.instability, "Tip changes and streaks between scan lines");
        if let Some(fb) = r.fb_agreement {
            bar(ui, "Fwd/bwd agree", fb, "Forward and backward images match (1 = identical)");
        }
        if let Some(b) = r.blunt {
            bar(ui, "Bluntness", b, "Features wider than a sharp tip gives (atomic-scale frames only)");
        }
    });
    if !r.ghosts.is_empty() {
        let g: Vec<String> = r
            .ghosts
            .iter()
            .take(3)
            .filter(|g| g.strength >= 0.1)
            .map(|g| format!("({:+.2}, {:+.2}) nm", g.offset_nm[0], g.offset_nm[1]))
            .collect();
        if !g.is_empty() {
            ui.label(format!("Copy offset{}: {}", if g.len() > 1 { "s" } else { "" }, g.join(", ")))
                .on_hover_text("Every feature appears again shifted by this (either way). Arrows on the image.");
            if r.apexes >= 3 {
                ui.weak(format!("Looks like {} apexes", r.apexes));
            }
        }
    }
    ui.weak(format!("{:.3} nm/px analysed \u{b7} signal/noise {:.1}", r.px_nm, r.snr));
}

/// Edit the thresholds; true if changed.
fn params_ui(ui: &mut Ui, p: &mut TipParams) -> bool {
    let mut changed = false;
    egui::Grid::new("tip_params_grid").num_columns(2).show(ui, |ui| {
        let mut row = |ui: &mut Ui, name: &str, v: &mut f32| {
            ui.label(name);
            changed |= ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).fixed_decimals(3)).changed();
            ui.end_row();
        };
        row(ui, "Multi-tip: suspect from", &mut p.suspect_multi);
        row(ui, "Multi-tip: poor from", &mut p.poor_multi);
        row(ui, "Instability: suspect from", &mut p.suspect_instability);
        row(ui, "Instability: poor from", &mut p.poor_instability);
        row(ui, "Fwd/bwd: suspect below", &mut p.min_fb);
        row(ui, "Bluntness: suspect from", &mut p.suspect_blunt);
        ui.label("Featureless below signal/noise");
        changed |= ui.add(egui::DragValue::new(&mut p.min_snr).range(0.0..=10.0).speed(0.01)).changed();
        ui.end_row();
    });
    changed
}

/// Grey image for labelling: each line levelled, 1–99 % contrast.
fn gray_image(data: &[f32], size: [u32; 2]) -> egui::ColorImage {
    let (w, h) = (size[0] as usize, size[1] as usize);
    let mut d = data.to_vec();
    for y in 0..h {
        let row = &mut d[y * w..(y + 1) * w];
        let fin: Vec<f32> = row.iter().copied().filter(|v| v.is_finite()).collect();
        if fin.is_empty() {
            continue;
        }
        let m = fin.iter().sum::<f32>() / fin.len() as f32;
        row.iter_mut().for_each(|v| *v -= m);
    }
    let mut fin: Vec<f32> = d.iter().copied().filter(|v| v.is_finite()).collect();
    fin.sort_by(f32::total_cmp);
    let (lo, hi) =
        if fin.is_empty() { (0., 1.) } else { (fin[fin.len() / 100], fin[(fin.len() * 99 / 100).min(fin.len() - 1)]) };
    let span = (hi - lo).max(1e-30);
    let px: Vec<Color32> = d
        .iter()
        .map(|v| {
            if v.is_finite() {
                let g = (((v - lo) / span).clamp(0., 1.) * 255.) as u8;
                Color32::from_gray(g)
            } else {
                Color32::from_rgb(40, 0, 40)
            }
        })
        .collect();
    egui::ColorImage { size: [w, h], pixels: px, source_size: egui::vec2(w as f32, h as f32) }
}
