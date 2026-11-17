//! "Polish images": a separate window that turns several stamps (e.g. a
//! survey grid) into one clean picture, with a live preview. Tiles can be
//! lined up (automatically and checked by eye, seam by seam), fixed one by
//! one, and the result saved as a PNG or put back on the map.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crossbeam::channel::{Receiver, Sender};
use egui::{Color32, DragValue, Key, Modifiers, Slider, Stroke, Ui};
use glam::{DAffine2, DVec2};
use uuid::Uuid;

use crate::{
    analysis::{
        polish::{self, Blend, Channel, Cleaned, Fill, LevelMatch, Look, Prepared, Source, Stats, TileEdit, TileSettings},
        gds_prior::GdsRings,
        polish_align::{self, Pair, RingAnchor, RingMiss},
    },
    scan_view::processing::{self, ColorMap},
};

/// Longest side of the preview, pixels.
const PREVIEW_SIDE: u32 = 1600;
/// Largest picture put on the map (GPU texture limit).
const MAP_SIDE: u32 = 8192;

/// A finished picture for the map: put it on the map as an image layer.
pub struct ForMap {
    pub name: String,
    pub image: image::RgbaImage,
    /// Map position of the top-left corner and the pixel size, nm.
    pub top_left: DVec2,
    pub px: f64,
    /// Stamps it was made from, to hide if asked.
    pub sources: Vec<Uuid>,
    pub hide_sources: bool,
}

type Edits = Arc<Vec<TileEdit>>;

struct Job {
    gen: u64,
    sources: Arc<Vec<Source>>,
    tile: TileSettings,
    edits: Edits,
    look: Look,
}

enum Done {
    Preview {
        gen: u64,
        image: egui::ColorImage,
        stats: Stats,
        px: f64,
        top_left: DVec2,
        prepared: Arc<Prepared>,
        key: (TileSettings, Edits),
    },
    Failed {
        gen: u64,
        msg: String,
    },
}

/// What a background alignment found: pair matches, and ring matches
/// against the GDS when asked for.
struct AlignFound {
    pairs: Vec<Pair>,
    rings: Option<Vec<Result<RingAnchor, RingMiss>>>,
}

enum Output {
    Saved(String),
    Map(ForMap),
}

/// The seam check: two tiles of a pair laid over each other.
struct Check {
    /// Index into the pairs.
    k: usize,
    /// Correction being set (move of b relative to a, nm).
    d: DVec2,
    tex: Option<(usize, [egui::TextureHandle; 2])>,
    zoom: f32,
    /// World nm.
    pan: DVec2,
    opacity: f32,
    blink: bool,
}

pub struct PolishWindow {
    open: bool,
    name: String,
    sources: Arc<Vec<Source>>,
    tile: TileSettings,
    look: Look,
    edits: Vec<TileEdit>,
    /// Selected tiles; the last one is shown under "Selected tile".
    selected: Vec<usize>,
    show_outlines: bool,
    /// Off: tiles shown where the map has them (moves kept, just not used).
    show_aligned: bool,
    /// Off: plain min-to-max contrast (no flattening, evening out, clipping,
    /// gamma or equalisation).
    stretch_on: bool,
    /// Tiles being Shift+dragged: drawn from the last preview while moving.
    dragging: bool,
    /// 0: full resolution.
    save_side: u32,
    hide_sources: bool,
    // Preview worker.
    jobs: Option<Sender<Job>>,
    results: Option<Receiver<Done>>,
    latest: Arc<AtomicU64>,
    gen: u64,
    shown_gen: u64,
    requested: Option<(TileSettings, Look, Vec<TileEdit>)>,
    texture: Option<egui::TextureHandle>,
    /// Edits the preview texture was made with.
    tex_edits: Edits,
    stats: Stats,
    preview_px: f64,
    preview_top_left: DVec2,
    prepared: Option<((TileSettings, Edits), Arc<Prepared>)>,
    zoom: f32,
    pan: egui::Vec2,
    // Alignment.
    pairs: Vec<Pair>,
    /// Automatic search range, share of a tile.
    align_radius: f32,
    align_rx: Option<Receiver<AlignFound>>,
    /// GDS ring systems on the map when the window was opened.
    gds: Arc<Vec<GdsRings>>,
    /// Per tile: where the GDS rings want it (from "Line up rings with GDS").
    anchors: Vec<Option<RingAnchor>>,
    check: Option<Check>,
    moves: Vec<(Uuid, DVec2)>,
    // Saving and putting on the map run in the background.
    out_tx: Sender<Result<Output, String>>,
    out_rx: Receiver<Result<Output, String>>,
    busy_out: bool,
    status: Option<String>,
    for_map: Vec<ForMap>,
}

impl Default for PolishWindow {
    fn default() -> Self {
        let (out_tx, out_rx) = crossbeam::channel::unbounded();
        Self {
            open: false,
            name: String::new(),
            sources: Arc::new(Vec::new()),
            tile: TileSettings::default(),
            look: Look { color_map: crate::theme::prefs().color_map, ..Look::default() },
            edits: Vec::new(),
            selected: Vec::new(),
            show_outlines: false,
            show_aligned: true,
            stretch_on: true,
            dragging: false,
            save_side: 0,
            hide_sources: true,
            jobs: None,
            results: None,
            latest: Arc::new(AtomicU64::new(0)),
            gen: 0,
            shown_gen: 0,
            requested: None,
            texture: None,
            tex_edits: Arc::new(Vec::new()),
            stats: Stats::default(),
            preview_px: 1.,
            preview_top_left: DVec2::ZERO,
            prepared: None,
            zoom: 1.,
            pan: egui::Vec2::ZERO,
            pairs: Vec::new(),
            align_radius: 0.08,
            align_rx: None,
            gds: Arc::new(Vec::new()),
            anchors: Vec::new(),
            check: None,
            moves: Vec::new(),
            out_tx,
            out_rx,
            busy_out: false,
            status: None,
            for_map: Vec::new(),
        }
    }
}

impl PolishWindow {
    /// Open the window on these stamps (topmost first). The settings stay as
    /// they were last time, except the first time, which starts from the
    /// first stamp's map settings. Hand edits and checks start afresh.
    pub fn open_with(
        &mut self,
        sources: Vec<Source>,
        name: &str,
        first_settings: Option<(processing::Background, processing::Filters)>,
        gds: Vec<GdsRings>,
    ) {
        if sources.is_empty() {
            return;
        }
        if self.sources.is_empty() {
            if let Some((bg, f)) = first_settings {
                self.tile.background = bg;
                self.tile.filters = f;
            }
            // Hills about a third of a tile across.
            let t = &sources[0].transform.matrix2;
            self.look.flatten_nm = (t.x_axis.length().max(t.y_axis.length()) / 3.).max(1.);
        }
        self.pairs = polish_align::find_pairs(&sources);
        self.edits = vec![TileEdit::default(); sources.len()];
        self.anchors = vec![None; sources.len()];
        self.gds = Arc::new(gds);
        self.sources = Arc::new(sources);
        self.name = name.to_string();
        self.open = true;
        self.selected.clear();
        self.dragging = false;
        self.show_aligned = true;
        self.check = None;
        self.align_rx = None;
        self.requested = None;
        self.prepared = None;
        self.texture = None;
        self.zoom = 1.;
        self.pan = egui::Vec2::ZERO;
        self.status = None;
    }

    /// Pictures the user asked to put on the map.
    pub fn take_for_map(&mut self) -> Vec<ForMap> {
        std::mem::take(&mut self.for_map)
    }

    /// Stamp moves the user asked to apply on the map (stamp, world nm).
    pub fn take_moves(&mut self) -> Vec<(Uuid, DVec2)> {
        std::mem::take(&mut self.moves)
    }

    fn start_worker(&mut self) {
        if self.jobs.is_some() {
            return;
        }
        let (jtx, jrx) = crossbeam::channel::unbounded::<Job>();
        let (rtx, rrx) = crossbeam::channel::unbounded::<Done>();
        let latest = self.latest.clone();
        std::thread::Builder::new()
            .name("polish preview".into())
            .spawn(move || {
                let mut cleaned = Cleaned::default();
                while let Ok(mut job) = jrx.recv() {
                    while let Ok(newer) = jrx.try_recv() {
                        job = newer;
                    }
                    let gen = job.gen;
                    let cancelled = || latest.load(Ordering::Relaxed) != gen;
                    let Some(prepared) = polish::prepare(&job.sources, &job.tile, &job.edits, &mut cleaned, &cancelled) else {
                        if !cancelled() {
                            let _ = rtx.send(Done::Failed { gen, msg: "Nothing to show: no tile has image data".into() });
                            crate::app::request_repaint();
                        }
                        continue;
                    };
                    let prepared = Arc::new(prepared);
                    let Some(r) = polish::render(&prepared, &job.look, PREVIEW_SIDE, false, &cancelled) else { continue };
                    let image = egui::ColorImage::from_rgba_unmultiplied(
                        [r.image.width() as usize, r.image.height() as usize],
                        r.image.as_raw(),
                    );
                    let _ = rtx.send(Done::Preview {
                        gen,
                        image,
                        stats: r.stats,
                        px: r.px,
                        top_left: r.top_left,
                        prepared,
                        key: (job.tile, job.edits),
                    });
                    crate::app::request_repaint();
                }
            })
            .expect("could not start the polish preview thread");
        self.jobs = Some(jtx);
        self.results = Some(rrx);
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let now = (self.tile, self.shown_look(), self.shown_edits());
        // While tiles are dragged the last preview is moved instead.
        if !self.dragging && self.requested.as_ref() != Some(&now) {
            self.start_worker();
            self.gen += 1;
            self.latest.store(self.gen, Ordering::Relaxed);
            if let Some(tx) = &self.jobs {
                let _ = tx.send(Job {
                    gen: self.gen,
                    sources: self.sources.clone(),
                    tile: now.0,
                    edits: Arc::new(now.2.clone()),
                    look: now.1,
                });
            }
            self.requested = Some(now);
        }
        while let Some(done) = self.results.as_ref().and_then(|r| r.try_recv().ok()) {
            match done {
                Done::Preview { gen, image, stats, px, top_left, prepared, key } => {
                    self.prepared = Some((key, prepared));
                    if gen < self.shown_gen {
                        continue;
                    }
                    self.shown_gen = gen;
                    self.tex_edits = self.prepared.as_ref().map(|(k, _)| k.1.clone()).unwrap_or_default();
                    self.stats = stats;
                    self.preview_px = px;
                    self.preview_top_left = top_left;
                    match &mut self.texture {
                        Some(t) => t.set(image, egui::TextureOptions::LINEAR),
                        None => self.texture = Some(ctx.load_texture("polish_preview", image, egui::TextureOptions::LINEAR)),
                    }
                }
                Done::Failed { gen, msg } => {
                    self.shown_gen = self.shown_gen.max(gen);
                    self.status = Some(msg);
                }
            }
        }
        if let Some(found) = self.align_rx.as_ref().and_then(|r| r.try_recv().ok()) {
            self.align_rx = None;
            for (p, f) in self.pairs.iter_mut().zip(found.pairs) {
                p.auto = f.auto;
                p.refused = f.refused;
            }
            let mut ring_note = String::new();
            if let Some(rings) = &found.rings {
                self.anchors = rings.iter().map(|r| r.ok()).collect();
                let got = self.anchors.iter().filter(|a| a.is_some()).count();
                let mut why: Vec<String> = Vec::new();
                for m in RingMiss::ALL {
                    let n = rings.iter().filter(|r| r.err() == Some(m)).count();
                    if n > 0 {
                        why.push(format!("{n} {}", m.label()));
                    }
                }
                ring_note = if got == 0 && rings.iter().all(|r| r.err() == Some(RingMiss::NoGds)) {
                    "; no visible GDS with rings covers these tiles".into()
                } else {
                    format!(
                        "; rings: {got} of {} tiles matched to the GDS{}",
                        rings.len(),
                        if why.is_empty() { String::new() } else { format!(" (not used: {})", why.join(", ")) }
                    )
                };
            }
            self.apply_alignment();
            self.show_aligned = true;
            let used = self.pairs.iter().filter(|p| p.refused.is_none() && p.auto.is_some()).count();
            let mut why: Vec<String> = Vec::new();
            for r in [
                polish_align::Refused::LookAlike,
                polish_align::Refused::Weak,
                polish_align::Refused::Outlier,
                polish_align::Refused::Different,
                polish_align::Refused::NoOverlap,
            ] {
                let n = self.pairs.iter().filter(|p| p.refused == Some(r)).count();
                if n > 0 {
                    why.push(format!("{n} {}", r.label()));
                }
            }
            self.status = Some(format!(
                "Aligned: {used} of {} overlaps matched{}{ring_note}",
                self.pairs.len(),
                if why.is_empty() { String::new() } else { format!(" (not used: {})", why.join(", ")) }
            ));
        }
        while let Ok(out) = self.out_rx.try_recv() {
            self.busy_out = false;
            match out {
                Ok(Output::Saved(msg)) => self.status = Some(msg),
                Ok(Output::Map(m)) => {
                    self.status = Some(format!("Put on the map as \u{201c}{}\u{201d}", m.name));
                    self.for_map.push(m);
                }
                Err(e) => self.status = Some(e),
            }
        }
    }

    fn busy(&self) -> bool {
        self.shown_gen < self.gen
    }

    /// Typical tile pixel size, nm.
    fn tile_px(&self) -> f64 {
        self.sources
            .first()
            .map(|s| s.transform.matrix2.x_axis.length() / s.forward.size[0].max(1) as f64)
            .unwrap_or(1.)
    }

    /// Where tile `i` is shown (its move left out with alignment off).
    fn transform_of(&self, i: usize) -> DAffine2 {
        polish::edit_transform(&self.sources[i], self.edits.get(i).filter(|_| self.show_aligned))
    }

    /// The edits as shown: without the moves when alignment is off.
    fn shown_edits(&self) -> Vec<TileEdit> {
        let mut e = self.edits.clone();
        if !self.show_aligned {
            for t in &mut e {
                t.shift = DVec2::ZERO;
            }
        }
        e
    }

    /// The look as shown: with the stretch off, a plain min-to-max scale.
    fn shown_look(&self) -> Look {
        if self.stretch_on {
            self.look
        } else {
            Look { flatten: 0., even: 0., clip_low: 0., clip_high: 0., gamma: 1., equalize: 0., ..self.look }
        }
    }

    /// Visible tiles, topmost first.
    fn draw_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.sources.len()).filter(|&i| !self.edits[i].hidden).collect();
        order.sort_by_key(|&i| !self.edits[i].on_top);
        order
    }

    /// Fit every tile's move to the pair corrections.
    fn apply_alignment(&mut self) {
        let active: Vec<bool> = self.edits.iter().map(|e| !e.hidden).collect();
        let px = self.tile_px();
        let x = polish_align::solve(self.sources.len(), &mut self.pairs, &active, &self.anchors, px);
        for (e, s) in self.edits.iter_mut().zip(x) {
            e.shift = s;
        }
    }

    /// Match the overlapping pairs in the background; with `rings`, also
    /// match every tile against the rings the GDS expects there.
    fn auto_align(&mut self, rings: bool) {
        let (tx, rx) = crossbeam::channel::unbounded();
        let sources = self.sources.clone();
        let mut pairs = self.pairs.clone();
        let radius = self.align_radius as f64;
        let (gds, tile) = (self.gds.clone(), self.tile);
        self.align_rx = Some(rx);
        self.status = Some(if rings { "Lining up the rings with the GDS\u{2026}" } else { "Aligning\u{2026}" }.into());
        std::thread::spawn(move || {
            for p in pairs.iter_mut() {
                match polish_align::measure(&sources[p.a], &sources[p.b], radius) {
                    Ok(m) => {
                        p.auto = Some(m);
                        p.refused = None;
                    }
                    Err(r) => {
                        p.auto = None;
                        p.refused = Some(r);
                    }
                }
            }
            let rings = rings.then(|| {
                std::thread::scope(|scope| {
                    let handles: Vec<_> = sources
                        .iter()
                        .map(|src| {
                            let gds = &gds;
                            scope.spawn(move || match polish::clean_one(src, &tile) {
                                Some((d, w, h)) => polish_align::ring_anchor(&d, w, h, src.transform, gds),
                                None => Err(RingMiss::NoData),
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap_or(Err(RingMiss::NoData))).collect()
                })
            });
            let _ = tx.send(AlignFound { pairs, rings });
            crate::app::request_repaint();
        });
    }

    /// Move the stamps on the map by the tiles' current moves, and take the
    /// moved positions as the new starting point here.
    fn move_stamps_on_map(&mut self) {
        let shifts: Vec<DVec2> = self.edits.iter().map(|e| e.shift).collect();
        self.moves = self
            .sources
            .iter()
            .zip(&shifts)
            .filter(|(_, s)| s.length() > 0.)
            .map(|(src, s)| (src.id, *s))
            .collect();
        let n = self.moves.len();
        let moved: Vec<Source> = self
            .sources
            .iter()
            .zip(&shifts)
            .map(|(src, s)| Source { transform: DAffine2::from_translation(*s) * src.transform, ..src.clone() })
            .collect();
        for p in self.pairs.iter_mut() {
            let rel = shifts[p.b] - shifts[p.a];
            if let Some(h) = &mut p.hand {
                *h -= rel;
            }
            if let Some(a) = &mut p.auto {
                a.0 -= rel;
            }
        }
        for (a, s) in self.anchors.iter_mut().zip(&shifts) {
            if let Some(a) = a {
                a.want -= *s;
            }
        }
        for e in self.edits.iter_mut() {
            e.shift = DVec2::ZERO;
        }
        self.sources = Arc::new(moved);
        self.status = Some(format!("{n} stamps moved on the map (Ctrl+Z in the main window undoes it)"));
    }

    /// Cleaned tiles for the current settings: the preview's if they match,
    /// else made again (in the calling background thread).
    /// What is shown is what is saved: alignment and stretch as toggled.
    fn output_inputs(&self) -> (Arc<Vec<Source>>, TileSettings, Edits, Option<Arc<Prepared>>) {
        let edits = self.shown_edits();
        let ready = self
            .prepared
            .as_ref()
            .filter(|((t, e), _)| *t == self.tile && **e == edits)
            .map(|(_, p)| p.clone());
        (self.sources.clone(), self.tile, Arc::new(edits), ready)
    }

    fn save(&mut self) {
        let file: String = self
            .name
            .chars()
            .map(|c| if c.is_alphanumeric() || " -_().".contains(c) { c } else { '_' })
            .collect();
        let dialog = crate::components::file_dialog_native::dialog()
            .add_filter("PNG image", &["png"])
            .set_title("Save polished image")
            .set_file_name(format!("{} polished.png", file.trim()));
        let (sources, tile, edits, ready) = self.output_inputs();
        let (look, side, tx) = (self.shown_look(), self.save_side, self.out_tx.clone());
        self.busy_out = true;
        std::thread::spawn(move || {
            let Some(path) = dialog.save_file() else {
                let _ = tx.send(Err("Not saved".into()));
                return;
            };
            let never = || false;
            let res = ready
                .or_else(|| polish::prepare(&sources, &tile, &edits, &mut Cleaned::default(), &never).map(Arc::new))
                .and_then(|p| polish::render(&p, &look, if side == 0 { 16000 } else { side }, false, &never))
                .ok_or_else(|| "Nothing to save: no tile has image data".to_string())
                .and_then(|r| {
                    r.image.save(&path).map_err(|e| format!("Could not save the image: {e}"))?;
                    Ok(Output::Saved(format!(
                        "Saved {} ({}\u{d7}{} px, {:.3} nm/px)",
                        path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
                        r.image.width(),
                        r.image.height(),
                        r.px
                    )))
                });
            let _ = tx.send(res);
            crate::app::request_repaint();
        });
    }

    fn put_on_map(&mut self) {
        let (sources, tile, edits, ready) = self.output_inputs();
        let ids: Vec<Uuid> = sources.iter().zip(edits.iter()).filter(|(_, e)| !e.hidden).map(|(s, _)| s.id).collect();
        let (look, tx) = (self.shown_look(), self.out_tx.clone());
        let name = format!("{} (polished)", self.name);
        let hide = self.hide_sources;
        self.busy_out = true;
        std::thread::spawn(move || {
            let never = || false;
            let res = ready
                .or_else(|| polish::prepare(&sources, &tile, &edits, &mut Cleaned::default(), &never).map(Arc::new))
                .and_then(|p| polish::render(&p, &look, MAP_SIDE, true, &never))
                .map(|r| Output::Map(ForMap { name, image: r.image, top_left: r.top_left, px: r.px, sources: ids, hide_sources: hide }))
                .ok_or_else(|| "Nothing to put on the map: no tile has image data".to_string());
            let _ = tx.send(res);
            crate::app::request_repaint();
        });
    }

    /// Show the window (its own OS window where the platform allows).
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        self.poll(ctx);
        let id = egui::ViewportId::from_hash_of("polish_images");
        let builder = egui::ViewportBuilder::default()
            .with_title(format!("Polish images \u{2013} {}", self.name))
            .with_inner_size([1300., 860.])
            .with_min_inner_size([760., 480.]);
        ctx.show_viewport_immediate(id, builder, |ctx, class| {
            if class == egui::ViewportClass::Embedded {
                let mut open = true;
                egui::Window::new("Polish images")
                    .open(&mut open)
                    .default_size([1000., 650.])
                    .resizable(true)
                    .show(ctx, |ui| self.contents(ui));
                self.open &= open;
            } else {
                egui::CentralPanel::default().show(ctx, |ui| self.contents(ui));
                if ctx.input(|i| i.viewport().close_requested()) {
                    self.open = false;
                }
            }
        });
    }

    fn contents(&mut self, ui: &mut Ui) {
        egui::TopBottomPanel::bottom("polish_bottom").show_inside(ui, |ui| {
            ui.add_space(4.);
            self.bottom_bar(ui);
            ui.add_space(2.);
        });
        egui::SidePanel::left("polish_controls")
            .resizable(true)
            .default_width(340.)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if self.check.is_some() {
                self.check_view(ui);
            } else {
                self.preview_toolbar(ui);
                self.preview(ui);
            }
        });
    }

    /// Quick before/after switches above the preview.
    fn preview_toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.stretch_on, "Stretch").on_hover_text(
                "Off: plain min-to-max contrast, to compare. Turns off flattening, evening out, \
                 clipping, gamma and equalisation (their settings are kept).",
            );
            ui.toggle_value(&mut self.show_aligned, "Alignment").on_hover_text(
                "Off: every tile where the map has it, to compare. The moves are kept and come \
                 back when you turn it on. Saving and \u{201c}Put on map\u{201d} use what is shown.",
            );
            ui.separator();
            let n = self.selected.len();
            if n > 0 {
                ui.weak(format!("{n} tile{} selected", if n == 1 { "" } else { "s" }));
                if ui.small_button("clear").on_hover_text("Esc").clicked() {
                    self.selected.clear();
                }
            }
            if ui.small_button("Select all").on_hover_text("Ctrl+A").clicked() {
                self.selected = self.draw_order();
            }
        });
    }

    fn bottom_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let ready = !self.sources.is_empty() && !self.busy_out;
            ui.label("Save size");
            egui::ComboBox::from_id_salt("polish_save_side")
                .selected_text(if self.save_side == 0 { "Full resolution".into() } else { format!("{} px", self.save_side) })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.save_side, 0, "Full resolution");
                    for s in [8000, 4000, 2000, 1000] {
                        ui.selectable_value(&mut self.save_side, s, format!("{s} px"));
                    }
                })
                .response
                .on_hover_text("Longest side of the saved picture. Full resolution: the finest pixel size among the stamps.");
            if ui.add_enabled(ready, egui::Button::new("\u{1F4BE} Save PNG\u{2026}")).clicked() {
                self.save();
            }
            ui.separator();
            if ui
                .add_enabled(ready, egui::Button::new("\u{2B07} Put on map"))
                .on_hover_text(
                    "Add the polished picture to the main map as an image layer, exactly where the \
                     tiles are here (no scale bar, empty parts transparent). It is saved with the \
                     project. The stamps themselves are not changed.",
                )
                .clicked()
            {
                self.put_on_map();
            }
            ui.checkbox(&mut self.hide_sources, "hide the stamps")
                .on_hover_text("Hide the stamps it was made from, so the polished picture is what you see.");
            if self.busy_out || self.align_rx.is_some() {
                ui.spinner();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(s) = &self.status {
                    ui.label(s);
                }
            });
        });
    }

    fn controls(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.strong(format!("{} stamps", self.sources.len()));
            if let Some((_, p)) = &self.prepared {
                ui.weak(format!("\u{b7} {} overlaps level-matched", p.pairs));
            }
        });
        ui.horizontal(|ui| {
            if ui
                .button("\u{2728} Make it pretty")
                .on_hover_text(
                    "A good starting point: line levelling, glitch lines fixed, tiles matched in \
                     height, tilt, bow and contrast, seams healed and blended, hills flattened, \
                     contrast evened out, a little equalisation.",
                )
                .clicked()
            {
                self.tile.background = processing::Background::LineLinear;
                self.tile.filters.lines = true;
                self.tile.level = LevelMatch::Bow;
                self.tile.match_contrast = true;
                self.tile.heal_all = true;
                self.tile.bridge = 0.08;
                self.tile.trim = self.tile.trim.max(0.01);
                self.look.blend = Blend::Feather;
                self.look.feather = 0.08;
                self.look.seam_blend = 0.4;
                self.look.flatten = 0.8;
                self.look.even = 0.5;
                self.look.equalize = 0.3;
                self.look.clip_low = 0.5;
                self.look.clip_high = 0.5;
            }
            if ui.button("Reset").on_hover_text("Settings back to the defaults (tile edits and alignment are kept)").clicked() {
                let keep = (self.look.flatten_nm, self.look.color_map);
                self.tile = TileSettings::default();
                self.look = Look { flatten_nm: keep.0, color_map: keep.1, ..Look::default() };
            }
        });
        ui.add_space(4.);

        if !self.selected.is_empty() {
            self.tile_editor(ui);
            ui.add_space(6.);
        }

        self.alignment_controls(ui);
        ui.add_space(6.);

        crate::theme::section_label(ui, "Each tile");
        egui::Grid::new("polish_tile").num_columns(2).spacing([8., 4.]).show(ui, |ui| {
            ui.label("Data");
            egui::ComboBox::from_id_salt("polish_channel")
                .selected_text(self.tile.channel.label())
                .show_ui(ui, |ui| {
                    for c in Channel::ALL {
                        ui.selectable_value(&mut self.tile.channel, c, c.label());
                    }
                })
                .response
                .on_hover_text("Average of forward and backward has less noise.");
            ui.end_row();
            processing::processing_rows(ui, "polish", &mut self.tile.background, &mut self.tile.filters);
            ui.label("Trim edges").on_hover_text(
                "Cut this much off every edge of each tile (ragged first lines, edge artefacts). \
                 Overlapping tiles fill in where they can.",
            );
            pct_slider(ui, &mut self.tile.trim, 0.0..=15.);
            ui.end_row();
        });
        ui.add_space(6.);

        crate::theme::section_label(ui, "Seams");
        egui::Grid::new("polish_seams").num_columns(2).spacing([8., 4.]).show(ui, |ui| {
            ui.label("Match height").on_hover_text(
                "Correct each tile so it agrees with its neighbours where they overlap, all tiles \
                 fitted together. Height: one offset each. + tilt: also a slope. + bow: also a \
                 curve (piezo creep, scanner bow).",
            );
            egui::ComboBox::from_id_salt("polish_level")
                .selected_text(self.tile.level.label())
                .show_ui(ui, |ui| {
                    for m in LevelMatch::ALL {
                        ui.selectable_value(&mut self.tile.level, m, m.label());
                    }
                });
            ui.end_row();
            ui.label("Match contrast").on_hover_text(
                "Also scale each tile's contrast to match its neighbours: tiles scanned with a \
                 blunter or sharper tip end up looking alike.",
            );
            ui.checkbox(&mut self.tile.match_contrast, "");
            ui.end_row();
            ui.label("Bridge gaps").on_hover_text(
                "Tiles that don't quite overlap but are this close (share of a tile) are still \
                 matched, edge to edge.",
            );
            pct_slider(ui, &mut self.tile.bridge, 0.0..=20.);
            ui.end_row();
            ui.label("Heal all seams").on_hover_text(
                "Bend every tile edge half way towards its neighbour along the whole seam, fading \
                 back to unchanged deeper in. Ctrl+click a seam in the preview to heal just that one \
                 fully.",
            );
            ui.checkbox(&mut self.tile.heal_all, "");
            ui.end_row();
            ui.label("Heal reach").on_hover_text("How far into a tile a healed seam's correction reaches.");
            pct_slider(ui, &mut self.tile.heal_reach, 5.0..=50.);
            ui.end_row();
            ui.label("Overlaps");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.look.blend, Blend::Feather, "Blend")
                    .on_hover_text("Fade from one tile into the next");
                ui.selectable_value(&mut self.look.blend, Blend::Topmost, "Topmost")
                    .on_hover_text("The top tile wins, as on the map");
            });
            ui.end_row();
            ui.label("Detail fade").on_hover_text("Width of the fade for fine detail, share of a tile. Keep small to avoid ghosting.");
            ui.add_enabled_ui(self.look.blend == Blend::Feather, |ui| pct_slider(ui, &mut self.look.feather, 0.0..=50.));
            ui.end_row();
            ui.label("Shading fade").on_hover_text(
                "Two-band blending: the broad shading of the tiles (hills, brightness) fades over \
                 this much of a tile, while fine detail keeps the narrow fade. 0 = off.",
            );
            pct_slider(ui, &mut self.look.seam_blend, 0.0..=50.);
            ui.end_row();
        });
        ui.add_space(6.);

        crate::theme::section_label(ui, "Contrast");
        egui::Grid::new("polish_contrast").num_columns(2).spacing([8., 4.]).show(ui, |ui| {
            ui.label("Hill size").on_hover_text("Size of the features 'Flatten hills' and 'Even out' work over.");
            ui.add(DragValue::new(&mut self.look.flatten_nm).range(1.0..=1e6).speed(1.).suffix(" nm"));
            ui.end_row();
            ui.label("Flatten hills").on_hover_text(
                "Remove large-scale hills and bowls across the whole picture so steps and fine \
                 features stand out.",
            );
            ui.add(Slider::new(&mut self.look.flatten, 0.0..=1.));
            ui.end_row();
            ui.label("Even out").on_hover_text(
                "Even out the contrast across the picture: areas where features look faint (e.g. \
                 rings on one side) are brought up to match the rest.",
            );
            ui.add(Slider::new(&mut self.look.even, 0.0..=1.));
            ui.end_row();
            ui.label("Clip dark").on_hover_text("Share of values shown fully dark");
            ui.add(Slider::new(&mut self.look.clip_low, 0.0..=20.).suffix(" %").logarithmic(true).smallest_positive(0.01));
            ui.end_row();
            ui.label("Clip bright").on_hover_text("Share of values shown fully bright");
            ui.add(Slider::new(&mut self.look.clip_high, 0.0..=20.).suffix(" %").logarithmic(true).smallest_positive(0.01));
            ui.end_row();
            ui.label("Gamma").on_hover_text("Below 1 brightens the darks, above 1 darkens the brights");
            ui.add(Slider::new(&mut self.look.gamma, 0.2..=5.).logarithmic(true));
            ui.end_row();
            ui.label("Equalize").on_hover_text("Spread the values evenly over the colour scale (histogram equalisation)");
            ui.add(Slider::new(&mut self.look.equalize, 0.0..=1.));
            ui.end_row();
        });
        histogram(ui, &self.stats);
        ui.add_space(6.);

        crate::theme::section_label(ui, "Look");
        egui::Grid::new("polish_look").num_columns(2).spacing([8., 4.]).show(ui, |ui| {
            ui.label("Colours");
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("polish_cmap")
                    .selected_text(self.look.color_map.label())
                    .show_ui(ui, |ui| {
                        for m in ColorMap::ALL {
                            ui.selectable_value(&mut self.look.color_map, m, m.label());
                        }
                    });
                ui.checkbox(&mut self.look.invert, "Invert");
            });
            ui.end_row();
            ui.label("Empty parts");
            egui::ComboBox::from_id_salt("polish_fill")
                .selected_text(self.look.fill.label())
                .show_ui(ui, |ui| {
                    for f in Fill::ALL {
                        ui.selectable_value(&mut self.look.fill, f, f.label());
                    }
                });
            ui.end_row();
            ui.label("Scale bar");
            ui.checkbox(&mut self.look.scale_bar, "");
            ui.end_row();
        });
    }

    fn alignment_controls(&mut self, ui: &mut Ui) {
        crate::theme::section_label(ui, "Alignment");
        let checked = self.pairs.iter().filter(|p| p.hand.is_some()).count();
        let n = self.pairs.len();
        ui.horizontal(|ui| {
            ui.label("Search \u{b1}").on_hover_text(
                "How far from where the map puts a tile the automatic match looks (share of a \
                 tile). Keep it small on repeating patterns, so a look-alike spot can't be picked.",
            );
            pct_slider(ui, &mut self.align_radius, 1.0..=30.);
        });
        ui.horizontal(|ui| {
            let idle = self.align_rx.is_none() && n > 0;
            if ui
                .add_enabled(idle, egui::Button::new("\u{1F50D} Auto-align"))
                .on_hover_text(
                    "Match every overlapping pair by its image within the search range, refuse weak \
                     matches and look-alikes, then fit all tiles together. Pairs you checked by eye \
                     count far more than automatic ones.",
                )
                .clicked()
            {
                self.auto_align(false);
            }
            if ui
                .add_enabled(n > 0, egui::Button::new("\u{1F441} Check seams by eye\u{2026}"))
                .on_hover_text(
                    "Step through every overlapping pair, the two tiles laid over each other, and \
                     line them up: drag or arrow keys, Enter to accept and go to the next.",
                )
                .clicked()
            {
                let k = self.pairs.iter().position(|p| p.hand.is_none()).unwrap_or(0);
                self.enter_check(k);
            }
        });
        ui.horizontal(|ui| {
            let idle = self.align_rx.is_none() && !self.sources.is_empty();
            let has_gds = !self.gds.is_empty();
            if ui
                .add_enabled(idle && has_gds, egui::Button::new("\u{25CE} Line up rings with GDS"))
                .on_hover_text(
                    "Make the rings smooth and continuous across the tiles: every tile is matched \
                     against the rings the GDS expects there and moved so its rings sit on the GDS \
                     rings, together with Auto-align for the overlaps (which settles the sliding \
                     along an arc that the rings alone can't see). Pairs checked by eye still count \
                     most. Matches that are unclear, or fit a neighbouring ring as well, are left out.",
                )
                .on_disabled_hover_text(if has_gds {
                    "Busy aligning"
                } else {
                    "No visible GDS with rings was on the map when this window was opened"
                })
                .clicked()
            {
                self.auto_align(true);
            }
            let anchored = self.anchors.iter().filter(|a| a.is_some()).count();
            if anchored > 0 {
                ui.weak(format!("{anchored} on the GDS"));
                if ui.small_button("forget").on_hover_text("Stop pulling tiles onto the GDS rings").clicked() {
                    self.anchors.iter_mut().for_each(|a| *a = None);
                    self.apply_alignment();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.weak(format!("{checked} of {n} pairs checked by eye"));
            if checked > 0 && ui.small_button("clear").clicked() {
                for p in &mut self.pairs {
                    p.hand = None;
                }
                self.apply_alignment();
            }
        });
        ui.horizontal(|ui| {
            if ui
                .button("Undo moves")
                .on_hover_text("Put every tile back where the map has it (checks are kept).")
                .clicked()
            {
                for e in &mut self.edits {
                    e.shift = DVec2::ZERO;
                }
            }
            let any = self.edits.iter().any(|e| e.shift.length() > 0.);
            if ui
                .add_enabled(any, egui::Button::new("Move stamps on map"))
                .on_hover_text("Move the stamps on the main map to these aligned positions (undo with Ctrl+Z there).")
                .clicked()
            {
                self.move_stamps_on_map();
            }
        });
        ui.checkbox(&mut self.show_outlines, "Show tile outlines");
    }

    fn tile_editor(&mut self, ui: &mut Ui) {
        self.selected.retain(|&i| i < self.sources.len());
        let Some(&i) = self.selected.last() else { return };
        let others: Vec<usize> = self.selected.iter().copied().filter(|&j| j != i).collect();
        let tf = self.transform_of(i);
        let span = (self.stats.hi - self.stats.lo).abs().max(f32::MIN_POSITIVE);
        crate::theme::section_label(ui, "Selected tile");
        ui.horizontal(|ui| {
            ui.strong(&self.sources[i].name);
            if !others.is_empty() {
                ui.weak(format!("+ {} more (moved together)", others.len()));
            }
            if ui.small_button("\u{2715}").on_hover_text("Deselect (Esc)").clicked() {
                self.selected.clear();
            }
        });
        let shift_before = self.edits[i].shift;
        let e = &mut self.edits[i];
        ui.horizontal(|ui| {
            ui.checkbox(&mut e.hidden, "Hide").on_hover_text("Leave this tile out");
            ui.checkbox(&mut e.on_top, "On top").on_hover_text("Draw this tile above the others");
            if ui.button("Reset tile").clicked() {
                let shift = e.shift;
                *e = TileEdit { shift, ..TileEdit::default() };
            }
        });
        egui::Grid::new("polish_tile_edit").num_columns(2).spacing([8., 4.]).show(ui, |ui| {
            ui.label("Brightness");
            let mut v = e.offset / span * 100.;
            if ui.add(Slider::new(&mut v, -100.0..=100.).suffix(" %")).changed() {
                e.offset = v / 100. * span;
            }
            ui.end_row();
            ui.label("Contrast");
            ui.add(Slider::new(&mut e.gain, 0.2..=5.).logarithmic(true));
            ui.end_row();
            for (k, label) in [(0, "Tilt across"), (1, "Tilt down")] {
                ui.label(label);
                let mut v = e.tilt[k] / span * 100.;
                if ui.add(Slider::new(&mut v, -200.0..=200.).suffix(" %")).changed() {
                    e.tilt[k] = v / 100. * span;
                }
                ui.end_row();
            }
            ui.label("Move").on_hover_text(
                "Or Shift+drag the tile in the preview. All selected tiles move together. \
                 Auto-align and seam checks set this again.",
            );
            ui.horizontal(|ui| {
                ui.add(DragValue::new(&mut e.shift.x).speed(0.5).suffix(" nm").prefix("x "));
                ui.add(DragValue::new(&mut e.shift.y).speed(0.5).suffix(" nm").prefix("y "));
            });
            ui.end_row();
            // Crop, labelled by where each edge is on the map.
            let mut edges: Vec<(u8, &str)> = (0..4u8).map(|k| (k, polish::edge_name(&tf, k))).collect();
            let rank = |n: &str| ["top", "bottom", "left", "right"].iter().position(|x| *x == n).unwrap_or(4);
            edges.sort_by_key(|(_, n)| rank(n));
            for (k, name) in edges {
                ui.label(format!("Crop {name}"));
                pct_slider(ui, &mut e.crop[k as usize], 0.0..=45.);
                ui.end_row();
            }
        });
        if !e.heals.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.label("Healed:");
                let mut remove = None;
                for (n, &k) in e.heals.iter().enumerate() {
                    if ui.small_button(format!("{} \u{2715}", polish::edge_name(&tf, k))).on_hover_text("Stop healing this edge").clicked() {
                        remove = Some(n);
                    }
                }
                if let Some(n) = remove {
                    e.heals.remove(n);
                }
            });
        }
        // A typed move takes the other selected tiles along.
        let moved = self.edits[i].shift - shift_before;
        if moved != DVec2::ZERO {
            for j in others {
                self.edits[j].shift += moved;
            }
        }
    }

    /// The tile edge forming the seam nearest to map point `p`: (tile, edge).
    fn seam_at(&self, p: DVec2) -> Option<(usize, u8)> {
        let tfs: Vec<(usize, DAffine2)> = self.draw_order().into_iter().map(|i| (i, self.transform_of(i))).collect();
        let inside = |t: &DAffine2, w: DVec2| {
            let l = t.inverse().transform_point2(w);
            l.x.abs() <= 0.5 && l.y.abs() <= 0.5
        };
        let mut best: Option<(f64, usize, u8)> = None;
        for (i, t) in &tfs {
            let l = t.inverse().transform_point2(p);
            if l.x.abs() > 0.7 || l.y.abs() > 0.7 {
                continue;
            }
            let size = DVec2::new(t.matrix2.x_axis.length(), t.matrix2.y_axis.length());
            for e in 0..4u8 {
                let (dist, along, scale) = match e {
                    0 => ((l.x + 0.5).abs(), l.y, size.x),
                    1 => ((0.5 - l.x).abs(), l.y, size.x),
                    2 => ((l.y + 0.5).abs(), l.x, size.y),
                    _ => ((0.5 - l.y).abs(), l.x, size.y),
                };
                if along.abs() > 0.5 || dist > 0.15 {
                    continue;
                }
                // Something across the edge?
                let out = match e {
                    0 => DVec2::new(-0.52, along),
                    1 => DVec2::new(0.52, along),
                    2 => DVec2::new(along, -0.52),
                    _ => DVec2::new(along, 0.52),
                };
                let w = t.transform_point2(out);
                if !tfs.iter().any(|(j, tj)| j != i && inside(tj, w)) {
                    continue;
                }
                let nm = dist * scale;
                if best.is_none_or(|b| nm < b.0) {
                    best = Some((nm, *i, e));
                }
            }
        }
        best.map(|b| (b.1, b.2))
    }

    fn preview(&mut self, ui: &mut Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0., Color32::from_gray(24));
        // Checkerboard so transparent parts show.
        let cell = 16.;
        let (nx, ny) = ((rect.width() / cell).ceil() as i32, (rect.height() / cell).ceil() as i32);
        for j in 0..ny {
            for i in (j % 2..nx).step_by(2) {
                let r = egui::Rect::from_min_size(rect.min + egui::vec2(i as f32 * cell, j as f32 * cell), egui::vec2(cell, cell));
                painter.rect_filled(r.intersect(rect), 0., Color32::from_gray(32));
            }
        }
        let Some(tex) = &self.texture else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Working\u{2026}", egui::FontId::proportional(16.), Color32::GRAY);
            return;
        };
        let size = tex.size_vec2();
        let fit = (rect.width() / size.x).min(rect.height() / size.y) * 0.98;
        let mods = ui.input(|i| i.modifiers);
        if resp.double_clicked() {
            self.zoom = 1.;
            self.pan = egui::Vec2::ZERO;
        }
        if let Some(pos) = resp.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0. {
                let old = self.zoom;
                self.zoom = (self.zoom * (scroll / 300.).exp()).clamp(0.2, 40.);
                // Keep the point under the mouse still.
                let c = rect.center() + self.pan;
                self.pan += (pos - c) * (1. - self.zoom / old);
            }
        }
        let shown = size * fit * self.zoom;
        let img_rect = egui::Rect::from_center_size(rect.center() + self.pan, shown);
        // Screen ↔ map.
        let k = shown.x / size.x;
        let (tl, px) = (self.preview_top_left, self.preview_px);
        let to_screen = |w: DVec2| {
            img_rect.min + egui::vec2(((w.x - tl.x) / px) as f32 * k, ((tl.y - w.y) / px) as f32 * k)
        };
        let to_world = |s: egui::Pos2| {
            let d = s - img_rect.min;
            DVec2::new(tl.x + (d.x / k) as f64 * px, tl.y - (d.y / k) as f64 * px)
        };
        let tex_id = tex.id();
        let tile_at = |me: &Self, w: DVec2| {
            me.draw_order().into_iter().find(|&i| {
                let l = me.transform_of(i).inverse().transform_point2(w);
                l.x.abs() <= 0.5 && l.y.abs() <= 0.5
            })
        };
        // Shift+drag on a tile moves it (and the other selected tiles with
        // it); a plain drag pans.
        if resp.drag_started() {
            self.dragging = false;
            let origin = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos());
            if let (true, Some(pos)) = (mods.shift, origin) {
                if let Some(t) = tile_at(self, to_world(pos)) {
                    if !self.show_aligned {
                        self.status = Some("Turn Alignment on (above the preview) to move tiles".into());
                    } else {
                        if !self.selected.contains(&t) {
                            self.selected = vec![t];
                        }
                        self.dragging = true;
                    }
                }
            }
        }
        if resp.dragged() {
            let d = resp.drag_delta();
            if self.dragging {
                let mv = DVec2::new((d.x / k) as f64 * px, -(d.y / k) as f64 * px);
                for &i in &self.selected {
                    self.edits[i].shift += mv;
                }
            } else {
                self.pan += d;
            }
        }
        if self.dragging && (resp.drag_stopped() || !ui.input(|i| i.pointer.any_down())) {
            self.dragging = false;
        }
        if resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let w = to_world(pos);
                if mods.command {
                    match self.seam_at(w) {
                        Some((i, e)) => {
                            if !self.edits[i].heals.contains(&e) {
                                self.edits[i].heals.push(e);
                            }
                            self.selected = vec![i];
                            self.status = Some(format!(
                                "Healing the {} edge of {} (undo: \u{2715} next to it under Selected tile)",
                                polish::edge_name(&self.transform_of(i), e),
                                self.sources[i].name
                            ));
                        }
                        None => self.status = Some("No seam there: Ctrl+click right on a tile edge where another tile meets it".into()),
                    }
                } else if mods.shift {
                    // Add to or take out of the selection.
                    if let Some(t) = tile_at(self, w) {
                        match self.selected.iter().position(|&j| j == t) {
                            Some(n) => {
                                self.selected.remove(n);
                            }
                            None => self.selected.push(t),
                        }
                    }
                } else {
                    self.selected = tile_at(self, w).into_iter().collect();
                }
            }
        }
        if !ui.ctx().wants_keyboard_input() {
            if ui.input(|i| i.key_pressed(Key::Escape)) {
                self.selected.clear();
            }
            if resp.hovered() && ui.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::A)) {
                self.selected = self.draw_order();
            }
        }
        painter.image(tex_id, img_rect, egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(1., 1.)), Color32::WHITE);
        // Tiles being moved: their part of the last preview, lifted off its
        // old place and drawn where it is going.
        if self.dragging {
            let corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].map(|(x, y)| DVec2::new(x, y));
            let moving: Vec<usize> = self.draw_order().into_iter().rev().filter(|i| self.selected.contains(i)).collect();
            let old_of = |i: usize| {
                let shift = self.tex_edits.get(i).map_or(DVec2::ZERO, |e| e.shift);
                DAffine2::from_translation(shift) * self.sources[i].transform
            };
            for &i in &moving {
                let old = old_of(i);
                let pts: Vec<egui::Pos2> = corners.iter().map(|&c| to_screen(old.transform_point2(c))).collect();
                painter.add(egui::Shape::convex_polygon(pts, Color32::from_black_alpha(150), Stroke::NONE));
            }
            for &i in &moving {
                let (old, new) = (old_of(i), self.transform_of(i));
                let mut mesh = egui::Mesh::with_texture(tex_id);
                for &c in &corners {
                    let w = old.transform_point2(c);
                    let uv = egui::pos2(((w.x - tl.x) / px) as f32 / size.x, ((tl.y - w.y) / px) as f32 / size.y);
                    mesh.vertices.push(egui::epaint::Vertex { pos: to_screen(new.transform_point2(c)), uv, color: Color32::WHITE });
                }
                mesh.add_triangle(0, 1, 2);
                mesh.add_triangle(0, 2, 3);
                painter.add(egui::Shape::mesh(mesh));
            }
        }
        // Tile outlines: all of them if asked, the selected ones always.
        let accent = crate::theme::accent(ui);
        for i in 0..self.sources.len() {
            let sel = self.selected.contains(&i);
            if !(sel || self.show_outlines) || self.edits[i].hidden {
                continue;
            }
            let t = self.transform_of(i);
            let c = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].map(|(x, y)| to_screen(t.transform_point2(DVec2::new(x, y))));
            let stroke = if sel { Stroke::new(2., accent) } else { Stroke::new(1., Color32::from_white_alpha(90)) };
            painter.add(egui::Shape::closed_line(c.to_vec(), stroke));
            // Healed edges in green.
            for &e in &self.edits[i].heals {
                let (a, b) = match e {
                    0 => (c[0], c[3]),
                    1 => (c[1], c[2]),
                    2 => (c[0], c[1]),
                    _ => (c[3], c[2]),
                };
                painter.line_segment([a, b], Stroke::new(3., Color32::from_rgb(90, 220, 120)));
            }
        }
        if self.busy() {
            painter.text(rect.right_top() + egui::vec2(-10., 10.), egui::Align2::RIGHT_TOP, "updating\u{2026}", egui::FontId::proportional(13.), Color32::LIGHT_GRAY);
        }
        painter.text(
            rect.left_bottom() + egui::vec2(8., -8.),
            egui::Align2::LEFT_BOTTOM,
            "click: select tile \u{b7} Shift+click: add/remove \u{b7} Shift+drag: move selected tiles \u{b7} Ctrl+click a seam: heal it \u{b7} drag: pan \u{b7} scroll: zoom \u{b7} double-click: fit",
            egui::FontId::proportional(12.),
            Color32::GRAY,
        );
    }

    // ---------- Seam check by eye ----------

    fn enter_check(&mut self, k: usize) {
        let Some(p) = self.pairs.get(k) else { return };
        let d = p.correction().unwrap_or(self.edits[p.b].shift - self.edits[p.a].shift);
        let like = self.check.take();
        self.check = Some(Check {
            k,
            d,
            tex: like.as_ref().and_then(|c| c.tex.clone()),
            zoom: 1.,
            pan: DVec2::ZERO,
            opacity: like.as_ref().map_or(0.5, |c| c.opacity),
            blink: like.as_ref().is_some_and(|c| c.blink),
        });
    }

    fn pair_textures(&self, ctx: &egui::Context, p: &Pair) -> [egui::TextureHandle; 2] {
        [p.a, p.b].map(|i| {
            let img = match polish::clean_one(&self.sources[i], &self.tile) {
                Some((d, w, h)) => gray_image(&d, w, h),
                None => egui::ColorImage::from_gray([1, 1], &[0]),
            };
            ctx.load_texture(format!("polish_check_{i}"), img, egui::TextureOptions::LINEAR)
        })
    }

    fn check_view(&mut self, ui: &mut Ui) {
        let n = self.pairs.len();
        let Some(k) = self.check.as_ref().map(|c| c.k).filter(|&k| k < n) else {
            self.check = None;
            return;
        };
        let p = self.pairs[k].clone();
        if self.check.as_ref().unwrap().tex.as_ref().is_none_or(|t| t.0 != k) {
            let tex = self.pair_textures(ui.ctx(), &p);
            self.check.as_mut().unwrap().tex = Some((k, tex));
        }
        let checked = self.pairs.iter().filter(|q| q.hand.is_some()).count();
        let srcs = self.sources.clone();
        let (sa, sb) = (&srcs[p.a], &srcs[p.b]);
        let step = sb.transform.matrix2.x_axis.length() / sb.forward.size[0].max(1) as f64;
        let mut go: Option<isize> = None;
        let mut accept = false;
        let mut leave = false;
        // Keys (not while typing in a field).
        if !ui.ctx().wants_keyboard_input() {
            ui.input_mut(|i| {
                if i.consume_key(Modifiers::NONE, Key::Enter) {
                    accept = true;
                }
                if i.consume_key(Modifiers::NONE, Key::Backspace) {
                    go = Some(-1);
                }
                if i.consume_key(Modifiers::NONE, Key::Tab) {
                    go = Some(1);
                }
                if i.consume_key(Modifiers::NONE, Key::Escape) {
                    leave = true;
                }
                if i.consume_key(Modifiers::NONE, Key::A) {
                    if let Some(a) = p.auto {
                        self.check.as_mut().unwrap().d = a.0;
                    }
                }
                if i.consume_key(Modifiers::NONE, Key::B) {
                    let c = self.check.as_mut().unwrap();
                    c.blink = !c.blink;
                }
                let s = if i.modifiers.shift { 0.25 * step } else { step };
                let c = self.check.as_mut().unwrap();
                for (key, dv) in [
                    (Key::ArrowLeft, DVec2::new(-s, 0.)),
                    (Key::ArrowRight, DVec2::new(s, 0.)),
                    (Key::ArrowUp, DVec2::new(0., s)),
                    (Key::ArrowDown, DVec2::new(0., -s)),
                ] {
                    if i.key_pressed(key) {
                        c.d += dv;
                    }
                }
            });
        }
        // Header.
        ui.horizontal(|ui| {
            ui.strong(format!("Seam {} of {n}", k + 1));
            ui.label(format!("{}  \u{2194}  {}", sa.name, sb.name));
            ui.weak(format!("\u{b7} {checked} checked"));
            match (&p.hand, &p.auto, p.refused) {
                (Some(_), _, _) => {
                    ui.colored_label(Color32::from_rgb(90, 220, 120), "\u{2714} checked by eye");
                }
                (None, Some((_, s)), None) => {
                    ui.weak(format!("auto match {s:.2}"));
                }
                (None, _, Some(r)) => {
                    ui.colored_label(Color32::from_rgb(240, 170, 60), format!("auto: {}", r.label()));
                }
                _ => {
                    ui.weak("not auto-aligned yet");
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.button("\u{25C0} Back").on_hover_text("Backspace").clicked() {
                go = Some(-1);
            }
            if ui.button("\u{2714} OK, next").on_hover_text("Enter: this pair lines up; go to the next").clicked() {
                accept = true;
            }
            if ui.button("Skip \u{25B6}").on_hover_text("Tab: next pair without checking this one").clicked() {
                go = Some(1);
            }
            if ui.add_enabled(p.auto.is_some(), egui::Button::new("Use auto")).on_hover_text("A: back to the automatic match").clicked() {
                if let Some(a) = p.auto {
                    self.check.as_mut().unwrap().d = a.0;
                }
            }
            if ui.add_enabled(p.hand.is_some(), egui::Button::new("Uncheck")).clicked() {
                self.pairs[k].hand = None;
                self.apply_alignment();
            }
            ui.separator();
            let c = self.check.as_mut().unwrap();
            ui.checkbox(&mut c.blink, "Blink (B)");
            ui.add_enabled(!c.blink, Slider::new(&mut c.opacity, 0.0..=1.).show_value(false).text("second tile"));
            if ui.button("Done").on_hover_text("Esc: back to the preview").clicked() {
                leave = true;
            }
        });
        ui.weak(
            "Grey: first tile. Orange: second tile. Drag it or use the arrow keys (Shift: \u{bc} px) until the \
             features line up; with Blink on it is right when nothing jumps. Enter accepts and moves on. \
             Scroll: zoom, right-drag: pan.",
        );
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let c = self.check.as_mut().unwrap();
        let ta = sa.transform;
        let tb = DAffine2::from_translation(c.d) * sb.transform;
        // View: the shared strip, generously framed.
        let center = polish_align::shared_center(&ta, &tb);
        let tile_size = ta.matrix2.x_axis.length().max(ta.matrix2.y_axis.length());
        let share = polish_align::shared(&ta, &tb).max(0.05);
        let frame = tile_size * (share.sqrt() * 1.3).clamp(0.35, 1.3);
        let scale = (rect.width().min(rect.height()) as f64 / frame) * c.zoom as f64;
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0. {
                c.zoom = (c.zoom * (scroll / 250.).exp()).clamp(0.3, 30.);
            }
        }
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            c.d += DVec2::new(d.x as f64, -d.y as f64) / scale;
        }
        if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            let d = resp.drag_delta();
            c.pan -= DVec2::new(d.x as f64, -d.y as f64) / scale;
        }
        let view_c = center + c.pan;
        let to_screen = |w: DVec2| {
            let r = (w - view_c) * scale;
            rect.center() + egui::vec2(r.x as f32, -r.y as f32)
        };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0., Color32::from_gray(15));
        let Some((_, tex)) = &c.tex else { return };
        let orange = |a: f32| Color32::from_rgba_unmultiplied(255, 170, 70, (a.clamp(0., 1.) * 255.) as u8);
        let show_b_only = c.blink && (ui.input(|i| i.time) * 2.).floor() as i64 % 2 == 1;
        if c.blink {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
        if !show_b_only {
            textured_quad(&painter, &tex[0], &ta, &to_screen, Color32::WHITE);
        }
        if c.blink {
            if show_b_only {
                textured_quad(&painter, &tex[1], &tb, &to_screen, orange(1.));
            }
        } else {
            textured_quad(&painter, &tex[1], &tb, &to_screen, orange(c.opacity));
        }
        let outline = |t: &DAffine2, col: Color32| {
            let pts = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].map(|(x, y)| to_screen(t.transform_point2(DVec2::new(x, y))));
            painter.add(egui::Shape::closed_line(pts.to_vec(), Stroke::new(1., col)));
        };
        outline(&ta, Color32::GRAY);
        outline(&tb, orange(1.));
        if let Some((a, _)) = p.auto {
            outline(&(DAffine2::from_translation(a) * sb.transform), Color32::from_rgb(80, 200, 255));
        }
        let d = c.d;
        if accept {
            self.pairs[k].hand = Some(d);
            self.apply_alignment();
            go = Some(1);
        }
        if let Some(delta) = go {
            let next = k as isize + delta;
            if next < 0 || next >= n as isize {
                self.status = Some(format!("Seam check: {} of {n} checked by eye", self.pairs.iter().filter(|q| q.hand.is_some()).count()));
                self.check = None;
            } else {
                self.enter_check(next as usize);
            }
        }
        if leave {
            self.check = None;
        }
    }
}

/// A slider showing a share (0..1) as percent.
fn pct_slider(ui: &mut Ui, v: &mut f32, range: std::ops::RangeInclusive<f32>) {
    let mut pct = *v * 100.;
    if ui.add(Slider::new(&mut pct, range).suffix(" %").step_by(0.5)).changed() {
        *v = pct / 100.;
    }
}

/// Grey picture of a tile, 1 % clipped at each end. Row 0 is local y −0.5.
fn gray_image(d: &[f32], w: usize, h: usize) -> egui::ColorImage {
    let mut sorted: Vec<f32> = d.iter().copied().filter(|x| x.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    let pick = |q: f64| sorted.get(((sorted.len().max(1) - 1) as f64 * q) as usize).copied().unwrap_or(0.);
    let (lo, hi) = (pick(0.01), pick(0.99));
    let span = (hi - lo).max(f32::MIN_POSITIVE);
    let bytes: Vec<u8> = d.iter().map(|&x| if x.is_finite() { (((x - lo) / span).clamp(0., 1.) * 255.) as u8 } else { 0 }).collect();
    egui::ColorImage::from_gray([w, h], &bytes)
}

/// Draw `tex` (row 0 at local y −0.5) where `t` puts it on the map.
fn textured_quad(painter: &egui::Painter, tex: &egui::TextureHandle, t: &DAffine2, to_screen: &dyn Fn(DVec2) -> egui::Pos2, tint: Color32) {
    let mut mesh = egui::Mesh::with_texture(tex.id());
    for ((x, y), (u, v)) in [((-0.5, -0.5), (0., 0.)), ((0.5, -0.5), (1., 0.)), ((0.5, 0.5), (1., 1.)), ((-0.5, 0.5), (0., 1.))] {
        mesh.vertices.push(egui::epaint::Vertex { pos: to_screen(t.transform_point2(DVec2::new(x, y))), uv: egui::pos2(u, v), color: tint });
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.add(egui::Shape::mesh(mesh));
}

/// Value histogram with the stretch limits marked.
fn histogram(ui: &mut Ui, s: &Stats) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(320.), 56.), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 3., ui.visuals().extreme_bg_color);
    let max = s.hist.iter().copied().max().unwrap_or(0).max(1) as f32;
    let n = s.hist.len().max(1) as f32;
    let col = ui.visuals().weak_text_color();
    for (i, &c) in s.hist.iter().enumerate() {
        let x0 = rect.left() + rect.width() * i as f32 / n;
        let x1 = rect.left() + rect.width() * (i + 1) as f32 / n;
        let hgt = rect.height() * (c as f32 / max).sqrt();
        p.rect_filled(egui::Rect::from_min_max(egui::pos2(x0, rect.bottom() - hgt), egui::pos2(x1, rect.bottom())), 0., col);
    }
    let (r0, r1) = s.range;
    if r1 > r0 {
        for v in [s.lo, s.hi] {
            let x = rect.left() + rect.width() * ((v - r0) / (r1 - r0)).clamp(0., 1.);
            p.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], Stroke::new(1.5, crate::theme::accent(ui)));
        }
    }
}
