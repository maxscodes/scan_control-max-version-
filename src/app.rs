use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    sync::LazyLock,
};

use crate::{
    components::{
        file_dialog_native::{ObjectImportDialog, ProjectOpenDialog, ProjectSaveDialog},
        recent_files,
        selectable_list::{SelectableEntry, SelectableList},
    },
    connection::{nanonis::NanonisConnection, Connection},
    project::{Persistant, ProjectDb},
    scan_view::{
        world_delta_transform, BorderRectangle, ImageEncoder, ScaleBar, ScanView, ScanViewCtx,
    },
    theme::{self, ThemeChoice},
    utils::vec_interop::{IntoEgui as _, IntoGlam as _},
    undo_queue::{StateModify, UndoQueue},
    view_object::{Object, Placement},
};
use egui::{Align2, Button, Color32, Layout, MenuBar, Modifiers, RichText, Stroke, Ui};
use glam::{DAffine2, DVec2};
use itertools::{izip, Itertools};
use redb::{ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition, WriteTransaction};
use tracing::{error, info};
use uuid::Uuid;

pub const COLOR_MAP_SIZE: usize = 256;

pub static CONFIG_DIR: LazyLock<directories::ProjectDirs> = LazyLock::new(|| {
    let base_dir = directories::ProjectDirs::from("", "qsi", "scan_control").unwrap();
    std::fs::create_dir_all(base_dir.config_local_dir()).unwrap();
    base_dir
});

/// For background work (e.g. stamp analysis) to ask for a repaint when done.
static REPAINT: std::sync::OnceLock<egui::Context> = std::sync::OnceLock::new();

pub fn request_repaint() {
    if let Some(ctx) = REPAINT.get() {
        ctx.request_repaint();
    }
}

pub struct MyApp {
    app_state: AppState,
    image_encoder: ImageEncoder,
    undo_queue: UndoQueue<AppState>,
    /// Map position (nm) under the mouse, for the status bar.
    cursor_world: Option<DVec2>,
    import_file_dialog: ObjectImportDialog,
    project_save_dialog: ProjectSaveDialog,
    project_open_dialog: ProjectOpenDialog,
    pending_connections: Vec<Box<dyn Connection>>,
    active_connection: Option<Box<dyn Connection>>,
    project: ProjectDb,
    /// Zoom the map to this "goto" transform on the next frame.
    pending_goto: Option<DAffine2>,
    /// GDS position before the last "Shift GDS", for Undo.
    gds_shift_undo: Option<(Uuid, DAffine2)>,
    /// Where each object was when nothing was being moved; a difference
    /// once the user lets go becomes an undoable move.
    placements: HashMap<Uuid, Placement>,
    /// Take the current placements as settled without recording a move
    /// (after undo/redo or opening a project).
    resync_placements: bool,
    /// "Align stamps": lines up the selected stamps by their overlaps.
    aligner: crate::analysis::mosaic::Aligner,
    align_status: Option<(String, std::time::Instant)>,
    /// "Save as one image": result messages from the background save.
    export_tx: crossbeam::channel::Sender<String>,
    export_rx: crossbeam::channel::Receiver<String>,
}

pub struct AppState {
    scan_view: ScanView,
    object_list: SelectableList<Object>,
}

impl MyApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let _ = REPAINT.set(cc.egui_ctx.clone());
        crate::components::file_dialog_native::set_main_window(cc);
        let project = ProjectDb::new_temp().unwrap();
        let wgpu = cc.wgpu_render_state.as_ref().unwrap();
        let image_encoder = ImageEncoder::new(wgpu);
        let nanonis_connection = Box::new(NanonisConnection::new(cc.egui_ctx.clone(), "localhost"));
        let object_list = SelectableList::new();
        let import_file_dialog = ObjectImportDialog::new();
        let project_save_dialog = ProjectSaveDialog::new();
        let project_open_dialog = ProjectOpenDialog::new();

        let export_channel = crossbeam::channel::unbounded();
        Self {
            app_state: AppState {
                scan_view: ScanView::new(&image_encoder),
                object_list,
            },
            import_file_dialog,
            image_encoder,
            undo_queue: UndoQueue::new(),
            cursor_world: None,
            pending_goto: None,
            gds_shift_undo: None,
            placements: HashMap::new(),
            resync_placements: true,
            aligner: Default::default(),
            align_status: None,
            export_tx: export_channel.0,
            export_rx: export_channel.1,
            pending_connections: vec![nanonis_connection],
            active_connection: None,
            project,
            project_save_dialog,
            project_open_dialog,
        }
    }
    pub fn mod_state<T: StateModify<AppState>>(&mut self, modifier: T) {
        self.undo_queue.push(&mut self.app_state, modifier);
    }
    fn undo(&mut self) {
        self.undo_queue.undo(&mut self.app_state);
        self.resync_placements = true;
    }
    fn redo(&mut self) {
        self.undo_queue.redo(&mut self.app_state);
        self.resync_placements = true;
    }

    /// Save the stamps at `indexes` (topmost first) as one picture laid out
    /// as on the map; asks where to save it.
    fn export_image(&mut self, indexes: &[usize], name: &str) {
        let pieces: Vec<crate::analysis::mosaic_image::Piece> = indexes
            .iter()
            .filter_map(|&i| match &**self.app_state.object_list.get(i)? {
                Object::ScanImage { image, .. } => Some(crate::analysis::mosaic_image::Piece {
                    transform: image.transform,
                    data: image.forward_data.clone(),
                }),
                _ => None,
            })
            .collect();
        if pieces.is_empty() {
            return;
        }
        let file: String = name
            .chars()
            .map(|c| if c.is_alphanumeric() || " -_().".contains(c) { c } else { '_' })
            .collect();
        let dialog = crate::components::file_dialog_native::dialog()
            .add_filter("PNG image", &["png"])
            .set_title("Save as one image")
            .set_file_name(format!("{}.png", file.trim()));
        let tx = self.export_tx.clone();
        std::thread::spawn(move || {
            let Some(path) = dialog.save_file() else { return };
            let n = pieces.len();
            let msg = match crate::analysis::mosaic_image::render(&pieces, 8000) {
                Some(img) => match img.save(&path) {
                    Ok(()) => format!(
                        "Saved {n} stamps as {} ({}\u{d7}{} px)",
                        path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
                        img.width(),
                        img.height()
                    ),
                    Err(e) => format!("Could not save the image: {e}"),
                },
                None => "Nothing to save: the stamps have no image data".into(),
            };
            let _ = tx.send(msg);
        });
    }

    /// Record moves of map objects (dragging, typed positions, Shift GDS,
    /// …) for Undo, once each move is finished: no mouse button held and no
    /// field being edited.
    fn track_moves(&mut self, ctx: &egui::Context) {
        let busy = ctx.input(|i| i.pointer.any_down()) || ctx.wants_keyboard_input();
        if busy && !self.resync_placements {
            return;
        }
        let now: HashMap<Uuid, Placement> = self
            .app_state
            .object_list
            .iter()
            .filter_map(|o| o.placement().map(|p| (o.uuid(), p)))
            .collect();
        if !std::mem::take(&mut self.resync_placements) {
            let moves: Vec<(Uuid, Placement, Placement)> = now
                .iter()
                .filter_map(|(id, p)| {
                    let before = *self.placements.get(id)?;
                    (before != *p).then_some((*id, before, *p))
                })
                .collect();
            if !moves.is_empty() {
                self.mod_state(MoveObjectsModifier(moves));
            }
        }
        self.placements = now;
    }

    /// Commit the current objects (GDS, images, scan area and its history)
    /// to the open project database.
    fn write_project(&mut self) -> Result<(), Box<dyn Error>> {
        let txn = self.project.db().begin_write()?;
        self.app_state.object_list.db_update(&txn)?;
        txn.commit()?;
        Ok(())
    }
    fn save_project(&mut self) {
        if let Err(e) = self.write_project() {
            error!("saving project failed: {e}");
            return;
        }
        if self.project.is_temp() {
            self.project_save_dialog.select_path();
        } else {
            self.verify_saved();
        }
    }
    /// Save As / first save: write the latest state (it may have changed while
    /// the dialog was open), then copy the database to `path`.
    fn save_project_as(&mut self, path: std::path::PathBuf) -> Result<(), Box<dyn Error>> {
        self.write_project()?;
        self.project.save_as(path)?;
        recent_files::add_project(self.project.current_path());
        self.verify_saved();
        Ok(())
    }
    fn project_name(&self) -> String {
        if self.project.is_temp() {
            "Untitled project".into()
        } else {
            self.project
                .current_path()
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "project".into())
        }
    }
    fn window_title(&self) -> String {
        format!("Scan Control v{} - {}", crate::version::version(), self.project_name())
    }
    fn show_status_bar(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let status = self.active_connection.as_ref().map(|c| c.status());
            let (color, text) = match (&status, self.pending_connections.is_empty()) {
                (Some(_), _) => (theme::OK, "Connected"),
                (None, false) => (theme::WARN, "Connecting\u{2026}"),
                (None, true) => (theme::IDLE, "Offline"),
            };
            theme::status_pill(ui, color, text, false);
            if let Some(s) = &status {
                ui.separator();
                ui.label(&s.scan);
                if let Some(b) = s.bias {
                    ui.separator();
                    ui.label(format!("Bias {b:.3} V"));
                }
                if let Some(i) = s.current {
                    ui.separator();
                    ui.label(format!("I {:.1} pA", i * 1e12));
                }
                if let Some(t) = s.tip_nm {
                    ui.separator();
                    ui.label(format!("Tip ({:.3}, {:.3}) \u{b5}m", t.x * 1e-3, t.y * 1e-3));
                }
            }
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(self.project_name()).weak());
                if let Some(p) = self.cursor_world {
                    ui.separator();
                    ui.monospace(format!("{:>9.3}, {:>9.3} \u{b5}m", p.x * 1e-3, p.y * 1e-3));
                }
                // Result of the last "Align stamps" / "Save as image", for a while.
                if let Some((text, at)) = &self.align_status {
                    if at.elapsed().as_secs() < 15 {
                        ui.separator();
                        ui.label(RichText::new(text).color(theme::accent(ui)))
                            .on_hover_text(text);
                        ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
                    }
                }
            });
        });
    }
    /// Check that the saved file lists every object currently loaded.
    fn verify_saved(&self) {
        let expected = self.app_state.object_list.len() as u64;
        let saved = self
            .project
            .db()
            .begin_read()
            .map_err(Box::<dyn Error>::from)
            .and_then(|txn| Ok(txn.open_table(OBJECT_LIST_TABLE)?.len()?));
        let path = self.project.current_path().display();
        match saved {
            Ok(n) if n == expected => info!("saved {n} objects to {path}"),
            Ok(n) => error!("project {path} holds {n} objects, expected {expected}"),
            Err(e) => error!("could not verify project {path}: {e}"),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match &mut self.active_connection {
            Some(conn) => conn.update(&mut self.app_state.object_list, &self.image_encoder),
            None => {
                for i in 0..self.pending_connections.len() {
                    if self.pending_connections[i]
                        .poll_connected(&mut self.app_state.object_list, &self.image_encoder)
                    {
                        self.active_connection = Some(self.pending_connections.remove(i));
                    }
                }
            }
        }
        while let Some(object) = self.import_file_dialog.try_recv_object() {
            let entry = object.into_entry(uuid::Uuid::new_v4());
            self.app_state.object_list.push(entry);
        }
        if let Some(summary) = self.aligner.poll(&mut self.app_state.object_list) {
            self.align_status = Some((summary, std::time::Instant::now()));
        }
        if let Ok(msg) = self.export_rx.try_recv() {
            self.align_status = Some((msg, std::time::Instant::now()));
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::CTRL, egui::Key::S)) {
            self.save_project();
        }
        if let Some(path) = self.project_save_dialog.try_recv_path() {
            match self.save_project_as(path) {
                Ok(()) => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.window_title()));
                }
                Err(e) => error!("save as failed: {e}"),
            }
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::CTRL, egui::Key::O)) {
            self.project_open_dialog.pick_file();
        }
        if let Some(project) = self.project_open_dialog.try_recv_project() {
            // Read before switching, so a bad file doesn't replace the
            // current project with an empty one.
            let loaded = project
                .db()
                .begin_write()
                .map_err(Box::<dyn Error>::from)
                .and_then(|txn| {
                    SelectableList::<Object>::db_read(Uuid::new_v4(), &txn, &self.image_encoder)
                });
            let path = project.current_path().display().to_string();
            match loaded {
                Ok(list) => {
                    info!("opened {path}: {} objects", list.len());
                    recent_files::add_project(project.current_path());
                    self.project = project;
                    self.app_state.object_list = list;
                    self.resync_placements = true;
                    if let Some(conn) = self.active_connection.take() {
                        self.pending_connections.push(conn);
                    }
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.window_title()));
                }
                Err(e) => error!("could not open project {path}: {e}"),
            }
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::F11)) {
            let is_fs = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!is_fs));
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::F)) {
            let indexes = self
                .app_state
                .object_list
                .iter_selected_indexes()
                .collect_vec();
            self.mod_state(MoveForwardModifier(indexes));
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::B)) {
            let indexes = self
                .app_state
                .object_list
                .iter_selected_indexes()
                .collect_vec();
            self.mod_state(MoveBackwardModifier(indexes));
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::Z)) {
            self.undo();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::Y)) {
            self.redo();
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, egui::Key::Delete)) {
            let idxs = self
                .app_state
                .object_list
                .iter_selected_indexes()
                .collect_vec();
            self.mod_state(DeleteObjectsModifier::new(idxs));
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, egui::Key::T)) {
            theme::toggle_theme(ctx);
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, egui::Key::Home)) {
            self.pending_goto = self.scan_frame_goto();
        }
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            MenuBar::new().ui(ui, |ui| {
                file_menu_button(ui, self);
                view_menu_button(ui, self);
                ui.separator();
                if ui
                    .add_enabled(self.undo_queue.can_undo(), Button::new("\u{21b6} Undo"))
                    .on_hover_text("Undo the last move, delete or reorder (Ctrl+Z)")
                    .clicked()
                {
                    self.undo();
                }
                if ui
                    .add_enabled(self.undo_queue.can_redo(), Button::new("\u{21b7} Redo"))
                    .on_hover_text("Redo (Ctrl+Y)")
                    .clicked()
                {
                    self.redo();
                }
                ui.separator();
                let selected: Vec<Uuid> = self
                    .app_state
                    .object_list
                    .iter_selected()
                    .filter(|o| matches!(&***o, Object::ScanImage { .. }))
                    .map(|o| o.uuid())
                    .collect();
                if ui
                    .add_enabled(
                        selected.len() >= 2 && !self.aligner.busy(),
                        Button::new(if self.aligner.busy() { "Aligning\u{2026}" } else { "Align stamps" }),
                    )
                    .on_hover_text(
                        "Line up the selected stamps by matching the image where they overlap \
                         (e.g. survey tiles). Only their positions change; Undo puts them back.",
                    )
                    .on_disabled_hover_text("Select two or more overlapping stamps")
                    .clicked()
                {
                    self.aligner.start(&self.app_state.object_list, &selected);
                }
                if ui
                    .add_enabled(!selected.is_empty(), Button::new("Save as image\u{2026}"))
                    .on_hover_text(
                        "Save the selected stamps as one PNG, laid out as on the map \
                         (e.g. a survey grid). Tip: click a group in Layers to select it all.",
                    )
                    .on_disabled_hover_text("Select the stamps to save")
                    .clicked()
                {
                    let idxs: Vec<usize> = self.app_state.object_list.iter_selected_indexes().rev().collect();
                    self.export_image(&idxs, "stamps");
                }
                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(self.scan_frame_goto().is_some(), Button::new("\u{26F6} Fit scan"))
                        .on_hover_text("Zoom the map to the current scan frame (Home)")
                        .clicked()
                    {
                        self.pending_goto = self.scan_frame_goto();
                    }
                    let dark = ui.visuals().dark_mode;
                    let icon = if dark { "\u{2600}" } else { "\u{1F319}" };
                    if ui
                        .button(icon)
                        .on_hover_text("Switch light / dark (T)")
                        .clicked()
                    {
                        theme::toggle_theme(ui.ctx());
                    }
                    ui.label(
                        RichText::new(format!(
                            "v{} \u{b7} updated {}",
                            crate::version::version(),
                            crate::version::UPDATED
                        ))
                        .small()
                        .weak(),
                    );
                });
            });
        });
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            self.show_status_bar(ui);
        });
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let mut new_world_transform = self.pending_goto.take();

                let mut cursor_world = None;
                let scan_view_resp = self.app_state.scan_view.show(ui, |ui| {
                    if let Some(ctx) = ui.data(|m| m.get_temp::<ScanViewCtx>(egui::Id::new(()))) {
                        cursor_world = ctx
                            .screen_response
                            .hover_pos()
                            .map(|p| ctx.world2egui().inverse().transform_point2(p.to_glam()));
                        if theme::prefs().show_grid {
                            draw_grid(ui, &ctx);
                        }
                    }
                    let objects = &mut self.app_state.object_list;
                    transform_objects(ui, objects);
                    for i in 0..objects.len() {
                        if *objects[i].hidden_mut() {
                            continue;
                        }
                        objects[i].show(ui);
                        let maybe_resp = objects[i].resp_group.response(ctx);
                        if let Some(tran) = objects[i].border_transform() {
                            if maybe_resp.as_ref().is_some_and(|resp| resp.hovered()) {
                                BorderRectangle {
                                    transform: tran,
                                    color: Color32::LIGHT_BLUE,
                                    dashed: false,
                                }
                                .show(ui);
                            }
                            if objects[i].selected {
                                BorderRectangle {
                                    transform: tran,
                                    color: Color32::GREEN,
                                    dashed: false,
                                }
                                .show(ui);
                            }
                        }
                        if maybe_resp.is_some_and(|resp| resp.double_clicked()) {
                            new_world_transform = Some(objects[i].goto_transform());
                        }
                    }
                    // Stamps taken before the GDS reference existed: remember
                    // the GDS as it is now, so later moves can be detected.
                    let gds = crate::analysis::gds_prior::collect(objects);
                    if !gds.is_empty() {
                        for obj in objects.iter_mut() {
                            if let Object::ScanImage { image, .. } = &mut **obj {
                                if image.gds_ref.is_none() {
                                    image.gds_ref = crate::analysis::gds_prior::predict_any(&gds, image.transform)
                                        .map(|(gi, _)| (gds[gi].uuid, gds[gi].transform));
                                }
                            }
                        }
                    }
                    if theme::prefs().show_gds_prior && !gds.is_empty() {
                        for obj in objects.iter() {
                            if let Object::ScanImage { image, hidden: false, .. } = &**obj {
                                if image.unit == "m" {
                                    if let Some(p) = crate::analysis::gds_prior::stamp_prior(&gds, image) {
                                        crate::analysis::gds_prior::draw_stamp(ui, image.transform, &p);
                                    }
                                }
                            }
                        }
                    }
                    // Ring Assist results saved with stamps (works offline too).
                    let show_rings = objects
                        .iter()
                        .find_map(|o| o.as_scan_area())
                        .is_none_or(|area| area.show_stamp_rings);
                    if show_rings {
                        for obj in objects.iter() {
                            if let Object::ScanImage { image, hidden: false, .. } = &**obj {
                                if image.ring.is_some() || image.ring_label.is_some() {
                                    crate::analysis::ring_stamp::draw(
                                        ui,
                                        image.transform,
                                        image.ring.as_ref(),
                                        image.ring_label.as_ref(),
                                    );
                                }
                            }
                        }
                    }
                    if let Some(object) = objects.get_hovered(ui.ctx()) {
                        if let Some(tran) = object.border_transform() {
                            BorderRectangle {
                                transform: tran,
                                color: Color32::LIGHT_BLUE,
                                dashed: true,
                            }
                            .show(ui);
                        }
                    }
                    for image in objects.iter_selected() {
                        if let Some(tran) = image.border_transform() {
                            BorderRectangle {
                                transform: tran,
                                color: Color32::GREEN,
                                dashed: true,
                            }
                            .show(ui);
                        }
                    }
                    if let Some(conn) = &mut self.active_connection {
                        conn.show_image_view_overlay(ui, objects);
                    }
                    ScaleBar.show(ui);
                });
                self.cursor_world = cursor_world;
                if let Some(tf) = new_world_transform {
                    // Fill most of the map area, whatever the window size.
                    let r = scan_view_resp.rect;
                    let s = (0.9 * r.width().min(r.height()) as f64).max(100.);
                    let tf = DAffine2::from_scale(DVec2 { x: s, y: -s }) * tf.inverse();
                    self.app_state.scan_view.world_transform = tf;
                }
                if scan_view_resp.clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.app_state.object_list.clear_selected();
                };
                egui::Window::new("Layers")
                    .frame(theme::overlay_frame(ctx))
                    .constrain_to(ui.min_rect())
                    // Starts at the map's top-left; drag the title to move it.
                    .default_pos(ui.min_rect().left_top() + egui::Vec2::new(5., 5.))
                    .movable(true)
                    .default_size([200., 400.])
                    .resizable(true)
                    .scroll([false, true])
                    .show(ctx, |ui| {
                        let list = &mut self.app_state.object_list;
                        let is_stamp = |o: &Object| matches!(o, Object::ScanImage { .. });
                        let (any_hidden, any_shown) = list.hidden_state_where(is_stamp);
                        let n_selected = list.iter_selected_indexes().count();
                        let hint = "Ctrl+click to pick layers, Shift+click for a range. \
                                    The eye of a selected layer hides or shows all selected.";
                        egui::Grid::new("layers_visibility").num_columns(3).spacing([6., 4.]).show(ui, |ui| {
                            ui.weak("All stamps");
                            if ui.add_enabled(any_hidden, Button::new("Show").small()).clicked() {
                                list.set_hidden_where(is_stamp, false);
                            }
                            if ui.add_enabled(any_shown, Button::new("Hide").small()).clicked() {
                                list.set_hidden_where(is_stamp, true);
                            }
                            ui.end_row();
                            ui.weak(if n_selected > 0 { format!("{n_selected} selected") } else { "Selected".into() })
                                .on_hover_text(hint);
                            if ui
                                .add_enabled(n_selected > 0, Button::new("Show").small())
                                .on_disabled_hover_text(hint)
                                .clicked()
                            {
                                list.set_selected_hidden(false);
                            }
                            if ui
                                .add_enabled(n_selected > 0, Button::new("Hide").small())
                                .on_disabled_hover_text(hint)
                                .clicked()
                            {
                                list.set_selected_hidden(true);
                            }
                            ui.end_row();
                        });
                        ui.separator();
                        let export = self.app_state.object_list.show_grouped(ui, &Object::group_mut);
                        if let Some(members) = export {
                            let name = members
                                .first()
                                .and_then(|&i| self.app_state.object_list.get_mut(i))
                                .and_then(|o| o.group_mut().and_then(|g| g.clone()))
                                .unwrap_or_else(|| "stamps".into());
                            self.export_image(&members, &name);
                        }
                    })
                    .map(|resp| {
                        resp.response.context_menu(|ui| {
                            if ui.button("Show all").clicked() {
                                for obj in &mut self.app_state.object_list.iter_mut() {
                                    *obj.hidden_mut() = false;
                                }
                            }
                        })
                    });
                self.show_gds_moved_warning(ctx, ui.min_rect());
                let mut new_top = ui.clip_rect().top();
                if let Some(conn) = &mut self.active_connection {
                    if let Some(resp) = egui::Window::new("Scan Control")
                        .vscroll(true)
                        .frame(theme::overlay_frame(ctx))
                        .constrain_to(ui.min_rect())
                        .anchor(Align2::RIGHT_TOP, egui::Vec2::new(5., 5.))
                        .resizable(false)
                        .show(ctx, |ui| {
                            ui.set_width(theme::SIDE_PANEL_WIDTH);
                            conn.show_menu(
                                ui,
                                &mut self.app_state.object_list,
                                &self.image_encoder,
                            );
                        })
                    {
                        new_top = resp.response.rect.bottom();
                    }
                }
                let selected = self
                    .app_state
                    .object_list
                    .iter_selected_indexes()
                    .collect_vec();
                // Several stamps selected (e.g. a grid or a sequence): one
                // window whose processing changes apply to all of them.
                let stamps = selected
                    .iter()
                    .copied()
                    .filter(|&i| matches!(&*self.app_state.object_list[i], Object::ScanImage { .. }))
                    .collect_vec();
                let multi = stamps.len() >= 2;
                if multi {
                    let mut rect = ui.min_rect();
                    rect.set_top(new_top);
                    if let Some(resp) = egui::Window::new(format!("{} stamps", stamps.len()))
                        .vscroll(true)
                        .id(egui::Id::new("multi_stamp_menu"))
                        .frame(theme::overlay_frame(ctx))
                        .constrain_to(rect)
                        .anchor(Align2::RIGHT_TOP, egui::Vec2::new(5., 5.))
                        .resizable(false)
                        .show(ctx, |ui| {
                            ui.set_width(theme::SIDE_PANEL_WIDTH);
                            ui.weak("Changes apply to every selected stamp.");
                            let mut images = self
                                .app_state
                                .object_list
                                .iter_mut()
                                .enumerate()
                                .filter(|(i, _)| stamps.contains(i))
                                .filter_map(|(_, o)| match &mut **o {
                                    Object::ScanImage { image, .. } => Some(image),
                                    _ => None,
                                })
                                .collect_vec();
                            crate::connection::display_controls(
                                ui,
                                &mut images,
                                &self.image_encoder,
                            );
                        })
                    {
                        new_top = resp.response.rect.bottom();
                    }
                }
                for i in selected {
                    if multi && stamps.contains(&i) {
                        continue;
                    }
                    if self.app_state.object_list[i].as_scan_area().is_some()
                        && self.active_connection.is_some()
                    {
                        continue;
                    }
                    let name = self.app_state.object_list[i].name();
                    let mut rect = ui.min_rect();
                    rect.set_top(new_top);
                    if let Some(resp) = egui::Window::new(name)
                        .vscroll(true)
                        .frame(theme::overlay_frame(ctx))
                        .constrain_to(rect)
                        .anchor(Align2::RIGHT_TOP, egui::Vec2::new(5., 5.))
                        .resizable(false)
                        .show(ctx, |ui| {
                            ui.set_width(theme::SIDE_PANEL_WIDTH);
                            self.app_state.object_list[i].show_menu(ui, &mut self.image_encoder);
                            if matches!(&*self.app_state.object_list[i], Object::Gds { .. }) {
                                gds_ring_check_ui(
                                    ui,
                                    &mut self.app_state.object_list,
                                    i,
                                    &mut self.gds_shift_undo,
                                );
                            }
                        })
                    {
                        new_top = resp.response.rect.bottom();
                    }
                }
            });
        self.track_moves(ctx);
    }
}

impl MyApp {
    /// "Goto" transform for the current scan frame (or the whole scan range
    /// when no frame is known), for zooming the map to it.
    fn scan_frame_goto(&self) -> Option<DAffine2> {
        self.app_state.object_list.iter().find_map(|o| {
            let area = o.as_scan_area()?;
            Some(match area.image_transform {
                Some(image) => {
                    area.world_transform * image * DAffine2::from_scale(DVec2::new(1., -1.))
                }
                None => o.goto_transform(),
            })
        })
    }

    /// Warn when the GDS has moved (or been replaced) since stamps were
    /// taken, with a way to put it back or accept the new position.
    fn show_gds_moved_warning(&mut self, ctx: &egui::Context, rect: egui::Rect) {
        let st = crate::analysis::gds_prior::ref_status(&self.app_state.object_list);
        if st.moved.is_empty() && st.replaced.is_empty() {
            return;
        }
        let objects = &mut self.app_state.object_list;
        egui::Window::new("GDS moved")
            .frame(theme::overlay_frame(ctx))
            .title_bar(false)
            .resizable(false)
            .constrain_to(rect)
            .anchor(Align2::CENTER_TOP, egui::Vec2::new(0., 5.))
            .show(ctx, |ui| {
                ui.set_max_width(360.);
                if let Some(c) = st.worst {
                    let mut what = format!("by up to {:.2} \u{b5}m at the stamps", c.shift_nm * 1e-3);
                    if c.rot_deg.abs() > 0.05 {
                        what += &format!(", rotated {:+.2}\u{b0}", c.rot_deg);
                    }
                    if (c.scale - 1.).abs() > 1e-3 {
                        what += &format!(", scaled \u{d7}{:.3}", c.scale);
                    }
                    ui.label(
                        RichText::new(format!(
                            "\u{26A0} The GDS has moved since {} stamp{} were taken ({what}).",
                            st.moved.len(),
                            if st.moved.len() == 1 { "" } else { "s" }
                        ))
                        .color(theme::WARN),
                    );
                    ui.label(
                        RichText::new("Ring predictions use the GDS where it is now.").weak(),
                    );
                    ui.horizontal(|ui| {
                        if let Some((id, old)) = st.restore {
                            if ui.button("Move GDS back").clicked() {
                                for o in objects.iter_mut() {
                                    if let Object::Gds { image, .. } = &mut **o {
                                        if image.uuid() == id {
                                            image.transform = old;
                                        }
                                    }
                                }
                            }
                        }
                        if ui
                            .button("Keep new position")
                            .on_hover_text("The stamps now refer to the GDS where it is")
                            .clicked()
                        {
                            set_gds_refs(objects, &st.moved);
                        }
                    });
                }
                if !st.replaced.is_empty() {
                    ui.label(
                        RichText::new(format!(
                            "\u{26A0} {} stamp{} were taken with a GDS that is no longer in the project.",
                            st.replaced.len(),
                            if st.replaced.len() == 1 { "" } else { "s" }
                        ))
                        .color(theme::WARN),
                    );
                    if ui.button("Use the current GDS").clicked() {
                        set_gds_refs(objects, &st.replaced);
                    }
                }
            });
    }
}

/// Point these stamps' GDS reference at the GDS as it is now.
fn set_gds_refs(objects: &mut SelectableList<Object>, stamps: &[usize]) {
    let gds = crate::analysis::gds_prior::collect(objects);
    for &i in stamps {
        if let Object::ScanImage { image, .. } = &mut *objects[i] {
            image.gds_ref = crate::analysis::gds_prior::predict_any(&gds, image.transform)
                .map(|(gi, _)| (gds[gi].uuid, gds[gi].transform));
        }
    }
}

/// Ring check for a GDS: how its predictions compare with the stamps' scan
/// bands, and a suggested shift if the stamps say it is off.
fn gds_ring_check_ui(
    ui: &mut Ui,
    objects: &mut SelectableList<Object>,
    i: usize,
    undo: &mut Option<(Uuid, DAffine2)>,
) {
    use crate::analysis::gds_prior::{self, GDS_COLOR};
    let Object::Gds { image, .. } = &*objects[i] else { return };
    let id = image.uuid();
    let systems = image.ring_systems();
    ui.separator();
    ui.label(RichText::new("Ring check").strong());
    if systems.is_empty() {
        ui.label(RichText::new("No ring pattern found in this GDS.").weak());
        return;
    }
    let gds = gds_prior::collect(objects);
    let check = gds_prior::alignment_check(objects, &gds, id);
    ui.label(format!(
        "{} ring system{} found; {} Z stamp{} on them.",
        systems.len(),
        if systems.len() == 1 { "" } else { "s" },
        check.stamps,
        if check.stamps == 1 { "" } else { "s" }
    ));
    if check.with_bands > 0 {
        ui.label(format!(
            "Scan bands agree with the GDS on {}/{} stamps (median {:.0}\u{b0}).",
            check.agree, check.with_bands, check.median_deg
        ));
    }
    let mut shift = None;
    match check.estimate {
        Some(e) if e.significant() => {
            ui.label(
                RichText::new(format!(
                    "The scans suggest moving the GDS by ({:+.2}, {:+.2}) \u{b5}m, \u{b1}{:.1} \u{b5}m (from {} stamps).",
                    e.offset.x * 1e-3,
                    e.offset.y * 1e-3,
                    e.sigma.max_element() * 1e-3,
                    e.used
                ))
                .color(GDS_COLOR),
            );
            if ui
                .button("Shift GDS")
                .on_hover_text("Move the GDS by this amount. Undo puts it back.")
                .clicked()
            {
                shift = Some(e.offset);
            }
        }
        Some(e) => {
            ui.label(
                RichText::new(format!(
                    "No clear offset (best guess ({:+.2}, {:+.2}) \u{b1}{:.1} \u{b5}m from {} stamps).",
                    e.offset.x * 1e-3,
                    e.offset.y * 1e-3,
                    e.sigma.max_element() * 1e-3,
                    e.used
                ))
                .weak(),
            );
        }
        None => {
            ui.label(
                RichText::new("Needs 3 or more stamps with clear bands to estimate an offset.")
                    .weak(),
            );
        }
    }
    let can_undo = undo.is_some_and(|(u, _)| u == id);
    let mut restore = None;
    if can_undo && ui.button("Undo shift").clicked() {
        restore = undo.take().map(|(_, t)| t);
    }
    let new = {
        let Object::Gds { image, .. } = &mut *objects[i] else { return };
        let before = image.transform;
        if let Some(off) = shift {
            *undo = Some((id, before));
            image.transform.translation += off;
        }
        if let Some(t) = restore {
            image.transform = t;
        }
        (image.transform != before).then_some(image.transform)
    };
    // A deliberate move: the stamps follow it, so no "GDS moved" warning.
    if let Some(t) = new {
        for o in objects.iter_mut() {
            if let Object::ScanImage { image, .. } = &mut **o {
                if image.gds_ref.is_some_and(|(u, _)| u == id) {
                    image.gds_ref = Some((id, t));
                }
            }
        }
    }
}

fn transform_objects(ui: &mut Ui, object_list: &mut SelectableList<Object>) {
    if ui.input(|i| i.modifiers.ctrl) {
        let [rotate, scale, translate] = world_delta_transform(ui);
        let tf = if object_list.iter_selected().all(|ent| ent.is_scalable()) {
            rotate * scale * translate
        } else {
            rotate * translate
        };
        for i in object_list.iter_selected_indexes().collect_vec() {
            object_list[i].apply_transform(tf);
        }
    }
}

fn view_menu_button(ui: &mut Ui, app: &mut MyApp) {
    ui.menu_button("View", |ui| {
        let current = theme::prefs().theme;
        for (choice, label) in [
            (ThemeChoice::Dark, "Dark theme"),
            (ThemeChoice::Light, "Light theme"),
            (ThemeChoice::System, "Follow system"),
        ] {
            if ui.radio(current == choice, label).clicked() {
                theme::set_theme(ui.ctx(), choice);
            }
        }
        ui.separator();
        let mut grid = theme::prefs().show_grid;
        if ui.checkbox(&mut grid, "Show grid").changed() {
            theme::update_prefs(|p| p.show_grid = grid);
        }
        let mut gds = theme::prefs().show_gds_prior;
        if ui
            .checkbox(&mut gds, "Show GDS prediction on stamps")
            .on_hover_text("Dashed violet arrow: direction to the ring centre from where the GDS sits on the map")
            .changed()
        {
            theme::update_prefs(|p| p.show_gds_prior = gds);
        }
        if let Some(area) = app
            .app_state
            .object_list
            .iter_mut()
            .find_map(|o| o.as_scan_area_mut())
        {
            ui.checkbox(&mut area.show_stamp_rings, "Show ring results on stamps");
            ui.checkbox(&mut area.show_history, "Show coarse move history");
            ui.add_enabled_ui(area.show_history, |ui| {
                ui.checkbox(&mut area.show_history_boxes, "    with frame boxes");
            });
        }
        ui.separator();
        if ui
            .add_enabled(app.scan_frame_goto().is_some(), Button::new("Fit scan frame  (Home)"))
            .clicked()
        {
            app.pending_goto = app.scan_frame_goto();
            ui.close();
        }
    });
}

/// Faint world grid with labels at a "nice" spacing for the current zoom.
fn draw_grid(ui: &Ui, ctx: &ScanViewCtx) {
    let w2s = ctx.world2egui();
    let px_per_nm = w2s.matrix2.x_axis.length();
    if !(px_per_nm > 0.) {
        return;
    }
    // Aim for lines about 120 px apart: 1, 2 or 5 x 10^n nm.
    let raw = 120. / px_per_nm;
    let pow = 10f64.powf(raw.log10().floor());
    let step = [1., 2., 5., 10.]
        .into_iter()
        .map(|m| m * pow)
        .find(|s| *s >= raw)
        .unwrap_or(10. * pow);
    let s2w = w2s.inverse();
    let r = ctx.rect;
    let corners = [r.left_top(), r.right_top(), r.left_bottom(), r.right_bottom()]
        .map(|p| s2w.transform_point2(p.to_glam()));
    let min = corners.iter().fold(DVec2::INFINITY, |a, &b| a.min(b));
    let max = corners.iter().fold(DVec2::NEG_INFINITY, |a, &b| a.max(b));
    let color = if ui.visuals().dark_mode {
        Color32::from_white_alpha(22)
    } else {
        Color32::from_black_alpha(28)
    };
    let label = ui.visuals().weak_text_color();
    let painter = ui.painter_at(r);
    let stroke = Stroke::new(1., color);
    let fmt = |v: f64| {
        let um = v * 1e-3;
        if step >= 1000. { format!("{um:.0} \u{b5}m") } else { format!("{um:.2} \u{b5}m") }
    };
    let (x0, x1) = ((min.x / step).floor() as i64, (max.x / step).ceil() as i64);
    let (y0, y1) = ((min.y / step).floor() as i64, (max.y / step).ceil() as i64);
    if (x1 - x0) > 400 || (y1 - y0) > 400 {
        return;
    }
    for k in x0..=x1 {
        let x = k as f64 * step;
        let a = w2s.transform_point2(DVec2::new(x, min.y)).to_egui_pos2();
        let b = w2s.transform_point2(DVec2::new(x, max.y)).to_egui_pos2();
        painter.line_segment([a, b], stroke);
        let top = if a.y < b.y { a } else { b };
        painter.text(
            egui::pos2(top.x + 3., r.top() + 2.),
            Align2::LEFT_TOP,
            fmt(x),
            egui::FontId::proportional(10.),
            label,
        );
    }
    for k in y0..=y1 {
        let y = k as f64 * step;
        let a = w2s.transform_point2(DVec2::new(min.x, y)).to_egui_pos2();
        let b = w2s.transform_point2(DVec2::new(max.x, y)).to_egui_pos2();
        painter.line_segment([a, b], stroke);
        let left = if a.x < b.x { a } else { b };
        painter.text(
            egui::pos2(r.left() + 3., left.y - 2.),
            Align2::LEFT_BOTTOM,
            fmt(y),
            egui::FontId::proportional(10.),
            label,
        );
    }
}

fn file_menu_button(ui: &mut Ui, app: &mut MyApp) {
    ui.menu_button("File", |ui| {
        if ui.button("Open").clicked() {
            app.project_open_dialog.pick_file();
        }
        let recent = recent_files::recent();
        ui.add_enabled_ui(!recent.projects.is_empty(), |ui| {
            ui.menu_button("Open Recent", |ui| {
                for (i, path) in recent.projects.iter().enumerate() {
                    let text = format!("{}  {}", i + 1, recent_files::label(path));
                    if ui
                        .button(text)
                        .on_hover_text(path.display().to_string())
                        .clicked()
                    {
                        app.project_open_dialog.open_path(path.clone());
                    }
                }
                ui.separator();
                if ui.button("Clear list").clicked() {
                    recent_files::clear_projects();
                }
            });
        });
        ui.separator();
        if ui.button("Save").clicked() {
            app.save_project();
        }
        if ui.button("Save As").clicked() {
            // The current state is written when the path comes back.
            app.project_save_dialog.select_path();
        }
        ui.separator();
        if ui.add(Button::new("Import Files")).clicked() {
            let tf = import_transform(app);
            app.import_file_dialog
                .pick_files(app.image_encoder.clone(), tf);
        }
        ui.add_enabled_ui(!recent.imports.is_empty(), |ui| {
            ui.menu_button("Import Recent", |ui| {
                for path in &recent.imports {
                    if ui
                        .button(recent_files::label(path))
                        .on_hover_text(path.display().to_string())
                        .clicked()
                    {
                        let tf = import_transform(app);
                        app.import_file_dialog.import_paths(
                            vec![path.clone()],
                            app.image_encoder.clone(),
                            tf,
                        );
                    }
                }
                ui.separator();
                if ui.button("Clear list").clicked() {
                    recent_files::clear_imports();
                }
            });
        });
    });
}

/// Where imported files land: the centre of the current view.
fn import_transform(app: &MyApp) -> DAffine2 {
    let (scale, _, translation) = app
        .app_state
        .scan_view
        .world_transform
        .to_scale_angle_translation();
    DAffine2::from_scale_angle_translation(
        1e3 / (scale.abs()),
        0.,
        -translation / scale.abs() * DVec2::new(1., -1.),
    )
}

struct DeleteObjectsModifier {
    imgs: Vec<SelectableEntry<Object>>,
    idxs: Vec<usize>,
}
impl DeleteObjectsModifier {
    pub fn new(idxs: Vec<usize>) -> Self {
        Self {
            imgs: Vec::with_capacity(idxs.len()),
            idxs,
        }
    }
}
impl StateModify<AppState> for DeleteObjectsModifier {
    fn redo(&mut self, state: &mut AppState) -> bool {
        for idx in self.idxs.iter().rev() {
            self.imgs.push(state.object_list.remove(*idx));
        }
        true
    }
    fn undo(&mut self, state: &mut AppState) {
        for (idx, img) in izip!(&self.idxs, self.imgs.drain(..).rev()) {
            state.object_list.insert(*idx, img);
        }
    }
}

/// Objects moved on the map: (object, from, to).
struct MoveObjectsModifier(Vec<(Uuid, Placement, Placement)>);
impl MoveObjectsModifier {
    fn place(state: &mut AppState, id: Uuid, p: Placement) {
        if let Some(o) = state.object_list.iter_mut().find(|o| o.uuid() == id) {
            o.set_placement(p);
        }
    }
}
impl StateModify<AppState> for MoveObjectsModifier {
    fn redo(&mut self, state: &mut AppState) -> bool {
        for &(id, _, to) in &self.0 {
            Self::place(state, id, to);
        }
        true
    }
    fn undo(&mut self, state: &mut AppState) {
        for &(id, from, _) in &self.0 {
            Self::place(state, id, from);
        }
    }
}

struct MoveForwardModifier(Vec<usize>);
impl StateModify<AppState> for MoveForwardModifier {
    fn redo(&mut self, state: &mut AppState) -> bool {
        self.0 = state.object_list.move_indexes_down(&self.0);
        !self.0.is_empty()
    }

    fn undo(&mut self, state: &mut AppState) {
        self.0 = state.object_list.move_indexes_up(&self.0)
    }
}
struct MoveBackwardModifier(Vec<usize>);
impl StateModify<AppState> for MoveBackwardModifier {
    fn redo(&mut self, state: &mut AppState) -> bool {
        self.0 = state.object_list.move_indexes_up(&self.0);
        !self.0.is_empty()
    }

    fn undo(&mut self, state: &mut AppState) {
        self.0 = state.object_list.move_indexes_down(&self.0)
    }
}

const OBJECT_LIST_TABLE: TableDefinition<Uuid, u64> = TableDefinition::new("object_list_table_v1");
impl Persistant for SelectableList<Object> {
    fn db_update<'t>(&self, txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>> {
        let desired: HashMap<Uuid, u64> = self
            .iter()
            .enumerate()
            .map(|(i, item)| (item.uuid(), i as u64))
            .collect();
        let mut table = txn.open_table(OBJECT_LIST_TABLE)?;
        for entry in table.extract_if(|id, _| !desired.contains_key(&id))? {
            let (uuid, _) = entry?;
            let uuid = uuid.value();
            Object::db_remove(uuid, txn)?;
        }
        for (id, index) in desired {
            if let Some(mut index_mut) = table.get_mut(id)? {
                info!("{id} exists");
                if index_mut.value() != index {
                    index_mut.insert(index)?;
                }
                self[index as usize].db_update(txn)?;
                continue;
            }
            info!("{id} doesnt exist, inserting");
            table.insert(id, index)?;
            self[index as usize].db_insert(txn)?;
        }
        Ok(())
    }

    fn db_remove<'t>(_id: Uuid, _txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn db_insert<'t>(&self, _txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn db_read<'t>(
        _id: Uuid,
        txn: &'t WriteTransaction,
        encoder: &ImageEncoder,
    ) -> Result<Self, Box<dyn Error>> {
        let table = txn.open_table(OBJECT_LIST_TABLE)?;
        let mut ids = BTreeMap::new();
        for entry in table.iter()? {
            let (id, ind) = entry?;
            ids.insert(ind.value(), id.value());
        }
        let mut object_list = SelectableList::new();
        for id in ids.values().copied() {
            let object = Object::db_read(id, txn, encoder)?;
            let entry = object.into_entry(uuid::Uuid::new_v4());
            object_list.push(entry);
        }
        Ok(object_list)
    }

    fn db_dump_stats<'t>(txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>> {
        let table = txn.open_table(OBJECT_LIST_TABLE)?;
        let len = table.len()?;
        println!("---- Object List: {len} items ----");
        Object::db_dump_stats(txn)?;
        Ok(())
    }
}
