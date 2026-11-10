use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    sync::LazyLock,
};

use crate::{
    components::{
        file_dialog_native::{ObjectImportDialog, ProjectOpenDialog, ProjectSaveDialog},
        selectable_list::{SelectableEntry, SelectableList},
    },
    connection::{nanonis::NanonisConnection, Connection},
    project::{Persistant, ProjectDb},
    scan_view::{world_delta_transform, BorderRectangle, ImageEncoder, ScaleBar, ScanView},
    undo_queue::{StateModify, UndoQueue},
    view_object::Object,
};
use egui::{
    widgets, Align2, Button, Color32, Frame, Layout, MenuBar, Modifiers, Shadow, ThemePreference,
    Ui,
};
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

pub struct MyApp {
    app_state: AppState,
    image_encoder: ImageEncoder,
    undo_queue: UndoQueue<AppState>,
    current_theme: ThemePreference,
    import_file_dialog: ObjectImportDialog,
    project_save_dialog: ProjectSaveDialog,
    project_open_dialog: ProjectOpenDialog,
    pending_connections: Vec<Box<dyn Connection>>,
    active_connection: Option<Box<dyn Connection>>,
    project: ProjectDb,
}

pub struct AppState {
    scan_view: ScanView,
    object_list: SelectableList<Object>,
}

impl MyApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let project = ProjectDb::new_temp().unwrap();
        let wgpu = cc.wgpu_render_state.as_ref().unwrap();
        let image_encoder = ImageEncoder::new(wgpu);
        let nanonis_connection = Box::new(NanonisConnection::new(cc.egui_ctx.clone(), "localhost"));
        let object_list = SelectableList::new();
        let import_file_dialog = ObjectImportDialog::new();
        let project_save_dialog = ProjectSaveDialog::new();
        let project_open_dialog = ProjectOpenDialog::new();

        Self {
            app_state: AppState {
                scan_view: ScanView::new(&image_encoder),
                object_list,
            },
            import_file_dialog,
            image_encoder,
            undo_queue: UndoQueue::new(),
            current_theme: cc.egui_ctx.theme().into(),
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
        self.verify_saved();
        Ok(())
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
            let entry = SelectableEntry::new(
                uuid::Uuid::new_v4(),
                object,
                |img| img.list_atoms(),
                |obj| obj.hidden_mut(),
            );
            self.app_state.object_list.push(entry);
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::CTRL, egui::Key::S)) {
            self.save_project();
        }
        if let Some(path) = self.project_save_dialog.try_recv_path() {
            match self.save_project_as(path) {
                Ok(()) => {
                    let file_name = self.project.current_path().file_name().unwrap().display();
                    let new_title = format!("Scan Control - {file_name}");
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title(new_title));
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
                    self.project = project;
                    self.app_state.object_list = list;
                    if let Some(conn) = self.active_connection.take() {
                        self.pending_connections.push(conn);
                    }
                    let file_name = self.project.current_path().file_name().unwrap().display();
                    let new_title = format!("Scan Control - {file_name}");
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title(new_title));
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
            self.undo_queue.undo(&mut self.app_state);
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::Y)) {
            self.undo_queue.redo(&mut self.app_state);
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
            let new_theme = match self.current_theme {
                ThemePreference::Dark => ThemePreference::Light,
                ThemePreference::Light => ThemePreference::Dark,
                ThemePreference::System => unreachable!(),
            };
            ctx.set_theme(new_theme);
            self.current_theme = new_theme;
        }
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            MenuBar::new().ui(ui, |ui| {
                file_menu_button(ui, self);
                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                    widgets::global_theme_preference_switch(ui);
                });
            });
        });
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            let tr = self.app_state.scan_view.world_transform.inverse();
            let (scale, _, translation) = tr.to_scale_angle_translation();
            ui.label(format!("scale: {scale}, translation: {translation}"));
        });
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let mut new_world_transform = None;

                let scan_view_resp = self.app_state.scan_view.show(ui, |ui| {
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
                if let Some(tf) = new_world_transform {
                    let tf = DAffine2::from_scale(DVec2 {
                        x: 0.5e3,
                        y: -0.5e3,
                    }) * tf.inverse();
                    self.app_state.scan_view.world_transform = tf;
                }
                if scan_view_resp.clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.app_state.object_list.clear_selected();
                };
                egui::Window::new("Layers")
                    .frame(
                        Frame::window(&ctx.style())
                            .multiply_with_opacity(0.5)
                            .shadow(Shadow::NONE),
                    )
                    .constrain_to(ui.min_rect())
                    .anchor(Align2::LEFT_TOP, egui::Vec2::new(5., 5.))
                    .default_size([200., 400.])
                    .resizable(true)
                    .scroll([false, true])
                    .show(ctx, |ui| {
                        let vis = &mut ui.style_mut().visuals.widgets.inactive;
                        vis.weak_bg_fill = vis.weak_bg_fill.gamma_multiply(0.5);
                        self.app_state.object_list.show(ui);
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
                let mut new_top = ui.clip_rect().top();
                if let Some(conn) = &mut self.active_connection {
                    new_top = egui::Window::new("Scan Control")
                        .frame(
                            Frame::window(&ctx.style())
                                .multiply_with_opacity(0.5)
                                .shadow(Shadow::NONE),
                        )
                        .constrain_to(ui.min_rect())
                        .anchor(Align2::RIGHT_TOP, egui::Vec2::new(5., 5.))
                        .resizable(false)
                        .show(ctx, |ui| {
                            let vis = &mut ui.style_mut().visuals.widgets.inactive;
                            vis.weak_bg_fill = vis.weak_bg_fill.gamma_multiply(0.5);
                            conn.show_menu(
                                ui,
                                &mut self.app_state.object_list,
                                &self.image_encoder,
                            );
                        })
                        .unwrap()
                        .response
                        .rect
                        .bottom();
                }
                let selected = self
                    .app_state
                    .object_list
                    .iter_selected_indexes()
                    .collect_vec();
                for i in selected {
                    if self.app_state.object_list[i].as_scan_area().is_some()
                        && self.active_connection.is_some()
                    {
                        continue;
                    }
                    let name = self.app_state.object_list[i].name();
                    let mut rect = ui.min_rect();
                    rect.set_top(new_top);
                    new_top = egui::Window::new(name)
                        .frame(Frame::window(&ctx.style()).multiply_with_opacity(0.5))
                        .constrain_to(rect)
                        .anchor(Align2::RIGHT_TOP, egui::Vec2::new(5., 5.))
                        .resizable(false)
                        .show(ctx, |ui| {
                            self.app_state.object_list[i].show_menu(ui, &mut self.image_encoder);
                        })
                        .unwrap()
                        .response
                        .rect
                        .bottom();
                }
            });
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

fn file_menu_button(ui: &mut Ui, app: &mut MyApp) {
    ui.menu_button("File", |ui| {
        if ui.button("Open").clicked() {
            app.project_open_dialog.pick_file();
        }
        if ui.button("Save").clicked() {
            app.save_project();
        }
        if ui.button("Save As").clicked() {
            // The current state is written when the path comes back.
            app.project_save_dialog.select_path();
        }
        if ui.add(Button::new("Import Files")).clicked() {
            let (scale, _, translation) = app
                .app_state
                .scan_view
                .world_transform
                .to_scale_angle_translation();
            let tf = DAffine2::from_scale_angle_translation(
                1e3 / (scale.abs()),
                0.,
                -translation / scale.abs() * DVec2::new(1., -1.),
            );
            app.import_file_dialog
                .pick_files(app.image_encoder.clone(), tf);
        }
    });
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

pub trait OkTraceExt<T> {
    fn ok_trace(self) -> Option<T>;
}
impl<T, E: std::fmt::Display> OkTraceExt<T> for std::result::Result<T, E> {
    fn ok_trace(self) -> Option<T> {
        self.inspect_err(|e| error!("{e:#}")).ok()
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

    fn db_remove<'t>(id: Uuid, txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn db_insert<'t>(&self, txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn db_read<'t>(
        id: Uuid,
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
            let entry = SelectableEntry::new(
                uuid::Uuid::new_v4(),
                object,
                |img| img.list_atoms(),
                |obj| obj.hidden_mut(),
            );
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
