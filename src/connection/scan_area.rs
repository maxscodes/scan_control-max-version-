use std::{collections::VecDeque, path::PathBuf, sync::Arc};

use crossbeam::channel::{Receiver, Sender};

use egui::{epaint::PathStroke, Color32, Id, Shape, Stroke, Ui};
use glam::{DAffine2, DMat2, DVec2};
use redb::{ReadableTable as _, ReadableTableMetadata, TableDefinition};
use uuid::Uuid;

use crate::{
    analysis::sxm_match::{load_frame, SxmFrame},
    connection::{nanonis::ScanStatus, FrameData, LiveImage},
    project::Persistant,
    scan_view::{border::dashes_from_line, BorderRectangle, ImageEncoder, ScanViewCtx},
    utils::vec_interop::{IntoEgui, Projection},
};

pub struct ScanArea {
    pub world_transform: DAffine2,
    pub area_size: DVec2,
    pub image_transform: Option<DAffine2>,
    pub live_image: LiveImage,
    pub channel_opts: Vec<String>,
    pub channel_selected: Option<String>,
    pub scan_status: ScanStatus,
    pub tip_pos: Option<DVec2>,
    pub stamp: VecDeque<LiveImage>,
    /// Stamps made from .sxm files after the fact: (image, file name, channel).
    pub stamp_import: VecDeque<(LiveImage, String, String)>,
    /// Nanonis save folder, where the .sxm picker starts.
    pub session_dir: String,
    sxm_tx: Sender<SxmFrame>,
    sxm_rx: Receiver<SxmFrame>,
    pub stamp_name_base: String,
    pub course_move_history: Vec<DVec2>,
    /// Coarse moves and scan-region drags, for learning the calibration.
    pub coarse_log: crate::analysis::coarse_fit::CoarseLog,
    /// The user ctrl+dragged the scan region since this was last cleared.
    pub manual_drag: bool,
    pub show_history: bool,
    pub show_history_boxes: bool,
    /// Draw the Ring Assist result saved with each stamped image.
    pub show_stamp_rings: bool,
}
impl ScanArea {
    pub fn new(uuid: Uuid, encoder: &ImageEncoder, area_size: DVec2) -> Self {
        let live_image = LiveImage::new(uuid, encoder, DAffine2::IDENTITY);
        let (sxm_tx, sxm_rx) = crossbeam::channel::unbounded();
        Self {
            stamp_import: VecDeque::new(),
            session_dir: String::new(),
            sxm_tx,
            sxm_rx,
            world_transform: DAffine2::IDENTITY,
            area_size,
            image_transform: None,
            live_image,
            channel_opts: vec![],
            channel_selected: None,
            scan_status: ScanStatus::default(),
            tip_pos: None,
            stamp: VecDeque::new(),
            stamp_name_base: String::new(),
            course_move_history: Vec::new(),
            coarse_log: Default::default(),
            manual_drag: false,
            show_history: false,
            show_history_boxes: false,
            show_stamp_rings: true,
        }
    }
    pub fn uuid(&self) -> Uuid {
        self.live_image.uuid()
    }
    pub fn show(&mut self, ui: &mut Ui) {
        self.show_image(ui);
        // doesn't work on v5
        // self.show_scan_line(ui);
        self.show_image_border(ui);
        self.show_area_border(ui);
        self.show_tip(ui);
        if self.show_history {
            self.show_history(ui);
        }
    }
    fn show_image_border(&self, ui: &mut Ui) {
        let Some(image_tran) = self.image_transform else {
            return;
        };
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let transform = self.world_transform * image_tran;
        BorderRectangle {
            transform,
            color: Color32::RED,
            dashed: false,
        }
        .show(ui);
        let bottom_left = ((ctx.world2egui() * transform).transform_point2(DVec2::new(-0.5, 0.5))
            + DVec2::new(-1., 1.))
        .to_egui_pos2();
        ui.painter().circle_filled(bottom_left, 3., Color32::RED);
    }
    fn show_image(&mut self, ui: &mut Ui) {
        let Some(image_tran) = self.image_transform else {
            return;
        };
        self.live_image.transform = self.world_transform * image_tran;
        self.live_image.show_image(ui);
    }
    fn show_area_border(&self, ui: &mut Ui) {
        BorderRectangle {
            transform: self.world_transform * DAffine2::from_scale(self.area_size),
            color: Color32::from_rgb(0xe3, 0xb6, 0x2d),
            dashed: false,
        }
        .show(ui);
    }
    #[allow(dead_code)] // end-of-line wait: not available on Nanonis v5
    fn show_scan_line(&self, ui: &mut Ui) {
        let Some(image_tran) = self.image_transform else {
            return;
        };
        let Some(y) = self
            .scan_status
            .scan_line_position(self.live_image.size(), self.live_image.line_dir)
        else {
            return;
        };
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let tf = ctx.world2egui() * self.world_transform * image_tran;
        let p0 = tf.project_pos2(DVec2::new(-0.5, y)).to_egui_pos2();
        let p1 = tf.project_pos2(DVec2::new(0.5, y)).to_egui_pos2();

        ui.painter().extend(dashes_from_line(
            ctx.rect,
            &[p0, p1],
            PathStroke {
                width: 1.0,
                color: egui::epaint::ColorMode::Solid(Color32::BLUE),
                kind: egui::StrokeKind::Middle,
            },
            3. * 5.,
            1. * 5.,
        ));
    }
    fn show_tip(&self, ui: &mut Ui) {
        let Some(tip_pos) = self.tip_pos else {
            return;
        };
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let tf = ctx.world2egui() * self.world_transform;
        let center = tf.project_pos2(tip_pos).to_egui_pos2();
        ui.painter().circle_filled(center, 3., Color32::BLUE);
    }
    /// Channel, display settings and stamping. Map display options (move
    /// history, ring results) are in the View menu.
    pub fn show_menu(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        self.show_channel_control(ui);
        self.live_image.show_menu(ui, encoder);
        ui.add_space(4.);
        self.show_stamp_button(ui, encoder);
    }
    fn show_stamp_button(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        ui.columns(2, |c| {
            let w = c[0].available_width();
            if c[0]
                .add_sized([w, 24.], egui::Button::new("\u{1F4F7} Stamp"))
                .on_hover_text("Paste the current image onto the map")
                .clicked()
            {
                self.stamp.push_front(self.live_image.stamp(encoder));
            }
            let w = c[1].available_width();
            if c[1]
                .add_sized([w, 24.], egui::Button::new("From .sxm\u{2026}"))
                .on_hover_text(
                    "Stamp saved .sxm scans you forgot to stamp live. They are placed \
                     using the current map position, so pick scans taken since the \
                     last coarse move.",
                )
                .clicked()
            {
                self.pick_sxm_files();
            }
        });
        if !self.stamp_name_base.is_empty() {
            ui.label(egui::RichText::new(format!("Name: {}", self.stamp_name_base)).small().weak())
                .on_hover_text("Stamps are named after the Nanonis file base name");
        }
    }
    /// Ask for .sxm files and load them in the background; see
    /// [`Self::poll_sxm_imports`].
    fn pick_sxm_files(&self) {
        let mut dialog = crate::components::file_dialog_native::dialog()
            .add_filter("Nanonis scan", &["sxm"])
            .set_title("Stamp from .sxm files");
        let dir = PathBuf::from(self.session_dir.trim());
        if !self.session_dir.trim().is_empty() && dir.is_dir() {
            dialog = dialog.set_directory(dir);
        }
        let tx = self.sxm_tx.clone();
        let channel = self.channel_selected.clone();
        std::thread::spawn(move || {
            for path in dialog.pick_files().unwrap_or_default() {
                match load_frame(&path, channel.as_deref()) {
                    Ok(frame) => {
                        let _ = tx.send(frame);
                    }
                    Err(e) => tracing::error!("cannot stamp {}: {e}", path.display()),
                }
            }
            crate::app::request_repaint();
        });
    }
    /// Turn loaded .sxm files into stamps, queued in `stamp_import`.
    pub fn poll_sxm_imports(&mut self, encoder: &ImageEncoder) {
        while let Ok(f) = self.sxm_rx.try_recv() {
            let data = |d: Vec<f32>| FrameData { size: f.size, data: Arc::new(d.into_boxed_slice()) };
            let image = self.live_image.stamp_from_data(
                encoder,
                self.world_transform * f.frame,
                data(f.forward),
                data(f.backward),
                f.unit,
            );
            tracing::info!("stamped {} ({}) from file", f.name, f.channel);
            self.stamp_import.push_back((image, f.name, f.channel));
        }
    }
    fn show_channel_control(&mut self, ui: &mut Ui) {
        let mut selection = self.channel_selected.as_deref();
        let changed = ui
            .horizontal(|ui| {
                ui.label("Channel");
                egui::ComboBox::new((self.live_image.uuid(), "combo_box"), "")
                    .width(ui.available_width() - 8.)
                    .selected_text(selection.unwrap_or("None"))
                    .show_ui(ui, |ui| {
                        // `any` would stop drawing at the first click.
                        let mut clicked = false;
                        for opt in &self.channel_opts {
                            clicked |= ui.selectable_value(&mut selection, Some(opt), opt).clicked();
                        }
                        clicked
                    })
                    .inner
                    .unwrap_or(false)
            })
            .inner;
        if changed {
            self.channel_selected = selection.map(|s| s.to_string());
        }
    }
    fn show_history(&self, ui: &mut Ui) {
        if self.show_history_boxes{
            let mut rel_pos = DVec2::ZERO;
            for real_world_move in self.course_move_history.iter().rev().copied() {
                rel_pos -= real_world_move;
                let real_world_transform =
                    DAffine2::from_scale_angle_translation(self.area_size, 0., rel_pos);
                BorderRectangle {
                    transform: self.world_transform * real_world_transform,
                    color: Color32::YELLOW,
                    dashed: false,
                }
                .show(ui);
            }
        }
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let real_to_screen = ctx.world2egui() * self.world_transform;
        let mut rel_pos = DVec2::ZERO;
        let mut last_rel_pos = DVec2::ZERO;
        for real_world_move in self.course_move_history.iter().rev().copied() {
            rel_pos -= real_world_move;
            let p0 = real_to_screen.transform_point2(rel_pos);
            let p1 = real_to_screen.transform_point2(last_rel_pos);
            let v_move = p0 - p1;
            let v_side = v_move.normalize() * f64::min(15., v_move.length() / 2.);
            let p2 = DMat2::from_angle(0.5) * v_side + p1;
            let p3 = DMat2::from_angle(-0.5) * v_side + p1;
            let p0 = p0.to_egui_pos2();
            let p1 = p1.to_egui_pos2();
            let p2 = p2.to_egui_pos2();
            let p3 = p3.to_egui_pos2();
            ui.painter()
                .line_segment([p0, p1], Stroke::new(2., Color32::ORANGE));
            ui.painter().add(Shape::convex_polygon(
                vec![p1, p2, p3],
                Color32::ORANGE,
                Stroke::NONE,
            ));
            last_rel_pos = rel_pos;
        }
    }
}

const TRANSFORM_TABLE: TableDefinition<Uuid, [f64; 6]> =
    TableDefinition::new("scanarea_transform_table_v1");
const AREA_SIZE_TABLE: TableDefinition<Uuid, [f64; 2]> =
    TableDefinition::new("scanarea_area_size_table_v1");
const HISTORY_TABLE: TableDefinition<Uuid, Vec<[f64; 2]>> =
    TableDefinition::new("scanarea_history_table_v1");
/// Optional: the coarse move log (JSON), once there is one.
const COARSE_LOG_TABLE: TableDefinition<Uuid, &str> =
    TableDefinition::new("scanarea_coarse_log_v1");

fn coarse_log_json(log: &crate::analysis::coarse_fit::CoarseLog) -> Option<String> {
    (!log.moves.is_empty() || !log.drags.is_empty())
        .then(|| serde_json::to_string(log).ok())
        .flatten()
}

impl Persistant for ScanArea {
    fn db_update<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let id = self.uuid();

        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        let mut tran_data = tran_table.get_mut(id)?.expect("");
        if tran_data.value() != self.world_transform.to_cols_array() {
            tran_data.insert(self.world_transform.to_cols_array())?;
        }
        let mut area_table = txn.open_table(AREA_SIZE_TABLE)?;
        let mut area_data = area_table.get_mut(id)?.expect("");
        if area_data.value() != self.area_size.to_array() {
            area_data.insert(self.area_size.to_array())?;
        }
        let mut history_table = txn.open_table(HISTORY_TABLE)?;
        let mut history_data = history_table.get_mut(id)?.expect("");
        let history: Vec<[f64; 2]> = self
            .course_move_history
            .iter()
            .map(|v| v.to_array())
            .collect();
        if history_data.value() != history {
            history_data.insert(&history)?;
        }
        drop(history_data);
        drop(history_table);
        let mut log_table = txn.open_table(COARSE_LOG_TABLE)?;
        let stored = log_table.get(id)?.map(|v| v.value().to_string());
        let wanted = coarse_log_json(&self.coarse_log);
        if stored != wanted {
            match wanted {
                Some(j) => log_table.insert(id, j.as_str())?,
                None => log_table.remove(id)?,
            };
        }

        Ok(())
    }

    fn db_remove<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        tran_table.remove(id)?;
        let mut area_table = txn.open_table(AREA_SIZE_TABLE)?;
        area_table.remove(id)?;
        let mut history_table = txn.open_table(HISTORY_TABLE)?;
        history_table.remove(id)?;
        let mut log_table = txn.open_table(COARSE_LOG_TABLE)?;
        log_table.remove(id)?;
        Ok(())
    }

    fn db_insert<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let id = self.uuid();
        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        tran_table.insert(id, self.world_transform.to_cols_array())?;
        let mut area_table = txn.open_table(AREA_SIZE_TABLE)?;
        area_table.insert(id, self.area_size.to_array())?;
        let mut history_table = txn.open_table(HISTORY_TABLE)?;
        history_table.insert(
            id,
            &self
                .course_move_history
                .iter()
                .map(|v| v.to_array())
                .collect(),
        )?;
        if let Some(j) = coarse_log_json(&self.coarse_log) {
            let mut log_table = txn.open_table(COARSE_LOG_TABLE)?;
            log_table.insert(id, j.as_str())?;
        }
        Ok(())
    }

    fn db_read<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
        encoder: &ImageEncoder,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tran_table = txn.open_table(TRANSFORM_TABLE)?;
        let tran_data = tran_table.get(id)?.expect("").value();
        let area_table = txn.open_table(AREA_SIZE_TABLE)?;
        let area_data = area_table.get(id)?.expect("").value();
        let history_table = txn.open_table(HISTORY_TABLE)?;
        let history_data = history_table.get(id)?.expect("").value();
        let log_table = txn.open_table(COARSE_LOG_TABLE)?;
        let coarse_log = log_table
            .get(id)?
            .and_then(|v| serde_json::from_str(v.value()).ok())
            .unwrap_or_default();
        let mut scan_area = Self::new(id, encoder, DVec2::from_array(area_data));
        scan_area.world_transform = DAffine2::from_cols_array(&tran_data);
        scan_area.course_move_history = history_data
            .iter()
            .map(|arr| DVec2::from_array(*arr))
            .collect();
        scan_area.coarse_log = coarse_log;
        Ok(scan_area)
    }

    fn db_dump_stats<'t>(
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Scan Area:");
        let transform_len = txn.open_table(TRANSFORM_TABLE)?.len()?;
        let area_size_len = txn.open_table(AREA_SIZE_TABLE)?.len()?;
        let history_len = txn.open_table(HISTORY_TABLE)?.len()?;
        println!("  transform table: {transform_len} items");
        println!("  area size table: {area_size_len} items");
        println!("  history table: {history_len} items");
        Ok(())
    }
}
