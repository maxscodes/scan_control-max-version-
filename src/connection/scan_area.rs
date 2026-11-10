use std::collections::VecDeque;

use egui::{epaint::PathStroke, Color32, Id, Shape, Stroke, Ui};
use glam::{DAffine2, DMat2, DVec2};
use redb::{ReadableTable as _, ReadableTableMetadata, TableDefinition};
use uuid::Uuid;

use crate::{
    connection::{nanonis::ScanStatus, LiveImage},
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
    pub stamp_name_base: String,
    pub course_move_history: Vec<DVec2>,
    pub show_history: bool,
    pub show_history_boxes: bool,
}
impl ScanArea {
    pub fn new(uuid: Uuid, encoder: &ImageEncoder, area_size: DVec2) -> Self {
        let live_image = LiveImage::new(uuid, encoder, DAffine2::IDENTITY);
        Self {
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
            show_history: false,
            show_history_boxes: false,
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
    pub fn show_menu(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        ui.set_max_width(140.);
        self.show_channel_control(ui);
        self.live_image.show_menu(ui, encoder);
        self.show_stamp_button(ui, encoder);
        ui.separator();
        ui.checkbox(&mut self.show_history, "Show history");
        if self.show_history {
            ui.indent("history_boxes_check", |ui|{
                ui.checkbox(&mut self.show_history_boxes, "Show boxes");
            });
        }
    }
    fn show_stamp_button(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        let font_id = egui::TextStyle::Body.resolve(ui.style());
        let galley = ui.painter().layout_no_wrap(
            self.stamp_name_base.clone(),
            font_id.clone(),
            ui.visuals().text_color(),
        );
        let padding = ui.spacing().button_padding.x * 2.0;
        let width = galley.size().x + padding;

        ui.horizontal(|ui| {
            if ui.button("Stamp").clicked() {
                self.stamp.push_front(self.live_image.stamp(encoder));
            }
            ui.add_enabled(
                false,
                egui::TextEdit::singleline(&mut self.stamp_name_base).desired_width(width),
            );
        });
    }
    fn show_channel_control(&mut self, ui: &mut Ui) {
        let mut selection = self.channel_selected.as_ref().map(|s| s.as_str());
        if egui::ComboBox::new((self.live_image.uuid(), "combo_box"), "")
            .selected_text(selection.unwrap_or_default())
            .show_ui(ui, |ui| {
                self.channel_opts
                    .iter()
                    .map(|opt| ui.selectable_value(&mut selection, Some(opt), opt))
                    .any(|resp| resp.clicked())
            })
            .inner
            .is_some_and(|clicked| clicked)
        {
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
        let mut scan_area = Self::new(id, encoder, DVec2::from_array(area_data));
        scan_area.world_transform = DAffine2::from_cols_array(&tran_data);
        scan_area.course_move_history = history_data
            .iter()
            .map(|arr| DVec2::from_array(*arr))
            .collect();
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
