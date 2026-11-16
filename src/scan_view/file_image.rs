use std::path::Path;

use eframe::egui_wgpu;
use egui::{Color32, Id, Response, Sense, Ui};
use glam::{DAffine2, DMat3, DVec2};
use image::{DynamicImage, EncodableLayout, GenericImageView, RgbaImage};
use image_compute::file_image::FileImageBuffers;
use itertools::Itertools;
use redb::{ReadableTable, ReadableTableMetadata as _, TableDefinition};
use uuid::Uuid;

use crate::{
    project::Persistant,
    scan_view::{callbacks::FileImageCallback, view::ScanViewCtx, ImageEncoder},
    utils::vec_interop::{IntoEgui as _, IntoGlam as _, Projection},
};

pub struct FileImage {
    pub transform: DMat3,
    buffers: FileImageBuffers,
    pub local_points: [glam::DVec2; 4],
    pub world_points: [glam::DVec2; 4],
    data: DynamicImage,
    editing: bool,
    uuid: Uuid,
}
impl FileImage {
    pub fn new_from_data(
        uuid: Uuid,
        image_encoder: &ImageEncoder,
        size: [u32; 2],
        data: Vec<u8>,
        transform: DMat3,
        world_points: [glam::DVec2; 4],
        local_points: [glam::DVec2; 4],
    ) -> Self {
        let img = RgbaImage::from_vec(size[0], size[1], data).unwrap().into();
        let buffers = FileImageBuffers::new(
            &image_encoder.wgpu_state.device,
            &image_encoder.wgpu_state.queue,
            &img,
        );
        Self {
            transform,
            buffers,
            local_points,
            world_points,
            editing: false,
            uuid,
            data: img,
        }
    }
    pub fn new(image_encoder: &ImageEncoder, path: impl AsRef<Path>, transform: DMat3) -> Self {
        let img = image::open(&path).unwrap();
        let buffers = FileImageBuffers::new(
            &image_encoder.wgpu_state.device,
            &image_encoder.wgpu_state.queue,
            &img,
        );
        let local_points = [
            glam::DVec2::new(-0.5, -0.5),
            glam::DVec2::new(0.5, -0.5),
            glam::DVec2::new(0.5, 0.5),
            glam::DVec2::new(-0.5, 0.5),
        ];
        let world_points = local_points
            .iter()
            .map(|p| transform.project_pos2(*p))
            .collect_array()
            .unwrap();
        Self {
            transform,
            buffers,
            local_points,
            world_points,
            editing: false,
            uuid: Uuid::new_v4(),
            data: img,
        }
    }
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }
    pub fn show(&mut self, ui: &mut Ui) {
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let callback = egui_wgpu::Callback::new_paint_callback(
            ctx.rect,
            FileImageCallback {
                transform: self.transform,
                image_buffers: self.buffers.clone(),
            },
        );
        ui.painter().add(callback);
        if self.editing {
            for (i, c) in POINT_COLORS.into_iter().enumerate() {
                let p = &mut self.world_points[i];
                let screen_pos = ctx.world2egui().project_pos2(*p).to_egui_pos2();
                let resp = drag_point((self.uuid, i), ui, screen_pos, 8., c)
                    .on_hover_cursor(egui::CursorIcon::Move);
                if resp.dragged_by(egui::PointerButton::Primary) {
                    *p += ctx
                        .world2egui()
                        .inverse()
                        .project_vec2(resp.drag_delta().to_glam());
                    self.update_transform();
                } else if resp.dragged_by(egui::PointerButton::Secondary) {
                    *p += ctx
                        .world2egui()
                        .inverse()
                        .project_vec2(resp.drag_delta().to_glam());
                    self.update_local_points();
                }
            }
        }
    }
    fn update_local_points(&mut self) {
        self.local_points = self
            .world_points
            .iter()
            .map(|p| self.transform.inverse().project_pos2(*p))
            .collect_array()
            .unwrap();
    }
    fn update_transform(&mut self) {
        let mut a = nalgebra::SMatrix::<f64, 8, 8>::zeros();
        let mut b = nalgebra::SVector::<f64, 8>::zeros();

        for i in 0..4 {
            let x = self.local_points[i].x;
            let y = self.local_points[i].y;
            let u = self.world_points[i].x;
            let v = self.world_points[i].y;

            let r0 = 2 * i;
            let r1 = r0 + 1;

            a[(r0, 0)] = x;
            a[(r0, 1)] = y;
            a[(r0, 2)] = 1.0;
            a[(r0, 6)] = -u * x;
            a[(r0, 7)] = -u * y;
            b[r0] = u;

            a[(r1, 3)] = x;
            a[(r1, 4)] = y;
            a[(r1, 5)] = 1.0;
            a[(r1, 6)] = -v * x;
            a[(r1, 7)] = -v * y;
            b[r1] = v;
        }

        if let Some(h) = a.lu().solve(&b) {
            self.transform =
                DMat3::from_cols_array(&[h[0], h[3], h[6], h[1], h[4], h[7], h[2], h[5], 1.0]);
        }
    }
    pub fn show_menu(&mut self, ui: &mut Ui) {
        if self.editing {
            if ui.button("Lock").clicked() {
                self.editing = false;
            }
            if ui.button("Flip X").clicked() {
                let tran = DAffine2::from_translation(self.center());
                let tf = DAffine2::from_scale(DVec2::new(-1., 1.));
                let tf = tran * tf * tran.inverse();
                self.transform_world_points(tf);
            }
            if ui.button("Flip Y").clicked() {
                let tran = DAffine2::from_translation(self.center());
                let tf = DAffine2::from_scale(DVec2::new(1., -1.));
                let tf = tran * tf * tran.inverse();
                self.transform_world_points(tf);
            }
        } else {
            if ui.button("Anchor Transform").clicked() {
                self.editing = true;
            }
        }
    }
    pub fn center(&self) -> glam::DVec2 {
        self.world_points.iter().sum::<glam::DVec2>() / 4.
    }
    /// Put the image's corners at these map positions.
    pub fn set_world_points(&mut self, points: [DVec2; 4]) {
        self.world_points = points;
        self.update_transform();
    }
    pub fn transform_world_points(&mut self, tf: DAffine2) {
        for p in &mut self.world_points {
            *p = tf.project_pos2(*p);
        }
        self.update_transform();
    }
}

const POINT_COLORS: [Color32; 4] = [
    Color32::MAGENTA,
    Color32::MAGENTA,
    Color32::MAGENTA,
    Color32::MAGENTA,
];

fn drag_point(
    id_salt: impl std::hash::Hash,
    ui: &mut Ui,
    center: egui::Pos2,
    radius: f32,
    color: impl Into<Color32>,
) -> Response {
    let resp = ui.interact(
        egui::Rect::from_center_size(
            center,
            egui::Vec2 {
                x: radius * 2.,
                y: radius * 2.,
            },
        ),
        egui::Id::new(id_salt),
        Sense::HOVER | Sense::DRAG,
    );

    if resp.hovered() {
        ui.painter()
            .circle_filled(center, radius * 1.3, Color32::CYAN);
    }
    ui.painter().circle_filled(center, radius, color);

    resp
}

const TRANSFORM_TABLE: TableDefinition<Uuid, [f64; 9]> =
    TableDefinition::new("fileimage_transform_table_v1");
const ANCHOR_TABLE: TableDefinition<Uuid, [[f64; 2]; 8]> =
    TableDefinition::new("fileimage_anchor_table_v1");
const SIZE_TABLE: TableDefinition<Uuid, [u32; 2]> = TableDefinition::new("fileimage_size_table_v1");
const DATA_TABLE: TableDefinition<Uuid, &[u8]> = TableDefinition::new("fileimage_data_table_v1");

impl Persistant for FileImage {
    fn db_update<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let id = self.uuid();

        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        let mut tran_data = tran_table.get_mut(id)?.expect("");
        if tran_data.value() != self.transform.to_cols_array() {
            tran_data.insert(self.transform.to_cols_array())?;
        }

        let mut anchor_table = txn.open_table(ANCHOR_TABLE)?;
        let mut anchor_data = anchor_table.get_mut(id)?.expect("");
        let current: [[f64; 2]; 8] = self
            .local_points
            .iter()
            .chain(self.world_points.iter())
            .map(|p| p.to_array())
            .collect_array()
            .unwrap();
        if anchor_data.value() != current {
            anchor_data.insert(current)?;
        }
        Ok(())
    }

    fn db_remove<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        tran_table.remove(id)?;
        let mut anchor_table = txn.open_table(ANCHOR_TABLE)?;
        anchor_table.remove(id)?;
        let mut size_table = txn.open_table(SIZE_TABLE)?;
        size_table.remove(id)?;
        let mut data_table = txn.open_table(DATA_TABLE)?;
        data_table.remove(id)?;
        Ok(())
    }

    fn db_insert<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let id = self.uuid();
        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        tran_table.insert(id, self.transform.to_cols_array())?;
        let mut anchor_table = txn.open_table(ANCHOR_TABLE)?;
        let current: [[f64; 2]; 8] = self
            .local_points
            .iter()
            .chain(self.world_points.iter())
            .map(|p| p.to_array())
            .collect_array()
            .unwrap();
        anchor_table.insert(id, current)?;
        let mut size_table = txn.open_table(SIZE_TABLE)?;
        let dims = self.data.dimensions();
        size_table.insert(id, [dims.0, dims.1])?;
        let mut data_table = txn.open_table(DATA_TABLE)?;
        data_table.insert(id, self.data.to_rgba8().as_bytes())?;
        Ok(())
    }

    fn db_read<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
        encoder: &ImageEncoder,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tran_table = txn.open_table(TRANSFORM_TABLE)?;
        let tran = tran_table.get(id)?.unwrap().value();
        let anchor_table = txn.open_table(ANCHOR_TABLE)?;
        let anchors = anchor_table.get(id)?.unwrap().value();
        let size_table = txn.open_table(SIZE_TABLE)?;
        let size = size_table.get(id)?.unwrap().value();
        let data_table = txn.open_table(DATA_TABLE)?;
        let data = data_table.get(id)?.unwrap().value().to_vec();
        let local_points = anchors[..4]
            .iter()
            .map(|v| DVec2::from_array(*v))
            .collect_array()
            .unwrap();
        let world_points = anchors[4..]
            .iter()
            .map(|v| DVec2::from_array(*v))
            .collect_array()
            .unwrap();
        Ok(Self::new_from_data(
            id,
            encoder,
            size,
            data,
            DMat3::from_cols_array(&tran),
            world_points,
            local_points,
        ))
    }
    
    fn db_dump_stats<'t>(txn: &'t redb::WriteTransaction) -> Result<(), Box<dyn std::error::Error>> {
        println!("File Image:");
        let transform_table_len = txn.open_table(TRANSFORM_TABLE)?.len()?;
        let anchor_table_len = txn.open_table(ANCHOR_TABLE)?.len()?;
        let size_table_len = txn.open_table(SIZE_TABLE)?.len()?;
        let data_table_len = txn.open_table(DATA_TABLE)?.len()?;
        println!("  transform table: {transform_table_len} items");
        println!("  anchor table: {anchor_table_len} items");
        println!("  size table: {size_table_len} items");
        println!("  data table: {data_table_len} items");
        Ok(())
    }
}
