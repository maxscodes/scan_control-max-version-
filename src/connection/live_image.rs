use core::f32;
use std::{fmt::Debug, io::Write, sync::Arc};

use egui::{Color32, DragValue, Id, Response, Ui, WidgetText};
use glam::{DAffine2, DVec2};
use image_compute::image_compute::{FitData, FitType};
use itertools::Itertools;
use nanonis_tcp::LineDir;
use redb::{ReadableTable as _, ReadableTableMetadata as _, TableDefinition};
use tracing::info;
use uuid::Uuid;

use crate::{
    components::{
        combo_box::{combo_box, ComboBoxType},
        EngFmt,
    },
    project::Persistant,
    scan_view::{
        static_image::NormType, BorderRectangle, ImageEncoder, ScanViewCtx, ScanViewImage,
    },
    utils::vec_interop::IntoEgui,
};

pub struct LiveImage {
    image_view: ScanViewImage,
    pub transform: DAffine2,
    pub std_dev: f32,
    pub fit_type: FitType,
    pub norm_type: NormType,
    pub line_dir: LineDir,
    pub forward_data: FrameData,
    pub backward_data: FrameData,
    pub unit: String,
    uuid: Uuid,
}

impl LiveImage {
    pub fn new(uuid: Uuid, encoder: &ImageEncoder, transform: DAffine2) -> Self {
        let norm_type = NormType::FullScale;
        let std_dev = 3.;
        let empty_data = FrameData::default();
        Self {
            image_view: ScanViewImage::new(
                Uuid::new_v4(),
                encoder,
                empty_data.size,
                transform,
                norm_type.combined(std_dev),
            ),
            norm_type,
            std_dev,
            line_dir: LineDir::Forward,
            fit_type: FitType::MeanSubtract,
            forward_data: empty_data.clone(),
            backward_data: empty_data,
            transform,
            unit: "".into(),
            uuid,
        }
    }
    pub fn show_image(&mut self, ui: &mut Ui) -> Response {
        self.image_view.transform = self.transform;
        self.image_view.norm_type = self.norm_type.combined(self.std_dev);
        let resp = self.image_view.show(ui);
        resp
    }
    pub fn show_menu(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        self.show_fit_control(ui, encoder);
        self.show_normalization_control(ui);
        self.show_line_dir_control(ui, encoder);
        self.show_metadata(ui);
    }
    pub fn show_metadata(&self, ui: &mut Ui) {
        let norm = self.image_view.norm_data.read();
        let fit = self.image_view.fit_data.read();
        if let (Some(norm), Some(fit)) = (norm.as_ref(), fit.as_ref()) {
            ui.label(format!(
                "Range:    {:.2}{}",
                EngFmt(norm.max - norm.min),
                self.unit
            ));
            ui.label(format!("Std Dev: {:.2}{}", EngFmt(norm.stddev), self.unit));
            if let FitData::PlaneFitSubtract {
                x_slope, y_slope, ..
            } = fit
            {
                ui.label(format!("X Slope: {:.2}{}", EngFmt(*x_slope), self.unit));
                ui.label(format!("Y Slope: {:.2}{}", EngFmt(*y_slope), self.unit));
            }
        }
    }
    pub fn show_fit_control(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        if combo_box(
            ui,
            (self.image_view.uuid(), "fit type"),
            &mut self.fit_type,
            &(),
        ) {
            self.update_texture(encoder);
        };
    }
    pub fn show_normalization_control(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            combo_box(
                ui,
                (self.image_view.uuid(), "norm type"),
                &mut self.norm_type,
                &(),
            );
            if self.norm_type == NormType::StdDev {
                ui.add(
                    DragValue::new(&mut self.std_dev)
                        .range((0.)..=(9.))
                        .speed(0.01),
                );
            }
        });
    }
    pub fn show_line_dir_control(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        let text = match self.line_dir {
            LineDir::Forward => "Forward",
            LineDir::Backward => "Backward",
        };
        if ui.button(text).clicked() {
            match self.line_dir {
                LineDir::Forward => self.line_dir = LineDir::Backward,

                LineDir::Backward => self.line_dir = LineDir::Forward,
            }
            self.write_and_update_texture(encoder);
        }
    }
    pub fn update_texture(&self, encoder: &ImageEncoder) {
        self.image_view.write_texture(encoder, self.fit_type);
    }
    pub fn write_and_update_texture(&mut self, encoder: &ImageEncoder) {
        let new_data = match self.line_dir {
            LineDir::Forward => &self.forward_data,
            LineDir::Backward => &self.backward_data,
        };
        if self.image_view.size() != new_data.size {
            self.image_view = self.image_view.resized(encoder, new_data.size);
        }
        self.image_view
            .write_lines(encoder, .., |buf| buf.copy_from_slice(&new_data.data))
            .unwrap();
        self.update_texture(encoder);
    }
    pub fn clear_texture(&mut self, encoder: &ImageEncoder) {
        self.image_view.clear(encoder);
    }
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }
    pub fn size(&self) -> [u32; 2] {
        self.image_view.size()
    }
    pub fn stamp(&mut self, encoder: &ImageEncoder) -> Self {
        let mut image_view = ScanViewImage::new(
            Uuid::new_v4(),
            encoder,
            self.image_view.size(),
            self.transform,
            self.image_view.norm_type,
        );
        std::mem::swap(&mut self.image_view, &mut image_view);
        self.write_and_update_texture(encoder);
        Self {
            image_view,
            transform: self.transform,
            std_dev: self.std_dev,
            fit_type: self.fit_type,
            norm_type: self.norm_type,
            line_dir: self.line_dir,
            forward_data: self.forward_data.clone(),
            backward_data: self.backward_data.clone(),
            unit: self.unit.clone(),
            uuid: Uuid::new_v4(),
        }
    }
}

impl ComboBoxType for LineDir {
    type Ctx = ();

    fn opt_atoms(&self, _ctx: &Self::Ctx) -> impl Into<WidgetText> {
        match self {
            LineDir::Forward => "Forward",
            LineDir::Backward => "Backward",
        }
    }

    fn options(_ctx: &Self::Ctx) -> impl Iterator<Item = Self> {
        [LineDir::Forward, LineDir::Backward].into_iter()
    }
}

#[derive(Clone)]
pub struct FrameData {
    pub size: [u32; 2],
    pub data: Arc<Box<[f32]>>,
}
impl Default for FrameData {
    fn default() -> Self {
        Self {
            size: [2, 2],
            data: Arc::new(vec![f32::NAN; 4].into_boxed_slice()),
        }
    }
}

const TRANSFORM_TABLE: TableDefinition<Uuid, [f64; 6]> =
    TableDefinition::new("liveimage_transform_table_v1");
const STD_DEV_TABLE: TableDefinition<Uuid, f32> = TableDefinition::new("liveimage_stddev_table_v1");
const DIR_TABLE: TableDefinition<Uuid, u32> = TableDefinition::new("liveimage_dir_table_v1");
const FIT_TABLE: TableDefinition<Uuid, u8> = TableDefinition::new("liveimage_fit_table_v1");
const NORM_TABLE: TableDefinition<Uuid, u8> = TableDefinition::new("liveimage_norm_table_v1");
const DATA_TABLE: TableDefinition<Uuid, [FrameData; 2]> =
    TableDefinition::new("liveimage_data_table_v1");
const UNIT_TABLE: TableDefinition<Uuid, &str> = TableDefinition::new("liveimage_unit_table_v1");

impl Persistant for LiveImage {
    fn db_update<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let id = self.uuid();

        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        let mut tran_data = tran_table
            .get_mut(id)?
            .expect(&format!("{id} did not exist"));
        if tran_data.value() != self.transform.to_cols_array() {
            tran_data.insert(self.transform.to_cols_array())?;
        }

        let mut stddev_table = txn.open_table(STD_DEV_TABLE)?;
        let mut stddev_data = stddev_table.get_mut(id)?.expect("");
        if stddev_data.value() != self.std_dev {
            stddev_data.insert(self.std_dev)?;
        }

        let mut dir_table = txn.open_table(DIR_TABLE)?;
        let mut dir_data = dir_table.get_mut(id)?.expect("");
        if dir_data.value() != Into::<u32>::into(self.line_dir) {
            dir_data.insert(&self.line_dir.into())?;
        }

        let mut fit_table = txn.open_table(FIT_TABLE)?;
        let mut fit_data = fit_table.get_mut(id)?.expect("");
        if fit_data.value() != Into::<u8>::into(self.fit_type) {
            fit_data.insert(&self.fit_type.into())?;
        }

        let mut norm_table = txn.open_table(NORM_TABLE)?;
        let mut norm_data = norm_table.get_mut(id)?.expect("");
        if norm_data.value() != Into::<u8>::into(self.norm_type) {
            norm_data.insert(&self.norm_type.into())?;
        }

        Ok(())
    }

    fn db_remove<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        tran_table.remove(id)?;
        let mut stddev_table = txn.open_table(STD_DEV_TABLE)?;
        stddev_table.remove(id)?;
        let mut dir_table = txn.open_table(DIR_TABLE)?;
        dir_table.remove(id)?;
        let mut fit_table = txn.open_table(FIT_TABLE)?;
        fit_table.remove(id)?;
        let mut norm_table = txn.open_table(NORM_TABLE)?;
        norm_table.remove(id)?;
        let mut unit_table = txn.open_table(UNIT_TABLE)?;
        unit_table.remove(id)?;
        let mut data_table = txn.open_table(DATA_TABLE)?;
        data_table.remove(id)?;

        Ok(())
    }

    fn db_insert<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let id = self.uuid();
        info!("Inserting LiveImage {id}");
        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        tran_table.insert(id, self.transform.to_cols_array())?;
        let mut stddev_table = txn.open_table(STD_DEV_TABLE)?;
        stddev_table.insert(id, self.std_dev)?;
        let mut dir_table = txn.open_table(DIR_TABLE)?;
        dir_table.insert(id, &self.line_dir.into())?;
        let mut fit_table = txn.open_table(FIT_TABLE)?;
        fit_table.insert(id, &self.fit_type.into())?;
        let mut norm_table = txn.open_table(NORM_TABLE)?;
        norm_table.insert(id, &self.norm_type.into())?;
        let mut unit_table = txn.open_table(UNIT_TABLE)?;
        unit_table.insert(id, self.unit.as_str())?;
        let mut data_table = txn.open_table(DATA_TABLE)?;
        data_table.insert(id, [self.forward_data.clone(), self.backward_data.clone()])?;
        Ok(())
    }

    fn db_read<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
        encoder: &ImageEncoder,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tran_table = txn.open_table(TRANSFORM_TABLE)?;
        let tran_data = tran_table.get(id)?.unwrap().value();
        let stddev_table = txn.open_table(STD_DEV_TABLE)?;
        let stddev_data = stddev_table.get(id)?.unwrap().value();
        let dir_table = txn.open_table(DIR_TABLE)?;
        let dir_data = dir_table.get(id)?.unwrap().value();
        let fit_table = txn.open_table(FIT_TABLE)?;
        let fit_data = fit_table.get(id)?.unwrap().value();
        let norm_table = txn.open_table(NORM_TABLE)?;
        let norm_data = norm_table.get(id)?.unwrap().value();
        let unit_table = txn.open_table(UNIT_TABLE)?;
        let unit_data = unit_table.get(id)?.unwrap().value().to_string();
        let data_table = txn.open_table(DATA_TABLE)?;
        let frame_data = data_table.get(id)?.unwrap().value();
        let mut image = Self::new(id, encoder, DAffine2::from_cols_array(&tran_data));
        image.std_dev = stddev_data;
        image.line_dir = LineDir::try_from(dir_data).unwrap();
        image.fit_type = FitType::try_from(fit_data).unwrap();
        image.norm_type = NormType::try_from(norm_data).unwrap();
        image.unit = unit_data;
        [image.forward_data, image.backward_data] = frame_data;
        image.write_and_update_texture(encoder);
        Ok(image)
    }
    
    fn db_dump_stats<'t>(txn: &'t redb::WriteTransaction) -> Result<(), Box<dyn std::error::Error>> {
        println!("Live Image:");
        let transform_table_len = txn.open_table(TRANSFORM_TABLE)?.len()?;
        let std_dev_table_len = txn.open_table(STD_DEV_TABLE)?.len()?;
        let dir_table_len = txn.open_table(DIR_TABLE)?.len()?;
        let fit_table_len = txn.open_table(FIT_TABLE)?.len()?;
        let norm_table_len = txn.open_table(NORM_TABLE)?.len()?;
        let data_table_len = txn.open_table(DATA_TABLE)?.len()?;
        let unit_table_len = txn.open_table(UNIT_TABLE)?.len()?;
        println!("  transform table: {transform_table_len} items");
        println!("  std_dev table: {std_dev_table_len} items");
        println!("  dir table: {dir_table_len} items");
        println!("  fit table: {fit_table_len} items");
        println!("  norm table: {norm_table_len} items");
        println!("  data table: {data_table_len} items");
        println!("  unit table: {unit_table_len} items");
        Ok(())
        
    }
}

impl redb::Value for FrameData {
    type SelfType<'a> = FrameData;

    type AsBytes<'a> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let s0 = u32::from_le_bytes(data[0..][..4].try_into().unwrap());
        let s1 = u32::from_le_bytes(data[4..][..4].try_into().unwrap());
        let mut buf = Vec::new();
        for chunk in data[8..].iter().copied().chunks(4).into_iter() {
            let data: [u8; 4] = chunk.collect_array().unwrap();
            buf.push(f32::from_le_bytes(data));
        }
        FrameData {
            size: [s0, s1],
            data: Arc::new(buf.into_boxed_slice()),
        }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        let mut buf = Vec::new();
        buf.extend(value.size[0].to_le_bytes());
        buf.extend(value.size[1].to_le_bytes());
        buf.extend(value.data.iter().map(|v| v.to_le_bytes()).flatten());
        buf
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("scan_control_frame_data")
    }
}

impl Debug for FrameData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameData")
            .field("size", &self.size)
            .finish()
    }
}
