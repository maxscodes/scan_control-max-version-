use core::f64;
use std::{collections::BTreeMap, path::Path};

use eframe::egui_wgpu;
use egui::{Color32, DragValue, Id, Ui};
use gdsr::{Cell, Element, Library};
use glam::{DAffine2, DVec2, Vec2};
use image_compute::gds_image::GDSImageBuffers;
use itertools::Itertools;
use redb::{ReadableTable, ReadableTableMetadata, TableDefinition};
use uuid::Uuid;

use crate::{
    project::Persistant,
    scan_view::{callbacks::GDSImageCallback, view::ScanViewCtx, ImageEncoder},
    utils::vec_interop::Projection,
};

pub struct GDSImage {
    uuid: Uuid,
    pub transform: DAffine2,
    pub scale: f64,
    buffers: BTreeMap<u16, GDSImageBuffers>,
    colors: BTreeMap<u16, Color32>,
    polys: BTreeMap<u16, Vec<Vec<Vec2>>>,
    /// Ring systems found in the polygons (computed on first use).
    rings: std::sync::OnceLock<std::sync::Arc<Vec<crate::analysis::gds_prior::RingSystem>>>,
}

impl GDSImage {
    pub fn new_from_file(
        encoder: &ImageEncoder,
        path: impl AsRef<Path>,
        transform: DAffine2,
    ) -> Self {
        let gds = gdsr::Library::read_file(path, None).unwrap();
        let mut polys = BTreeMap::new();
        for cell in gds.cells().values() {
            draw_cell(&mut polys, &gds, DAffine2::IDENTITY, cell);
        }
        let (min, max) = polys.values().flatten().flatten().fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(min, max), v| {
                (
                    Vec2 {
                        x: v.x.min(min.x),
                        y: v.y.min(min.y),
                    },
                    Vec2 {
                        x: v.x.max(max.x),
                        y: v.y.max(max.y),
                    },
                )
            },
        );
        let center = (max + min) / 2.;
        for vert in polys.values_mut().flatten().flatten() {
            *vert -= center;
        }
        Self::new_from_data(Uuid::new_v4(), encoder, polys, transform)
    }
    pub fn new_from_data(
        uuid: Uuid,
        encoder: &ImageEncoder,
        polys: BTreeMap<u16, Vec<Vec<Vec2>>>,
        transform: DAffine2,
    ) -> Self {
        let (min, max) = polys.values().flatten().flatten().fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(min, max), v| {
                (
                    Vec2 {
                        x: v.x.min(min.x),
                        y: v.y.min(min.y),
                    },
                    Vec2 {
                        x: v.x.max(max.x),
                        y: v.y.max(max.y),
                    },
                )
            },
        );
        let scale = min.distance(max) as f64 / 2f64.sqrt();
        let colors = polys
            .keys()
            .map(|layer| {
                (
                    *layer,
                    *COLORS.into_iter().cycle().nth(*layer as usize).unwrap(),
                )
            })
            .collect();
        let buffers = polys
            .iter()
            .map(|(layer, polys)| {
                (
                    *layer,
                    GDSImageBuffers::new(&encoder.wgpu_state.device, polys),
                )
            })
            .collect();
        Self {
            transform,
            buffers,
            colors,
            scale,
            uuid,
            polys,
            rings: std::sync::OnceLock::new(),
        }
    }
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }
    /// Polygons per GDS layer, in this image's local nm (apply `transform`
    /// for world coordinates).
    pub fn polygons(&self) -> &BTreeMap<u16, Vec<Vec<Vec2>>> {
        &self.polys
    }
    /// Concentric ring systems in this GDS (for the Ring Assist prior).
    pub fn ring_systems(&self) -> std::sync::Arc<Vec<crate::analysis::gds_prior::RingSystem>> {
        self.rings
            .get_or_init(|| std::sync::Arc::new(crate::analysis::gds_prior::find_ring_systems(&self.polys)))
            .clone()
    }
    pub fn show(&self, ui: &mut Ui) {
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        for (layer, bufs) in &self.buffers {
            let callback = egui_wgpu::Callback::new_paint_callback(
                ctx.rect,
                GDSImageCallback {
                    transform: self.transform.into(),
                    image_buffers: bufs.clone(),
                    color: *self.colors.get(layer).unwrap(),
                },
            );
            ui.painter().add(callback);
        }
    }
    pub fn show_menu(&mut self, ui: &mut Ui) {
        let (scale, angle, tran) = self.transform.to_scale_angle_translation();
        let mut scalar = scale.x;
        ui.horizontal(|ui| {
            ui.label("Scale multiplier: ");
            ui.add(
                DragValue::new(&mut scalar)
                    .range(0.5..=2.0)
                    .speed(0.001)
                    .min_decimals(3),
            )
        });
        self.transform =
            DAffine2::from_scale_angle_translation(DVec2::new(scalar, scalar), angle, tran);
    }
}

fn draw_cell(
    polys: &mut BTreeMap<u16, Vec<Vec<glam::Vec2>>>,
    lib: &Library,
    transform: DAffine2,
    cell: &Cell,
) {
    for elem in cell.iter_elements() {
        draw_element(polys, lib, transform, elem);
    }
}

fn draw_element(
    polys: &mut BTreeMap<u16, Vec<Vec<glam::Vec2>>>,
    lib: &Library,
    transform: DAffine2,
    elem: &Element,
) {
    #[allow(unused_variables)]
    match elem {
        gdsr::Element::Path(path) => todo!(),
        gdsr::Element::Polygon(polygon) => {
            let points = polygon
                .points()
                .iter()
                .dropping_back(1)
                .map(|p| transform.project_pos2(p.to_world()).as_vec2())
                .collect_vec();
            let layer = polygon.layer().value();
            polys.entry(layer).or_default().push(points);
        }
        gdsr::Element::Box(gds_box) => todo!(),
        gdsr::Element::Node(node) => todo!(),
        gdsr::Element::Text(text) => todo!(),
        gdsr::Element::Reference(reference) => {
            let origin = reference.grid().origin().to_world();
            let transform = transform
                * DAffine2::from_angle_translation(
                    reference.grid().angle() / 180. * 3.14159,
                    origin,
                );
            match reference.instance() {
                gdsr::Instance::Cell(cell_name) => {
                    let cell = lib.get_cell(cell_name).unwrap();
                    draw_cell(polys, lib, transform, cell);
                }
                gdsr::Instance::Element(elem) => {
                    draw_element(polys, lib, transform, elem);
                }
            }
        }
    }
}

trait GDSPoint: Copy {
    fn to_world(self) -> glam::DVec2;
}
impl GDSPoint for gdsr::Point {
    fn to_world(self) -> glam::DVec2 {
        glam::DVec2 {
            x: self.x().to_world(),
            y: self.y().to_world(),
        }
    }
}

trait GDSUnit: Copy {
    fn to_world(self) -> f64;
}
impl GDSUnit for gdsr::Unit {
    fn to_world(self) -> f64 {
        match self {
            gdsr::Unit::Integer(gdsr::IntegerUnit { value, units }) => {
                let scale = units * 1e9;
                value as f64 * scale
            }
            #[allow(unused_variables)]
            gdsr::Unit::Float(gdsr::FloatUnit { value, units }) => todo!(),
        }
    }
}

const COLORS: &[Color32] = &[
    Color32::from_rgb(0xff, 0x9d, 0x9d),
    Color32::from_rgb(0xff, 0x80, 0xa8),
    Color32::from_rgb(0xc0, 0x80, 0xff),
    Color32::from_rgb(0x95, 0x80, 0xff),
    Color32::from_rgb(0x80, 0x86, 0xff),
    Color32::from_rgb(0x80, 0xa8, 0xff),
];

const TRANSFORM_TABLE: TableDefinition<Uuid, [f64; 6]> =
    TableDefinition::new("gds_transform_table_v1");
const LAYER_TABLE: TableDefinition<Uuid, Vec<u16>> = TableDefinition::new("gds_layer_table_v1");
const DATA_TABLE: TableDefinition<(Uuid, u16), Vec<Vec<[f32; 2]>>> =
    TableDefinition::new("gds_data_table_v1");

impl Persistant for GDSImage {
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

        Ok(())
    }

    fn db_remove<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        tran_table.remove(id)?;
        let mut layer_table = txn.open_table(LAYER_TABLE)?;
        let layers = layer_table.get(id)?.unwrap().value();
        layer_table.remove(id)?;
        let mut data_table = txn.open_table(DATA_TABLE)?;
        for layer in layers {
            data_table.remove((id, layer))?;
        }
        Ok(())
    }

    fn db_insert<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let id = self.uuid();
        let mut tran_table = txn.open_table(TRANSFORM_TABLE)?;
        tran_table.insert(id, self.transform.to_cols_array())?;
        let mut layer_table = txn.open_table(LAYER_TABLE)?;
        layer_table.insert(id, &self.buffers.keys().copied().collect())?;
        let mut data_table = txn.open_table(DATA_TABLE)?;
        for (layer, data) in self.polys.iter() {
            let data = data
                .iter()
                .map(|poly| poly.iter().map(|vert| vert.to_array()).collect())
                .collect();
            data_table.insert((id, *layer), &data)?;
        }

        Ok(())
    }

    fn db_read<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
        encoder: &ImageEncoder,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tran_table = txn.open_table(TRANSFORM_TABLE)?;
        let tran_data = tran_table.get(id)?.unwrap().value();
        let layer_table = txn.open_table(LAYER_TABLE)?;
        let layer_data = layer_table.get(id)?.unwrap().value();
        let data_table = txn.open_table(DATA_TABLE)?;
        let mut polys = BTreeMap::new();
        for layer in layer_data {
            let data = data_table.get((id, layer))?.unwrap().value();
            let data = data
                .into_iter()
                .map(|poly| {
                    poly.into_iter()
                        .map(|vert| Vec2::from_array(vert))
                        .collect_vec()
                })
                .collect_vec();
            polys.insert(layer, data);
        }
        Ok(Self::new_from_data(
            id,
            encoder,
            polys,
            DAffine2::from_cols_array(&tran_data),
        ))
    }

    fn db_dump_stats<'t>(
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("GDS Image:");
        let transform_table_len = txn.open_table(TRANSFORM_TABLE)?.len()?;
        let layer_table_len = txn.open_table(LAYER_TABLE)?.len()?;
        let data_table_len = txn.open_table(DATA_TABLE)?.len()?;
        println!("  transform table: {transform_table_len} items");
        println!("  layer table: {layer_table_len} items");
        println!("  data table: {data_table_len} items");
        Ok(())
    }
}
