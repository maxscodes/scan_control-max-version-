use crate::analysis::ring_stamp::{RingLabel, RingStamp};
use core::f32;
use std::{fmt::Debug, sync::Arc};

use egui::{DragValue, Response, Ui, WidgetText};
use glam::DAffine2;
use image_compute::image_compute::FitData;
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
        processing::{self, Background, Filters},
        static_image::NormType,
        ImageEncoder, ScanViewImage,
    },
};

pub struct LiveImage {
    image_view: ScanViewImage,
    pub transform: DAffine2,
    pub std_dev: f32,
    pub background: Background,
    /// Display filters. They only change the drawn image: `forward_data`
    /// and `backward_data` stay as recorded.
    pub filters: Filters,
    pub norm_type: NormType,
    /// 0..1, how opaque the image is drawn on the map.
    pub opacity: f32,
    pub line_dir: LineDir,
    pub forward_data: FrameData,
    pub backward_data: FrameData,
    pub unit: String,
    /// Ring Assist result recorded when this image was stamped.
    pub ring: Option<RingStamp>,
    /// The user's training label for this stamp.
    pub ring_label: Option<RingLabel>,
    /// The GDS (uuid and transform) on the map when this stamp was taken or
    /// first checked, to warn if the GDS is moved later.
    pub gds_ref: Option<(Uuid, DAffine2)>,
    /// Where the stamp was put and after which coarse move, to learn the
    /// coarse calibration from drags.
    pub coarse_ref: Option<crate::analysis::coarse_fit::CoarseRef>,
    /// Tip Check result for this stamp.
    pub tip: Option<crate::analysis::tip_quality::TipReport>,
    /// Band direction in the forward frame, measured in the background on
    /// first use.
    bands: Arc<std::sync::OnceLock<Option<crate::analysis::gds_prior::Bands>>>,
    bands_queued: std::sync::atomic::AtomicBool,
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
            opacity: 1.,
            line_dir: LineDir::Forward,
            background: Background::Raw,
            filters: Filters::default(),
            forward_data: empty_data.clone(),
            backward_data: empty_data,
            transform,
            unit: "".into(),
            ring: None,
            ring_label: None,
            gds_ref: None,
            coarse_ref: None,
            tip: None,
            bands: Default::default(),
            bands_queued: Default::default(),
            uuid,
        }
    }
    /// Broad band direction of a stamped Z frame. Measured once, on a
    /// background thread so drawing never waits: `None` until it's ready.
    pub fn bands(&self) -> Option<crate::analysis::gds_prior::Bands> {
        if let Some(b) = self.bands.get() {
            return *b;
        }
        if !self.bands_queued.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let size = self.forward_data.size;
            if self.unit != "m" || size[0] == 0 {
                let _ = self.bands.set(None);
                return None;
            }
            let nm_per_px = self.transform.matrix2.x_axis.length() / size[0] as f64;
            let cell = self.bands.clone();
            let data = self.forward_data.data.clone();
            crate::analysis::gds_prior::queue_bands(Box::new(move || {
                let _ = cell.set(crate::analysis::gds_prior::band_orientation(&data, size, nm_per_px));
            }));
        }
        None
    }
    pub fn show_image(&mut self, ui: &mut Ui) -> Response {
        self.image_view.transform = self.transform;
        self.image_view.norm_type = self.norm_type.combined(self.std_dev);
        self.image_view.opacity = self.opacity;
        let resp = self.image_view.show(ui);
        resp
    }
    pub fn show_menu(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        display_controls(ui, &mut [self], encoder);
        self.show_metadata(ui);
    }
    pub fn display_settings(&self) -> DisplaySettings {
        DisplaySettings {
            background: self.background,
            filters: self.filters,
            norm_type: self.norm_type,
            std_dev: self.std_dev,
            opacity: self.opacity,
            line_dir: self.line_dir,
        }
    }
    /// Apply only the settings that differ between `from` and `to`, so
    /// changing one setting on several images keeps their other settings.
    pub fn apply_display_change(
        &mut self,
        from: &DisplaySettings,
        to: &DisplaySettings,
        encoder: &ImageEncoder,
    ) {
        let before = self.display_settings();
        if from.background != to.background {
            self.background = to.background;
        }
        let (f0, f1, f) = (&from.filters, &to.filters, &mut self.filters);
        if f0.smooth != f1.smooth {
            f.smooth = f1.smooth;
        }
        if f0.smooth_px != f1.smooth_px {
            f.smooth_px = f1.smooth_px;
        }
        if f0.sharpen != f1.sharpen {
            f.sharpen = f1.sharpen;
        }
        if f0.sharpen_amount != f1.sharpen_amount {
            f.sharpen_amount = f1.sharpen_amount;
        }
        if f0.lines != f1.lines {
            f.lines = f1.lines;
        }
        if f0.lines_threshold != f1.lines_threshold {
            f.lines_threshold = f1.lines_threshold;
        }
        if f0.blobs != f1.blobs {
            f.blobs = f1.blobs;
        }
        if f0.blob_px != f1.blob_px {
            f.blob_px = f1.blob_px;
        }
        if from.norm_type != to.norm_type {
            self.norm_type = to.norm_type;
        }
        if from.std_dev != to.std_dev {
            self.std_dev = to.std_dev;
        }
        if from.opacity != to.opacity {
            self.opacity = to.opacity;
        }
        if from.line_dir != to.line_dir {
            self.line_dir = to.line_dir;
        }
        let after = self.display_settings();
        // Contrast is applied when drawing; the rest needs the image redone.
        if (before.background, before.filters, before.line_dir)
            != (after.background, after.filters, after.line_dir)
        {
            self.write_and_update_texture(encoder);
        }
    }
    pub fn show_metadata(&self, ui: &mut Ui) {
        let norm = self.image_view.norm_data.read();
        let fit = self.image_view.fit_data.read();
        if let (Some(norm), Some(fit)) = (norm.as_ref(), fit.as_ref()) {
            let unit = &self.unit;
            egui::Grid::new((self.uuid, "metadata_grid"))
                .num_columns(2)
                .spacing([8., 2.])
                .show(ui, |ui| {
                    let row = |ui: &mut Ui, name: &str, v: f32| {
                        ui.weak(name);
                        ui.monospace(format!("{:.2}{unit}", EngFmt(v)));
                        ui.end_row();
                    };
                    row(ui, "Range", norm.max - norm.min);
                    row(ui, "Std dev", norm.stddev);
                    if let FitData::PlaneFitSubtract { x_slope, y_slope, .. } = fit {
                        row(ui, "X slope", *x_slope);
                        row(ui, "Y slope", *y_slope);
                    }
                });
        }
        if let Some(t) = &self.tip {
            ui.label(
                egui::RichText::new(format!("Tip: {}", t.summary()))
                    .color(crate::connection::nanonis::tip_check::verdict_color(&t.verdict)),
            );
            for n in &t.notes {
                ui.weak(format!("  {n}"));
            }
        }
        if let Some(r) = &self.ring {
            ui.label(egui::RichText::new(format!("Ring: {}", r.summary())).color(r.color()));
            ui.label(format!(
                "  inliers {:.2}, radius {:.1} µm{}",
                r.inlier_fraction,
                r.radius_um,
                if r.valid_fraction < 0.98 {
                    format!(", {:.0}% scanned", r.valid_fraction * 100.)
                } else {
                    String::new()
                }
            ));
        }
    }
    pub fn update_texture(&self, encoder: &ImageEncoder) {
        self.image_view.write_texture(encoder, self.background.gpu_fit());
    }
    pub fn write_and_update_texture(&mut self, encoder: &ImageEncoder) {
        let new_data = match self.line_dir {
            LineDir::Forward => &self.forward_data,
            LineDir::Backward => &self.backward_data,
        };
        if self.image_view.size() != new_data.size {
            self.image_view = self.image_view.resized(encoder, new_data.size);
        }
        // Processing works on a copy; the recorded data is never changed.
        let processed =
            processing::process(&new_data.data, new_data.size, self.background, self.filters);
        let src: &[f32] = processed.as_deref().unwrap_or(&new_data.data);
        self.image_view
            .write_lines(encoder, .., |buf| buf.copy_from_slice(src))
            .unwrap();
        self.update_texture(encoder);
    }
    pub fn clear_texture(&mut self, encoder: &ImageEncoder) {
        self.image_view.clear(encoder);
    }
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }
    #[allow(dead_code)] // for ScanArea::show_scan_line, disabled on v5
    pub fn size(&self) -> [u32; 2] {
        self.image_view.size()
    }
    /// A stamp made from saved data (an .sxm file), shown with this
    /// image's display settings.
    pub fn stamp_from_data(
        &self,
        encoder: &ImageEncoder,
        transform: DAffine2,
        forward: FrameData,
        backward: FrameData,
        unit: String,
    ) -> Self {
        let mut img = Self::new(Uuid::new_v4(), encoder, transform);
        img.std_dev = self.std_dev;
        img.background = self.background;
        img.filters = self.filters;
        img.norm_type = self.norm_type;
        img.opacity = self.opacity;
        img.line_dir = self.line_dir;
        img.forward_data = forward;
        img.backward_data = backward;
        img.unit = unit;
        img.write_and_update_texture(encoder);
        img
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
            background: self.background,
            filters: self.filters,
            norm_type: self.norm_type,
            opacity: self.opacity,
            line_dir: self.line_dir,
            forward_data: self.forward_data.clone(),
            backward_data: self.backward_data.clone(),
            unit: self.unit.clone(),
            ring: None,
            ring_label: None,
            gds_ref: None,
            coarse_ref: None,
            tip: None,
            bands: Default::default(),
            bands_queued: Default::default(),
            uuid: Uuid::new_v4(),
        }
    }
}

/// The display settings of a scan image that can be shared across stamps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplaySettings {
    pub background: Background,
    pub filters: Filters,
    pub norm_type: NormType,
    pub std_dev: f32,
    pub opacity: f32,
    pub line_dir: LineDir,
}

/// Processing, contrast and colour controls. The first image's settings are
/// shown; a change is applied to every image in `images` (only the setting
/// that was changed, so mixed selections keep their other settings).
pub fn display_controls(ui: &mut Ui, images: &mut [&mut LiveImage], encoder: &ImageEncoder) {
    let Some(first) = images.first() else { return };
    let id = first.image_view.uuid();
    let before = first.display_settings();
    let mut s = before;
    egui::Grid::new((id, "display_grid"))
        .num_columns(2)
        .spacing([8., 4.])
        .show(ui, |ui| {
            ui.label("Direction");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut s.line_dir, LineDir::Forward, "Forward");
                ui.selectable_value(&mut s.line_dir, LineDir::Backward, "Backward");
            })
            .response
            .on_hover_text("Show the forward or backward scan");
            ui.end_row();

            processing::processing_rows(ui, id, &mut s.background, &mut s.filters);

            ui.label("Contrast");
            ui.horizontal(|ui| {
                combo_box(ui, (id, "norm type"), &mut s.norm_type, &());
                if s.norm_type == NormType::StdDev {
                    ui.add(DragValue::new(&mut s.std_dev).range((0.)..=(9.)).speed(0.01));
                }
            });
            ui.end_row();

            ui.label("Opacity");
            let mut pct = s.opacity * 100.;
            if ui
                .add(egui::Slider::new(&mut pct, 5.0..=100.0).suffix(" %").fixed_decimals(0))
                .on_hover_text("See through the image to check it lines up with what is under it")
                .changed()
            {
                s.opacity = (pct / 100.).clamp(0.05, 1.);
            }
            ui.end_row();

            processing::color_scale_ui(ui);
        });

    // Undo and reset. Processing never changes the recorded data, so going
    // back only needs the earlier settings.
    let hist_id = egui::Id::new("display_settings_history");
    let mut hist: DisplayHistory = ui.data(|d| d.get_temp(hist_id)).unwrap_or_default();
    let now = ui.input(|i| i.time);
    let can_undo = images
        .iter()
        .any(|img| hist.0.get(&img.uuid()).is_some_and(|(v, _)| !v.is_empty()));
    let mut undo = false;
    let mut reset = false;
    ui.horizontal(|ui| {
        undo = ui
            .add_enabled(can_undo, egui::Button::new("\u{21b6} Undo"))
            .on_hover_text("Go back to the processing before the last change")
            .clicked();
        reset = ui
            .button("Original")
            .on_hover_text(
                "Show the image as recorded: no background removal, filters or \
                 contrast changes (Undo brings the processing back)",
            )
            .clicked();
    });
    if reset {
        s.background = Background::Raw;
        s.filters = Filters::default();
        s.norm_type = NormType::FullScale;
        s.opacity = 1.;
    }
    if undo {
        for img in images.iter_mut() {
            let entry = hist.0.entry(img.uuid()).or_default();
            if let Some(prev) = entry.0.pop() {
                let cur = img.display_settings();
                img.apply_display_change(&cur, &prev, encoder);
            }
            entry.1 = f64::NEG_INFINITY;
        }
    } else if s != before {
        for img in images.iter_mut() {
            // A drag changes the value every frame: one undo step for it.
            let entry = hist.0.entry(img.uuid()).or_default();
            if reset || now - entry.1 > 0.7 {
                entry.0.push(img.display_settings());
                if entry.0.len() > 50 {
                    entry.0.remove(0);
                }
            }
            entry.1 = if reset { f64::NEG_INFINITY } else { now };
            if reset {
                let cur = img.display_settings();
                let target = DisplaySettings {
                    background: s.background,
                    filters: s.filters,
                    norm_type: s.norm_type,
                    opacity: s.opacity,
                    ..cur
                };
                img.apply_display_change(&cur, &target, encoder);
            } else {
                img.apply_display_change(&before, &s, encoder);
            }
        }
    }
    ui.data_mut(|d| d.insert_temp(hist_id, hist));
}

/// Earlier display settings per image, for Undo, and when each was last
/// changed.
#[derive(Clone, Default)]
struct DisplayHistory(std::collections::HashMap<Uuid, (Vec<DisplaySettings>, f64)>);

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
impl FrameData {
    /// Same size and the same values bit for bit (NaN, unscanned pixels,
    /// compare equal), so a re-downloaded unchanged frame can be skipped.
    pub fn same_as(&self, other: &FrameData) -> bool {
        self.size == other.size
            && (Arc::ptr_eq(&self.data, &other.data)
                || bytemuck::cast_slice::<f32, u32>(&self.data)
                    == bytemuck::cast_slice::<f32, u32>(&other.data))
    }
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
/// Optional: only images with filters changed from the defaults have a row.
const FILTER_TABLE: TableDefinition<Uuid, [f32; 6]> =
    TableDefinition::new("liveimage_filter_table_v1");
/// Optional: only images with blob removal changed from the defaults.
const BLOB_TABLE: TableDefinition<Uuid, [f32; 2]> = TableDefinition::new("liveimage_blob_table_v1");
const NORM_TABLE: TableDefinition<Uuid, u8> = TableDefinition::new("liveimage_norm_table_v1");
/// Optional: only images drawn see-through have a row.
const OPACITY_TABLE: TableDefinition<Uuid, f32> =
    TableDefinition::new("liveimage_opacity_table_v1");
const DATA_TABLE: TableDefinition<Uuid, [FrameData; 2]> =
    TableDefinition::new("liveimage_data_table_v1");
const UNIT_TABLE: TableDefinition<Uuid, &str> = TableDefinition::new("liveimage_unit_table_v1");
/// Optional: only stamps analysed by Ring Assist have a row.
const RING_TABLE: TableDefinition<Uuid, [f64; 10]> = TableDefinition::new("liveimage_ring_table_v1");
/// Optional: only stamps the user labelled have a row.
const RING_LABEL_TABLE: TableDefinition<Uuid, [f64; 3]> =
    TableDefinition::new("liveimage_ringlabel_table_v1");
/// Optional: the GDS (uuid, transform) the stamp was taken with.
const GDS_REF_TABLE: TableDefinition<Uuid, (Uuid, [f64; 6])> =
    TableDefinition::new("liveimage_gdsref_table_v1");
/// Optional: where a live stamp was put (JSON), for coarse learning.
const COARSE_REF_TABLE: TableDefinition<Uuid, &str> =
    TableDefinition::new("liveimage_coarseref_table_v1");
/// Optional: the Tip Check result (JSON).
const TIP_TABLE: TableDefinition<Uuid, &str> = TableDefinition::new("liveimage_tip_table_v1");

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
        if fit_data.value() != Into::<u8>::into(self.background) {
            fit_data.insert(&self.background.into())?;
        }
        drop(fit_data);
        drop(fit_table);

        let mut filter_table = txn.open_table(FILTER_TABLE)?;
        let stored = filter_table.get(id)?.map(|v| v.value());
        let wanted = (!self.filters.is_default()).then(|| self.filters.to_array());
        if stored != wanted {
            match wanted {
                Some(a) => filter_table.insert(id, a)?,
                None => filter_table.remove(id)?,
            };
        }
        drop(filter_table);

        let mut blob_table = txn.open_table(BLOB_TABLE)?;
        let stored = blob_table.get(id)?.map(|v| v.value());
        let wanted = self.filters.blob_array();
        if stored != wanted {
            match wanted {
                Some(a) => blob_table.insert(id, a)?,
                None => blob_table.remove(id)?,
            };
        }
        drop(blob_table);

        let mut norm_table = txn.open_table(NORM_TABLE)?;
        let mut norm_data = norm_table.get_mut(id)?.expect("");
        if norm_data.value() != Into::<u8>::into(self.norm_type) {
            norm_data.insert(&self.norm_type.into())?;
        }
        drop(norm_data);
        drop(norm_table);

        let mut opacity_table = txn.open_table(OPACITY_TABLE)?;
        let stored = opacity_table.get(id)?.map(|v| v.value());
        let wanted = (self.opacity < 1.).then_some(self.opacity);
        if stored != wanted {
            match wanted {
                Some(a) => opacity_table.insert(id, a)?,
                None => opacity_table.remove(id)?,
            };
        }
        drop(opacity_table);

        let mut ring_table = txn.open_table(RING_TABLE)?;
        let stored = ring_table.get(id)?.map(|v| v.value());
        let wanted = self.ring.map(|r| r.to_array());
        if stored != wanted {
            match wanted {
                Some(a) => ring_table.insert(id, a)?,
                None => ring_table.remove(id)?,
            };
        }
        drop(ring_table);

        let mut label_table = txn.open_table(RING_LABEL_TABLE)?;
        let stored = label_table.get(id)?.map(|v| v.value());
        let wanted = self.ring_label.map(|l| l.to_array());
        if stored != wanted {
            match wanted {
                Some(a) => label_table.insert(id, a)?,
                None => label_table.remove(id)?,
            };
        }
        drop(label_table);

        let mut gds_table = txn.open_table(GDS_REF_TABLE)?;
        let stored = gds_table.get(id)?.map(|v| v.value());
        let wanted = self.gds_ref.map(|(u, t)| (u, t.to_cols_array()));
        if stored != wanted {
            match wanted {
                Some(v) => gds_table.insert(id, v)?,
                None => gds_table.remove(id)?,
            };
        }
        drop(gds_table);

        let mut cref_table = txn.open_table(COARSE_REF_TABLE)?;
        let stored = cref_table.get(id)?.map(|v| v.value().to_string());
        let wanted = self.coarse_ref.and_then(|c| serde_json::to_string(&c).ok());
        if stored != wanted {
            match wanted {
                Some(j) => cref_table.insert(id, j.as_str())?,
                None => cref_table.remove(id)?,
            };
        }
        drop(cref_table);

        let mut tip_table = txn.open_table(TIP_TABLE)?;
        let stored = tip_table.get(id)?.map(|v| v.value().to_string());
        let wanted = self.tip.as_ref().and_then(|t| serde_json::to_string(t).ok());
        if stored != wanted {
            match wanted {
                Some(j) => tip_table.insert(id, j.as_str())?,
                None => tip_table.remove(id)?,
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
        let mut stddev_table = txn.open_table(STD_DEV_TABLE)?;
        stddev_table.remove(id)?;
        let mut dir_table = txn.open_table(DIR_TABLE)?;
        dir_table.remove(id)?;
        let mut fit_table = txn.open_table(FIT_TABLE)?;
        fit_table.remove(id)?;
        let mut filter_table = txn.open_table(FILTER_TABLE)?;
        filter_table.remove(id)?;
        let mut blob_table = txn.open_table(BLOB_TABLE)?;
        blob_table.remove(id)?;
        let mut norm_table = txn.open_table(NORM_TABLE)?;
        norm_table.remove(id)?;
        let mut opacity_table = txn.open_table(OPACITY_TABLE)?;
        opacity_table.remove(id)?;
        let mut unit_table = txn.open_table(UNIT_TABLE)?;
        unit_table.remove(id)?;
        let mut data_table = txn.open_table(DATA_TABLE)?;
        data_table.remove(id)?;
        let mut ring_table = txn.open_table(RING_TABLE)?;
        ring_table.remove(id)?;
        let mut label_table = txn.open_table(RING_LABEL_TABLE)?;
        label_table.remove(id)?;
        let mut gds_table = txn.open_table(GDS_REF_TABLE)?;
        gds_table.remove(id)?;
        let mut cref_table = txn.open_table(COARSE_REF_TABLE)?;
        cref_table.remove(id)?;
        let mut tip_table = txn.open_table(TIP_TABLE)?;
        tip_table.remove(id)?;

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
        fit_table.insert(id, &self.background.into())?;
        if !self.filters.is_default() {
            let mut filter_table = txn.open_table(FILTER_TABLE)?;
            filter_table.insert(id, self.filters.to_array())?;
        }
        if let Some(a) = self.filters.blob_array() {
            let mut blob_table = txn.open_table(BLOB_TABLE)?;
            blob_table.insert(id, a)?;
        }
        let mut norm_table = txn.open_table(NORM_TABLE)?;
        norm_table.insert(id, &self.norm_type.into())?;
        if self.opacity < 1. {
            let mut opacity_table = txn.open_table(OPACITY_TABLE)?;
            opacity_table.insert(id, self.opacity)?;
        }
        let mut unit_table = txn.open_table(UNIT_TABLE)?;
        unit_table.insert(id, self.unit.as_str())?;
        let mut data_table = txn.open_table(DATA_TABLE)?;
        data_table.insert(id, [self.forward_data.clone(), self.backward_data.clone()])?;
        if let Some(r) = &self.ring {
            let mut ring_table = txn.open_table(RING_TABLE)?;
            ring_table.insert(id, r.to_array())?;
        }
        if let Some(l) = &self.ring_label {
            let mut label_table = txn.open_table(RING_LABEL_TABLE)?;
            label_table.insert(id, l.to_array())?;
        }
        if let Some((u, t)) = self.gds_ref {
            let mut gds_table = txn.open_table(GDS_REF_TABLE)?;
            gds_table.insert(id, (u, t.to_cols_array()))?;
        }
        if let Some(j) = self.coarse_ref.and_then(|c| serde_json::to_string(&c).ok()) {
            let mut cref_table = txn.open_table(COARSE_REF_TABLE)?;
            cref_table.insert(id, j.as_str())?;
        }
        if let Some(j) = self.tip.as_ref().and_then(|t| serde_json::to_string(t).ok()) {
            let mut tip_table = txn.open_table(TIP_TABLE)?;
            tip_table.insert(id, j.as_str())?;
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
        let stddev_table = txn.open_table(STD_DEV_TABLE)?;
        let stddev_data = stddev_table.get(id)?.unwrap().value();
        let dir_table = txn.open_table(DIR_TABLE)?;
        let dir_data = dir_table.get(id)?.unwrap().value();
        let fit_table = txn.open_table(FIT_TABLE)?;
        let fit_data = fit_table.get(id)?.unwrap().value();
        let filter_table = txn.open_table(FILTER_TABLE)?;
        let blob_table = txn.open_table(BLOB_TABLE)?;
        let filters = Filters::from_arrays(
            filter_table.get(id)?.map(|v| v.value()),
            blob_table.get(id)?.map(|v| v.value()),
        );
        let norm_table = txn.open_table(NORM_TABLE)?;
        let norm_data = norm_table.get(id)?.unwrap().value();
        let opacity_table = txn.open_table(OPACITY_TABLE)?;
        let opacity = opacity_table.get(id)?.map(|v| v.value()).unwrap_or(1.);
        let unit_table = txn.open_table(UNIT_TABLE)?;
        let unit_data = unit_table.get(id)?.unwrap().value().to_string();
        let data_table = txn.open_table(DATA_TABLE)?;
        let frame_data = data_table.get(id)?.unwrap().value();
        let ring_table = txn.open_table(RING_TABLE)?;
        let ring = ring_table.get(id)?.map(|v| RingStamp::from_array(v.value()));
        let label_table = txn.open_table(RING_LABEL_TABLE)?;
        let ring_label = label_table.get(id)?.map(|v| RingLabel::from_array(v.value()));
        let gds_table = txn.open_table(GDS_REF_TABLE)?;
        let gds_ref = gds_table
            .get(id)?
            .map(|v| v.value())
            .map(|(u, t)| (u, DAffine2::from_cols_array(&t)));
        let cref_table = txn.open_table(COARSE_REF_TABLE)?;
        let coarse_ref = cref_table.get(id)?.and_then(|v| serde_json::from_str(v.value()).ok());
        let mut image = Self::new(id, encoder, DAffine2::from_cols_array(&tran_data));
        image.coarse_ref = coarse_ref;
        let tip_table = txn.open_table(TIP_TABLE)?;
        image.tip = tip_table.get(id)?.and_then(|v| serde_json::from_str(v.value()).ok());
        image.std_dev = stddev_data;
        image.line_dir = LineDir::try_from(dir_data).unwrap();
        image.background = Background::try_from(fit_data).unwrap_or(Background::Raw);
        image.filters = filters;
        image.norm_type = NormType::try_from(norm_data).unwrap();
        image.opacity = opacity;
        image.unit = unit_data;
        image.ring = ring;
        image.ring_label = ring_label;
        image.gds_ref = gds_ref;
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
        let ring_table_len = txn.open_table(RING_TABLE)?.len()?;
        println!("  ring table: {ring_table_len} items");
        let label_table_len = txn.open_table(RING_LABEL_TABLE)?.len()?;
        println!("  ring label table: {label_table_len} items");
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
