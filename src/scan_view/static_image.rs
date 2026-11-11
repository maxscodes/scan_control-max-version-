use core::f32;

use egui::{DragValue, Response, Ui};
use glam::DAffine2;
use image_compute::image_compute::{FitData, FitType, NormalizationType};
use itertools::izip;
use nanonis_tcp::LineDir;
use uuid::Uuid;

use crate::{
    components::{
        combo_box::{combo_box, ComboBoxType},
        EngFmt,
    },
    connection::backing::BufferState,
    scan_view::{ImageEncoder, ScanViewImage},
};

pub struct StaticImage {
    image_view: ScanViewImage,
    channel: String,
    pub buffers: BufferState,
    pub line_dir: LineDir,
    pub transform: DAffine2,
    pub fit_type: FitType,
    pub norm_type: NormType,
    pub std_dev: f32,
    pub name: String,
}
impl StaticImage {
    pub fn new(
        encoder: &ImageEncoder,
        transform: DAffine2,
        channel: String,
        buffers: BufferState,
    ) -> Self {
        let image_data = ScanViewImage::new(
            Uuid::new_v4(),
            encoder,
            [buffers.size[1] as u32, buffers.size[0] as u32],
            transform,
            NormalizationType::FullScale,
        );
        Self {
            line_dir: LineDir::Forward,
            image_view: image_data,
            fit_type: FitType::MeanSubtract,
            norm_type: NormType::FullScale,
            std_dev: 1.5,
            transform,
            name: String::new(),
            channel,
            buffers,
        }
    }
    pub fn size(&self) -> [u32; 2] {
        self.image_view.size()
    }
    pub fn update_texture(&self, image_encoder: &ImageEncoder) {
        let src = match self.line_dir {
            LineDir::Forward => &self.buffers.buf_f,
            LineDir::Backward => &self.buffers.buf_b,
        };
        self.image_view
            .write_lines(image_encoder, .., |buf| buf.copy_from_slice(src))
            .unwrap();
        self.image_view.write_texture(image_encoder, self.fit_type);
    }
    pub fn show(&mut self, ui: &mut Ui) -> Response {
        self.image_view.norm_type = self.norm_type.combined(self.std_dev);
        self.image_view.transform = self.transform;
        self.image_view.show(ui)
    }
    pub fn show_image_menu(&mut self, ui: &mut Ui, image_encoder: &mut ImageEncoder) {
        let vis = &mut ui.style_mut().visuals.widgets.inactive;
        vis.weak_bg_fill = vis.weak_bg_fill.gamma_multiply(0.5);
        if combo_box(
            ui,
            (self.image_view.uuid(), "fit type"),
            &mut self.fit_type,
            &(),
        ) {
            self.update_texture(image_encoder);
        }
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
        if ui.button(self.line_dir.opt_atoms(&())).clicked() {
            match self.line_dir {
                LineDir::Forward => self.line_dir = LineDir::Backward,
                LineDir::Backward => self.line_dir = LineDir::Forward,
            }
            self.update_texture(image_encoder);
        }
        ui.label(format!("Channel: {}", self.channel));
        let norm = self.image_view.norm_data.read();
        let fit = self.image_view.fit_data.read();
        if let (Some(norm), Some(fit)) = (norm.as_ref(), fit.as_ref()) {
            let mean = fit.mean();
            ui.label(format!("Max:     {:.2}", EngFmt(norm.max)));
            ui.label(format!("Gap:    {:.2}", EngFmt(norm.max - norm.min)));
            ui.label(format!("Min:     {:.2}", EngFmt(norm.min)));
            ui.label(format!("Mean:    {:.2}", EngFmt(mean)));
            ui.label(format!("Std Dev: {:.2}", EngFmt(norm.stddev)));
            match fit {
                FitData::PlaneFitSubtract {
                    x_slope, y_slope, ..
                } => {
                    ui.label(format!("X Slope: {:.2}", EngFmt(*x_slope)));
                    ui.label(format!("Y Slope: {:.2}", EngFmt(*y_slope)));
                }
                FitData::MeanSubtract { .. } => {}
                FitData::LineMeanSubtract { means } => {
                    for m in means {
                        ui.label(format!("{:.2}", EngFmt(*m)));
                    }
                }
                FitData::LineFitSubtract { means, slopes } => {
                    for (m, s) in izip!(means, slopes) {
                        ui.label(format!("{:.2}  {:.2}", EngFmt(*m), EngFmt(*s)));
                    }
                }
            }
        }
    }
}

impl ComboBoxType for FitType {
    type Ctx = ();

    fn opt_atoms(&self, _ctx: &()) -> impl Into<egui::WidgetText> {
        match self {
            FitType::MeanSubtract => "Raw",
            FitType::LineMeanSubtract => "Subtract Average",
            FitType::LineFitSubtract => "Subtract Linear Fit",
            FitType::PlaneFitSubtract => "Subtract Plane",
        }
    }

    fn options(_ctx: &()) -> impl Iterator<Item = Self> {
        [
            FitType::MeanSubtract,
            FitType::LineMeanSubtract,
            FitType::LineFitSubtract,
            FitType::PlaneFitSubtract,
        ]
        .into_iter()
    }
}

#[derive(
    Debug, PartialEq, Eq, Clone, Copy, num_enum::TryFromPrimitive, num_enum::IntoPrimitive,
)]
#[repr(u8)]
pub enum NormType {
    FullScale = 0,
    StdDev = 1,
}
impl NormType {
    pub fn combined(&self, std_dev: f32) -> NormalizationType {
        match self {
            NormType::FullScale => image_compute::image_compute::NormalizationType::FullScale,
            NormType::StdDev => image_compute::image_compute::NormalizationType::StdDev(std_dev),
        }
    }
}
impl ComboBoxType for NormType {
    type Ctx = ();
    fn opt_atoms(&self, _ctx: &()) -> impl Into<egui::WidgetText> {
        match self {
            NormType::FullScale => "Full Scale",
            NormType::StdDev => "Std Dev",
        }
    }

    fn options(_ctx: &()) -> impl Iterator<Item = Self> {
        [NormType::FullScale, NormType::StdDev].into_iter()
    }
}

// pub fn combo_box<'s, T: Eq + Copy>(
//     ui: &mut Ui,
//     id_salt: impl Hash,
//     types: &[(T, &'s str)],
//     data: &mut T,
// ) -> bool {
//     ComboBox::new((id_salt, "combo_box"), "")
//         .selected_text(
//             types
//                 .iter()
//                 .find(|(t, _)| *t == *data)
//                 .map(|(_, name)| *name)
//                 .unwrap_or(""),
//         )
//         .show_ui(ui, |ui| {
//             let clicked = types
//                 .iter()
//                 .map(|(typ, name)| ui.selectable_value(data, *typ, *name))
//                 .any(|resp| resp.clicked());
//             clicked
//         })
//         .inner
//         .is_some_and(|clicked| clicked)
// }
