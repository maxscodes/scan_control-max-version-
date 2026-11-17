use std::{ops::RangeBounds, sync::Arc};

use eframe::egui_wgpu;
use egui::{mutex::RwLock, Id, Response, Sense, Ui};
use glam::DAffine2;
use image_compute::{
    buffers::BufferOpError,
    image_compute::{FitData, FitType, ImageComputeBuffers, NormalizationType, NormalizeData},
};
use uuid::Uuid;

use crate::{
    scan_view::{callbacks::ImageCallback, view::ScanViewCtx, ImageEncoder},
    utils::{
        response_group::neutral_response,
        vec_interop::{IntoGlam as _, Projection},
    },
};

pub struct ScanViewImage {
    uuid: Uuid,
    pub transform: DAffine2,
    pub norm_type: NormalizationType,
    /// 0..1, how opaque the image is drawn on the map.
    pub opacity: f32,
    image_buffers: ImageComputeBuffers,
    pub fit_data: Arc<RwLock<Option<FitData>>>,
    pub norm_data: Arc<RwLock<Option<NormalizeData>>>,
}
impl ScanViewImage {
    pub fn new(
        uuid: Uuid,
        image_encoder: &ImageEncoder,
        size: [u32; 2],
        transform: DAffine2,
        norm_type: NormalizationType,
    ) -> Self {
        let image_buffers = ImageComputeBuffers::new(
            &image_encoder.wgpu_state.device,
            &image_encoder.wgpu_state.queue,
            None,
            size,
        );
        Self {
            uuid,
            transform,
            image_buffers,
            norm_type,
            opacity: 1.,
            fit_data: Arc::new(RwLock::new(None)),
            norm_data: Arc::new(RwLock::new(None)),
        }
    }
    pub fn resized(&self, image_encoder: &ImageEncoder, size: [u32; 2]) -> Self {
        let mut img = Self::new(self.uuid, image_encoder, size, self.transform, self.norm_type);
        img.opacity = self.opacity;
        img
    }
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }
    pub fn show(&self, ui: &mut Ui) -> Response {
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let resp = ui
            .input(|i| i.pointer.latest_pos())
            .and_then(|pos| {
                let [x, y] = (ctx.world2egui() * self.transform)
                    .inverse()
                    .project_pos2(pos.to_glam())
                    .abs()
                    .into();
                (x < 0.5 && y < 0.5).then(|| {
                    ui.interact(
                        ctx.rect,
                        egui::Id::new(self.uuid),
                        Sense::focusable_noninteractive() | Sense::click(),
                    )
                })
            })
            .unwrap_or_else(|| neutral_response(ui, egui::Id::new(self.uuid)));
        let callback = egui_wgpu::Callback::new_paint_callback(
            ctx.rect,
            ImageCallback {
                norm_type: self.norm_type,
                opacity: self.opacity,
                transform: self.transform,
                image_buffers: self.image_buffers.clone(),
            },
        );
        ui.painter().add(callback);
        resp
    }
    pub fn write_texture(&self, image_encoder: &ImageEncoder, fit_type: FitType) {
        let mut encoder = image_encoder.wgpu_state.device.create_command_encoder(
            &wgpu::wgt::CommandEncoderDescriptor {
                label: Some("write texture"),
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            image_encoder.pipeline.dispatch(
                &image_encoder.wgpu_state.device,
                &mut pass,
                &self.image_buffers,
                fit_type,
            );
        }
        image_encoder.wgpu_state.queue.submit([encoder.finish()]);
        {
            let norm_data = self.norm_data.clone();
            let fit_data = self.fit_data.clone();
            self.image_buffers
                .download_normalize_data(
                    &image_encoder.wgpu_state.device,
                    &image_encoder.wgpu_state.queue,
                    move |data| *norm_data.write() = Some(data),
                )
                .ok();
            self.image_buffers
                .download_fit_data(
                    &image_encoder.wgpu_state.device,
                    &image_encoder.wgpu_state.queue,
                    fit_type,
                    move |data| *fit_data.write() = Some(data),
                )
                .ok();
        }
    }
    pub fn clear(&self, image_encoder: &ImageEncoder) {
        let mut encoder = image_encoder
            .wgpu_state
            .device
            .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            image_encoder.pipeline.dispatch_clear_texture(
                &image_encoder.wgpu_state.device,
                &mut pass,
                &self.image_buffers,
            );
        }
        image_encoder.wgpu_state.queue.submit([encoder.finish()]);
        *self.norm_data.write() = None;
        *self.fit_data.write() = None;
    }
    pub fn write_lines(
        &self,
        image_encoder: &ImageEncoder,
        lines: impl RangeBounds<u32>,
        callback: impl Fn(&mut [f32]),
    ) -> Result<(), BufferOpError> {
        self.image_buffers
            .write_lines_range(&image_encoder.wgpu_state.queue, lines, callback)
    }
    pub fn size(&self) -> [u32; 2] {
        self.image_buffers.size()
    }
}
