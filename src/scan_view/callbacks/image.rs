use eframe::egui_wgpu::{self, CallbackTrait};
use glam::DAffine2;
use image_compute::image_compute::{ImageComputeBuffers, NormalizationType};

use crate::scan_view::callbacks::global::GlobalResources;

pub struct ImageCallback {
    pub transform: DAffine2,
    pub norm_type: NormalizationType,
    /// 0..1
    pub opacity: f32,
    pub image_buffers: ImageComputeBuffers,
}
impl CallbackTrait for ImageCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Set the new world transform
        self.image_buffers
            .write_world_transform(queue, self.transform);
        self.image_buffers
            .write_display(queue, self.norm_type, self.opacity)
            .unwrap();
        vec![]
    }
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let global_res = callback_resources
            .get::<GlobalResources>()
            .expect("GlobalResources not initialized");

        // Draw the image to the screen
        global_res.scan_image_pipeline.draw(
            render_pass,
            &self.image_buffers,
            &global_res.scan_image_buffers,
        );
    }
}
