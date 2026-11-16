use std::sync::Arc;

use eframe::egui_wgpu::RenderState;
use egui::{Id, Ui};
use glam::{DAffine2, DVec2};
use image_compute::image_compute::ImageComputePipeline;

pub mod border;
mod callbacks;
mod file_image;
pub mod gds_image;
pub mod processing;
mod scale_bar;
mod scan_image;
pub mod static_image;
mod view;

pub use border::BorderRectangle;
pub use file_image::FileImage;
pub use gds_image::GDSImage;
pub use scale_bar::ScaleBar;
pub use scan_image::ScanViewImage;
pub use view::ScanView;
// TODO: remove this pub use
pub use view::ScanViewCtx;

use crate::utils::vec_interop::{IntoGlam as _, Projection as _};

#[derive(Clone)]
pub struct ImageEncoder {
    pipeline: Arc<ImageComputePipeline>,
    wgpu_state: RenderState,
}
impl ImageEncoder {
    pub fn new(wgpu_state: &RenderState) -> Self {
        let pipeline = Arc::new(ImageComputePipeline::new(&wgpu_state.device));
        Self {
            pipeline,
            wgpu_state: wgpu_state.clone(),
        }
    }
}

pub fn world_delta_transform(ui: &Ui) -> [DAffine2; 3] {
    let ctx = ui
        .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
        .unwrap();
    let center = ctx
        .world2egui()
        .inverse()
        .transform_point2(ctx.rect.center().to_glam());
    let screen_to_world = ctx.world2egui().inverse();
    let response = ctx.screen_response;
    // Calculate the dragging transform
    let drag = if response.dragged_by(egui::PointerButton::Primary) {
        let screen_drag = response.drag_delta().to_glam();
        let world_drag = screen_to_world.project_vec2(screen_drag);
        DAffine2::from_translation(world_drag)
    } else {
        DAffine2::IDENTITY
    };
    // Calculate the rotation transform
    let rotate = if response.dragged_by(egui::PointerButton::Secondary) {
        let cursor_pos =
            screen_to_world.project_pos2(response.interact_pointer_pos().unwrap().to_glam());
        let cursor_vec = cursor_pos - center;
        let drag_vec = screen_to_world.project_vec2(response.drag_delta().to_glam());
        let angle = cursor_vec.perp_dot(drag_vec) / cursor_vec.length_squared();

        let trans = DAffine2::from_translation(center);
        let rot = DAffine2::from_angle(angle as f64);
        trans * rot * trans.inverse()
    } else {
        DAffine2::IDENTITY
    };

    // Calculate the Zooming transform
    let zoom = {
        let scalar = (ui.input(|is| is.raw_scroll_delta).y / 100.).exp() as f64;
        let scale = DAffine2::from_scale(DVec2::splat(scalar).into());
        let trans = DAffine2::from_translation(center);
        trans * scale * trans.inverse()
    };
    [rotate, zoom, drag]
}
