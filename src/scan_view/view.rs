
use eframe::{egui_wgpu::Callback, wgpu::TextureFormat};
use egui::{Color32, Id, Response, Ui};
use glam::{DAffine2, DVec2, Vec2};

use crate::{
    app::COLOR_MAP_SIZE,
    scan_view::{callbacks::GlobalCallback, processing::ColorMap, ImageEncoder},
    utils::vec_interop::IntoGlam as _,
};

// #[derive(Clone)]
pub struct ScanView {
    pub world_transform: DAffine2,
    target_format: TextureFormat,
    new_color_map: Option<Box<[egui::Color32; COLOR_MAP_SIZE]>>,
    /// Palette last sent to the GPU.
    applied_color_map: Option<(ColorMap, bool)>,
}
impl ScanView {
    pub fn show<R>(
        &mut self,
        ui: &mut egui::Ui,
        add_contents: impl FnOnce(&mut Ui) -> R,
    ) -> Response {
        let prefs = crate::theme::prefs();
        let wanted = (prefs.color_map, prefs.color_invert);
        if self.applied_color_map != Some(wanted) {
            self.new_color_map = Some(wanted.0.table(wanted.1));
            self.applied_color_map = Some(wanted);
        }
        let fill = if ui.style().visuals.dark_mode {
            Color32::from_rgb(0x0e, 0x10, 0x13)
        } else {
            Color32::from_rgb(0xdf, 0xe3, 0xe8)
        };
        egui::Frame::canvas(ui.style())
            .fill(fill)
            .inner_margin(0)
            .corner_radius(0)
            .outer_margin(0)
            .show(ui, |ui| {
                let (rect, response) =
                    ui.allocate_at_least(ui.available_size_before_wrap(), egui::Sense::all());
                let screen_transform = self.handle_inputs(ui, response.clone());
                ui.painter().add(Callback::new_paint_callback(
                    rect,
                    GlobalCallback {
                        target_format: self.target_format,
                        screen_transform,
                        new_color_map: std::mem::take(&mut self.new_color_map),
                    },
                ));
                let ctx = ScanViewCtx {
                    rect,
                    world_transform: self.world_transform,
                    screen_response: response.clone(),
                };
                ui.data_mut(|map| map.insert_temp(Id::new(()), ctx));
                add_contents(ui);
                ui.data_mut(|map| map.remove::<ScanViewCtx>(Id::new(())));
                response
            })
            .inner
    }
    fn handle_inputs(&mut self, ui: &mut egui::Ui, response: egui::Response) -> DAffine2 {
        // update the world transform using the calculated transforms
        if ui.input(|i| !i.modifiers.ctrl) {
            let tf = transform_from_response(&response, ui);
            self.world_transform = tf * self.world_transform;
        }

        // calculate the screen transform
        let rect = response.rect;
        let screen_transform =
            DAffine2::from_scale(rect.size().to_glam() * DVec2::new(0.5, -0.5)).inverse();

        screen_transform * self.world_transform
    }
    pub fn new(image_encoder: &ImageEncoder) -> Self {
        let mut world_transform = DAffine2::IDENTITY;
        world_transform.matrix2.y_axis[1] = -1.0;
        Self {
            // Set from the preferences on the first `show`.
            new_color_map: None,
            applied_color_map: None,
            world_transform,
            target_format: image_encoder.wgpu_state.target_format,
        }
    }
}

#[derive(Clone)]
pub struct ScanViewCtx {
    pub rect: egui::Rect,
    pub world_transform: DAffine2,
    pub screen_response: Response,
}

impl ScanViewCtx {
    pub fn world2egui(&self) -> DAffine2 {
        DAffine2::from_translation(self.rect.center().to_glam()) * self.world_transform
    }
}

pub fn transform_from_response(response: &Response, ui: &Ui) -> DAffine2 {
    let rect = response.rect;
    // Calculate the dragging transform
    let drag = if response.dragged_by(egui::PointerButton::Primary) {
        DAffine2::from_translation(response.drag_delta().to_glam())
    } else {
        DAffine2::IDENTITY
    };
    // Calculate the rotation transform
    let rotate = if response.dragged_by(egui::PointerButton::Secondary) {
        let cursor_pos = (response.interact_pointer_pos().unwrap() - rect.center()).to_glam();
        let drag_vec = response.drag_delta().to_glam();
        let angle = cursor_pos.perp_dot(drag_vec) / cursor_pos.length_squared();
        DAffine2::from_angle(angle as f64)
    } else {
        DAffine2::IDENTITY
    };

    // Calculate the Zooming transform
    let zoom = if let Some(window_pos) = response.hover_pos() {
        let scalar = (ui.input(|is| is.smooth_scroll_delta).y / 100.).exp();
        let scale = DAffine2::from_scale(Vec2::splat(scalar).into());
        let trans = DAffine2::from_translation((window_pos - rect.center()).to_glam());
        trans * scale * trans.inverse()
    } else {
        DAffine2::IDENTITY
    };

    rotate * zoom * drag
}
