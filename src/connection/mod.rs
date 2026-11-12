pub mod backing;
mod live_image;
pub mod queue;
mod scan_area;
mod shared_state;

use egui::Ui;
pub use live_image::{FrameData, LiveImage};
pub use scan_area::ScanArea;

use crate::{components::selectable_list::SelectableList, scan_view::ImageEncoder, view_object};
pub mod nanonis;

pub trait Connection {
    fn poll_connected(
        &mut self,
        object_list: &mut SelectableList<view_object::Object>,
        encoder: &ImageEncoder,
    ) -> bool;
    fn update(
        &mut self,
        object_list: &mut SelectableList<view_object::Object>,
        encoder: &ImageEncoder,
    );
    fn show_menu(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<view_object::Object>,
        encoder: &ImageEncoder,
    );
    fn show_image_view_overlay(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<view_object::Object>,
    );
    /// Summary for the status bar.
    fn status(&self) -> StatusInfo {
        StatusInfo::default()
    }
}

/// What the status bar shows about a live connection.
#[derive(Clone, Debug, Default)]
pub struct StatusInfo {
    /// e.g. "Scanning ⬇", "Paused", "Idle".
    pub scan: String,
    /// Volts.
    pub bias: Option<f32>,
    /// Amps.
    pub current: Option<f32>,
    /// Tip position in nm (map coordinates).
    pub tip_nm: Option<glam::DVec2>,
}
