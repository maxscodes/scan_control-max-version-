pub mod backing;
mod live_image;
pub mod queue;
mod scan_area;
mod shared_state;

use egui::Ui;
pub use live_image::LiveImage;
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
}
