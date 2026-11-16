//! Display types shared by scan images: background fit and contrast choice.

use image_compute::image_compute::{FitType, NormalizationType};

use crate::components::combo_box::ComboBoxType;

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
