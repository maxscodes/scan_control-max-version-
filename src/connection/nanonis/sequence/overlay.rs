use egui::{Align2, Color32, FontId, Id, Ui};
use glam::{DAffine2, DVec2, IVec2};

use crate::{
    connection::{
        nanonis::{
            course_motion::CourseMotionState,
            sequence::{
                editor::{DragTarget, SequenceState},
                model::{compile, CoarseSpec, Dir, Frame, Position, Step},
            },
        },
        ScanArea,
    },
    scan_view::{world_delta_transform, BorderRectangle, ScanViewCtx},
    utils::vec_interop::IntoEgui as _,
};

const PLAN_COLOR: Color32 = Color32::from_rgb(0xd0, 0x5c, 0xe3);
/// The entry being edited by dragging.
const DRAG_COLOR: Color32 = Color32::from_rgb(0xff, 0x9c, 0xff);

impl SequenceState {
    /// Draw where each remaining scan will happen: a numbered dashed outline
    /// per Scan step, offset by the coarse moves predicted before it. In drag
    /// mode, ctrl+drag moves the selected entry's frame or coarse move.
    pub fn show_overlay(&mut self, ui: &mut Ui, scan_area: &ScanArea, course: &CourseMotionState) {
        if !self.window_open && self.run.is_none() {
            return;
        }
        // Hidden after finishing, until the sequence or selection changes.
        if let Some((seq, sel)) = &self.plan_hidden {
            if self.run.is_none() && *seq == self.seq && *sel == self.selected {
                return;
            }
            self.plan_hidden = None;
        }
        let Some(transform) = scan_area.image_transform else {
            return;
        };
        let editing = self.run.is_none();
        let drag_entry = self
            .drag
            .and(self.selected)
            .filter(|&i| editing && i < self.seq.entries.len());

        // Apply this frame's drag before drawing so the outline follows the mouse.
        if let Some(i) = drag_entry {
            if !course.menu_open() {
                self.apply_drag(ui, i, scan_area, course, Frame::from_transform(transform));
            }
        }

        let (steps, start) = match &self.run {
            Some(run) => (run.steps.clone(), run.current),
            None => (compile(&self.seq, 0, Dir::Down), 0),
        };
        let steps2real = course.steps2real_world();
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let world2screen = ctx.world2egui();

        let mut frame = Frame::from_transform(transform);
        let mut offset = DVec2::ZERO;
        for &(entry, step) in steps.iter().skip(start) {
            match step {
                Step::CoarseMove { steps } => {
                    offset += steps2real * IVec2::from_array(steps).as_dvec2();
                }
                Step::SetFrame(spec) => frame = spec.apply(frame),
                Step::Scan(_) => {
                    let dragging = drag_entry == Some(entry);
                    let color = if dragging { DRAG_COLOR } else { PLAN_COLOR };
                    let area_offset =
                        scan_area.world_transform * DAffine2::from_translation(offset);
                    let transform = area_offset * frame.to_transform();
                    BorderRectangle {
                        transform,
                        color,
                        dashed: !dragging,
                    }
                    .show(ui);
                    if dragging && self.drag == Some(DragTarget::Coarse) {
                        // Show the whole scan range after the coarse move.
                        BorderRectangle {
                            transform: area_offset * DAffine2::from_scale(scan_area.area_size),
                            color,
                            dashed: true,
                        }
                        .show(ui);
                    }
                    let corner = (world2screen * transform)
                        .transform_point2(DVec2::new(-0.5, -0.5))
                        .to_egui_pos2();
                    ui.painter().text(
                        corner,
                        Align2::LEFT_BOTTOM,
                        format!("{}", entry + 1),
                        FontId::proportional(14.),
                        color,
                    );
                }
                _ => {}
            }
        }
    }

    /// Frame at the start of entry `target`, before its own SetFrame.
    fn frame_before(&self, target: usize, current: Frame) -> Frame {
        let mut frame = current;
        for (entry, step) in compile(&self.seq, 0, Dir::Down) {
            if entry >= target {
                break;
            }
            if let Step::SetFrame(spec) = step {
                frame = spec.apply(frame);
            }
        }
        frame
    }

    fn apply_drag(
        &mut self,
        ui: &Ui,
        i: usize,
        scan_area: &ScanArea,
        course: &CourseMotionState,
        current: Frame,
    ) {
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let ctrl = ui.input(|i| i.modifiers.ctrl);
        if !(ctrl && ctx.screen_response.dragged_by(egui::PointerButton::Primary)) {
            self.coarse_drag_real = None;
            return;
        }
        let [_, _, translate] = world_delta_transform(ui);
        // World delta -> scan-area (piezo, nm) coordinates.
        let delta = scan_area
            .world_transform
            .inverse()
            .transform_vector2(translate.translation);
        if delta == DVec2::ZERO {
            return;
        }
        match self.drag {
            Some(DragTarget::Frame) => {
                let base = self.frame_before(i, current);
                let e = &mut self.seq.entries[i];
                e.frame.position = match e.frame.position {
                    Position::Keep => {
                        Position::At([base.center_x + delta.x, base.center_y + delta.y])
                    }
                    Position::At([x, y]) => Position::At([x + delta.x, y + delta.y]),
                    Position::Shift([x, y]) => Position::Shift([x + delta.x, y + delta.y]),
                };
            }
            Some(DragTarget::Coarse) => {
                let steps2real = course.steps2real_world();
                let e = &mut self.seq.entries[i];
                let mut c = e.coarse.unwrap_or(CoarseSpec::default());
                // Accumulate in nm so small drags add up to whole steps.
                let real = self
                    .coarse_drag_real
                    .unwrap_or(steps2real * IVec2::from_array(c.steps).as_dvec2())
                    + delta;
                self.coarse_drag_real = Some(real);
                c.steps = course.real_to_steps(real).to_array();
                e.coarse = Some(c);
            }
            None => {}
        }
    }
}
