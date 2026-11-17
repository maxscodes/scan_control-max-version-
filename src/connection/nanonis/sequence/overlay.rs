use egui::{Align2, Color32, FontId, Id, Stroke, Ui};
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
        if self.plan_off || (!self.window_open && self.run.is_none()) {
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
                if self.drag == Some(DragTarget::Tip) {
                    self.apply_tip_edit(ui, i, scan_area, course);
                } else {
                    self.apply_drag(ui, i, scan_area, course, Frame::from_transform(transform));
                }
            }
        }

        let (steps, start) = match &self.run {
            Some(run) => (run.steps.clone(), run.current),
            None => (compile(&self.seq, 0, Dir::Down), 0),
        };
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let world2screen = ctx.world2egui();

        let mut frame = Frame::from_transform(transform);
        let mut offset = DVec2::ZERO;
        // Tip positions: (entry, number within the entry, last point on screen).
        let mut tip_count: Option<(usize, usize, egui::Pos2)> = None;
        for (entry, step) in steps.iter().skip(start) {
            let entry = *entry;
            match step {
                Step::MoveTip(p) => {
                    let color = if drag_entry == Some(entry) { DRAG_COLOR } else { PLAN_COLOR };
                    let to_screen = world2screen * scan_area.world_transform * DAffine2::from_translation(offset);
                    let at = to_screen.transform_point2(DVec2::from_array(*p)).to_egui_pos2();
                    let k = match tip_count {
                        Some((e, k, last)) if e == entry => {
                            ui.painter().line_segment([last, at], Stroke::new(1., color.gamma_multiply(0.6)));
                            k + 1
                        }
                        _ => 1,
                    };
                    tip_count = Some((entry, k, at));
                    let painter = ui.painter();
                    painter.circle_stroke(at, 5., Stroke::new(2., color));
                    painter.circle_filled(at, 1.5, color);
                    painter.text(
                        at + egui::vec2(6., -4.),
                        Align2::LEFT_BOTTOM,
                        format!("{}.{k}", entry + 1),
                        FontId::proportional(12.),
                        color,
                    );
                }
                Step::CoarseMove { steps } => {
                    offset += self.predict_move(IVec2::from_array(*steps), course);
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

    /// Where a coarse move of the run will take the tip (piezo nm): with
    /// what a tracked survey grid has learned so far, else the coarse
    /// calibration. Tiles already done stay where they are; this only
    /// places the ones still to come.
    fn predict_move(&self, steps: IVec2, course: &CourseMotionState) -> DVec2 {
        match &self.grid_track {
            Some(t) if self.run.is_some() && self.survey.track => t.predict(steps),
            _ => course.steps_to_real(steps),
        }
    }

    /// Coarse offset (nm) in effect for entry `target`'s tip moves: every
    /// coarse move up to and including its own.
    fn offset_through(&self, target: usize, course: &CourseMotionState) -> DVec2 {
        compile(&self.seq, 0, Dir::Down)
            .into_iter()
            .take_while(|(entry, _)| *entry <= target)
            .filter_map(|(_, step)| match step {
                Step::CoarseMove { steps } => Some(course.steps_to_real(IVec2::from_array(steps))),
                _ => None,
            })
            .sum()
    }

    /// Ctrl+click adds a tip position (or removes the one under the mouse),
    /// ctrl+drag moves one.
    fn apply_tip_edit(&mut self, ui: &Ui, i: usize, scan_area: &ScanArea, course: &CourseMotionState) {
        /// Pick radius on screen, points.
        const PICK: f64 = 10.;
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();
        let resp = &ctx.screen_response;
        if !ui.input(|i| i.modifiers.ctrl) {
            self.tip_drag = None;
            return;
        }
        let to_screen = ctx.world2egui()
            * scan_area.world_transform
            * DAffine2::from_translation(self.offset_through(i, course));
        let Some(pointer) = resp.interact_pointer_pos().or(resp.hover_pos()) else {
            return;
        };
        let pointer = DVec2::new(pointer.x as f64, pointer.y as f64);
        let at = to_screen.inverse().transform_point2(pointer);
        let points = &mut self.seq.entries[i].tip.points;
        let nearest = points
            .iter()
            .enumerate()
            .map(|(k, p)| (k, to_screen.transform_point2(DVec2::from_array(*p)).distance(pointer)))
            .filter(|(_, d)| *d <= PICK)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(k, _)| k);
        if resp.drag_started_by(egui::PointerButton::Primary) {
            self.tip_drag = nearest;
        }
        if resp.dragged_by(egui::PointerButton::Primary) {
            if let Some(p) = self.tip_drag.and_then(|k| points.get_mut(k)) {
                *p = at.to_array();
            }
        } else {
            self.tip_drag = None;
        }
        if resp.clicked() {
            match nearest {
                Some(k) => {
                    points.remove(k);
                }
                None => points.push(at.to_array()),
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
                let e = &mut self.seq.entries[i];
                let mut c = e.coarse.unwrap_or(CoarseSpec::default());
                // Accumulate in nm so small drags add up to whole steps.
                let real = self
                    .coarse_drag_real
                    .unwrap_or(course.steps_to_real(IVec2::from_array(c.steps)))
                    + delta;
                self.coarse_drag_real = Some(real);
                c.steps = course.real_to_steps(real).to_array();
                e.coarse = Some(c);
            }
            Some(DragTarget::Tip) | None => {}
        }
    }
}
