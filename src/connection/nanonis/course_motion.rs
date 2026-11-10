use core::f64;
use std::{
    fs::File,
    io::{Read, Seek, Write},
    path::Path,
};

use egui::{
    epaint::PathStroke, Button, CollapsingHeader, Color32, DragValue, Frame, Id, Shadow, Shape,
    Stroke, Ui, Vec2,
};
use glam::{DAffine2, DMat2, DVec2, IVec2};
use itertools::Itertools as _;
use tracing::info;

use crate::{
    app::CONFIG_DIR,
    components::{selectable_list::SelectableList, si_drag::si_drag_value},
    connection::{
        nanonis::command_channel::CommandChannelSender, shared_state::SharedState, ScanArea,
    },
    scan_view::{world_delta_transform, BorderRectangle, ScanViewCtx},
    utils::vec_interop::IntoEgui as _,
    view_object::Object,
};

// #[derive(Clone)]
pub struct CourseMoveCommand {
    pub steps: IVec2,
    pub group: u32,
    pub auto_withdraw: bool,
    pub auto_approach: bool
}

pub struct CourseMotionState {
    menu_active: bool,
    move_target: DVec2,
    calib_matrix: DMat2,
    voltages: SharedState<DVec2>,
    move_sender: CommandChannelSender<CourseMoveCommand, ()>,
    currently_moving: bool,
    group: u32,
    last_x_move: Option<(DVec2, IVec2)>,
    last_y_move: Option<(DVec2, IVec2)>,
    cfg_file: File,
    auto_withdraw: bool,
    auto_approach: bool
}
impl CourseMotionState {
    pub fn new(
        voltages: &SharedState<DVec2>,
        move_sender: &CommandChannelSender<CourseMoveCommand, ()>,
    ) -> Self {
        let mut cfg_path = CONFIG_DIR.config_local_dir().to_path_buf();
        cfg_path.push("course_matrix.bin");
        let mut cfg_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(cfg_path)
            .unwrap();
        let mut buf = Vec::new();
        cfg_file.read_to_end(&mut buf).unwrap();
        let calib_matrix = if buf.is_empty() {
            let calib_matrix = DMat2::IDENTITY * 1e-6;
            cfg_file
                .write_all(bytemuck::bytes_of(&calib_matrix))
                .unwrap();
            calib_matrix
        } else {
            *bytemuck::from_bytes(&buf)
        };

        Self {
            menu_active: false,
            move_target: DVec2::ZERO,
            calib_matrix,
            voltages: voltages.clone(),
            move_sender: move_sender.clone(),
            currently_moving: false,
            group: 1,
            last_x_move: None,
            last_y_move: None,
            cfg_file,
            auto_withdraw: true,
            auto_approach: true
        }
    }
    /// Record a finished coarse move: update the calibration bookkeeping, the
    /// move history, and shift the scan area on the map. Used both by this
    /// menu and by the sequence runner.
    pub fn apply_completed_move(&mut self, scan_area: &mut ScanArea, steps: IVec2) {
        if steps.y == 0 {
            let last_pos = scan_area.world_transform.translation;
            self.last_x_move = Some((last_pos, steps));
        } else {
            self.last_x_move = None;
        }
        if steps.x == 0 {
            let last_pos = scan_area.world_transform.translation;
            self.last_y_move = Some((last_pos, steps));
        } else {
            self.last_y_move = None;
        }
        let real_world_move = self.steps2real_world() * steps.as_dvec2();
        scan_area.course_move_history.push(real_world_move);
        scan_area.world_transform =
            scan_area.world_transform * DAffine2::from_translation(real_world_move);
    }
    /// Motor group index (0-based) as sent to Motor.StartMove.
    pub fn group_index(&self) -> u32 {
        self.group - 1
    }
    /// `locked` disables starting a move (e.g. while a sequence is running).
    pub fn show_menu(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>, locked: bool) {
        if self.menu_active {
            object_list.clear_selected();
        }
        if self.move_sender.poll_complete().is_some() {
            self.currently_moving = false;
            if let Some(scan_area) = object_list
                .iter_mut()
                .find_map(|entry| entry.as_scan_area_mut())
            {
                let steps = self.get_steps();
                self.apply_completed_move(scan_area, steps);
            };
            self.move_target = DVec2::ZERO;
        }
        if ui
            .add_enabled(!self.menu_active, egui::Button::new("Course Motion Menu"))
            .clicked()
        {
            self.menu_active = true;
            self.move_target = DVec2::ZERO;
        }
        let mut menu_active = self.menu_active;
        egui::Window::new("Course Motion")
            .frame(
                Frame::window(&ui.ctx().style())
                    .multiply_with_opacity(0.5)
                    .shadow(Shadow::NONE),
            )
            .auto_sized()
            .collapsible(false)
            .enabled(!self.currently_moving)
            .open(&mut menu_active)
            .show(&ui.ctx(), |ui| {
                let mut voltages = *self.voltages.peek();
                let mut steps = self.get_steps();
                ui.horizontal(|ui| {
                    ui.label("X steps: ");
                    ui.add_enabled(voltages.x > 0., DragValue::new(&mut steps.x));
                    ui.label("Y steps: ");
                    ui.add_enabled(voltages.y > 0., DragValue::new(&mut steps.y));
                });
                ui.shrink_width_to_current();
                self.write_steps(steps);
                ui.add_space(12.);
                ui.checkbox(&mut self.auto_withdraw, "Auto withdraw");
                ui.checkbox(&mut self.auto_approach, "Auto approach");
                ui.add_space(12.);
                let mut button_rect = ui.cursor();
                button_rect.set_height(32.);
                button_rect = button_rect.scale_from_center2(Vec2::new(0.5, 1.));
                ui.horizontal(|ui| {
                    let steps = self.get_steps();
                    if ui
                        .add_enabled_ui(!locked && (steps.x.abs() > 0 || steps.y.abs() > 0), |ui| {
                            ui.put(button_rect, Button::new("Execute Move"))
                        })
                        .inner
                        .clicked()
                    {
                        info!("Executing course move {steps:?} on group {}", self.group);
                        self.currently_moving = true;
                        self.move_sender.send(CourseMoveCommand {
                            steps,
                            group: self.group - 1,
                            auto_withdraw: self.auto_withdraw,
                            auto_approach: self.auto_approach
                        });
                    }
                    if self.currently_moving {
                        ui.spinner();
                    }
                });
                if locked {
                    ui.colored_label(Color32::ORANGE, "Sequence running");
                }
                ui.separator();
                CollapsingHeader::new("Config")
                    .default_open(false)
                    .show_unindented(ui, |ui| {
                        ui.label("Group:");
                        ui.add(DragValue::new(&mut self.group).speed(0).range(1..=4));
                        ui.label("Driver Amplitudes:");
                        ui.horizontal(|ui| {
                            ui.add_enabled(false, DragValue::new(&mut voltages.x));
                            ui.add_enabled(false, DragValue::new(&mut voltages.y));
                        });
                        ui.label("Calibration Matrix:");
                        egui::Grid::new("course cal grid").show(ui, |ui| {
                            ui.add(si_drag_value(&mut self.calib_matrix.x_axis.x));
                            ui.add(si_drag_value(&mut self.calib_matrix.y_axis.x));
                            ui.end_row();
                            ui.add(si_drag_value(&mut self.calib_matrix.x_axis.y));
                            ui.add(si_drag_value(&mut self.calib_matrix.y_axis.y));
                        });
                        ui.label("Auto Calibration:");
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled_ui(self.last_x_move.is_some(), |ui| {
                                    ui.button("Calibrate X")
                                })
                                .inner
                                .clicked()
                            {
                                if let Some(scan_area) = object_list
                                    .iter_mut()
                                    .find_map(|entry| entry.as_scan_area_mut())
                                {
                                    let (last_pos, steps) = self.last_x_move.take().unwrap();
                                    let world_motion =
                                        scan_area.world_transform.translation - last_pos;
                                    let real_world_motion = scan_area
                                        .world_transform
                                        .inverse()
                                        .transform_vector2(world_motion);
                                    self.calib_matrix.x_axis =
                                        real_world_motion / (steps.x as f64 * voltages.x) * 1e-9;
                                }
                            }
                            if ui
                                .add_enabled_ui(self.last_y_move.is_some(), |ui| {
                                    ui.button("Calibrate Y")
                                })
                                .inner
                                .clicked()
                            {
                                if let Some(scan_area) = object_list
                                    .iter_mut()
                                    .find_map(|entry| entry.as_scan_area_mut())
                                {
                                    let (last_pos, steps) = self.last_y_move.take().unwrap();
                                    let world_motion =
                                        scan_area.world_transform.translation - last_pos;
                                    let real_world_motion = scan_area
                                        .world_transform
                                        .inverse()
                                        .transform_vector2(world_motion);
                                    self.calib_matrix.y_axis =
                                        real_world_motion / (steps.y as f64 * voltages.y) * 1e-9;
                                }
                            }
                        });
                    });
            });
        if menu_active == false && self.menu_active == true {
            match self.update_config_file() {
                Err(e) => {
                    tracing::error!("error saving config: {}", e)
                }
                Ok(_) => {}
            }
        }
        self.menu_active = menu_active;
    }

    fn update_config_file(&mut self) -> std::io::Result<()> {
        self.cfg_file.seek(std::io::SeekFrom::Start(0))?;
        self.cfg_file
            .write_all(bytemuck::bytes_of(&self.calib_matrix))?;
        Ok(())
    }
    pub fn show_overlay(&mut self, ui: &mut Ui, object_list: &mut SelectableList<Object>) {
        let Some(scan_area) = object_list
            .iter_mut()
            .find_map(|entry| entry.as_scan_area_mut())
        else {
            return;
        };
        if self.menu_active {
            if ui.input(|i| i.modifiers.ctrl) {
                let [_, _, translate] = world_delta_transform(ui);
                let world_translate = translate.translation;
                let scan_world_translate = scan_area
                    .world_transform
                    .inverse()
                    .transform_vector2(world_translate);
                self.move_target += scan_world_translate;
            }
            self.show_current_move(ui, scan_area);
        }
    }
    fn show_current_move(&self, ui: &mut Ui, scan_area: &ScanArea) {
        let ctx = ui
            .data(|map| map.get_temp::<ScanViewCtx>(Id::new(())))
            .unwrap();

        let world2screen = ctx.world2egui();
        let course_world2screen = |p: DVec2| {
            (world2screen * scan_area.world_transform)
                .transform_point2(self.steps2real_world() * p)
                .to_egui_pos2()
        };
        let steps = self.get_steps();

        std::iter::once(DVec2::ZERO)
            .chain(course_path_iter(steps))
            .map(course_world2screen)
            .tuple_windows()
            .for_each(|(a, b)| {
                ui.painter()
                    .line_segment([a, b], Stroke::new(1., Color32::ORANGE));
            });
        course_path_iter(steps)
            .map(course_world2screen)
            .for_each(|point| {
                ui.painter().circle_filled(point, 4., Color32::ORANGE);
            });

        let real_course_move = self.steps2real_world() * steps.as_dvec2();
        let move_transform =
            DAffine2::from_scale_angle_translation(scan_area.area_size, 0., real_course_move);
        BorderRectangle {
            transform: scan_area.world_transform * move_transform,
            color: Color32::from_rgb(0xe3, 0xb6, 0x2d),
            dashed: false,
        }
        .show(ui);
    }
    pub fn steps2real_world(&self) -> DMat2 {
        let voltages = *self.voltages.peek();
        self.calib_matrix * DMat2::from_diagonal(voltages) * 1e9
    }
    pub fn get_steps(&self) -> IVec2 {
        self.real_to_steps(self.move_target)
    }
    /// Nearest whole number of motor steps for a real-world (nm) move.
    pub fn real_to_steps(&self, real: DVec2) -> IVec2 {
        let m = self.steps2real_world();
        let mat = nalgebra::Matrix2::<f64>::from(m);
        let inv = DMat2::from(mat.pseudo_inverse(f64::EPSILON * 10.).unwrap());
        (inv * real).round().as_ivec2()
    }
    /// The Course Motion window is open (it owns ctrl+drag on the map).
    pub fn menu_open(&self) -> bool {
        self.menu_active
    }
    fn write_steps(&mut self, steps: IVec2) {
        let change = steps - self.get_steps();
        self.move_target += self.steps2real_world() * change.as_dvec2();
    }
}

fn course_path_iter(steps: IVec2) -> impl Iterator<Item = DVec2> {
    integer_iter(steps.x)
        .map(move |x| IVec2 { x, y: 0 })
        .chain(integer_iter(steps.y).map(move |y| IVec2 { x: steps.x, y }))
        .map(|v| v.as_dvec2())
}
fn integer_iter(end: i32) -> impl Iterator<Item = i32> {
    let del = end.signum();
    (0..end.abs()).map(move |v| (v + 1) * del)
}
