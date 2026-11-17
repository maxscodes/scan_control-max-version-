mod channel_state;
mod coarse_cal;
mod coarse_learn;
mod command_channel;
pub mod course_motion;
mod fine_motion;
mod lithography;
mod pair_view;
mod ring_assist;
mod scan_control;
mod scan_status;
mod sequence;
mod stamp_names;
pub mod tip_check;
mod spectroscopy;
mod worker;

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

pub use channel_state::ChannelState;
use egui::Ui;
use glam::{DAffine2, DVec2};
use itertools::Itertools;
use nanonis_tcp::LineDir;
pub use scan_status::ScanStatus;
use tracing::trace;
use uuid::Uuid;

use crate::{
    components::selectable_list::SelectableList,
    connection::{
        live_image::{FrameData, LiveImage},
        nanonis::{
            course_motion::CourseMotionState,
            fine_motion::FineMotionState,
            lithography::{LithographyState, LithographyWorker},
            ring_assist::RingAssistState,
            scan_control::{PanelInput, Readback, ScanControlState},
            sequence::{SequenceAction, SequenceState, SequenceWorker},
            spectroscopy::{SpectroscopyState, SpectroscopyWorker},
            worker::{FastStatusWorker, FrameWorker, LineWorker, SlowStatusWorker, Worker as _},
        },
        queue::{overwrite_queue, OverwriteQueueSender},
        scan_area::ScanArea,
        shared_state::SharedState,
        Connection, StatusInfo,
    },
    scan_view::ImageEncoder,
    theme,
    view_object::{self, Object},
};

pub struct NanonisConnection {
    forward_data: SharedState<FrameData>,
    backward_data: SharedState<FrameData>,
    image_transform: SharedState<DAffine2>,
    area_size: SharedState<DVec2>,
    channel_state: SharedState<ChannelState>,
    scan_status: SharedState<ScanStatus>,
    base_name: SharedState<String>,
    /// Nanonis save folder, to name stamps after their .sxm files.
    session_path: SharedState<String>,
    stamp_namer: stamp_names::StampNamer,
    frame_queue_tx: OverwriteQueueSender<LineDir>,
    tip_pos: SharedState<DVec2>,
    readback: SharedState<Readback>,
    /// Scan-area world transform as of the last update, for the status bar.
    last_world_transform: DAffine2,
    slow_status_init: Arc<AtomicBool>,
    fast_status_init: Arc<AtomicBool>,
    course_menu: CourseMotionState,
    fine_motion: FineMotionState,
    scan_control: ScanControlState,
    sequence: SequenceState,
    spectroscopy: SpectroscopyState,
    lithography: LithographyState,
    ring_assist: RingAssistState,
    tip_check: tip_check::TipCheckState,
    pending_stamp: Option<PendingStamp>,
    /// Lines up survey tiles when a survey ends.
    aligner: crate::analysis::mosaic::Aligner,
}

/// A stamp requested by the sequence runner, waiting for a fresh full frame.
struct PendingStamp {
    since: Instant,
    got_forward: bool,
    got_backward: bool,
}
/// Stamp with whatever data is present if the fresh frame doesn't arrive.
const STAMP_FALLBACK: Duration = Duration::from_secs(10);

impl Connection for NanonisConnection {
    fn poll_connected(
        &mut self,
        object_list: &mut SelectableList<view_object::Object>,
        encoder: &ImageEncoder,
    ) -> bool {
        if !self.slow_status_init.load(Ordering::SeqCst)
            || !self.fast_status_init.load(Ordering::SeqCst)
        {
            return false;
        }
        if let Some(scan_area) = object_list
            .iter_mut()
            .find_map(|obj| obj.as_scan_area_mut())
        {
            self.sync_scan_area(scan_area);
            return true;
        }
        let mut scan_area = ScanArea::new(Uuid::new_v4(), encoder, *self.area_size.read());
        self.sync_scan_area(&mut scan_area);
        object_list.push(Object::ScanArea { image: scan_area, hidden: false }.into_entry("area"));
        true
    }

    fn update(
        &mut self,
        object_list: &mut SelectableList<view_object::Object>,
        encoder: &ImageEncoder,
    ) {
        let gds = crate::analysis::gds_prior::collect(object_list);
        let Some((index, scan_area)) = object_list
            .iter_mut()
            .enumerate()
            .find_map(|(i, entry)| entry.as_scan_area_mut().map(|area| (i, area)))
        else {
            return;
        };
        self.last_world_transform = scan_area.world_transform;
        self.update_channel(scan_area, encoder);
        self.update_area_size(scan_area);
        self.update_image_transform(scan_area);
        let (got_forward, got_backward) =
            self.update_image_data(&mut scan_area.live_image, encoder);
        self.update_sequence(scan_area, encoder, got_forward, got_backward);
        let fine_lock = self.fine_lock();
        self.fine_motion.update(
            scan_area,
            encoder,
            &mut self.scan_control,
            &self.frame_queue_tx,
            got_forward,
            got_backward,
            fine_lock,
        );
        self.ring_assist.update(scan_area, got_forward, &gds);
        self.tip_check.update(scan_area, got_forward);
        self.update_scan_status(scan_area);
        self.update_tip_pos(scan_area);
        self.update_base_name(scan_area);
        let stamps = scan_area.stamp.drain(..).collect_vec();
        let coarse_now = (scan_area.coarse_log.last_move(), scan_area.world_transform);
        let piezo_frame = scan_area.image_transform;
        let stamp_channel = scan_area.channel_selected.clone();
        let time = time::UtcDateTime::now();
        let time = time::PrimitiveDateTime::new(time.date(), time.time());
        let base_name = time_fmt::format::format_date_time(&scan_area.stamp_name_base, time)
            .unwrap_or_else(|_| scan_area.stamp_name_base.clone());

        scan_area.poll_sxm_imports(encoder);
        for (mut stamp, file, channel) in scan_area.stamp_import.drain(..).collect_vec() {
            stamp.ring = self.ring_assist.stamp_result(
                Some(&channel),
                &stamp.forward_data.data,
                stamp.forward_data.size,
                stamp.transform,
            );
            stamp.gds_ref = crate::analysis::gds_prior::predict_any(&gds, stamp.transform)
                .map(|(gi, _)| (gds[gi].uuid, gds[gi].transform));
            let name = unique_name(object_list, &file);
            object_list.insert(
                index,
                Object::ScanImage { image: stamp, name, hidden: false, group: None }.into_entry(Uuid::new_v4()),
            );
        }

        let mut name_index = 0;
        for mut stamp in stamps {
            stamp.ring = self.ring_assist.stamp_result(
                stamp_channel.as_deref(),
                &stamp.forward_data.data,
                stamp.forward_data.size,
                stamp.transform,
            );
            // Remember where the GDS was, to warn if it is moved later.
            stamp.gds_ref = crate::analysis::gds_prior::predict_any(&gds, stamp.transform)
                .map(|(gi, _)| (gds[gi].uuid, gds[gi].transform));
            // And where the stamp was put, to learn the coarse calibration
            // from how far the user drags it.
            stamp.coarse_ref =
                Some(crate::analysis::coarse_fit::CoarseRef::new(coarse_now.0, coarse_now.1, stamp.transform));
            self.tip_check.queue_stamp(&stamp, piezo_frame);
            'try_again: loop {
                for obj in object_list.iter() {
                    let name = obj.name();
                    let Some(rest) = name.strip_prefix(&base_name) else {
                        continue;
                    };
                    if rest.is_empty() && name_index == 0 {
                        name_index += 1;
                        continue 'try_again;
                    }
                    let Some(existing_name_index) = rest
                        .strip_prefix("(")
                        .and_then(|rest| rest.strip_suffix(")"))
                        .and_then(|num_str| num_str.parse::<usize>().ok())
                    else {
                        continue;
                    };
                    if existing_name_index == name_index {
                        name_index += 1;
                        continue 'try_again;
                    }
                }
                break;
            }
            let name = if name_index == 0 {
                base_name.clone()
            } else {
                format!("{}({})", base_name, name_index)
            };
            name_index += 1;
            self.stamp_namer.add_new(stamp.uuid(), stamp.forward_data.data.clone(), stamp.forward_data.size);
            object_list.insert(
                index,
                Object::ScanImage { image: stamp, name, hidden: false, group: None }.into_entry(Uuid::new_v4()),
            );
        }
        self.name_stamps(object_list);
        self.tip_check.poll_stamps(object_list);
        // Put survey tiles in their survey's layer group (once their stamp
        // is on the list).
        let mut waiting = Vec::new();
        for (id, g) in self.sequence.take_tile_groups() {
            match object_list.iter_mut().find(|o| o.uuid() == id) {
                Some(o) => {
                    if let Some(group) = o.group_mut() {
                        *group = Some(g);
                    }
                }
                None => waiting.push((id, g)),
            }
        }
        self.sequence.keep_tile_groups(waiting);
        // Undone (or redone) automatic placements of grid tiles: move the
        // stamp, and the record of where it was put with it, so learning
        // from drags doesn't take it for a drag. The latest tile carries
        // the tip (scan area).
        for m in self.sequence.take_stamp_moves() {
            for o in object_list.iter_mut() {
                if let Object::ScanImage { image, .. } = &mut **o {
                    if image.uuid() == m.id {
                        image.transform.translation += m.delta;
                        if let Some(c) = &mut image.coarse_ref {
                            for a in [&mut c.stamp_at, &mut c.world_at] {
                                a[4] += m.delta.x;
                                a[5] += m.delta.y;
                            }
                        }
                    }
                }
            }
            if m.latest {
                if let Some(area) = object_list.iter_mut().find_map(|e| e.as_scan_area_mut()) {
                    area.world_transform.translation += m.delta;
                }
            }
        }
        // A tracked grid: tiles dragged on the map are where the tip really
        // was; the latest one carries the tip (scan area) with it.
        let ids = self.sequence.grid_tile_ids();
        if !ids.is_empty() {
            let pos: std::collections::HashMap<Uuid, glam::DVec2> = object_list
                .iter()
                .filter(|o| ids.contains(&o.uuid()))
                .map(|o| (o.uuid(), o.transform_center()))
                .collect();
            if let Some(delta) = self.sequence.grid_sync(|id| pos.get(&id).copied()) {
                // Learning from drags already moves the scan area with the
                // latest stamp: don't move it twice.
                if delta != glam::DVec2::ZERO && !self.course_menu.follows_latest_stamp() {
                    if let Some(area) = object_list.iter_mut().find_map(|e| e.as_scan_area_mut()) {
                        area.world_transform.translation += delta;
                    }
                }
            }
        }
        if let Some((c, n)) = self.sequence.take_grid_calibration() {
            self.course_menu.set_calibration(c, format!("survey grid ({n} moves)"));
        }
        // Line up the tiles of a survey that just ended.
        if let Some(ids) = self.sequence.take_finished_survey() {
            if self.aligner.start(object_list, &ids, true) {
                self.sequence.set_status(format!("Lining up {} survey tiles\u{2026}", ids.len()));
            }
        }
        if let Some(summary) = self.aligner.poll(object_list) {
            self.sequence.set_status(summary);
        }
        if self.sequence.take_lineup_undo_request() {
            let n = self.aligner.undo(object_list);
            self.sequence.set_status(format!("Line-up undone: {n} tiles put back"));
        }
        self.sequence.set_lineup_undo_available(self.aligner.can_undo());
    }
    fn show_menu(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<view_object::Object>,
        encoder: &ImageEncoder,
    ) {
        let Some(scan_area) = object_list
            .iter_mut()
            .find_map(|entry| entry.as_scan_area_mut())
        else {
            return;
        };
        let scan_status = scan_area.scan_status;
        // Only one tool may drive the tip at a time.
        let sequence = self
            .sequence
            .is_running()
            .then_some("A sequence is running");
        let spectroscopy = self
            .spectroscopy
            .is_running()
            .then_some("I(V) spectroscopy is running");
        let lithography = self
            .lithography
            .is_running()
            .then_some("Lithography is running");
        let scanning = scan_status
            .scanning
            .then_some("A scan is running; stop it first");
        let fine = self
            .fine_motion
            .is_running()
            .then_some("Fine Motion is moving");
        let fine_lock = self.fine_lock();
        let locked_by = sequence.or(spectroscopy).or(lithography);
        let locked = locked_by.is_some();
        // Scan controls stay usable during a sequence: using one holds it.
        let scan_locked_by = spectroscopy.or(lithography);

        // Status, scan buttons, bias/setpoint/current and progress: always shown.
        let was_compact = theme::prefs().scan_compact;
        let mut compact = was_compact;
        let rows = frame_rows(&scan_area.live_image.forward_data);
        let user_command = self.scan_control.show_menu(
            ui,
            PanelInput {
                status: &scan_status,
                sequence_active: self.sequence.is_running(),
                locked: scan_locked_by,
                readback: Some(*self.readback.peek()).filter(|_| self.slow_status_init.load(Ordering::SeqCst)),
                rows,
            },
            &mut compact,
        );
        if let Some(cmd) = user_command {
            // Hold the sequence before the command reaches Nanonis.
            if !matches!(cmd, scan_control::ScanCommand::Save) {
                self.sequence.interrupt(&format!("you used the scan controls ({})", cmd.describe()));
            }
            self.scan_control.request(cmd);
        }
        if compact != was_compact {
            theme::update_prefs(|p| p.scan_compact = compact);
        }

        // Full mode: image settings, stamp and the tool launchers. The tools
        // are always updated (their windows stay open); only their launch
        // buttons and status lines are hidden in compact mode.
        let show_tools = !compact;
        if show_tools {
            ui.separator();
            egui::CollapsingHeader::new(egui::RichText::new("Image").strong())
                .id_salt("scan_control_image")
                .default_open(true)
                .show_unindented(ui, |ui| scan_area.show_menu(ui, encoder));
            ui.separator();
            theme::section_label(ui, "TOOLS");
        }
        let mut tools_ui = ui.new_child(egui::UiBuilder::new().invisible().sizing_pass());
        let tools_ui: &mut Ui = if show_tools { ui } else { &mut tools_ui };
        let ui = tools_ui;
        self.tool_buttons(ui);
        ui.add_space(2.);
        let readback = Some(*self.readback.peek()).filter(|_| self.slow_status_init.load(Ordering::SeqCst));
        let line_times = readback
            .map(|r| (r.fwd_line_time as f64, r.bwd_line_time as f64))
            .filter(|&(f, _)| f > 0.);
        (self.course_menu.cal_progress, self.course_menu.cal_measuring) = self.sequence.calibration_progress();
        self.course_menu.cal_can_restore = self.sequence.can_restore_calibration_frame();
        self.course_menu.line_times = line_times;
        self.course_menu.learn_sync(ui, object_list);
        self.course_menu.show_menu(ui, object_list, locked);
        // Auto calibration runs as a sequence; its result goes back to the menu.
        if let Some(mut plan) = self.course_menu.take_calibration_request() {
            // Remember the frame size and resolution to put back afterwards.
            let rb = *self.readback.peek();
            let frame = object_list
                .iter()
                .find_map(|o| o.as_scan_area())
                .and_then(|a| a.image_transform)
                .map(sequence::Frame::from_transform);
            plan.restore = frame
                .filter(|_| rb.lines > 0 && rb.pixels > 0)
                .map(|f| ([f.width, f.height], [rb.pixels, rb.lines]));
            plan.restore_line_time_s = Some(rb.fwd_line_time as f64).filter(|&t| t > 0.);
            let old = self.course_menu.baseline();
            self.sequence.start_calibration(plan, self.course_menu.group_index(), old);
        }
        if self.course_menu.take_restore_request() {
            self.sequence.restore_calibration_frame();
        }
        if let Some(result) = self.sequence.take_calibration_result() {
            self.course_menu.set_calibration_result(result);
        }
        self.fine_motion.show_menu(ui, object_list, fine_lock);
        self.sequence.line_times = line_times;
        self.sequence.lines = readback.map(|r| r.lines).filter(|&l| l > 0);
        self.sequence
            .show_menu(ui, object_list, &self.course_menu, spectroscopy.or(lithography));
        self.spectroscopy
            .show_menu(ui, sequence.or(lithography).or(scanning).or(fine));
        self.lithography
            .show_menu(ui, object_list, encoder, sequence.or(spectroscopy).or(scanning).or(fine));
        self.ring_assist
            .show_menu(ui, object_list, &mut self.course_menu, locked_by);
        self.tip_check.show_menu(ui);
    }
    fn status(&self) -> StatusInfo {
        let status = *self.scan_status.peek();
        let rb = self
            .slow_status_init
            .load(Ordering::SeqCst)
            .then(|| *self.readback.peek());
        StatusInfo {
            scan: self.scan_control.status_text(&status),
            bias: rb.map(|r| r.bias),
            current: rb.map(|r| r.current),
            tip_nm: self
                .fast_status_init
                .load(Ordering::SeqCst)
                .then(|| self.last_world_transform.transform_point2(*self.tip_pos.peek())),
        }
    }
    fn show_image_view_overlay(
        &mut self,
        ui: &mut Ui,
        object_list: &mut SelectableList<view_object::Object>,
    ) {
        self.course_menu.show_overlay(ui, object_list);
        self.ring_assist.show_label_overlay(ui, object_list);
        self.tip_check.show_overlay(ui, object_list);
        if let Some(scan_area) = object_list
            .iter_mut()
            .find_map(|entry| entry.as_scan_area_mut())
        {
            self.sequence.show_overlay(ui, scan_area, &self.course_menu);
            // Only one tool takes ctrl+click / ctrl+drag on the map.
            let map_free = !self.course_menu.menu_open() && !self.sequence.map_drag_active();
            self.lithography.show_overlay(ui, scan_area, map_free);
            self.ring_assist.show_overlay(ui, scan_area);
            let map_free = map_free && !self.lithography.map_edit_active();
            self.fine_motion.show_overlay(ui, scan_area, map_free);
        }
    }
}

impl NanonisConnection {
    /// The tool launchers: one toggle per tool window, two to a row.
    fn tool_buttons(&mut self, ui: &mut Ui) {
        let mut coarse = self.course_menu.menu_open();
        let mut fine = self.fine_motion.is_open();
        ui.columns(2, |c| {
            theme::tool_button(&mut c[0], &mut coarse, "Coarse Motion", "Coarse motor moves, auto approach and calibration");
            theme::tool_button(&mut c[1], &mut fine, "Fine Motion", "Move the scan frame or the tip within the piezo range");
            theme::tool_button(&mut c[0], self.sequence.open_mut(), "Sequence", "Plan and run scans, survey grids and tip moves");
            theme::tool_button(&mut c[1], self.spectroscopy.open_mut(), "Spectroscopy", "I(V) spectroscopy series at stepped Z");
            theme::tool_button(&mut c[0], self.lithography.open_mut(), "Lithography", "Write a GDS pattern with the tip");
            theme::tool_button(&mut c[1], self.ring_assist.open_mut(), "Ring Assist", "Find the ring centre in Z scans and suggest moves");
            theme::tool_button(&mut c[0], self.tip_check.open_mut(), "Tip Check", "Judge tip quality (multi-tip ghosts, tip changes) and label archive scans");
        });
        if coarse != self.course_menu.menu_open() {
            self.course_menu.set_open(coarse);
        }
        if fine != self.fine_motion.is_open() {
            self.fine_motion.set_open(fine);
        }
    }

    /// Name stamps after the .sxm file Nanonis saved the same scan to (and
    /// their training labels too). Stamps without a file name yet are looked
    /// up in the background.
    fn name_stamps(&mut self, object_list: &mut SelectableList<view_object::Object>) {
        for o in object_list.iter() {
            if let Object::ScanImage { image, name, .. } = &**o {
                if !stamp_names::is_file_name(name) && image.forward_data.size[0] > 0 {
                    self.stamp_namer.add_existing(
                        image.uuid(),
                        image.forward_data.data.clone(),
                        image.forward_data.size,
                    );
                }
            }
        }
        let session = self.session_path.read().clone();
        for (id, file) in self.stamp_namer.poll(&session) {
            let new = unique_name(object_list, &file);
            for o in object_list.iter_mut() {
                if let Object::ScanImage { image, name, .. } = &mut **o {
                    if image.uuid() == id {
                        *name = new.clone();
                        self.ring_assist.rename_label(id, &new);
                    }
                }
            }
        }
    }

    /// Why Fine Motion can't move now: another tool drives the tip.
    fn fine_lock(&self) -> Option<&'static str> {
        if self.sequence.is_running() {
            Some("A sequence is running")
        } else if self.spectroscopy.is_running() {
            Some("I(V) spectroscopy is running")
        } else if self.lithography.is_running() {
            Some("Lithography is running")
        } else if self.course_menu.is_moving() {
            Some("A coarse move is running")
        } else {
            None
        }
    }
    /// Copy the current Nanonis state into a (new or reloaded) scan area.
    fn sync_scan_area(&mut self, scan_area: &mut ScanArea) {
        scan_area.channel_opts = self.channel_state.read().channel_opts_names().collect();
        scan_area.area_size = *self.area_size.read();
        scan_area.image_transform = Some(*self.image_transform.read());
        scan_area.tip_pos = Some(*self.tip_pos.read());
        scan_area.stamp_name_base = self.base_name.read().clone();
        scan_area.session_dir = self.session_path.read().clone();
        scan_area.scan_status = *self.scan_status.read();
    }
    pub fn new(ctx: egui::Context, address: impl AsRef<str>) -> Self {
        let image_transform = SharedState::new_default();
        let area_size = SharedState::new_default();
        let channel_state = SharedState::new_default();
        let forward_data = SharedState::new_default();
        let backward_data = SharedState::new_default();
        let scan_status = SharedState::new_default();
        let base_name = SharedState::new_default();
        let session_path = SharedState::new_default();
        let tip_pos = SharedState::new_default();
        let readback = SharedState::new_default();
        let course_voltages = SharedState::new_default();
        let course_freqs = SharedState::new_default();
        let (frame_queue_tx, frame_queue_rx) = overwrite_queue(2);
        let slow_status_init = Arc::new(AtomicBool::new(false));
        let fast_status_init = Arc::new(AtomicBool::new(false));
        let (move_sender, move_receiver) = command_channel::command_channel();

        let course_menu = CourseMotionState::new(&course_voltages, &course_freqs, &move_sender);
        let (scan_sender, scan_receiver) = command_channel::command_channel();
        let scan_control = ScanControlState::new(&scan_sender);
        let (seq_cmd_tx, seq_cmd_rx) = crossbeam::channel::unbounded();
        let (seq_event_tx, seq_event_rx) = crossbeam::channel::unbounded();
        let sequence = SequenceState::new(seq_cmd_tx, seq_event_rx);
        let (spec_cmd_tx, spec_cmd_rx) = crossbeam::channel::unbounded();
        let (spec_event_tx, spec_event_rx) = crossbeam::channel::unbounded();
        let plot_choice: crate::connection::nanonis::spectroscopy::SharedPlotChoice = Default::default();
        let spectroscopy = SpectroscopyState::new(spec_cmd_tx, spec_event_rx, &plot_choice);
        let (litho_cmd_tx, litho_cmd_rx) = crossbeam::channel::unbounded();
        let (litho_event_tx, litho_event_rx) = crossbeam::channel::unbounded();
        let lithography = LithographyState::new(litho_cmd_tx, litho_event_rx);

        let address = address.as_ref();
        LineWorker::new(&frame_queue_tx, &scan_status).run(address, 6500);
        FrameWorker::new(
            &ctx,
            &forward_data,
            &backward_data,
            frame_queue_rx,
            &channel_state,
            &scan_status,
        )
        .run(address, 6500);
        FastStatusWorker::new(&ctx, &image_transform, &tip_pos, &fast_status_init)
            .run(address, 6500);
        SlowStatusWorker::new(
            &ctx,
            &area_size,
            &channel_state,
            &scan_status,
            &base_name,
            &session_path,
            &course_voltages,
            &course_freqs,
            &move_receiver,
            &scan_receiver,
            &readback,
            &slow_status_init,
        )
        .run(address, 6500);
        SequenceWorker::new(&ctx, seq_cmd_rx, seq_event_tx, &plot_choice).run(address, 6500);
        SpectroscopyWorker::new(&ctx, spec_cmd_rx, spec_event_tx).run(address, 6500);
        LithographyWorker::new(&ctx, litho_cmd_rx, litho_event_tx).run(address, 6500);
        Self {
            image_transform,
            area_size,
            channel_state,
            backward_data,
            forward_data,
            frame_queue_tx,
            scan_status,
            tip_pos,
            readback,
            last_world_transform: DAffine2::IDENTITY,
            slow_status_init,
            base_name,
            session_path,
            stamp_namer: Default::default(),
            course_menu,
            fine_motion: FineMotionState::new(),
            scan_control,
            sequence,
            spectroscopy,
            lithography,
            ring_assist: RingAssistState::new(),
            tip_check: tip_check::TipCheckState::new(),
            pending_stamp: None,
            aligner: Default::default(),
            fast_status_init,
        }
    }
    fn update_tip_pos(&mut self, scan_area: &mut ScanArea) {
        if let Some(tip_pos) = self.tip_pos.read_new().as_deref().copied() {
            scan_area.tip_pos = Some(tip_pos);
        }
    }
    fn update_scan_status(&mut self, scan_area: &mut ScanArea) {
        if let Some(scan_status) = self.scan_status.read_new().as_deref().copied() {
            scan_area.scan_status = scan_status;
        }
    }
    /// Returns whether new (forward, backward) data arrived.
    fn update_image_data(
        &mut self,
        live_image: &mut LiveImage,
        encoder: &ImageEncoder,
    ) -> (bool, bool) {
        // While scanning, frames are re-downloaded about every 100 ms, mostly
        // unchanged (a line takes longer). Only a changed frame is processed
        // and uploaded again; "got" still reports every arrival, as stamping
        // waits for a fresh download.
        let mut got = (false, false);
        if let Some(forward_data) = self.forward_data.read_new() {
            got.0 = true;
            if !live_image.forward_data.same_as(&forward_data) {
                live_image.forward_data = forward_data.clone();
                if live_image.line_dir == LineDir::Forward {
                    live_image.write_and_update_texture(encoder);
                }
            }
        }
        if let Some(backward_data) = self.backward_data.read_new() {
            got.1 = true;
            if !live_image.backward_data.same_as(&backward_data) {
                live_image.backward_data = backward_data.clone();
                if live_image.line_dir == LineDir::Backward {
                    live_image.write_and_update_texture(encoder);
                }
            }
        }
        got
    }
    /// Handle sequence runner events that touch the scan area, and finish a
    /// pending auto-stamp once a fresh full frame has been downloaded.
    fn update_sequence(
        &mut self,
        scan_area: &mut ScanArea,
        encoder: &ImageEncoder,
        got_forward: bool,
        got_backward: bool,
    ) {
        for action in self.sequence.poll_events() {
            match action {
                SequenceAction::CoarseMoved(steps) => {
                    self.course_menu.apply_completed_move(scan_area, steps);
                    // A tracked survey grid places the tip with what it has
                    // learned so far (only the tiles still to come move).
                    let calib = self.course_menu.calibration();
                    if let Some(extra) = self.sequence.grid_moved(steps, &calib) {
                        let d = scan_area.world_transform.matrix2 * extra;
                        scan_area.world_transform.translation += d;
                    }
                }
                SequenceAction::SpecStarted(base) => self.spectroscopy.sequence_series_started(&base),
                SequenceAction::SpecCurve(curve) => self.spectroscopy.sequence_curve(curve),
                SequenceAction::SpecDone(ok, message, files) => {
                    self.spectroscopy.sequence_message(ok, message, files)
                }
                SequenceAction::StampRequest => {
                    if scan_area.channel_selected.is_none() {
                        // Nothing displayed, nothing to stamp.
                        self.sequence.stamp_done();
                        continue;
                    }
                    Self::request_full_frame(&self.frame_queue_tx);
                    self.pending_stamp = Some(PendingStamp {
                        since: Instant::now(),
                        got_forward: false,
                        got_backward: false,
                    });
                    // Arrival is checked below on later frames.
                    return;
                }
            }
        }
        let Some(pending) = &mut self.pending_stamp else {
            return;
        };
        pending.got_forward |= got_forward;
        pending.got_backward |= got_backward;
        let fresh = pending.got_forward && pending.got_backward;
        if fresh || pending.since.elapsed() > STAMP_FALLBACK {
            if !fresh {
                tracing::warn!("fresh frame not received, stamping current data");
            }
            let saved = self.sequence.take_scan_file();
            let is_tile = self.sequence.current_entry_is_tile();
            if self.sequence.calibration_active() || is_tile {
                // The Z forward image from the file Nanonis saved: the live
                // view may show another channel, or not have the whole
                // frame yet.
                let from_file = saved.as_deref().and_then(|p| {
                    match crate::analysis::sxm_match::load_frame(std::path::Path::new(p), Some("Z")) {
                        Ok(f) => Some(coarse_cal::CalFrame {
                            data: FrameData { size: f.size, data: std::sync::Arc::new(f.forward.into_boxed_slice()) },
                            frame: f.frame,
                            source: format!("{} ({})", f.name, f.channel),
                        }),
                        Err(e) => {
                            tracing::warn!("calibration: could not read {p}: {e}; using the live image");
                            None
                        }
                    }
                });
                let frame = from_file.or_else(|| {
                    scan_area.image_transform.map(|frame| coarse_cal::CalFrame {
                        data: scan_area.live_image.forward_data.clone(),
                        frame,
                        source: format!("live view ({})", scan_area.channel_selected.as_deref().unwrap_or("?")),
                    })
                });
                if let Some(frame) = frame {
                    if is_tile {
                        // Before StampDone, so the next move is corrected in time.
                        let (calib, volts) = (self.course_menu.calibration(), self.course_menu.volts());
                        let m2 = scan_area.world_transform.matrix2;
                        if let Some(off) = self.sequence.grid_tile_scanned(frame, calib, volts, m2) {
                            // The tip is where the match says: move the scan
                            // area there, so this tile is stamped in place.
                            let d = m2 * off;
                            scan_area.world_transform.translation += d;
                            scan_area.live_image.transform.translation += d;
                        }
                    } else {
                        self.sequence.capture_calibration_frame(frame);
                    }
                }
            }
            let stamp = scan_area.live_image.stamp(encoder);
            if self.sequence.current_entry_is_tile() {
                self.sequence.record_tile(stamp.uuid());
            }
            scan_area.stamp.push_front(stamp);
            self.pending_stamp = None;
            self.sequence.stamp_done();
        }
    }
    fn update_image_transform(&mut self, scan_area: &mut ScanArea) {
        if let Some(new_transform) = self.image_transform.read_new().as_deref().copied() {
            scan_area.image_transform = Some(new_transform);
            return;
        }
        self.image_transform.modify_conditional(
            |prev| scan_area.image_transform.is_some_and(|it| it != *prev),
            |old| *old = scan_area.image_transform.unwrap(),
        );
    }
    fn update_area_size(&mut self, scan_area: &mut ScanArea) {
        if let Some(new_size) = self.area_size.read_new().as_deref().copied() {
            scan_area.area_size = new_size;
        }
    }
    fn update_base_name(&mut self, scan_area: &mut ScanArea) {
        if let Some(new_name) = self.base_name.read_new().as_deref() {
            scan_area.stamp_name_base = new_name.clone();
        }
        if let Some(dir) = self.session_path.read_new().as_deref() {
            scan_area.session_dir = dir.clone();
        }
    }
    fn request_full_frame(tx: &OverwriteQueueSender<LineDir>) {
        trace!("requesting full frame");
        tx.send(LineDir::Forward);
        tx.send(LineDir::Backward);
    }
    fn update_channel(&mut self, scan_area: &mut ScanArea, encoder: &ImageEncoder) {
        // if nanonis sent a change to the channel
        if let Some(state) = self.channel_state.read_new() {
            scan_area.channel_opts = state.channel_opts_names().collect();
            if let Some(unit) = state.unit() {
                scan_area.live_image.unit = unit;
            }
            let new_selected_string = state.selected_as_string();
            if scan_area.channel_selected != new_selected_string {
                scan_area.channel_selected = new_selected_string;
                if state.selection.is_some() {
                    Self::request_full_frame(&self.frame_queue_tx);
                } else {
                    clear_live(&mut scan_area.live_image, encoder);
                }
            }
        }
        // if we changed the selected channel
        if self.channel_state.modify_conditional(
            |prev| prev.selected_as_string() != scan_area.channel_selected,
            |state| {
                match &scan_area.channel_selected {
                    Some(ch_name) => state.set_selection_by_name(&ch_name),
                    None => state.selection = None,
                };
                if let Some(unit) = state.unit() {
                    scan_area.live_image.unit = unit;
                }
            },
        ) {
            if scan_area.channel_selected.is_some() {
                Self::request_full_frame(&self.frame_queue_tx);
            } else {
                clear_live(&mut scan_area.live_image, encoder);
            }
        }
    }
}

/// `file`, or `file(2)`, `file(3)`, ... if a layer already has that name.
fn unique_name(object_list: &SelectableList<view_object::Object>, file: &str) -> String {
    let taken = |name: &str| object_list.iter().any(|o| o.name() == name);
    if !taken(file) {
        return file.to_string();
    }
    (2..).map(|k| format!("{file}({k})")).find(|n| !taken(n)).unwrap()
}

/// No channel shown: drop the picture and its data, so the same frame
/// downloaded again after picking a channel is drawn (unchanged frames are
/// otherwise skipped).
fn clear_live(live: &mut LiveImage, encoder: &ImageEncoder) {
    live.forward_data = FrameData::default();
    live.backward_data = FrameData::default();
    live.clear_texture(encoder);
}

/// (rows received, total rows) of a frame being scanned: rows whose middle
/// pixel already has data.
fn frame_rows(data: &FrameData) -> Option<(usize, usize)> {
    let [w, h] = data.size.map(|v| v as usize);
    if w == 0 || h == 0 || data.data.len() != w * h {
        return None;
    }
    let done = (0..h).filter(|&r| data.data[r * w + w / 2].is_finite()).count();
    Some((done, h))
}
