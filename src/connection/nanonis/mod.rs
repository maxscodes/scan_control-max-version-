mod channel_state;
mod command_channel;
mod course_motion;
mod lithography;
mod ring_assist;
mod scan_control;
mod scan_status;
mod sequence;
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
    components::selectable_list::{SelectableEntry, SelectableList},
    connection::{
        live_image::{FrameData, LiveImage},
        nanonis::{
            course_motion::CourseMotionState,
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
    frame_queue_tx: OverwriteQueueSender<LineDir>,
    tip_pos: SharedState<DVec2>,
    readback: SharedState<Readback>,
    /// Scan-area world transform as of the last update, for the status bar.
    last_world_transform: DAffine2,
    slow_status_init: Arc<AtomicBool>,
    fast_status_init: Arc<AtomicBool>,
    course_menu: CourseMotionState,
    scan_control: ScanControlState,
    sequence: SequenceState,
    spectroscopy: SpectroscopyState,
    lithography: LithographyState,
    ring_assist: RingAssistState,
    pending_stamp: Option<PendingStamp>,
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
        object_list.push(SelectableEntry::new(
            "area",
            Object::ScanArea {
                image: scan_area,
                hidden: false,
            },
            |img| img.list_atoms(),
            |obj| obj.hidden_mut(),
        ));
        true
    }

    fn update(
        &mut self,
        object_list: &mut SelectableList<view_object::Object>,
        encoder: &ImageEncoder,
    ) {
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
        self.ring_assist.update(scan_area, got_forward);
        self.update_scan_status(scan_area);
        self.update_tip_pos(scan_area);
        self.update_base_name(scan_area);
        let stamps = scan_area.stamp.drain(..).collect_vec();
        let stamp_channel = scan_area.channel_selected.clone();
        let time = time::UtcDateTime::now();
        let time = time::PrimitiveDateTime::new(time.date(), time.time());
        let base_name = time_fmt::format::format_date_time(&scan_area.stamp_name_base, time)
            .unwrap_or_else(|_| scan_area.stamp_name_base.clone());

        let mut name_index = 0;
        for mut stamp in stamps {
            stamp.ring = self.ring_assist.stamp_result(
                stamp_channel.as_deref(),
                &stamp.forward_data.data,
                stamp.forward_data.size,
                stamp.transform,
            );
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
            object_list.insert(
                index,
                SelectableEntry::new(
                    Uuid::new_v4(),
                    Object::ScanImage {
                        image: stamp,
                        name,
                        hidden: false,
                    },
                    |img| img.list_atoms(),
                    |obj| obj.hidden_mut(),
                ),
            );
        }
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
        let locked_by = sequence.or(spectroscopy).or(lithography);
        let locked = locked_by.is_some();

        // Status, scan buttons, bias/setpoint/current and progress: always shown.
        let was_compact = theme::prefs().scan_compact;
        let mut compact = was_compact;
        let rows = frame_rows(&scan_area.live_image.forward_data);
        self.scan_control.show_menu(
            ui,
            PanelInput {
                status: &scan_status,
                sequence_active: self.sequence.is_running(),
                locked: locked_by,
                readback: Some(*self.readback.peek()).filter(|_| self.slow_status_init.load(Ordering::SeqCst)),
                rows,
            },
            &mut compact,
        );
        if compact != was_compact {
            theme::update_prefs(|p| p.scan_compact = compact);
        }

        // Full mode: image settings, stamp, history and the tool launchers.
        // The tools are always updated (their windows stay open); only their
        // launch buttons are hidden in compact mode.
        let show_tools = !compact;
        if show_tools {
            ui.separator();
            scan_area.show_menu(ui, encoder);
            ui.separator();
            ui.label(egui::RichText::new("Tools").small().weak());
        }
        let mut tools_ui = ui.new_child(egui::UiBuilder::new().invisible().sizing_pass());
        let tools_ui: &mut Ui = if show_tools { ui } else { &mut tools_ui };
        let ui = tools_ui;
        self.course_menu.show_menu(ui, object_list, locked);
        self.sequence
            .show_menu(ui, object_list, &self.course_menu, spectroscopy.or(lithography));
        self.spectroscopy
            .show_menu(ui, sequence.or(lithography).or(scanning));
        self.lithography
            .show_menu(ui, object_list, sequence.or(spectroscopy).or(scanning));
        self.ring_assist.show_menu(ui, object_list);
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
        if let Some(scan_area) = object_list
            .iter_mut()
            .find_map(|entry| entry.as_scan_area_mut())
        {
            self.sequence.show_overlay(ui, scan_area, &self.course_menu);
            self.lithography.show_overlay(ui, scan_area);
            self.ring_assist.show_overlay(ui, scan_area);
        }
    }
}

impl NanonisConnection {
    /// Copy the current Nanonis state into a (new or reloaded) scan area.
    fn sync_scan_area(&mut self, scan_area: &mut ScanArea) {
        scan_area.channel_opts = self.channel_state.read().channel_opts_names().collect();
        scan_area.area_size = *self.area_size.read();
        scan_area.image_transform = Some(*self.image_transform.read());
        scan_area.tip_pos = Some(*self.tip_pos.read());
        scan_area.stamp_name_base = self.base_name.read().clone();
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
        let tip_pos = SharedState::new_default();
        let readback = SharedState::new_default();
        let course_voltages = SharedState::new_default();
        let (frame_queue_tx, frame_queue_rx) = overwrite_queue(2);
        let slow_status_init = Arc::new(AtomicBool::new(false));
        let fast_status_init = Arc::new(AtomicBool::new(false));
        let (move_sender, move_receiver) = command_channel::command_channel();

        let course_menu = CourseMotionState::new(&course_voltages, &move_sender);
        let (scan_sender, scan_receiver) = command_channel::command_channel();
        let scan_control = ScanControlState::new(&scan_sender);
        let (seq_cmd_tx, seq_cmd_rx) = crossbeam::channel::unbounded();
        let (seq_event_tx, seq_event_rx) = crossbeam::channel::unbounded();
        let sequence = SequenceState::new(seq_cmd_tx, seq_event_rx);
        let (spec_cmd_tx, spec_cmd_rx) = crossbeam::channel::unbounded();
        let (spec_event_tx, spec_event_rx) = crossbeam::channel::unbounded();
        let spectroscopy = SpectroscopyState::new(spec_cmd_tx, spec_event_rx);
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
            &course_voltages,
            &move_receiver,
            &scan_receiver,
            &readback,
            &slow_status_init,
        )
        .run(address, 6500);
        SequenceWorker::new(&ctx, seq_cmd_rx, seq_event_tx).run(address, 6500);
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
            course_menu,
            scan_control,
            sequence,
            spectroscopy,
            lithography,
            ring_assist: RingAssistState::new(),
            pending_stamp: None,
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
        let mut got = (false, false);
        if let Some(forward_data) = self.forward_data.read_new() {
            got.0 = true;
            live_image.forward_data = forward_data.clone();
            if live_image.line_dir == LineDir::Forward {
                live_image.write_and_update_texture(encoder);
            }
        }
        if let Some(backward_data) = self.backward_data.read_new() {
            got.1 = true;
            live_image.backward_data = backward_data.clone();
            if live_image.line_dir == LineDir::Backward {
                live_image.write_and_update_texture(encoder);
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
            scan_area
                .stamp
                .push_front(scan_area.live_image.stamp(encoder));
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
                    scan_area.live_image.clear_texture(encoder);
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
                scan_area.live_image.clear_texture(encoder);
            }
        }
    }
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
