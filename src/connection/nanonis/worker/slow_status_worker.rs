use std::{
    f64, i32,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use eyre::Context;
use glam::{DVec2, IVec2};
use itertools::Itertools as _;
use nanonis_tcp::{
    blocking::NanonisTcp,
    error::{NanonisTcpError, NanonisTcpResult},
    ActionType, MotorAxis, MotorDir,
};
use tracing::debug;

use crate::connection::{
    nanonis::{
        channel_state::ChannelState, command_channel::CommandChannelReciever,
        course_motion::CourseMoveCommand, scan_control::ScanCommand, worker::Worker, ScanStatus,
    },
    shared_state::SharedState,
};

const DEBUG_COURSE: bool = false;

pub struct SlowStatusWorker {
    ctx: egui::Context,
    area_size: SharedState<DVec2>,
    channel_state: SharedState<ChannelState>,
    scan_status: SharedState<ScanStatus>,
    base_name: SharedState<String>,
    course_voltages: SharedState<DVec2>,
    course_reciever: CommandChannelReciever<CourseMoveCommand, ()>,
    scan_reciever: CommandChannelReciever<ScanCommand, ()>,
    init: Arc<AtomicBool>,
}
impl SlowStatusWorker {
    pub fn new(
        ctx: &egui::Context,
        area_size: &SharedState<DVec2>,
        channel_state: &SharedState<ChannelState>,
        scan_status: &SharedState<ScanStatus>,
        base_name: &SharedState<String>,
        course_voltages: &SharedState<DVec2>,
        course_reciever: &CommandChannelReciever<CourseMoveCommand, ()>,
        scan_reciever: &CommandChannelReciever<ScanCommand, ()>,
        init: &Arc<AtomicBool>,
    ) -> Self {
        Self {
            ctx: ctx.clone(),
            area_size: area_size.clone(),
            channel_state: channel_state.clone(),
            scan_status: scan_status.clone(),
            base_name: base_name.clone(),
            course_voltages: course_voltages.clone(),
            course_reciever: course_reciever.clone(),
            scan_reciever: scan_reciever.clone(),
            init: init.clone(),
        }
    }
    fn execute_scan_command(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        let Some(cmd) = self.scan_reciever.try_recv() else {
            return Ok(());
        };
        let current_dir = self.scan_status.peek().scan_dir;
        let (action, dir) = match cmd {
            ScanCommand::Start(dir) => (ActionType::Start, dir),
            ScanCommand::Pause => (ActionType::Pause, current_dir),
            ScanCommand::Resume => (ActionType::Resume, current_dir),
            ScanCommand::Stop => (ActionType::Stop, current_dir),
        };
        debug!("Executing scan action {action:?} {dir:?}");
        let result = conn.scan_action(action, dir).and_then(|_| {
            if let ScanCommand::Start(dir) = cmd {
                self.scan_status.modify_conditional(
                    |prev| prev.scan_dir != dir,
                    |prev| prev.scan_dir = dir,
                );
            }
            self.update_scan_status(conn)
        });
        self.scan_reciever.send_response(());
        self.ctx.request_repaint();
        result
    }
    fn execute_course_move(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        if let Some(CourseMoveCommand {
            steps,
            group,
            auto_withdraw,
            auto_approach
        }) = self.course_reciever.try_recv()
        {
            let args = move_args_from_steps(steps);
            debug!("Executing {args:?} on group index {group}");
            let result = if DEBUG_COURSE {
                std::thread::sleep(Duration::from_secs(2));
                Ok(())
            } else {
                || -> NanonisTcpResult<()> {
                    if auto_withdraw {
                        conn.zctrl_withdraw(true, None)?;
                    }
                    if let Some((dir, steps)) = args[0] {
                        conn.motor_start_move(dir, steps, group, true)?;
                    }
                    if let Some((dir, steps)) = args[1] {
                        conn.motor_start_move(dir, steps, group, true)?;
                    }
                    if auto_approach {
                        conn.zctrl_onoffset(true)?;
                    }
                    Ok(())
                }()
            };
            self.course_reciever.send_response(());
            result?;
        }
        Ok(())
    }
    fn update_signal_names(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        // In V5e the scan buffer channel indexes are signal slot indexes
        // (0-23), so names must come from the slot list, not the full
        // 128-signal list (which would label Z as e.g. "Input 15 (V)").
        // Older controllers without Signals.InSlotsGet use full-list indexes.
        let resp_names = match conn.signals_in_slots_get() {
            Ok(resp) => resp.names,
            Err(NanonisTcpError::Api(e)) => {
                tracing::warn!("Signals.InSlotsGet unavailable ({e}), using Signals.NamesGet");
                conn.signals_names_get()?.names
            }
            Err(e) => return Err(e),
        };
        if self.channel_state.peek().signal_names() != resp_names {
            let names = Arc::new(resp_names.into_boxed_slice());
            self.channel_state
                .modify(|prev| prev.write_signal_names(names));
        }
        Ok(())
    }
    fn update_area_transform(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        let piezo_range = conn.piezo_range_get()?;
        let area_transform = DVec2::new(
            piezo_range.range_x as f64 * 1e9,
            piezo_range.range_y as f64 * 1e9,
        );
        if self.area_size.modify_conditional(
            |prev| *prev != area_transform,
            |prev| *prev = area_transform,
        ) {
            self.ctx.request_repaint();
        }
        Ok(())
    }
    fn update_scan_status(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        let status = conn.scan_status_get()?;
        if self.scan_status.modify_conditional(
            |prev| prev.scanning != status.running,
            |prev| prev.scanning = status.running,
        ) {
            self.ctx.request_repaint();
        };
        Ok(())
    }
    fn update_channel_opts(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        let buffer = conn.scan_buffer_get()?;
        let channels = buffer
            .channel_indexes
            .into_iter()
            .map(|v| v as usize)
            .collect_vec();
        if self.channel_state.peek().options() != channels {
            self.channel_state
                .modify(|prev| prev.modify_opts(|opts| *opts = channels));
            self.ctx.request_repaint();
        }
        Ok(())
    }
    fn update_base_name(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        let frame_meta = conn.scan_props_get()?;
        if self.base_name.modify_conditional(
            |prev| *prev != frame_meta.series_name,
            |prev| *prev = frame_meta.series_name.clone(),
        ) {
            self.ctx.request_repaint();
        }
        Ok(())
    }
    fn update_course_voltages(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<()> {
        let err_pred = |e: &NanonisTcpError| matches!(e, NanonisTcpError::Api(_));
        let default = if DEBUG_COURSE { 5. } else { 0. };
        let x_data = conn
            .motor_freq_amp_get(MotorAxis::X)
            .map(|resp| resp.amplitude as f64)
            .or_else(|e| err_pred(&e).then_some(default).ok_or(e))?;
        let y_data = conn
            .motor_freq_amp_get(MotorAxis::Y)
            .map(|resp| resp.amplitude as f64)
            .or_else(|e| err_pred(&e).then_some(default).ok_or(e))?;
        let new_data = DVec2::new(x_data, y_data);
        self.course_voltages
            .modify_conditional(|prev| *prev != new_data, |prev| *prev = new_data);
        Ok(())
    }
}
impl Worker for SlowStatusWorker {
    fn work(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        self.update_area_transform(conn)
            .context("failed update_area_transform")?;
        self.update_scan_status(conn)
            .context("failed update_scan_status")?;
        self.update_channel_opts(conn)
            .context("failed update_channel_opts")?;
        self.update_base_name(conn)
            .context("failed update_base_name")?;
        self.update_course_voltages(conn)
            .context("failed update_course_voltages")?;
        self.execute_scan_command(conn)
            .context("failed execute_scan_command")?;
        self.execute_course_move(conn)
            .context("failed execute_course_move")?;
        Ok(())
    }
    fn init(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        self.update_signal_names(conn)
            .context("failed update_signal_names")?;
        self.update_area_transform(conn)
            .context("failed update_area_transform")?;
        self.update_scan_status(conn)
            .context("failed update_scan_status")?;
        self.update_channel_opts(conn)
            .context("failed update_channel_opts")?;
        self.update_base_name(conn)
            .context("failed update_base_name")?;
        self.update_course_voltages(conn)
            .context("failed update_course_voltages")?;
        self.init.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn name(&self) -> String {
        "Slow Status Worker".to_string()
    }
}

pub(crate) fn move_args_from_steps(steps: IVec2) -> [Option<(MotorDir, u16)>; 2] {
    let x_args = match steps.x {
        0 => None,
        steps @ 1..=i32::MAX => Some((MotorDir::XPos, steps.abs() as u16)),
        steps @ i32::MIN..=-1 => Some((MotorDir::XNeg, steps.abs() as u16)),
    };
    let y_args = match steps.y {
        0 => None,
        steps @ 1..=i32::MAX => Some((MotorDir::YPos, steps.abs() as u16)),
        steps @ i32::MIN..=-1 => Some((MotorDir::YNeg, steps.abs() as u16)),
    };
    [x_args, y_args]
}
