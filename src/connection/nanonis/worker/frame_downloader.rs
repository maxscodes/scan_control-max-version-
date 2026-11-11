use std::sync::Arc;

use eyre::Context;
use nanonis_tcp::{
    blocking::NanonisTcp, commands::scan::FrameDataGrabResponse, LineDir,
};
use tracing::trace;

use crate::connection::{
    live_image::FrameData,
    nanonis::{ChannelState, ScanStatus, worker::Worker},
    queue::OverwriteQueueReceiver,
    shared_state::SharedState,
};

pub struct FrameWorker {
    ctx: egui::Context,
    forward_data: SharedState<FrameData>,
    backward_data: SharedState<FrameData>,
    queue: OverwriteQueueReceiver<LineDir>,
    channel_state: SharedState<ChannelState>,
    scan_status: SharedState<ScanStatus>,
}
impl FrameWorker {
    pub fn new(
        ctx: &egui::Context,
        forward_data: &SharedState<FrameData>,
        backward_data: &SharedState<FrameData>,
        queue: OverwriteQueueReceiver<LineDir>,
        channel_state: &SharedState<ChannelState>,
        scan_status: &SharedState<ScanStatus>,
    ) -> Self {
        Self {
            ctx: ctx.clone(),
            forward_data: forward_data.clone(),
            backward_data: backward_data.clone(),
            channel_state: channel_state.clone(),
            scan_status: scan_status.clone(),
            queue,
        }
    }
}
impl Worker for FrameWorker {
    fn init(&mut self, _conn: &mut NanonisTcp) -> eyre::Result<()> {
        Ok(())
    }

    fn work(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        let dir = self.queue.recv();
        let Some(ch) = self.channel_state.read().selection else {
            return Ok(());
        };
        trace!("downloading frame {} {:?}", ch, dir);
        let resp = conn.scan_frame_data_grab(ch as u32, dir).context("failed scan_frame_data_grab")?;
        self.scan_status.modify_conditional(
            |prev| prev.scan_dir != resp.scan_dir,
            |prev| prev.scan_dir = resp.scan_dir,
        );
        let mut frame = FrameData::from(resp);
        if (frame.size[0] * frame.size[1]) == 0 {
            frame = FrameData::default();
        }
        match dir {
            LineDir::Forward => self.forward_data.write(frame),
            LineDir::Backward => self.backward_data.write(frame),
        }
        self.ctx.request_repaint();
        Ok(())
    }
    fn name(&self) -> String {
        "Frame Downloader".to_string()
    }
}

impl From<FrameDataGrabResponse> for FrameData {
    fn from(frame: FrameDataGrabResponse) -> Self {
        Self {
            size: [
                frame.scan_data.size[1] as u32,
                frame.scan_data.size[0] as u32,
            ],
            data: Arc::new(frame.scan_data.data.into_boxed_slice()),
        }
    }
}
