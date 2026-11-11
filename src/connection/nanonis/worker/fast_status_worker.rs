use core::f64;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use eyre::Context;
use glam::{DAffine2, DVec2};
use nanonis_tcp::{
    blocking::NanonisTcp,
    commands::{
        self,
        scan::{FrameGetResponse, FrameSetArgs},
    },
    error::NanonisTcpResult,
};

use crate::connection::{nanonis::worker::Worker, shared_state::SharedState};

pub struct FastStatusWorker {
    ctx: egui::Context,
    image_transform: SharedState<DAffine2>,
    tip_pos: SharedState<DVec2>,
    init: Arc<AtomicBool>
}
impl FastStatusWorker {
    pub fn new(
        ctx: &egui::Context,
        image_transform: &SharedState<DAffine2>,
        tip_pos: &SharedState<DVec2>,
        init: &Arc<AtomicBool>,
    ) -> Self {
        Self {
            ctx: ctx.clone(),
            image_transform: image_transform.clone(),
            tip_pos: tip_pos.clone(),
            init: init.clone(),
        }
    }
    fn update_image_transform(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<bool> {
        if let Some(new_transform) = self.image_transform.read_new().as_deref().copied() {
            let frame_args = frame_from_transform(new_transform);
            conn.call::<commands::scan::FrameSet>(&frame_args)?;
            return Ok(false);
        }
        let frame = conn.scan_frame_get()?;
        let new_transform = transform_from_frame(&frame);
        let changed = self
            .image_transform
            .modify_conditional(|prev| *prev != new_transform, |val| *val = new_transform);
        Ok(changed)
    }
    fn update_tip_pos(&mut self, conn: &mut NanonisTcp) -> NanonisTcpResult<bool> {
        let resp = conn.fol_me_xy_pos_get(false)?;
        let new_pos = DVec2::new(resp.x_pos * 1e9, resp.y_pos * 1e9);
        let changed = self
            .tip_pos
            .modify_conditional(|prev| *prev != new_pos, |prev| *prev = new_pos);
        Ok(changed)
    }
}
impl Worker for FastStatusWorker {
    fn work(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        let mut update = false;
        update |= self.update_image_transform(conn).context("failed update_image_transform")?;
        update |= self.update_tip_pos(conn).context("failed update_tip_pos")?;
        if update {
            self.ctx.request_repaint();
        }
        Ok(())
    }
    fn init(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        self.update_image_transform(conn).context("failed update_image_transform")?;
        self.update_tip_pos(conn).context("failed update_tip_pos")?;
        self.init.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn name(&self) -> String {
        "Fast Status Worker".to_string()
    }
    fn interval(&self) -> std::time::Duration {
        // ~33 Hz is plenty for the tip marker and frame outline.
        std::time::Duration::from_millis(30)
    }
}

fn transform_from_frame(frame: &FrameGetResponse) -> DAffine2 {
    DAffine2::from_scale_angle_translation(
        DVec2 {
            x: frame.width as f64 * 1e9,
            y: frame.height as f64 * -1e9,
        },
        frame.angle as f64 / 180. * -f64::consts::PI,
        DVec2 {
            x: frame.center_x as f64 * 1e9,
            y: frame.center_y as f64 * 1e9,
        },
    )
}

fn frame_from_transform(transform: DAffine2) -> FrameSetArgs {
    let (
        DVec2 {
            x: scale_x,
            y: scale_y,
        },
        mut angle,
        DVec2 {
            x: trans_x,
            y: trans_y,
        },
    ) = transform.to_scale_angle_translation();
    if scale_x < 0. {
        angle += f64::consts::PI;
    }
    FrameSetArgs {
        width: (scale_x * 1e-9).abs() as f32,
        height: (scale_y * 1e-9).abs() as f32,
        angle: (angle * 180. / -f64::consts::PI) as f32,
        center_x: (trans_x * 1e-9) as f32,
        center_y: (trans_y * 1e-9) as f32,
    }
}
