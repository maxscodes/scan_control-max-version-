use std::time::Duration;

use eyre::Context;
use nanonis_tcp::{blocking::NanonisTcp, LineDir, ScanMovementType};
use tracing::trace;

use crate::connection::{
    nanonis::{worker::Worker, OverwriteQueueSender, ScanStatus},
    shared_state::SharedState,
};

pub struct LineWorker {
    queue: OverwriteQueueSender<LineDir>,
    scan_status: SharedState<ScanStatus>,
}
impl LineWorker {
    pub fn new(
        queue: &OverwriteQueueSender<LineDir>,
        scan_status: &SharedState<ScanStatus>,
    ) -> Self {
        Self {
            queue: queue.clone(),
            scan_status: scan_status.clone(),
        }
    }
    pub fn line_wait(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        let resp = conn
            .scan_wait_end_of_line(None)
            .context("failed scan_wait_end_of_line")?;
        let dir = match resp.movement_type {
            ScanMovementType::Scan(dir) => dir,
            _ => return Ok(()),
        };
        trace!("line finished {} {:?}", resp.line_number, dir);
        self.scan_status.modify_conditional(
            |prev| prev.line_number != resp.line_number as u32 || prev.line_dir != dir,
            |prev| {
                prev.line_number = resp.line_number as u32;
                prev.scanning = true;
                prev.line_dir = dir;
            },
        );
        self.queue.send(dir);
        Ok(())
    }
    pub fn dumb_line_wait(&mut self, _conn: &mut NanonisTcp) -> eyre::Result<()> {
        if self.scan_status.peek().scanning {
            for _ in 0..5{
                std::thread::sleep(Duration::from_millis(100));
                trace!("Sending dumb line finished");
                self.queue.send(LineDir::Forward);
                self.queue.send(LineDir::Backward);
            }
        }
        Ok(())
    }
}
impl Worker for LineWorker {
    fn init(&mut self, _conn: &mut NanonisTcp) -> eyre::Result<()> {
        Ok(())
    }
    fn work(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()> {
        // self.line_wait(conn)?;
        self.dumb_line_wait(conn)?;
        Ok(())
    }
    fn name(&self) -> String {
        "Line Waiter".to_string()
    }
    fn interval(&self) -> Duration {
        // Without this the loop spins a CPU core while no scan is running.
        Duration::from_millis(100)
    }
}
