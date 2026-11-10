mod fast_status_worker;
mod frame_downloader;
mod line_worker;
mod slow_status_worker;

use std::{thread::JoinHandle, time::Duration};

pub use fast_status_worker::FastStatusWorker;
pub use frame_downloader::FrameWorker;
pub use line_worker::LineWorker;
use nanonis_tcp::{
    blocking::NanonisTcp,
    error::{NanonisTcpError, NanonisTcpResult},
};
pub(crate) use slow_status_worker::move_args_from_steps;
pub use slow_status_worker::SlowStatusWorker;
use tracing::{error, info, instrument};

pub trait Worker: Sized + Send + 'static {
    fn name(&self) -> String;
    fn init(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()>;
    fn work(&mut self, conn: &mut NanonisTcp) -> eyre::Result<()>;
    fn run(mut self, addr: impl AsRef<str>, port: u16) -> JoinHandle<()> {
        let addr = addr.as_ref().to_string();
        std::thread::Builder::new()
            .name(self.name())
            .spawn(move || self.run_inner(addr, port))
            .unwrap()
    }
    #[instrument(name = "worker", skip(self), fields(name = self.name()))]
    fn run_inner(&mut self, addr: String, port: u16) {
        'reconnect: loop {
            info!("connecting");
            let mut conn = loop {
                if let Ok(conn) = NanonisTcp::new((addr.as_str(), port)) {
                    break conn;
                }
            };
            info!("connected");
            'retry: loop {
                match self
                    .init(&mut conn)
                    .inspect_err(|e| error!("failed initializing: {:#}", e))
                {
                    Ok(_) => break,
                    Err(e) => match e.downcast::<NanonisTcpError>() {
                        Ok(NanonisTcpError::Api(_)) | Ok(NanonisTcpError::Codec(_)) => {
                            std::thread::sleep(Duration::from_secs(1));
                            continue 'retry;
                        }
                        Ok(NanonisTcpError::Io(_)) => {
                            std::thread::sleep(Duration::from_secs(1));
                            continue 'reconnect;
                        }
                        _ => {}
                    },
                }
            }
            info!("initialized");
            let mut num_retries = 0;
            'retry: loop {
                match self
                    .work(&mut conn)
                    .inspect_err(|e| error!("failed working: {:#}", e))
                {
                    Ok(_) => {
                        if num_retries != 0 {
                            info!("retry successful");
                        }
                        num_retries = 0;
                    }
                    Err(e) => match e.downcast::<NanonisTcpError>(){
                        Ok(NanonisTcpError::Api(_)) | Ok(NanonisTcpError::Codec(_)) => {
                            let dur = (2f32.powi(num_retries) * 1e-3).min(1.0);
                            let dur = Duration::from_secs_f32(dur);
                            info!("retrying after {dur:?}");
                            std::thread::sleep(dur);
                            num_retries += 1;
                            continue 'retry;
                        }
                        Ok(NanonisTcpError::Io(_)) => {
                            let dur = Duration::from_secs(1);
                            info!("reconnecting after {dur:?}");
                            std::thread::sleep(dur);
                            continue 'reconnect;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
