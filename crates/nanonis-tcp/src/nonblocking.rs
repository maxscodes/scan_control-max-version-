use std::time::Duration;

use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::{TcpStream, ToSocketAddrs},
};

use crate::{
    MotorAxis, MotorDir,
    codec::{ActionType, LineDir, ScanDir},
    commands::{Command, *},
    error::NanonisTcpResult,
    fsm::NanonisTcpFsm,
};

pub struct NanonisTcp {
    stream: TcpStream,
    buf: Vec<u8>,
}
impl NanonisTcp {
    pub async fn new(addr: impl ToSocketAddrs) -> std::io::Result<Self> {
        Ok(Self {
            stream: TcpStream::connect(addr).await?,
            buf: Vec::new(),
        })
    }
    pub async fn piezo_range_get(&mut self) -> NanonisTcpResult<piezo::RangeGetResponse> {
        self.call::<piezo::RangeGet>(&()).await
    }
    pub async fn scan_action(&mut self, action: ActionType, dir: ScanDir) -> NanonisTcpResult<()> {
        self.call::<scan::Action>(&scan::ActionArgs { action, dir })
            .await
    }
    pub async fn scan_status_get(&mut self) -> NanonisTcpResult<scan::StatusGetResponse> {
        self.call::<scan::StatusGet>(&()).await
    }
    pub async fn scan_wait_end_of_line(
        &mut self,
        timeout: Option<Duration>,
    ) -> NanonisTcpResult<scan::WaitEndOfLineResponse> {
        self.call::<scan::WaitEndOfLine>(&scan::WaitEndOfLineArgs { timeout })
            .await
    }
    pub async fn scan_wait_end_of_scan(
        &mut self,
        timeout: Option<Duration>,
    ) -> NanonisTcpResult<scan::WaitEndOfScanResponse> {
        self.call::<scan::WaitEndOfScan>(&scan::WaitEndOfScanArgs { timeout })
            .await
    }
    pub async fn scan_frame_set(
        &mut self,
        center_x: f32,
        center_y: f32,
        width: f32,
        height: f32,
        angle: f32,
    ) -> NanonisTcpResult<()> {
        self.call::<scan::FrameSet>(&scan::FrameSetArgs {
            center_x,
            center_y,
            width,
            height,
            angle,
        })
        .await
    }
    pub async fn scan_frame_get(&mut self) -> NanonisTcpResult<scan::FrameGetResponse> {
        self.call::<scan::FrameGet>(&()).await
    }
    pub async fn scan_xy_pos_get(
        &mut self,
        wait_newest: bool,
    ) -> NanonisTcpResult<scan::XYPosGetResponse> {
        self.call::<scan::XYPosGet>(&scan::XYPosGetArgs { wait_newest })
            .await
    }
    pub async fn fol_me_xy_pos_get(
        &mut self,
        wait_newest: bool,
    ) -> NanonisTcpResult<fol_me::XYPosGetResponse> {
        self.call::<fol_me::XYPosGet>(&fol_me::XYPosGetArgs { wait_newest })
            .await
    }
    pub async fn bias_set(&mut self, bias: f32) -> NanonisTcpResult<()> {
        self.call::<bias::Set>(&bias::SetArgs { bias }).await
    }
    pub async fn scan_props_get(&mut self) -> NanonisTcpResult<scan::PropsGetResponse> {
        self.call::<scan::PropsGet>(&()).await
    }
    pub async fn scan_buffer_get(&mut self) -> NanonisTcpResult<scan::BufferGetResponse> {
        self.call::<scan::BufferGet>(&()).await
    }
    pub async fn signals_names_get(&mut self) -> NanonisTcpResult<signals::NamesGetResponse> {
        self.call::<signals::NamesGet>(&()).await
    }
    pub async fn signals_in_slots_get(&mut self) -> NanonisTcpResult<signals::InSlotsGetResponse> {
        self.call::<signals::InSlotsGet>(&()).await
    }
    pub async fn current_get(&mut self) -> NanonisTcpResult<f32> {
        self.call::<current::Get>(&()).await.map(|r| r.current)
    }
    /// Save the current scan buffer to file (Nanonis naming and folder).
    pub async fn scan_save(
        &mut self,
        wait_until_saved: bool,
        timeout: Option<Duration>,
    ) -> NanonisTcpResult<bool> {
        self.call::<scan::Save>(&scan::SaveArgs { wait_until_saved, timeout }).await
            .map(|r| r.timed_out)
    }
    pub async fn scan_frame_data_grab(
        &mut self,
        channel_index: u32,
        data_dir: LineDir,
    ) -> NanonisTcpResult<scan::FrameDataGrabResponse> {
        self.call::<scan::FrameDataGrab>(&scan::FrameDataGrabArgs {
            channel_index,
            data_dir,
        })
        .await
    }
    pub async fn motor_start_move(
        &mut self,
        dir: MotorDir,
        num_steps: u16,
        group: u32,
        blocking: bool,
    ) -> NanonisTcpResult<()> {
        self.call::<motor::StartMove>(&motor::StartMoveArgs {
            dir,
            num_steps,
            group,
            blocking,
        })
        .await
    }
    pub async fn motor_freq_amp_get(
        &mut self,
        axis: MotorAxis,
    ) -> NanonisTcpResult<motor::FreqAmpGetResponse> {
        self.call::<motor::FreqAmpGet>(&motor::FreqAmpGetArgs { axis })
            .await
    }
    pub async fn zctrl_withdraw(
        &mut self,
        wait_finished: bool,
        timeout: Option<Duration>,
    ) -> NanonisTcpResult<()> {
        self.call::<zctrl::Withdraw>(&zctrl::WithdrawArgs {
            wait_finished,
            timeout,
        })
        .await
    }
    pub async fn zctrl_onoffset(&mut self, status: bool) -> NanonisTcpResult<()> {
        self.call::<zctrl::OnOffSet>(&zctrl::OnOffSetArgs { status })
            .await
    }
    pub async fn call<C: Command>(&mut self, args: &C::Args) -> NanonisTcpResult<C::Response> {
        let fsm = NanonisTcpFsm::<C>::new(&mut self.buf, args)?;
        self.stream.write_all(fsm.bytes_to_write()).await?;
        self.stream.flush().await?;
        let mut fsm = fsm.prepare_for_header()?;
        self.stream.read_exact(fsm.bytes_to_read_mut()).await?;
        let mut fsm = fsm.prepare_for_body()?;
        self.stream.read_exact(fsm.bytes_to_read_mut()).await?;
        fsm.parse_response()
    }
}
