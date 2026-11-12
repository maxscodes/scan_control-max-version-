use std::io::{Read as _, Write as _};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::codec::{ActionType, LineDir, ScanDir};
use crate::error::NanonisTcpResult;
use crate::fsm::NanonisTcpFsm;
use crate::{MotorAxis, MotorDir};
use crate::{commands::Command, commands::*};

pub struct NanonisTcp {
    stream: TcpStream,
    buf: Vec<u8>,
}
impl NanonisTcp {
    pub fn new(addr: impl ToSocketAddrs) -> std::io::Result<Self> {
        Ok(Self {
            stream: TcpStream::connect(addr)?,
            buf: Vec::new(),
        })
    }
    pub fn piezo_range_get(&mut self) -> NanonisTcpResult<piezo::RangeGetResponse> {
        self.call::<piezo::RangeGet>(&())
    }
    pub fn scan_action(&mut self, action: ActionType, dir: ScanDir) -> NanonisTcpResult<()> {
        self.call::<scan::Action>(&scan::ActionArgs { action, dir })
    }
    pub fn scan_status_get(&mut self) -> NanonisTcpResult<scan::StatusGetResponse> {
        self.call::<scan::StatusGet>(&())
    }
    pub fn scan_wait_end_of_line(
        &mut self,
        timeout: Option<Duration>,
    ) -> NanonisTcpResult<scan::WaitEndOfLineResponse> {
        self.call::<scan::WaitEndOfLine>(&scan::WaitEndOfLineArgs { timeout })
    }
    pub fn scan_wait_end_of_scan(
        &mut self,
        timeout: Option<Duration>,
    ) -> NanonisTcpResult<scan::WaitEndOfScanResponse> {
        self.call::<scan::WaitEndOfScan>(&scan::WaitEndOfScanArgs { timeout })
    }
    pub fn scan_frame_set(
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
    }
    pub fn scan_frame_get(&mut self) -> NanonisTcpResult<scan::FrameGetResponse> {
        self.call::<scan::FrameGet>(&())
    }
    pub fn scan_xy_pos_get(
        &mut self,
        wait_newest: bool,
    ) -> NanonisTcpResult<scan::XYPosGetResponse> {
        self.call::<scan::XYPosGet>(&scan::XYPosGetArgs { wait_newest })
    }
    pub fn fol_me_xy_pos_get(
        &mut self,
        wait_newest: bool,
    ) -> NanonisTcpResult<fol_me::XYPosGetResponse> {
        self.call::<fol_me::XYPosGet>(&fol_me::XYPosGetArgs { wait_newest })
    }
    /// Moves the tip to (x, y) in m.
    pub fn fol_me_xy_pos_set(
        &mut self,
        x_pos: f64,
        y_pos: f64,
        wait_end_of_move: bool,
    ) -> NanonisTcpResult<()> {
        self.call::<fol_me::XYPosSet>(&fol_me::XYPosSetArgs {
            x_pos,
            y_pos,
            wait_end_of_move,
        })
    }
    /// Sets the Follow Me tip speed in m/s (`custom_speed` false = use scan speed).
    pub fn fol_me_speed_set(&mut self, speed: f32, custom_speed: bool) -> NanonisTcpResult<()> {
        self.call::<fol_me::SpeedSet>(&fol_me::SpeedSetArgs {
            speed,
            custom_speed,
        })
    }
    pub fn fol_me_speed_get(&mut self) -> NanonisTcpResult<fol_me::SpeedGetResponse> {
        self.call::<fol_me::SpeedGet>(&())
    }
    pub fn bias_set(&mut self, bias: f32) -> NanonisTcpResult<()> {
        self.call::<bias::Set>(&bias::SetArgs { bias })
    }
    pub fn scan_props_get(&mut self) -> NanonisTcpResult<scan::PropsGetResponse> {
        self.call::<scan::PropsGet>(&())
    }
    pub fn scan_buffer_get(&mut self) -> NanonisTcpResult<scan::BufferGetResponse> {
        self.call::<scan::BufferGet>(&())
    }
    pub fn scan_buffer_set(
        &mut self,
        channel_indexes: Vec<i32>,
        px_per_line: usize,
        num_lines: usize,
    ) -> NanonisTcpResult<()> {
        self.call::<scan::BufferSet>(&scan::BufferSetArgs {
            channel_indexes,
            px_per_line,
            num_lines,
        })
    }
    pub fn scan_speed_get(&mut self) -> NanonisTcpResult<scan::SpeedGetResponse> {
        self.call::<scan::SpeedGet>(&())
    }
    pub fn scan_speed_set(&mut self, args: &scan::SpeedSetArgs) -> NanonisTcpResult<()> {
        self.call::<scan::SpeedSet>(args)
    }
    pub fn signals_names_get(&mut self) -> NanonisTcpResult<signals::NamesGetResponse> {
        self.call::<signals::NamesGet>(&())
    }
    pub fn signals_in_slots_get(&mut self) -> NanonisTcpResult<signals::InSlotsGetResponse> {
        self.call::<signals::InSlotsGet>(&())
    }
    pub fn current_get(&mut self) -> NanonisTcpResult<f32> {
        self.call::<current::Get>(&()).map(|r| r.current)
    }
    /// Save the current scan buffer to file (Nanonis naming and folder).
    pub fn scan_save(
        &mut self,
        wait_until_saved: bool,
        timeout: Option<Duration>,
    ) -> NanonisTcpResult<bool> {
        self.call::<scan::Save>(&scan::SaveArgs { wait_until_saved, timeout })
            .map(|r| r.timed_out)
    }
    pub fn scan_frame_data_grab(
        &mut self,
        channel_index: u32,
        data_dir: LineDir,
    ) -> NanonisTcpResult<scan::FrameDataGrabResponse> {
        self.call::<scan::FrameDataGrab>(&scan::FrameDataGrabArgs {
            channel_index,
            data_dir,
        })
    }
    pub fn motor_start_move(
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
    }
    pub fn motor_freq_amp_get(
        &mut self,
        axis: MotorAxis,
    ) -> NanonisTcpResult<motor::FreqAmpGetResponse> {
        self.call::<motor::FreqAmpGet>(&motor::FreqAmpGetArgs { axis })
    }
    pub fn zctrl_withdraw(
        &mut self,
        wait_finished: bool,
        timeout: Option<Duration>,
    ) -> NanonisTcpResult<()> {
        self.call::<zctrl::Withdraw>(&zctrl::WithdrawArgs {
            wait_finished,
            timeout,
        })
    }
    pub fn zctrl_onoffset(&mut self, status: bool) -> NanonisTcpResult<()> {
        self.call::<zctrl::OnOffSet>(&zctrl::OnOffSetArgs { status })
    }
    pub fn bias_get(&mut self) -> NanonisTcpResult<f32> {
        self.call::<bias::Get>(&()).map(|r| r.bias)
    }
    pub fn zctrl_setpnt_set(&mut self, setpoint: f32) -> NanonisTcpResult<()> {
        self.call::<zctrl::SetpntSet>(&zctrl::SetpntSetArgs { setpoint })
    }
    pub fn zctrl_setpnt_get(&mut self) -> NanonisTcpResult<f32> {
        self.call::<zctrl::SetpntGet>(&()).map(|r| r.setpoint)
    }
    pub fn zctrl_zpos_set(&mut self, z_pos: f32) -> NanonisTcpResult<()> {
        self.call::<zctrl::ZPosSet>(&zctrl::ZPosSetArgs { z_pos })
    }
    pub fn zctrl_zpos_get(&mut self) -> NanonisTcpResult<f32> {
        self.call::<zctrl::ZPosGet>(&()).map(|r| r.z_pos)
    }
    /// 1=Off, 2=On, 3=Hold, 4=Switching Off, 5=Safe Tip activated, 6=Withdrawing
    pub fn zctrl_status_get(&mut self) -> NanonisTcpResult<u16> {
        self.call::<zctrl::StatusGet>(&()).map(|r| r.status)
    }
    /// Runs a sweep with the Bias Spectroscopy module's settings and blocks
    /// until it finishes.
    pub fn bias_spectr_start(
        &mut self,
        get_data: bool,
        save_base_name: String,
    ) -> NanonisTcpResult<bias_spectr::StartResponse> {
        self.call::<bias_spectr::Start>(&bias_spectr::StartArgs {
            get_data,
            save_base_name,
        })
    }
    pub fn bias_spectr_stop(&mut self) -> NanonisTcpResult<()> {
        self.call::<bias_spectr::Stop>(&())
    }
    pub fn call<C: Command>(&mut self, args: &C::Args) -> NanonisTcpResult<C::Response> {
        let fsm = NanonisTcpFsm::<C>::new(&mut self.buf, args)?;
        self.stream.write_all(fsm.bytes_to_write())?;
        self.stream.flush()?;
        let mut fsm = fsm.prepare_for_header()?;
        self.stream.read_exact(fsm.bytes_to_read_mut())?;
        let mut fsm = fsm.prepare_for_body()?;
        self.stream.read_exact(fsm.bytes_to_read_mut())?;
        fsm.parse_response()
    }
}