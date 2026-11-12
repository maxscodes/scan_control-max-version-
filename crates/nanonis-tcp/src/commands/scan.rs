use std::time::Duration;

use macro_rules_attribute::apply;

use crate::{
    codec::{
        ActionType, CodecRead, CodecReadDerive, CodecWriteDerive, LineDir, ScanDir,
        ScanMovementType, Vec2D,
    },
    commands::Command,
};

/// Starts, stops, pauses or resumes a scan.
pub struct Action;
impl Command for Action {
    const NAME: &'static str = "Scan.Action";
    type Args = ActionArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct ActionArgs {
    /// sets which action to perform, where 0=Start, 1=Stop, 2=Pause, 3=Resume,
    /// 4=Freeze, 5=Unfreeze, 6=Go to Center
    pub action: ActionType,
    /// if 1, scan direction is set to up. If 0, direction is down
    pub dir: ScanDir,
}

/// Returns if the scan is running or not.
pub struct StatusGet;
impl Command for StatusGet {
    const NAME: &'static str = "Scan.StatusGet";
    type Args = ();
    type Response = StatusGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct StatusGetResponse {
    /// means that if it is 1, scan is running. If 0, scan is not running
    pub running: bool,
}

/// Returns the values of the X and Y signals
pub struct XYPosGet;
impl Command for XYPosGet {
    const NAME: &'static str = "Scan.XYPosGet";
    type Args = XYPosGetArgs;
    type Response = XYPosGetResponse;
}
#[apply(CodecWriteDerive)]
pub struct XYPosGetArgs {
    /// selects whether the function returns the next available signal value
    // or if it waits for a full period of new data. If False, this function returns a value 0 to Tap seconds after being
    // called. If True, the function discards the first oversampled signal value received but returns the second
    // value received. Thus, the function returns a value Tap to 2*Tap seconds after being called.
    pub wait_newest: bool,
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct XYPosGetResponse {
    pub x_pos: f32,
    pub y_pos: f32,
}

/// Waits for the End-of-Line.
/// This function returns only when an End-of-Line or timeout occurs (whichever occurs first).
pub struct WaitEndOfLine;
impl Command for WaitEndOfLine {
    const NAME: &'static str = "Scan.WaitEndOfLine";
    type Args = WaitEndOfLineArgs;
    type Response = WaitEndOfLineResponse;
}
#[apply(CodecWriteDerive)]
pub struct WaitEndOfLineArgs {
    /// sets how many milliseconds this function waits for an End-of-Scan. If –1, it waits indefinitely
    pub timeout: Option<Duration>,
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct WaitEndOfLineResponse {
    /// means that if it is 1, the function timed-out. If 0, it didn’t time-out
    pub timed_out: bool,
    /// the line number of the last completed line
    pub line_number: i32,
    /// can be forward (0) or backward (1) while scanning, moved to the scan
    /// frame center (2), or moved to start point of the scan frame right before starting to scan (3)
    pub movement_type: ScanMovementType,
    /// the pass number of the last completed line (relevant when MultiPass is enabled)
    pub pass_number: usize,
}

/// Waits for the End-of-Scan.
/// This function returns only when an End-of-Scan or timeout occurs (whichever occurs first).
pub struct WaitEndOfScan;
impl Command for WaitEndOfScan {
    const NAME: &'static str = "Scan.WaitEndOfScan";
    type Args = WaitEndOfScanArgs;
    type Response = WaitEndOfScanResponse;
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct WaitEndOfScanArgs {
    pub timeout: Option<Duration>,
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct WaitEndOfScanResponse {
    pub timed_out: bool,
    pub file_path: String,
}

/// Returns the scan frame parameters.
pub struct FrameGet;
impl Command for FrameGet {
    const NAME: &'static str = "Scan.FrameGet";
    type Args = ();
    type Response = FrameGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct FrameGetResponse {
    /// the X position of the scan frame center
    pub center_x: f32,
    /// the Y position of the scan frame center
    pub center_y: f32,
    /// the width of the scan frame
    pub width: f32,
    /// the height of the scan frame
    pub height: f32,
    /// the angle of the scan frame (positive angle means clockwise rotation)
    pub angle: f32,
}

/// Configures the scan frame parameters.
pub struct FrameSet;
impl Command for FrameSet {
    const NAME: &'static str = "Scan.FrameSet";
    type Args = FrameSetArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct FrameSetArgs {
    /// the X position of the scan frame center
    pub center_x: f32,
    /// the Y position of the scan frame center
    pub center_y: f32,
    /// the width of the scan frame
    pub width: f32,
    /// the height of the scan frame
    pub height: f32,
    /// the angle of the scan frame (positive angle means clockwise rotation)
    pub angle: f32,
}

/// Returns the scan buffer parameters.
pub struct BufferGet;
impl Command for BufferGet {
    const NAME: &'static str = "Scan.BufferGet";
    type Args = ();
    type Response = BufferGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct BufferGetResponse {
    /// the indexes of recorded channels. The index is comprised between 0
    /// and 127, and it corresponds to the full list of signals available in the system.
    /// To get the signal name and its corresponding index in the list of the 128 available signals in the Nanonis
    /// Controller, use the Signal.NamesGet function, or check the RT Idx value in the Signals Manager module.
    pub channel_indexes: Vec<i32>,
    /// the number of pixels per line
    pub px_per_line: usize,
    /// the number of scan lines
    pub num_lines: usize,
}

/// Returns some of the scan parameters.
pub struct PropsGet;
impl Command for PropsGet {
    const NAME: &'static str = "Scan.PropsGet";
    type Args = ();
    type Response = PropsGetResponse;
}
#[cfg(feature = "v5e")]
#[apply(CodecReadDerive)]
#[derive(Debug)]
pub struct PropsGetResponse {
    ///  indicates whether the scan continues or stops when a frame has been
    /// completed. 0 means Off, and 1 is On
    pub continuous_scan: bool,
    /// indicates whether the scan direction changes when a frame has been
    /// completed. 0 means Off, and 1 is On
    pub bouncy_scan: bool,
    /// defines the save behavior when a frame has been completed. "All" saves all the
    /// future images. "Next" only saves the next frame. 0 is All, 1 is Next, and 2 means Off
    pub autosave: u32,
    /// base name used for the saved images
    pub series_name: String,
    /// comment saved in the file
    pub comment: String,
    _modules_names_size: i32,
    /// an array of modules names strings
    pub modules_names: Vec<String>,
    /// an array containing the number of parameters per module
    pub params_per_mod: Vec<usize>,
    /// returns the parameters that are going to be saved in the header of the image
    /// files. Each row of parameters belongs to a different module.
    pub parameters: Vec2D<String>,
}
#[cfg(not(feature = "v5e"))]
#[apply(CodecReadDerive)]
#[derive(Debug)]
pub struct PropsGetResponse {
    ///  indicates whether the scan continues or stops when a frame has been
    /// completed. 0 means Off, and 1 is On
    pub continuous_scan: bool,
    /// indicates whether the scan direction changes when a frame has been
    /// completed. 0 means Off, and 1 is On
    pub bouncy_scan: bool,
    /// defines the save behavior when a frame has been completed. "All" saves all the
    /// future images. "Next" only saves the next frame. 0 is All, 1 is Next, and 2 means Off
    pub autosave: u32,
    /// base name used for the saved images
    pub series_name: String,
    /// comment saved in the file
    pub comment: String,
}

/// Returns the scan data of the selected frame.
pub struct FrameDataGrab;
impl Command for FrameDataGrab {
    const NAME: &'static str = "Scan.FrameDataGrab";
    type Args = FrameDataGrabArgs;
    type Response = FrameDataGrabResponse;
}
#[apply(CodecWriteDerive)]
pub struct FrameDataGrabArgs {
    /// Selects which channel to get the data from.
    /// The channel must be one of the acquired channels.
    /// The list of acquired channels while scanning can be configured by the function Scan.BufferSet.
    /// The index is comprised between 0 and 127, and it corresponds to the full list of signals available in the
    /// system.
    /// To get the signal name and its corresponding index in the list of the 128 available signals in the Nanonis
    /// Controller, use the Signal.NamesGet function, or check the RT Idx value in the Signals Manager module.
    pub channel_index: u32,
    /// Selects the data direction, where 1 is forward, and 0 is backward
    pub data_dir: LineDir,
}
#[derive(Debug, Clone)]
pub struct FrameDataGrabResponse {
    /// the name of the channel selected by Channel index
    pub channel_name: String,
    /// the scan frame data of the selected channel
    pub scan_data: Vec2D<f32>,
    /// the scan direction, where 1 is up, and 0 is down
    pub scan_dir: ScanDir,
}
impl CodecRead for FrameDataGrabResponse {
    fn codec_read(reader: &mut impl std::io::Read) -> std::io::Result<Self> {
        let channel_name = String::codec_read(reader)?;
        let scan_data = (!channel_name.is_empty())
            .then(|| <Vec2D<f32>>::codec_read(reader))
            .transpose()?
            .unwrap_or_default();
        let scan_dir = ScanDir::codec_read(reader)?;
        Ok(Self {
            channel_name,
            scan_data,
            scan_dir,
        })
    }
}

/// Configures the scan buffer parameters (recorded channels and resolution).
pub struct BufferSet;
impl Command for BufferSet {
    const NAME: &'static str = "Scan.BufferSet";
    type Args = BufferSetArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct BufferSetArgs {
    /// the indexes of recorded channels (see [`BufferGetResponse`])
    pub channel_indexes: Vec<i32>,
    /// the number of pixels per line
    pub px_per_line: usize,
    /// the number of scan lines
    pub num_lines: usize,
}

/// Returns the scan speed parameters.
pub struct SpeedGet;
impl Command for SpeedGet {
    const NAME: &'static str = "Scan.SpeedGet";
    type Args = ();
    type Response = SpeedGetResponse;
}
#[derive(Debug, Clone, Copy)]
#[apply(CodecReadDerive)]
pub struct SpeedGetResponse {
    /// forward linear speed in m/s
    pub fwd_speed: f32,
    /// backward linear speed in m/s
    pub bwd_speed: f32,
    /// forward time per line in s
    pub fwd_line_time: f32,
    /// backward time per line in s
    pub bwd_line_time: f32,
    /// which parameter is kept constant when the frame changes:
    /// 0 = no change, 1 = linear speed, 2 = time per line
    pub keep_constant: u16,
    /// backward speed relative to the forward speed
    pub speed_ratio: f32,
}

/// Configures the scan speed parameters.
pub struct SpeedSet;
impl Command for SpeedSet {
    const NAME: &'static str = "Scan.SpeedSet";
    type Args = SpeedSetArgs;
    type Response = ();
}
#[derive(Debug, Clone, Copy)]
#[apply(CodecWriteDerive)]
pub struct SpeedSetArgs {
    /// forward linear speed in m/s
    pub fwd_speed: f32,
    /// backward linear speed in m/s
    pub bwd_speed: f32,
    /// forward time per line in s
    pub fwd_line_time: f32,
    /// backward time per line in s
    pub bwd_line_time: f32,
    /// which parameter is kept constant: 0 = no change, 1 = linear speed,
    /// 2 = time per line
    pub keep_constant: u16,
    /// backward speed relative to the forward speed
    pub speed_ratio: f32,
}

/// Saves the current scan databuffer into file. If `wait_until_saved` is set,
/// returns only when the data has been saved or the timeout occurs.
pub struct Save;
impl Command for Save {
    const NAME: &'static str = "Scan.Save";
    type Args = SaveArgs;
    type Response = SaveResponse;
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct SaveArgs {
    pub wait_until_saved: bool,
    /// None waits indefinitely
    pub timeout: Option<Duration>,
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct SaveResponse {
    pub timed_out: bool,
}
