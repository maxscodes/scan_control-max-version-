use macro_rules_attribute::apply;

use crate::{CodecReadDerive, CodecWriteDerive, commands::Command};

/// Returns the values of the X and Y signals
pub struct XYPosGet;
impl Command for XYPosGet {
    const NAME: &'static str = "FolMe.XYPosGet";
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
    pub x_pos: f64,
    pub y_pos: f64,
}
/// Moves the tip to the given X and Y target coordinates (m).
pub struct XYPosSet;
impl Command for XYPosSet {
    const NAME: &'static str = "FolMe.XYPosSet";
    type Args = XYPosSetArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct XYPosSetArgs {
    pub x_pos: f64,
    pub y_pos: f64,
    /// if true, the function returns only once the tip has reached the target
    pub wait_end_of_move: bool,
}

/// Configures the tip speed when moving in Follow Me mode.
pub struct SpeedSet;
impl Command for SpeedSet {
    const NAME: &'static str = "FolMe.SpeedSet";
    type Args = SpeedSetArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct SpeedSetArgs {
    /// tip speed in m/s
    pub speed: f32,
    /// if true, `speed` is used; if false, the scan speed is used
    pub custom_speed: bool,
}

/// Returns the tip speed used in Follow Me mode.
pub struct SpeedGet;
impl Command for SpeedGet {
    const NAME: &'static str = "FolMe.SpeedGet";
    type Args = ();
    type Response = SpeedGetResponse;
}
#[derive(Debug, Clone, Copy)]
#[apply(CodecReadDerive)]
pub struct SpeedGetResponse {
    /// tip speed in m/s
    pub speed: f32,
    /// if true, the custom speed is used; if false, the scan speed
    pub custom_speed: bool,
}
