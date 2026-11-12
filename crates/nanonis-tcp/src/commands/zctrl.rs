use std::time::Duration;

use macro_rules_attribute::apply;

use crate::{CodecReadDerive, CodecWriteDerive, commands::Command};

/// Withdraws the tip. 
/// This function switches off the Z-Controller and then fully withdraws the tip (to the upper limit of the Z-piezo range).
pub struct Withdraw;
impl Command for Withdraw {
    const NAME: &'static str = "ZCtrl.Withdraw";
    type Args = WithdrawArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct WithdrawArgs {
    pub wait_finished: bool,
    pub timeout: Option<Duration>
}

/// Switches the Z-Controller On or Off.
pub struct OnOffSet;
impl Command for OnOffSet {
    const NAME: &'static str = "ZCtrl.OnOffSet";
    type Args = OnOffSetArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct OnOffSetArgs {
    pub status: bool,
}

/// Sets the setpoint of the Z-Controller.
pub struct SetpntSet;
impl Command for SetpntSet {
    const NAME: &'static str = "ZCtrl.SetpntSet";
    type Args = SetpntSetArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct SetpntSetArgs {
    pub setpoint: f32,
}

/// Returns the setpoint of the Z-Controller.
pub struct SetpntGet;
impl Command for SetpntGet {
    const NAME: &'static str = "ZCtrl.SetpntGet";
    type Args = ();
    type Response = SetpntGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct SetpntGetResponse {
    pub setpoint: f32,
}

/// Sets the Z position of the tip. The Z-Controller must be off.
pub struct ZPosSet;
impl Command for ZPosSet {
    const NAME: &'static str = "ZCtrl.ZPosSet";
    type Args = ZPosSetArgs;
    type Response = ();
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct ZPosSetArgs {
    /// Z position in m
    pub z_pos: f32,
}

/// Returns the current Z position of the tip.
pub struct ZPosGet;
impl Command for ZPosGet {
    const NAME: &'static str = "ZCtrl.ZPosGet";
    type Args = ();
    type Response = ZPosGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct ZPosGetResponse {
    /// Z position in m
    pub z_pos: f32,
}

/// Returns the status of the Z-Controller.
pub struct StatusGet;
impl Command for StatusGet {
    const NAME: &'static str = "ZCtrl.StatusGet";
    type Args = ();
    type Response = StatusGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct StatusGetResponse {
    /// 1=Off, 2=On, 3=Hold, 4=Switching Off, 5=Safe Tip activated, 6=Withdrawing
    pub status: u16,
}
