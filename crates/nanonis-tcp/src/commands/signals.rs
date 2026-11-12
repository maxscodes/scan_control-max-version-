use macro_rules_attribute::apply;

use crate::{codec::CodecReadDerive, commands::Command};

/// Returns the signals names list of the 128 signals available in the software.
/// The 128 signals are physical inputs, physical outputs and internal channels. By searching in the list the channel’s
/// name you are interested in, you can get its index (0-127).
pub struct NamesGet;
impl Command for NamesGet {
    const NAME: &'static str = "Signals.NamesGet";
    type Args = ();
    type Response = NamesGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct NamesGetResponse {
    _names_size: i32,
    /// an array of signals names strings
    pub names: Vec<String>,
}

/// Returns the names and indexes of the 24 signals assigned to the slots of the
/// acquisition software. In V5e, Scan.BufferGet and Scan.FrameDataGrab use the
/// slot index (0-23), so the slot names here are the ones matching those indexes.
pub struct InSlotsGet;
impl Command for InSlotsGet {
    const NAME: &'static str = "Signals.InSlotsGet";
    type Args = ();
    type Response = InSlotsGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct InSlotsGetResponse {
    _names_size: i32,
    /// the names of the signals in slots, in slot order
    pub names: Vec<String>,
    /// the index (0-127) of each slot's signal in the full Signals.NamesGet list
    pub indexes: Vec<i32>,
}
