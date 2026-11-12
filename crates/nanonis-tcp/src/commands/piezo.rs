use macro_rules_attribute::apply;

use crate::{codec::CodecReadDerive, commands::Command};

/// Sets the Bias voltage to the specified value.
pub struct RangeGet;
impl Command for RangeGet {
    const NAME: &'static str = "Piezo.RangeGet";
    type Args = ();
    type Response = RangeGetResponse;
}
#[apply(CodecReadDerive)]
pub struct RangeGetResponse {
    pub range_x: f32,
    pub range_y: f32,
    pub range_z: f32,
}
