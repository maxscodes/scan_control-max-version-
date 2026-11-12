use macro_rules_attribute::apply;

use crate::{codec::CodecReadDerive, codec::CodecWriteDerive, commands::Command};

/// Sets the Bias voltage to the specified value.
pub struct Set;
impl Command for Set {
    const NAME: &'static str = "Bias.Set";
    type Args = SetArgs;
    type Response = ();
}
#[apply(CodecWriteDerive)]
pub struct SetArgs {
    pub bias: f32,
}

/// Returns the Bias voltage value.
pub struct Get;
impl Command for Get {
    const NAME: &'static str = "Bias.Get";
    type Args = ();
    type Response = GetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct GetResponse {
    pub bias: f32,
}
