use macro_rules_attribute::apply;

use crate::{codec::CodecReadDerive, commands::Command};

/// Returns the tunneling current value.
pub struct Get;
impl Command for Get {
    const NAME: &'static str = "Current.Get";
    type Args = ();
    type Response = GetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct GetResponse {
    /// the tunneling current (A)
    pub current: f32,
}
