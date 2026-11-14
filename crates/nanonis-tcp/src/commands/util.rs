use macro_rules_attribute::apply;

use crate::{codec::CodecReadDerive, commands::Command};

/// Returns the session path (the folder Nanonis saves data into).
pub struct SessionPathGet;
impl Command for SessionPathGet {
    const NAME: &'static str = "Util.SessionPathGet";
    type Args = ();
    type Response = SessionPathGetResponse;
}
#[derive(Debug)]
#[apply(CodecReadDerive)]
pub struct SessionPathGetResponse {
    pub session_path: String,
}
