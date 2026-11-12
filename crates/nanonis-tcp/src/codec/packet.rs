use std::io::Read;

use macro_rules_attribute::apply;

use crate::{
    codec::{CodecRead, CodecReadDerive, CodecWriteDerive, fixed_string::FixedString},
    commands::Command,
    error::{NanonisTcpError, NanonisTcpResult},
};

#[derive(Debug)]
#[apply(CodecReadDerive)]
#[apply(CodecWriteDerive)]
pub struct Header {
    pub name: FixedString<32>,
    pub body_len: i32,
    pub response: u16,
    _pad: u16,
}
impl Header {
    pub const fn new_for_command<C: Command>(body_len: usize) -> Self {
        Self {
            name: const { FixedString::<32>::new_command_name::<C>() },
            body_len: body_len as i32,
            response: 1,
            _pad: 0,
        }
    }
}

#[derive(Debug)]
pub struct Footer {
    pub description: Option<String>,
}
impl Footer {
    pub fn into_result(self) -> NanonisTcpResult<()> {
        match self.description {
            Some(desc) => Err(NanonisTcpError::Api(desc)),
            None => Ok(()),
        }
    }
}
impl CodecRead for Footer {
    fn codec_read(reader: &mut impl Read) -> std::io::Result<Self> {
        let status = u32::codec_read(reader)?;
        let description = match status {
            0 => {
                let _ = i32::codec_read(reader)?;
                None
            }
            _ => {
                let mut desc = String::codec_read(reader)?;
                newline_replace(&mut desc);
                Some(desc)
            }
        };
        Ok(Self { description })
    }
}
fn newline_replace(string: &mut str) {
    // Safety:
    // we are replacing bytes which are guaranteed to be single byte codepoints
    // with spaces, which are also single byte codepoints.  Thus, the string
    // remains valid utf-8
    for byte in unsafe { string.as_bytes_mut() } {
        if *byte == b'\n' || *byte == b'\r' {
            *byte = b' ';
        }
    }
}
