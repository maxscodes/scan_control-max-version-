use std::marker::PhantomData;

use crate::{
    codec::{CodecRead, CodecWrite, Footer, Header},
    commands::Command,
    error::{CodecError, NanonisTcpResult},
};

pub struct HasArgs;
pub struct WantsHeader;
pub struct WantsBody;

pub struct NanonisTcpFsm<'b, C: Command, S = HasArgs> {
    buf: &'b mut Vec<u8>,
    _phantom: PhantomData<(C, S)>,
}
impl<'b, C: Command> NanonisTcpFsm<'b, C> {
    pub fn new(
        buf: &'b mut Vec<u8>,
        args: &C::Args,
    ) -> NanonisTcpResult<NanonisTcpFsm<'b, C, HasArgs>> {
        let header = Header::new_for_command::<C>(args.codec_len());
        buf.resize(header.codec_len() + header.body_len as usize, 0);
        let mut buf_view = buf.as_mut_slice();
        let expected = buf_view.len();
        header.codec_write(&mut buf_view)?;
        args.codec_write(&mut buf_view)?;
        if !buf_view.is_empty() {
            Err(CodecError::WriteLenMismatch {
                expected,
                wrote: expected - buf_view.len(),
            })?;
        }
        Ok(NanonisTcpFsm {
            buf,
            _phantom: PhantomData,
        })
    }
}
impl<'b, C: Command> NanonisTcpFsm<'b, C, HasArgs> {
    pub fn bytes_to_write(&self) -> &[u8] {
        self.buf.as_slice()
    }
    pub fn prepare_for_header(self) -> NanonisTcpResult<NanonisTcpFsm<'b, C, WantsHeader>> {
        self.buf.resize(40, 0u8);
        Ok(NanonisTcpFsm {
            buf: self.buf,
            _phantom: PhantomData,
        })
    }
}
impl<'b, C: Command> NanonisTcpFsm<'b, C, WantsHeader> {
    pub fn bytes_to_read_mut(&mut self) -> &mut [u8] {
        self.buf.as_mut_slice()
    }
    pub fn prepare_for_body(self) -> NanonisTcpResult<NanonisTcpFsm<'b, C, WantsBody>> {
        let header = Header::codec_read(&mut self.buf.as_slice())?;
        if &*header.name != C::NAME {
            Err(CodecError::NameMismatch {
                expected: C::NAME.into(),
                received: header.name.to_string(),
            })?;
        }
        self.buf.resize(header.body_len as usize, 0u8);
        Ok(NanonisTcpFsm {
            buf: self.buf,
            _phantom: PhantomData,
        })
    }
}
impl<C: Command> NanonisTcpFsm<'_, C, WantsBody> {
    pub fn bytes_to_read_mut(&mut self) -> &mut [u8] {
        self.buf.as_mut_slice()
    }
    pub fn parse_response(self) -> NanonisTcpResult<C::Response> {
        let mut response_bytes = self.buf.as_slice();
        let expected = response_bytes.len();
        let response = C::Response::codec_read(&mut response_bytes);
        let footer = Footer::codec_read(&mut response_bytes).or_else(|_| {
            response_bytes = self.buf.as_slice();
            Footer::codec_read(&mut response_bytes)
        })?;
        if !response_bytes.is_empty() {
            Err(CodecError::ReadLenMismatch {
                expected,
                parsed: expected - response_bytes.len(),
            })?;
        }
        footer.into_result()?;
        Ok(response?)
    }
}
