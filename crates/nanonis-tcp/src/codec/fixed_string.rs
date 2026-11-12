use std::{
    io::{Read, Write},
    ops::Deref,
};

use crate::{
    codec::{CodecRead, CodecWrite},
    commands::Command,
};

#[derive(Debug)]
pub struct FixedString<const N: usize> {
    bytes: [u8; N],
    len: usize,
}
impl<const N: usize> Deref for FixedString<N> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        // Safety:
        // The string is guaranteed to be valid utf8 on construction
        unsafe { str::from_utf8_unchecked(&self.bytes[..self.len]) }
    }
}
impl<const N: usize> FixedString<N> {
    pub const fn new_command_name<C: Command>() -> Self {
        let len = C::NAME.len();
        assert!(len <= N);
        let mut bytes = [0u8; N];
        let (string_part, _) = bytes.split_at_mut(len);
        string_part.copy_from_slice(C::NAME.as_bytes());
        Self { len, bytes }
    }
}
impl<const N: usize> CodecRead for FixedString<N> {
    fn codec_read(reader: &mut impl Read) -> std::io::Result<Self> {
        let mut bytes = [0u8; N];
        reader.read_exact(&mut bytes)?;
        let len = bytes.iter().position(|b| *b == 0).unwrap_or(N);
        str::from_utf8(&bytes[..len]).map_err(std::io::Error::other)?;
        Ok(Self { bytes, len })
    }
}
impl<const N: usize> CodecWrite for FixedString<N> {
    fn codec_write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.bytes)?;
        Ok(())
    }

    fn codec_len(&self) -> usize {
        N
    }
}
