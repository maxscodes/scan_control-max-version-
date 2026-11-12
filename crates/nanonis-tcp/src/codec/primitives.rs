use crate::codec::{CodecRead, CodecWrite};
use std::io::{Read, Write};

macro_rules! codec_primitive {
    ($t:ty) => {
        impl CodecRead for $t {
            #[inline]
            fn codec_read(reader: &mut impl Read) -> std::io::Result<Self> {
                let mut buf = [0u8; size_of::<Self>()];
                reader.read_exact(buf.as_mut_slice())?;
                Ok(Self::from_be_bytes(buf))
            }
        }
        impl CodecWrite for $t {
            #[inline]
            fn codec_write(&self, writer: &mut impl Write) -> std::io::Result<()> {
                writer.write_all(&self.to_be_bytes())
            }
            #[inline]
            fn codec_len(&self) -> usize {
                size_of::<Self>()
            }
        }
    };
}
codec_primitive!(i32);
codec_primitive!(u32);
codec_primitive!(u16);
codec_primitive!(f32);
codec_primitive!(f64);

impl CodecRead for () {
    #[inline]
    fn codec_read(_reader: &mut impl Read) -> std::io::Result<Self> {
        Ok(())
    }
}
impl CodecWrite for () {
    #[inline]
    fn codec_write(&self, _writer: &mut impl Write) -> std::io::Result<()> {
        Ok(())
    }

    #[inline]
    fn codec_len(&self) -> usize {
        0
    }
}

impl CodecRead for bool {
    fn codec_read(reader: &mut impl std::io::Read) -> std::io::Result<Self> {
        u32::codec_read(reader).map(|v| v != 0)
    }
}
impl CodecWrite for bool {
    fn codec_write(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        (*self as u32).codec_write(writer)
    }

    fn codec_len(&self) -> usize {
        size_of::<u32>()
    }
}
impl CodecRead for usize {
    fn codec_read(reader: &mut impl std::io::Read) -> std::io::Result<Self> {
        i32::codec_read(reader)?
            .try_into()
            .map_err(std::io::Error::other)
    }
}
impl CodecWrite for usize {
    fn codec_write(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        (*self as i32).codec_write(writer)
    }

    fn codec_len(&self) -> usize {
        size_of::<i32>()
    }
}
