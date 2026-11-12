use std::io::{Read, Write};

use crate::codec::{CodecRead, CodecWrite};

impl CodecRead for String {
    fn codec_read(reader: &mut impl Read) -> std::io::Result<Self> {
        let len = i32::codec_read(reader)?;
        let mut buf = vec![0u8; len as usize];
        reader.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(std::io::Error::other)
    }
}
impl CodecWrite for String {
    fn codec_write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        (self.len() as i32).codec_write(writer)?;
        writer.write_all(self.as_bytes())?;
        Ok(())
    }
    #[inline]
    fn codec_len(&self) -> usize {
        size_of::<i32>() + self.len()
    }
}

impl<T: CodecRead> CodecRead for Vec<T> {
    fn codec_read(reader: &mut impl Read) -> std::io::Result<Self> {
        (0..i32::codec_read(reader)?)
            .map(|_| T::codec_read(reader))
            .collect()
    }
}
impl<T: CodecWrite> CodecWrite for Vec<T> {
    fn codec_write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        (self.len() as i32).codec_write(writer)?;
        self.iter().try_for_each(|v| v.codec_write(writer))
    }
    #[inline]
    fn codec_len(&self) -> usize {
        size_of::<i32>() + self.iter().map(T::codec_len).sum::<usize>()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Vec2D<T> {
    pub size: [usize; 2],
    pub data: Vec<T>,
}
impl<T: CodecRead> CodecRead for Vec2D<T> {
    fn codec_read(reader: &mut impl Read) -> std::io::Result<Self> {
        let size = [usize::codec_read(reader)?, usize::codec_read(reader)?];
        let data = (0..size[0] * size[1])
            .map(|_| T::codec_read(reader))
            .collect::<std::io::Result<_>>()?;
        Ok(Self { size, data })
    }
}
impl<T: CodecWrite> CodecWrite for Vec2D<T> {
    fn codec_write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        (self.size[0]).codec_write(writer)?;
        (self.size[1]).codec_write(writer)?;
        self.data.iter().try_for_each(|v| v.codec_write(writer))
    }
    #[inline]
    fn codec_len(&self) -> usize {
        usize::codec_len(&0) * 2 + self.data.iter().map(T::codec_len).sum::<usize>()
    }
}
