use std::io::{Read, Write};

mod allocating;
mod api;
mod fixed_string;
mod packet;
mod primitives;

pub use allocating::*;
pub use api::*;
pub(crate) use packet::*;

pub(crate) trait CodecRead: Sized {
    fn codec_read(reader: &mut impl Read) -> std::io::Result<Self>;
}
pub(crate) trait CodecWrite: Sized {
    fn codec_write(&self, writer: &mut impl Write) -> std::io::Result<()>;
    fn codec_len(&self) -> usize;
}

macro_rules! CodecWriteDerive {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_attr:meta])*
                $field_vis:vis $field:ident : $ty:ty
            ),* $(,)?
        }
    ) => {
        // Reproduce the original struct unchanged
        $(#[$attr])*
        $vis struct $name {
            $(
                $(#[$field_attr])*
                $field_vis $field: $ty
            ),*
        }

        // Now do whatever you want with names + types
        impl crate::CodecWrite for $name {
            fn codec_write(&self, writer: &mut impl ::std::io::Write) -> ::std::io::Result<()> {
                $(crate::CodecWrite::codec_write(&self.$field, writer)?;)+
                Ok(())
            }

            fn codec_len(&self) -> usize {
                0 $(+ crate::CodecWrite::codec_len(&self.$field))+
            }
        }
    };
}
pub(crate) use CodecWriteDerive;

macro_rules! CodecReadDerive {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_attr:meta])*
                $field_vis:vis $field:ident : $ty:ty
            ),* $(,)?
        }
    ) => {
        // Reproduce the original struct unchanged
        $(#[$attr])*
        $vis struct $name {
            $(
                $(#[$field_attr])*
                $field_vis $field: $ty
            ),*
        }

        // Now do whatever you want with names + types
        impl crate::CodecRead for $name {
            fn codec_read(reader: &mut impl std::io::Read) -> std::io::Result<Self> {
                $(let $field = crate::CodecRead::codec_read(reader)?;)+
                Ok(Self {
                    $($field),+
                })
            }
        }
    };
}
pub(crate) use CodecReadDerive;
