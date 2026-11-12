use crate::codec::{CodecRead, CodecWrite};

pub mod bias;
pub mod bias_spectr;
pub mod current;
pub mod piezo;
pub mod scan;
pub mod signals;
pub mod motor;
pub mod fol_me;
pub mod zctrl;

#[allow(private_bounds)]
pub trait Command {
    const NAME: &'static str;
    type Args: CodecWrite;
    type Response: CodecRead;
}
