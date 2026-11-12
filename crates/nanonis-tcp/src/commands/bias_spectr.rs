use macro_rules_attribute::apply;

use crate::{
    codec::{CodecReadDerive, CodecWriteDerive, Vec2D},
    commands::Command,
};

/// Starts a bias spectroscopy in the Bias Spectroscopy module.
///
/// Sweep range, points, timing and recorded channels are taken from the
/// module's current settings. Blocks until the sweep has finished.
pub struct Start;
impl Command for Start {
    const NAME: &'static str = "BiasSpectr.Start";
    type Args = StartArgs;
    type Response = StartResponse;
}
#[derive(Debug)]
#[apply(CodecWriteDerive)]
pub struct StartArgs {
    /// if true, the function returns the acquired data
    pub get_data: bool,
    /// base name used by the saved files. If empty, the base name set in the
    /// module is used. Files are only saved if saving is enabled in the module.
    pub save_base_name: String,
}
#[derive(Debug, Clone)]
#[apply(CodecReadDerive)]
pub struct StartResponse {
    _names_size: i32,
    /// names of the acquired channels, in the same order as the data rows
    pub channel_names: Vec<String>,
    /// one row per channel, one column per sweep point
    pub data: Vec2D<f32>,
    /// the parameters of the sweep (see the Nanonis TCP protocol manual)
    pub parameters: Vec<f32>,
}

/// Stops the current bias spectroscopy measurement.
pub struct Stop;
impl Command for Stop {
    const NAME: &'static str = "BiasSpectr.Stop";
    type Args = ();
    type Response = ();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRead;

    #[test]
    fn start_response_decodes() {
        let names = ["Bias (V)", "Current (A)"];
        let mut b = Vec::new();
        let names_size: i32 = names.iter().map(|n| 4 + n.len() as i32).sum();
        b.extend(names_size.to_be_bytes());
        b.extend((names.len() as i32).to_be_bytes());
        for n in names {
            b.extend((n.len() as i32).to_be_bytes());
            b.extend(n.as_bytes());
        }
        // 2 rows (channels) x 3 columns (points)
        b.extend(2i32.to_be_bytes());
        b.extend(3i32.to_be_bytes());
        for v in [-1f32, 0., 1., -1e-10, 0., 1e-10] {
            b.extend(v.to_be_bytes());
        }
        b.extend(1i32.to_be_bytes());
        b.extend(0.5f32.to_be_bytes());

        let r = StartResponse::codec_read(&mut b.as_slice()).unwrap();
        assert_eq!(r.channel_names, names);
        assert_eq!(r.data.size, [2, 3]);
        assert_eq!(r.data.data[3], -1e-10);
        assert_eq!(r.parameters, vec![0.5]);
    }
}
