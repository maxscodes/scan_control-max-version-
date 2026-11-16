use std::fmt::Display;

use egui::emath;

#[repr(transparent)]
pub struct EngFmt<N: emath::Numeric>(pub N);
impl<N: emath::Numeric> Display for EngFmt<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = self.0.to_f64();
        let mag = (val.abs().log10() / 3.).floor();
        let scaled = val / (10f64).powf(mag * 3.);
        let suf = match mag as i32 {
            4 => Some("T"),
            3 => Some("G"),
            2 => Some("M"),
            1 => Some("k"),
            -1 => Some("m"),
            -2 => Some("μ"),
            -3 => Some("n"),
            -4 => Some("p"),
            -5 => Some("f"),
            _ => None,
        };
        if let Some(suf) = suf {
            f64::fmt(&scaled, f)?;
            write!(f, "{}", suf)?;
        } else {
            f64::fmt(&val, f)?;
        }
        Ok(())
    }
}
impl<N: emath::Numeric> EngFmt<N> {
    pub fn multplier_for_char(c: char) -> Option<N> {
        match c {
            'T' => Some(1e12),
            'G' => Some(1e9),
            'M' => Some(1e6),
            'k' => Some(1e3),
            'm' => Some(1e-3),
            'μ' => Some(1e-6),
            'u' => Some(1e-6),
            'n' => Some(1e-9),
            'p' => Some(1e-12),
            'f' => Some(1e-15),
            _ => None,
        }
        .map(|v| N::from_f64(v))
    }
}
