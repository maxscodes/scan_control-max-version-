//! Data types and pure helpers for the I(V) series: settings, curve data,
//! channel extraction from a BiasSpectr.Start response, and .dat export.

use std::{fmt::Write as _, path::Path};

use serde::{Deserialize, Serialize};

/// Settings for a series of I(V) curves at stepped tip heights. Sweep range,
/// points and recorded channels come from the Nanonis Bias Spectroscopy module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IvSeriesSettings {
    pub num_curves: u32,
    /// Passed to Nanonis as the save base name (curve `n` gets `_IV_BATCH_00n`)
    /// and used as the prefix of exported .dat files.
    pub base_name: String,
    /// Time the Z controller is left on to settle between curves, in s.
    /// Only used when `feedback_between` is set.
    pub delay_s: f64,
    /// Turn the Z controller on between curves to re-establish the setpoint
    /// before each Z step. Otherwise it stays off for the whole series.
    pub feedback_between: bool,
    /// Z change per curve in pm, relative to the reference height.
    pub z_step_pm: f64,
}
impl Default for IvSeriesSettings {
    fn default() -> Self {
        Self {
            num_curves: 5,
            base_name: "BiasSpec".into(),
            delay_s: 0.2,
            feedback_between: false,
            z_step_pm: 0.,
        }
    }
}
impl IvSeriesSettings {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_curves == 0 {
            return Err("Number of curves must be at least 1".into());
        }
        if self.base_name.trim().is_empty() {
            return Err("Base name cannot be blank".into());
        }
        if !(self.delay_s >= 0.) {
            return Err("Delay between curves cannot be negative".into());
        }
        if !self.z_step_pm.is_finite() {
            return Err("Z change per curve must be a number".into());
        }
        Ok(())
    }
    /// Save base name sent to Nanonis for curve `index` (0-based).
    pub fn nanonis_name(&self, index: usize) -> String {
        format!("{}_IV_BATCH_{:03}", self.base_name.trim(), index + 1)
    }
    /// Absolute Z target (m) for curve `index` (0-based), measured from the
    /// reference height. Absolute targets avoid accumulating readback error.
    pub fn z_target(&self, start_z_m: f64, index: usize) -> f64 {
        start_z_m + index as f64 * self.z_step_pm * 1e-12
    }
    /// Allowed Z readback error (m): 5 pm or 2 % of the step, whichever is larger.
    pub fn z_tolerance(&self) -> f64 {
        (5e-12f64).max(self.z_step_pm.abs() * 1e-12 * 0.02)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IvCurve {
    /// 0-based position in the series.
    pub index: usize,
    /// Measured Z just before the sweep, relative to the reference, in pm.
    pub z_offset_pm: f64,
    pub bias_v: Vec<f64>,
    pub current_pa: Vec<f64>,
    /// Lock-in / dI/dV channels (name, values in pA), same length as `bias_v`.
    pub didv_pa: Vec<(String, Vec<f64>)>,
    /// When the curve was acquired (UTC, ISO 8601).
    pub timestamp: String,
}
impl IvCurve {
    pub fn label(&self) -> String {
        format!("Z = {:+.1} pm", self.z_offset_pm)
    }
    pub fn didv(&self, channel: &str) -> Option<&[f64]> {
        self.didv_pa
            .iter()
            .find(|(n, _)| n == channel)
            .map(|(_, v)| v.as_slice())
    }
}

/// The channels of one sweep: (bias V, current pA, dI/dV channels in pA).
pub type IvData = (Vec<f64>, Vec<f64>, Vec<(String, Vec<f64>)>);

/// Split the sweep data into channels. `size` is `[rows, cols]` of the
/// row-major `data`; rows are expected to be channels, but a transposed
/// layout is accepted too.
pub fn split_channels(
    names: &[String],
    size: [usize; 2],
    data: &[f32],
) -> Result<Vec<Vec<f64>>, String> {
    let [rows, cols] = size;
    if rows * cols != data.len() {
        return Err(format!(
            "data size {rows}×{cols} does not match {} values",
            data.len()
        ));
    }
    let n = names.len();
    if rows == n {
        Ok(data
            .chunks(cols.max(1))
            .take(rows)
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect())
    } else if cols == n {
        Ok((0..cols)
            .map(|c| (0..rows).map(|r| data[r * cols + c] as f64).collect())
            .collect())
    } else {
        Err(format!(
            "{n} channel names but data is {rows}×{cols}; channels: {names:?}"
        ))
    }
}

/// Pick the bias, current and dI/dV channels by name, the same way the
/// original Python tool did. Current is converted from A to pA.
pub fn extract_iv(names: &[String], channels: &[Vec<f64>]) -> Result<IvData, String> {
    if names.is_empty() || channels.is_empty() {
        return Err("Bias Spectroscopy returned no data. Select at least the Bias and \
                    Current channels in the Nanonis Bias Spectroscopy module."
            .into());
    }
    let find = |words: &[&str]| {
        names.iter().position(|name| {
            let lower = name.to_lowercase();
            words.iter().any(|w| lower.contains(w))
        })
    };
    let voltage = find(&["bias", "voltage"]).or(Some(0));
    let current = find(&["current", "tunnel", "i (", "i_"])
        .or_else(|| (0..names.len()).find(|&i| Some(i) != voltage));
    let (Some(v), Some(c)) = (voltage, current) else {
        return Err(format!(
            "Need two recorded channels (bias and current). Channels returned: {names:?}"
        ));
    };
    let bias = channels[v].clone();
    let current = &channels[c];
    if bias.is_empty() || current.is_empty() {
        return Err("Nanonis returned an empty bias or current channel".into());
    }
    if bias.len() != current.len() {
        return Err(format!(
            "Channel length mismatch: {}={}, {}={}",
            names[v],
            bias.len(),
            names[c],
            current.len()
        ));
    }
    let current_pa = current.iter().map(|i| i * 1e12).collect();
    let didv = names
        .iter()
        .zip(channels)
        .filter(|(name, values)| {
            let lower = name.to_lowercase();
            ["didv", "di/dv", "li demod", "lix", "liy"]
                .iter()
                .any(|m| lower.contains(m))
                && values.len() == bias.len()
        })
        .map(|(name, values)| (name.clone(), values.iter().map(|x| x * 1e12).collect()))
        .collect();
    Ok((bias, current_pa, didv))
}

/// The dI/dV channel shown for "Auto": lock-in demod 1 X if present, else the first.
pub fn auto_didv_channel(names: &[String]) -> Option<&String> {
    names
        .iter()
        .find(|n| {
            let l = n.to_lowercase();
            l.contains("demod 1 x") || l.contains("lix")
        })
        .or(names.first())
}

/// Sorted union of the dI/dV channel names across all curves.
pub fn didv_channel_names(curves: &[IvCurve]) -> Vec<String> {
    let mut names: Vec<String> = curves
        .iter()
        .flat_map(|c| c.didv_pa.iter().map(|(n, _)| n.clone()))
        .collect();
    names.sort();
    names.dedup();
    names
}

/// File name for an exported curve: `<prefix>_<index>_<z>pm.dat`.
pub fn dat_file_name(prefix: &str, curve: &IvCurve) -> String {
    format!("{prefix}_{}_{:+.1}pm.dat", curve.index, curve.z_offset_pm)
}

/// Tab-separated .dat contents in the same layout as the Python tool.
pub fn dat_contents(prefix: &str, curve: &IvCurve, didv_channels: &[String]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Nanonis Spectroscopy Batch Curve Export");
    let _ = writeln!(s, "# Curve File Base Group Name: {prefix}");
    let _ = writeln!(s, "# Zero-indexed Sequence Counter: {}", curve.index);
    let _ = writeln!(s, "# Target Measured relative Z: {:+.3} pm", curve.z_offset_pm);
    let _ = writeln!(s, "# Timestamp: {}", curve.timestamp);
    let _ = writeln!(s, "# -------------------------------------------------");
    let mut columns = vec!["Bias_Voltage_V".to_string(), "Tunneling_Current_pA".to_string()];
    columns.extend(didv_channels.iter().map(|n| format!("{n}_pA")));
    let _ = writeln!(s, "{}", columns.join("\t"));
    for (i, (v, c)) in curve.bias_v.iter().zip(&curve.current_pa).enumerate() {
        let mut row = vec![fmt_g(*v), fmt_g(*c)];
        for name in didv_channels {
            row.push(curve.didv(name).and_then(|d| d.get(i)).map_or(String::new(), |x| fmt_g(*x)));
        }
        let _ = writeln!(s, "{}", row.join("\t"));
    }
    s
}

/// Write every curve as `<dir>/<prefix>_<index>_<z>pm.dat`. Returns the count.
pub fn save_curves(dir: &Path, prefix: &str, curves: &[IvCurve]) -> std::io::Result<usize> {
    let channels = didv_channel_names(curves);
    for curve in curves {
        std::fs::write(
            dir.join(dat_file_name(prefix, curve)),
            dat_contents(prefix, curve, &channels),
        )?;
    }
    Ok(curves.len())
}

/// Full precision without trailing noise (like Python's `{:.12g}`).
fn fmt_g(x: f64) -> String {
    if x == 0. || !x.is_finite() {
        return format!("{x}");
    }
    // Round to 12 significant digits first; the exponent is taken from that.
    let sci = format!("{x:.11e}");
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let trim = |s: &str| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if (-4..12).contains(&exp) {
        let decimals = (11 - exp).max(0) as usize;
        trim(&format!("{x:.decimals$}"))
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exp.abs())
    }
}

/// Current UTC time as ISO 8601 (`2026-11-10T14:03:07Z`).
pub fn utc_timestamp() -> String {
    let t = time::UtcDateTime::now();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        t.year(),
        u8::from(t.month()),
        t.day(),
        t.hour(),
        t.minute(),
        t.second()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: &[&str]) -> Vec<String> {
        n.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn split_rows_and_transposed() {
        let n = names(&["Bias (V)", "Current (A)"]);
        let rows = split_channels(&n, [2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(rows, vec![vec![1., 2., 3.], vec![4., 5., 6.]]);
        let cols = split_channels(&n, [3, 2], &[1., 4., 2., 5., 3., 6.]).unwrap();
        assert_eq!(cols, rows);
        assert!(split_channels(&n, [3, 3], &[0.; 9]).is_err());
    }

    #[test]
    fn extract_finds_channels_and_converts() {
        let n = names(&["Bias calc (V)", "Current (A)", "LI Demod 1 X (A)", "Z (m)"]);
        let ch = vec![
            vec![-1., 1.],
            vec![-2e-12, 3e-12],
            vec![1e-12, 1e-12],
            vec![0., 0.],
        ];
        let (v, i, d) = extract_iv(&n, &ch).unwrap();
        assert_eq!(v, vec![-1., 1.]);
        assert!((i[0] + 2.).abs() < 1e-9 && (i[1] - 3.).abs() < 1e-9);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].0, "LI Demod 1 X (A)");
        assert_eq!(auto_didv_channel(&[d[0].0.clone()]), Some(&d[0].0));
    }

    #[test]
    fn extract_falls_back_to_order() {
        let n = names(&["Sweep (V)", "Signal"]);
        let (v, i, _) = extract_iv(&n, &[vec![0., 1.], vec![1e-12, 2e-12]]).unwrap();
        assert_eq!(v, vec![0., 1.]);
        assert!((i[1] - 2.).abs() < 1e-9);
        assert!(extract_iv(&[], &[]).is_err());
    }

    #[test]
    fn z_targets_are_absolute() {
        let s = IvSeriesSettings {
            z_step_pm: -20.,
            ..Default::default()
        };
        let pm = |m: f64| (m * 1e12).round();
        assert_eq!(
            (0..4).map(|k| pm(s.z_target(0., k))).collect::<Vec<_>>(),
            vec![0., -20., -40., -60.]
        );
        assert_eq!(s.z_tolerance(), 5e-12);
        let big = IvSeriesSettings {
            z_step_pm: 1000.,
            ..Default::default()
        };
        assert!((big.z_tolerance() - 20e-12).abs() < 1e-18);
    }

    #[test]
    fn dat_layout() {
        let c = IvCurve {
            index: 1,
            z_offset_pm: -20.,
            bias_v: vec![-0.5, 0.5],
            current_pa: vec![-1.25, 2.],
            didv_pa: vec![("LIX".into(), vec![0.1, 0.2])],
            timestamp: "t".into(),
        };
        assert_eq!(dat_file_name("BiasSpec", &c), "BiasSpec_1_-20.0pm.dat");
        let s = dat_contents("BiasSpec", &c, &["LIX".into()]);
        let lines: Vec<_> = s.lines().collect();
        assert_eq!(lines[6], "Bias_Voltage_V\tTunneling_Current_pA\tLIX_pA");
        assert_eq!(lines[7], "-0.5\t-1.25\t0.1");
        assert_eq!(lines.len(), 9);
    }

    #[test]
    fn fmt_g_like_python() {
        assert_eq!(fmt_g(0.5), "0.5");
        assert_eq!(fmt_g(-1.25), "-1.25");
        assert_eq!(fmt_g(1e-15), "1e-15");
        assert_eq!(fmt_g(-1.5e-5), "-1.5e-05");
        assert_eq!(fmt_g(1.5e-4), "0.00015");
        assert_eq!(fmt_g(123.456), "123.456");
        assert_eq!(fmt_g(0.1 + 0.2), "0.3");
        assert_eq!(fmt_g(0.), "0");
    }
}
