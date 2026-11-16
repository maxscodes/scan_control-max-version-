//! Find the Nanonis .sxm file a stamped frame came from, by its data.
//!
//! A stamp is a copy of the live scan, and Nanonis saves the same scan as an
//! .sxm file. Comparing the data (any channel, forward or backward, either
//! line order) identifies the file without relying on timing.

use std::path::{Path, PathBuf};

/// All .sxm files under `dir` (recursively).
pub fn list_sxm(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("sxm")) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Number of sample points compared.
const SAMPLES: usize = 256;

/// Do two frames (same size) hold the same scan? Compared at a grid of
/// points, as a correlation over the points both have data for, so small
/// offsets or scaling don't matter.
fn same_scan(a: &[f32], b: &[f32], w: usize, h: usize) -> bool {
    let n = (SAMPLES as f64).sqrt() as usize;
    let (mut c, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0., 0., 0., 0., 0., 0.);
    for j in 0..n {
        for i in 0..n {
            let (x, y) = ((i * 2 + 1) * w / (2 * n), (j * 2 + 1) * h / (2 * n));
            let (u, v) = (a[y * w + x] as f64, b[y * w + x] as f64);
            if u.is_finite() && v.is_finite() {
                c += 1.;
                sx += u;
                sy += v;
                sxx += u * u;
                syy += v * v;
                sxy += u * v;
            }
        }
    }
    if c < 32. {
        return false;
    }
    let cxx = sxx - sx * sx / c;
    let cyy = syy - sy * sy / c;
    cxx > 0. && cyy > 0. && (sxy - sx * sy / c) / (cxx * cyy).sqrt() > 0.999
}

/// Rows in reverse order.
fn flip_rows(d: &[f32], w: usize) -> Vec<f32> {
    d.chunks(w).rev().flatten().copied().collect()
}

/// The image channels of one .sxm file.
pub struct Sxm {
    size: [u32; 2],
    channels: Vec<Box<[f32]>>,
}

impl Sxm {
    pub fn load(path: &Path) -> Option<Sxm> {
        let sxm = sxmfile::SXM::parse_file(path).ok()?;
        let size = sxm.get_image_size().ok()?;
        Some(Sxm { size, channels: sxm.data.into_iter().flatten().collect() })
    }
    /// Does this file hold the frame (any channel, either line order)?
    pub fn holds(&self, data: &[f32], size: [u32; 2]) -> bool {
        if self.size != size || data.len() != size[0] as usize * size[1] as usize {
            return false;
        }
        let (w, h) = (size[0] as usize, size[1] as usize);
        self.channels.iter().any(|ch| {
            ch.len() == w * h && (same_scan(data, ch, w, h) || same_scan(data, &flip_rows(ch, w), w, h))
        })
    }
}

/// Columns in reverse order.
fn flip_cols(d: &[f32], w: usize) -> Vec<f32> {
    d.chunks(w).flat_map(|r| r.iter().rev()).copied().collect()
}

/// Plain correlation of two equal-size frames over the points both have.
fn corr(a: &[f32], b: &[f32]) -> f64 {
    let (mut c, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0., 0., 0., 0., 0., 0.);
    for (&u, &v) in a.iter().zip(b) {
        let (u, v) = (u as f64, v as f64);
        if u.is_finite() && v.is_finite() {
            c += 1.;
            sx += u;
            sy += v;
            sxx += u * u;
            syy += v * v;
            sxy += u * v;
        }
    }
    let cxx = sxx - sx * sx / c;
    let cyy = syy - sy * sy / c;
    if c < 2. || cxx <= 0. || cyy <= 0. {
        return 0.;
    }
    (sxy - sx * sy / c) / (cxx * cyy).sqrt()
}

/// The backward scan of the channel whose forward scan is this frame,
/// turned to line up with the frame (rows as the frame has them, columns
/// whichever way fits the forward scan better).
#[cfg(test)] // used by the tests only
pub fn backward_for(path: &Path, data: &[f32], size: [u32; 2]) -> Option<Vec<f32>> {
    let sxm = sxmfile::SXM::parse_file(path).ok()?;
    if sxm.get_image_size().ok()? != size {
        return None;
    }
    let (w, h) = (size[0] as usize, size[1] as usize);
    for [f, b] in &sxm.data {
        if f.len() != w * h || b.len() != w * h {
            continue;
        }
        let rows_flipped = if same_scan(data, f, w, h) {
            false
        } else if same_scan(data, &flip_rows(f, w), w, h) {
            true
        } else {
            continue;
        };
        let b: Vec<f32> = if rows_flipped { flip_rows(b, w) } else { b.to_vec() };
        let flipped = flip_cols(&b, w);
        return Some(if corr(data, &flipped) > corr(data, &b) { flipped } else { b });
    }
    None
}

/// Does this .sxm file hold the frame?
#[cfg(test)] // used by the tests only
pub fn sxm_matches(path: &Path, data: &[f32], size: [u32; 2]) -> bool {
    Sxm::load(path).is_some_and(|s| s.holds(data, size))
}

/// The first file in `files` that holds the frame.
#[cfg(test)] // used by the tests only
pub fn find_matching_sxm(data: &[f32], size: [u32; 2], files: &[PathBuf]) -> Option<PathBuf> {
    if data.len() != size[0] as usize * size[1] as usize {
        return None;
    }
    files.iter().find(|p| sxm_matches(p, data, size)).cloned()
}

/// One channel of an .sxm file, laid out as a live frame (row 0 at the top)
/// and placed as Nanonis' scan frame was, for stamping after the fact.
pub struct SxmFrame {
    /// File name without extension, e.g. `20261112_0006`.
    pub name: String,
    pub channel: String,
    pub unit: String,
    pub size: [u32; 2],
    pub forward: Vec<f32>,
    pub backward: Vec<f32>,
    /// Frame in piezo coordinates (nm), as the live scan frame.
    pub frame: glam::DAffine2,
}

/// Load the channel named like `preferred` (the live channel), else Z,
/// else the first one.
pub fn load_frame(path: &Path, preferred: Option<&str>) -> Result<SxmFrame, String> {
    let sxm = sxmfile::SXM::parse_file(path).map_err(|e| format!("{e:#}"))?;
    let size = sxm.get_image_size().map_err(|e| e.to_string())?;
    let range = sxm.get_scan_range().map_err(|e| e.to_string())?;
    let center = sxm.get_scan_center().map_err(|e| e.to_string())?;
    let angle = sxm.get_scan_angle().unwrap_or(0.);
    let chans = sxm.get_channels().map_err(|e| e.to_string())?;
    let names: Vec<&str> = chans.iter().map(|(n, _)| n.trim()).collect();
    // Live names may carry a unit, e.g. "Z (m)".
    let like = |want: &str| {
        let want = want.trim();
        names.iter().position(|n| {
            want == *n || want.strip_prefix(*n).is_some_and(|r| r.starts_with(' '))
        })
    };
    let ci = preferred
        .and_then(like)
        .or_else(|| like("Z"))
        .filter(|&i| i < sxm.data.len())
        .unwrap_or(0);
    let [f, b] = sxm.data.get(ci).ok_or("file has no image data")?;
    let (w, h) = (size[0] as usize, size[1] as usize);
    if w * h == 0 || f.len() != w * h || b.len() != w * h {
        return Err("image size does not match its data".into());
    }
    // Nanonis stores the first-acquired line first: for an "up" scan that
    // is the bottom line.
    let up = sxm.get_metadata("SCAN_DIR").map(|d| d.trim() == "up").unwrap_or(false);
    let rows = |d: &[f32]| if up { flip_rows(d, w) } else { d.to_vec() };
    let forward = rows(f);
    let b = rows(b);
    let flipped = flip_cols(&b, w);
    let backward = if corr(&forward, &flipped) > corr(&forward, &b) { flipped } else { b };
    let frame = glam::DAffine2::from_scale_angle_translation(
        glam::DVec2::new(range[0] as f64 * 1e9, range[1] as f64 * -1e9),
        angle as f64 / 180. * -std::f64::consts::PI,
        glam::DVec2::new(center[0] as f64 * 1e9, center[1] as f64 * 1e9),
    );
    Ok(SxmFrame {
        name: path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "sxm".into()),
        channel: names.get(ci).copied().unwrap_or_default().to_string(),
        unit: chans.get(ci).map(|c| c.1.unit().trim().to_string()).unwrap_or_default(),
        size,
        forward,
        backward,
        frame,
    })
}

#[cfg(test)]
mod tests {
    /// .sxm files load in the same layout as the live stamps saved from them.
    /// `SXM_DIR=dir cargo test --bin scan_control sxm_frame_matches_stamps -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sxm_frame_matches_stamps() {
        use crate::analysis::ring_training::Dataset;
        let dir = std::env::var("SXM_DIR").unwrap_or(r"F:\Data\LT SPM\2026\November".into());
        let files = super::list_sxm(std::path::Path::new(&dir));
        let (mut n, mut same, mut flipped) = (0, 0, 0);
        for s in Dataset::load(Dataset::default_dir()).samples() {
            let Some(path) = files.iter().find(|f| f.file_stem().is_some_and(|x| x == s.record.source.as_str())) else {
                continue;
            };
            let Ok(f) = super::load_frame(path, Some("Z")) else { continue };
            let (w, h) = (f.size[0] as usize, f.size[1] as usize);
            if [s.record.w, s.record.h] != f.size {
                continue;
            }
            n += 1;
            if super::same_scan(&s.data, &f.forward, w, h) {
                same += 1;
            } else if super::same_scan(&s.data, &super::flip_rows(&f.forward, w), w, h) {
                flipped += 1;
                println!("{}: rows flipped", s.record.source);
            }
        }
        println!("{n} stamps: {same} same layout, {flipped} rows flipped");
        assert!(n > 0 && flipped == 0);
    }
}
