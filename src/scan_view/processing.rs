//! Display processing for scan images: background removal, smoothing and
//! sharpening, and the colour scale.
//!
//! The mean, line and plane fits run on the GPU (`image_compute`). The
//! parabola fits and the filters run here on the CPU before the data is
//! uploaded; the GPU then only subtracts the mean. Missing pixels (NaN, e.g.
//! lines not scanned yet) are ignored by every step and stay missing.

use egui::{Color32, DragValue, Ui, WidgetText};
use image_compute::image_compute::FitType;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::{
    app::COLOR_MAP_SIZE,
    components::combo_box::{combo_box, ComboBoxType},
};

/// Background removed before display. The numbers 0–3 match `FitType`, so
/// values saved before the parabola fits existed still load.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, num_enum::TryFromPrimitive, num_enum::IntoPrimitive,
)]
#[repr(u8)]
pub enum Background {
    Raw = 0,
    Plane = 1,
    LineMean = 2,
    LineLinear = 3,
    Parabola = 4,
    LineParabola = 5,
}

impl Background {
    /// The fit the GPU does after the CPU step.
    pub fn gpu_fit(self) -> FitType {
        match self {
            Background::Raw | Background::Parabola | Background::LineParabola => {
                FitType::MeanSubtract
            }
            Background::Plane => FitType::PlaneFitSubtract,
            Background::LineMean => FitType::LineMeanSubtract,
            Background::LineLinear => FitType::LineFitSubtract,
        }
    }
    fn hover(self) -> &'static str {
        match self {
            Background::Raw => "Only the mean is removed",
            Background::LineMean => "Each scan line is shifted to the same mean height",
            Background::LineLinear => "A straight line is fitted to each scan line and subtracted",
            Background::LineParabola => "A parabola is fitted to each scan line and subtracted",
            Background::Plane => "One plane is fitted to the whole image and subtracted",
            Background::Parabola => {
                "One parabolic surface (x², xy, y² terms) is fitted to the whole image and \
                 subtracted: removes piezo bow and curved backgrounds"
            }
        }
    }
}

impl ComboBoxType for Background {
    type Ctx = ();
    fn opt_atoms(&self, _ctx: &()) -> impl Into<WidgetText> {
        match self {
            Background::Raw => "Raw",
            Background::LineMean => "Line: mean",
            Background::LineLinear => "Line: linear",
            Background::LineParabola => "Line: parabola",
            Background::Plane => "Global: plane",
            Background::Parabola => "Global: parabola",
        }
    }
    fn options(_ctx: &()) -> impl Iterator<Item = Self> {
        [
            Background::Raw,
            Background::LineMean,
            Background::LineLinear,
            Background::LineParabola,
            Background::Plane,
            Background::Parabola,
        ]
        .into_iter()
    }
}

/// Optional filters applied before the background fit. They change only the
/// drawn image: the scan data (and any .sxm file) are never modified.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Filters {
    pub smooth: bool,
    /// Gaussian sigma in pixels.
    pub smooth_px: f32,
    pub sharpen: bool,
    /// Unsharp-mask strength.
    pub sharpen_amount: f32,
    /// Remove horizontal streaks (tip changes, feedback glitches).
    pub lines: bool,
    /// How far (in noise levels) a streak must stand out to be removed.
    pub lines_threshold: f32,
    /// Remove small spots (spikes, short tip pick-ups).
    pub blobs: bool,
    /// Largest spot removed, in pixels across.
    pub blob_px: f32,
}

impl Default for Filters {
    fn default() -> Self {
        Self {
            smooth: false,
            smooth_px: 1.,
            sharpen: false,
            sharpen_amount: 1.,
            lines: false,
            lines_threshold: 3.,
            blobs: false,
            blob_px: 6.,
        }
    }
}

impl Filters {
    pub const MAX_SMOOTH_PX: f32 = 5.;
    pub const MAX_BLOB_PX: f32 = 40.;
    /// Smooth, sharpen and line settings (the original saved format).
    pub fn to_array(self) -> [f32; 6] {
        [
            self.smooth as u8 as f32,
            self.smooth_px,
            self.sharpen as u8 as f32,
            self.sharpen_amount,
            self.lines as u8 as f32,
            self.lines_threshold,
        ]
    }
    /// Blob settings, saved separately; `None` when they are the defaults.
    pub fn blob_array(self) -> Option<[f32; 2]> {
        let d = Self::default();
        ((self.blobs, self.blob_px) != (d.blobs, d.blob_px))
            .then_some([self.blobs as u8 as f32, self.blob_px])
    }
    pub fn from_arrays(a: Option<[f32; 6]>, b: Option<[f32; 2]>) -> Self {
        let mut f = Self::default();
        if let Some(a) = a {
            f.smooth = a[0] != 0.;
            f.smooth_px = a[1].clamp(0.3, Self::MAX_SMOOTH_PX);
            f.sharpen = a[2] != 0.;
            f.sharpen_amount = a[3].clamp(0.1, 5.);
            f.lines = a[4] != 0.;
            f.lines_threshold = a[5].clamp(1., 20.);
        }
        if let Some(b) = b {
            f.blobs = b[0] != 0.;
            f.blob_px = b[1].clamp(1., Self::MAX_BLOB_PX);
        }
        f
    }
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
    pub fn any(&self) -> bool {
        self.smooth || self.sharpen || self.lines || self.blobs
    }
}

/// Background choice and filter toggles. Returns whether anything changed.
/// Background and filter controls in their own two-column grid.
pub fn processing_ui(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    background: &mut Background,
    filters: &mut Filters,
) -> bool {
    egui::Grid::new((&id, "processing_grid"))
        .num_columns(2)
        .spacing([8., 4.])
        .show(ui, |ui| processing_rows(ui, &id, background, filters))
        .inner
}

/// Background and filter rows (label | control) for a two-column
/// [`egui::Grid`] owned by the caller, so they line up with its other rows.
pub fn processing_rows(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    background: &mut Background,
    filters: &mut Filters,
) -> bool {
    let mut changed = false;
    ui.label("Background").on_hover_text(background.hover());
    let before = *background;
    combo_box(ui, (&id, "background"), background, &());
    changed |= *background != before;
    ui.end_row();

    // Grid label column: "Filters" on the first filter row only.
    let mut row_label = "Filters";
    let mut filter = |ui: &mut Ui, label: &str, on: &mut bool, value: DragValue, hover: &str| {
        ui.label(std::mem::take(&mut row_label));
        ui.horizontal(|ui| {
            changed |= ui.checkbox(on, label).changed();
            changed |= ui.add_enabled(*on, value).changed();
        })
        .response
        .on_hover_text(hover);
        ui.end_row();
    };
    filter(
        ui,
        "Fix lines",
        &mut filters.lines,
        DragValue::new(&mut filters.lines_threshold).range(1.0..=20.).speed(0.05),
        "Removes horizontal streaks (tip changes, feedback glitches): stretches of up to \
         three scan lines that jump away from the lines above and below are filled in \
         from those lines. Works while the scan is running (each line once two more \
         are in). The value is how many noise levels a streak must stand out: lower \
         removes more. Only the drawn image changes; the scan data and .sxm files are \
         never modified.",
    );
    filter(
        ui,
        "Remove blobs",
        &mut filters.blobs,
        DragValue::new(&mut filters.blob_px)
            .range(1.0..=Filters::MAX_BLOB_PX)
            .speed(0.1)
            .suffix(" px"),
        "Removes small sharp spots (spikes, short tip pick-ups) that stand out strongly \
         from their surroundings, and fills them in from around them. The value is the \
         largest spot removed, in pixels across: keep it below the size of real \
         features such as adsorbates. Only the drawn image changes.",
    );
    filter(
        ui,
        "Smooth",
        &mut filters.smooth,
        DragValue::new(&mut filters.smooth_px)
            .range(0.3..=Filters::MAX_SMOOTH_PX)
            .speed(0.02)
            .suffix(" px"),
        "Gaussian blur: reduces noise. The value is the blur radius (sigma).",
    );
    filter(
        ui,
        "Sharpen",
        &mut filters.sharpen,
        DragValue::new(&mut filters.sharpen_amount).range(0.1..=5.).speed(0.02),
        "Unsharp mask: boosts fine detail and edges. The value is the strength.",
    );
    changed
}

/// The processed copy of `data` (row-major, `size = [width, height]`, rows
/// are scan lines), or `None` when only the GPU fit is needed.
pub fn process(
    data: &[f32],
    size: [u32; 2],
    background: Background,
    filters: Filters,
) -> Option<Vec<f32>> {
    let cpu_bg = matches!(background, Background::Parabola | Background::LineParabola);
    if !cpu_bg && !filters.any() {
        return None;
    }
    let [w, h] = [size[0] as usize, size[1] as usize];
    if data.len() != w * h || w == 0 || h == 0 {
        return None;
    }
    // A copy: the caller's data (the scan as recorded) is never modified.
    let mut out = data.to_vec();
    if filters.lines {
        remove_line_streaks(&mut out, w, h, filters.lines_threshold);
    }
    if filters.blobs {
        // Detection works on line-levelled data so that line-to-line drift
        // doesn't hide spots; the offsets are put back afterwards.
        let offsets = level_rows(&mut out, w);
        remove_blobs(&mut out, w, h, filters.blob_px);
        for (row, o) in out.chunks_mut(w).zip(offsets) {
            row.iter_mut().for_each(|v| *v += o);
        }
    }
    if filters.smooth {
        out = gaussian_blur(&out, w, h, filters.smooth_px);
    }
    if filters.sharpen {
        let blurred = gaussian_blur(&out, w, h, 1.);
        for (v, b) in out.iter_mut().zip(&blurred) {
            *v += filters.sharpen_amount * (*v - b);
        }
    }
    match background {
        Background::Parabola => subtract_parabola(&mut out, w, h),
        Background::LineParabola => subtract_line_parabolas(&mut out, w, h),
        _ => {}
    }
    Some(out)
}

/// Median of `v` (reordered); the mean of the middle two for even lengths.
fn median(v: &mut [f32]) -> f32 {
    let (n, m) = (v.len(), v.len() / 2);
    let (lo, hi, _) = v.select_nth_unstable_by(m, f32::total_cmp);
    let hi = *hi;
    if n % 2 == 1 {
        hi
    } else {
        let lo = lo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        0.5 * (lo + hi)
    }
}

/// Subtracts each row's median and returns the medians (0 for empty rows).
fn level_rows(data: &mut [f32], w: usize) -> Vec<f32> {
    let mut buf = Vec::with_capacity(w);
    data.chunks_mut(w)
        .map(|row| {
            buf.clear();
            buf.extend(row.iter().copied().filter(|v| !v.is_nan()));
            if buf.is_empty() {
                return 0.;
            }
            let m = median(&mut buf);
            row.iter_mut().for_each(|v| *v -= m);
            m
        })
        .collect()
}

/// Pixel noise, from how much each pixel differs from its neighbours along
/// the scan line (unaffected by line jumps and by smooth features).
fn line_noise(d: &[f32], w: usize) -> Option<f32> {
    let mut v: Vec<f32> = d
        .chunks(w)
        .flat_map(|row| row.windows(3))
        .map(|t| (t[1] - 0.5 * (t[0] + t[2])).abs())
        .filter(|v| !v.is_nan())
        .collect();
    if v.is_empty() {
        return None;
    }
    // For white noise σ the second difference has spread σ·√1.5.
    let s = 1.4826 * median(&mut v) / 1.5f32.sqrt();
    (s > 0.).then_some(s)
}

/// Lines looked at on each side of a pixel when testing for a streak. A
/// streak up to half this thick still leaves a majority of clean lines.
const STREAK_SIDE: usize = 6;

/// How far `v` stands out from the values on *both* sides: the smaller of
/// the two differences to their medians when they have the same sign, else
/// 0 (an edge matches one side). `None` with fewer than 2 values a side.
fn stands_out(v: f32, a: &mut [f32], b: &mut [f32]) -> Option<f32> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let (da, db) = (v - median(a), v - median(b));
    Some(if da * db > 0. {
        if da.abs() < db.abs() { da } else { db }
    } else {
        0.
    })
}

/// How far each pixel stands out from the lines above and below, less how
/// far it stands out along its own line: a streak is flat along the line,
/// while a real bump stands out both ways. NaN where there are too few
/// lines around; 0 where it stands out less than `small` across.
fn streak_residual(d: &[f32], w: usize, h: usize, small: f32) -> Vec<f32> {
    let mut out = vec![f32::NAN; w * h];
    let mut a = Vec::with_capacity(STREAK_SIDE);
    let mut b = Vec::with_capacity(STREAK_SIDE);
    let ok = |u: &f32| !u.is_nan();
    for y in 0..h {
        for x in 0..w {
            let v = d[y * w + x];
            if v.is_nan() {
                continue;
            }
            a.clear();
            a.extend((y.saturating_sub(STREAK_SIDE)..y).map(|yy| d[yy * w + x]).filter(ok));
            b.clear();
            b.extend((y + 1..(y + 1 + STREAK_SIDE).min(h)).map(|yy| d[yy * w + x]).filter(ok));
            let Some(across) = stands_out(v, &mut a, &mut b) else { continue };
            // Too small to matter: skip the test along the line.
            if across.abs() < small {
                out[y * w + x] = 0.;
                continue;
            }
            let row = &d[y * w..(y + 1) * w];
            a.clear();
            a.extend(row[x.saturating_sub(STREAK_SIDE)..x].iter().copied().filter(ok));
            b.clear();
            b.extend(row[x + 1..(x + 1 + STREAK_SIDE).min(w)].iter().copied().filter(ok));
            // Curvature along the line, from the medians on each side.
            let side = |s: &mut Vec<f32>| (s.len() >= 2).then(|| median(s));
            let along = match (side(&mut a), side(&mut b)) {
                (Some(l), Some(r)) => v - 0.5 * (l + r),
                (Some(m), None) | (None, Some(m)) => v - m,
                (None, None) => 0.,
            };
            // Only the part standing out the same way as across counts.
            let along = if along * across > 0. { along.abs() } else { 0. };
            out[y * w + x] = across.signum() * (across.abs() - along).max(0.);
        }
    }
    out
}

/// Removes horizontal streaks: stretches of up to three scan lines that
/// stand out from the lines above *and* below by more than `threshold`
/// noise levels, over a run along the line, are filled in from those lines.
/// Noisy streaks count too (the run is tested on the median of the size of
/// the jump). `data` should be line-levelled.
fn remove_line_streaks(data: &mut [f32], w: usize, h: usize, threshold: f32) {
    if h < 5 || w < 5 {
        return;
    }
    // Take out the slow change from line to line (tilt, creep) so it
    // doesn't weaken the test; a running median ignores single-line jumps.
    let mut buf = Vec::with_capacity(w.max(2 * STREAK_SIDE + 1));
    let levels: Vec<f32> = data
        .chunks(w)
        .map(|row| {
            buf.clear();
            buf.extend(row.iter().copied().filter(|v| !v.is_nan()));
            if buf.is_empty() { f32::NAN } else { median(&mut buf) }
        })
        .collect();
    // The local slope is the median step between neighbouring line levels:
    // a jumped line or a lasting step (tip change, terrace) is only one or
    // two of those steps, so it isn't taken out with the slope.
    let steps: Vec<f32> = levels.windows(2).map(|p| p[1] - p[0]).collect();
    let half = 2 * STREAK_SIDE;
    let mut trend = Vec::with_capacity(h);
    let mut t = 0.;
    for y in 0..h {
        trend.push(t);
        if y + 1 < h {
            buf.clear();
            buf.extend(
                steps[y.saturating_sub(half)..(y + half + 1).min(h - 1)]
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan()),
            );
            if !buf.is_empty() {
                t += median(&mut buf);
            }
        }
    }
    for (row, t) in data.chunks_mut(w).zip(&trend) {
        row.iter_mut().for_each(|v| *v -= t);
    }
    remove_flat_streaks(data, w, h, threshold);
    for (row, t) in data.chunks_mut(w).zip(&trend) {
        row.iter_mut().for_each(|v| *v += t);
    }
}

fn remove_flat_streaks(data: &mut [f32], w: usize, h: usize, threshold: f32) {
    let Some(noise) = line_noise(data, w) else { return };
    // Half the length a streak must at least have.
    let hw = (w / 64).max(3);
    for _pass in 0..2 {
        let r = streak_residual(data, w, h, 0.5 * threshold * noise);
        // Images whose rows differ a lot anyway (e.g. atom rows along the
        // scan) need streaks to stand out from that too.
        let mut all: Vec<f32> = r.iter().filter(|v| !v.is_nan()).map(|v| v.abs()).collect();
        if all.is_empty() {
            return;
        }
        let spread = 1.4826 * median(&mut all);
        let limit = threshold * noise.max(spread);
        let half = 0.5 * limit;
        let mut mark = vec![false; w * h];
        let mut med = vec![f32::NAN; w];
        let mut buf = Vec::with_capacity(2 * hw + 1);
        for y in 0..h {
            let row = &r[y * w..(y + 1) * w];
            for x in 0..w {
                buf.clear();
                buf.extend(
                    row[x.saturating_sub(hw)..(x + hw + 1).min(w)]
                        .iter()
                        .filter(|v| !v.is_nan())
                        .map(|v| v.abs()),
                );
                med[x] = if buf.len() > hw { median(&mut buf) } else { f32::NAN };
            }
            let mut x = 0;
            while x < w {
                if !(med[x] > limit) {
                    x += 1;
                    continue;
                }
                let s = x;
                while x < w && med[x] > limit {
                    x += 1;
                }
                // The windows of [s, x) cover hw more on each side: keep
                // what stands out, then grow over the streak's ends.
                let mut a = s.saturating_sub(hw);
                while a < s && !(row[a].abs() > half) {
                    a += 1;
                }
                let mut b = (x + hw).min(w);
                while b > x && !(row[b - 1].abs() > half) {
                    b -= 1;
                }
                while a > 0 && row[a - 1].abs() > half {
                    a -= 1;
                }
                while b < w && row[b].abs() > half {
                    b += 1;
                }
                mark[y * w + a..y * w + b].fill(true);
                x = x.max(b);
            }
        }
        if !mark.contains(&true) {
            return;
        }
        fill_columns(data, w, h, &mark);
    }
}

/// Fills each marked stretch of a column by interpolating between the
/// unmarked pixels above and below it.
fn fill_columns(data: &mut [f32], w: usize, h: usize, mark: &[bool]) {
    for x in 0..w {
        let mut y = 0;
        while y < h {
            if !mark[y * w + x] {
                y += 1;
                continue;
            }
            let y0 = y;
            while y < h && mark[y * w + x] {
                y += 1;
            }
            let above = (y0 > 0).then(|| data[(y0 - 1) * w + x]).filter(|v| !v.is_nan());
            let below = (y < h).then(|| data[y * w + x]).filter(|v| !v.is_nan());
            for (k, yy) in (y0..y).enumerate() {
                let i = yy * w + x;
                if data[i].is_nan() {
                    continue;
                }
                data[i] = match (above, below) {
                    (Some(a), Some(b)) => a + (b - a) * (k + 1) as f32 / (y - y0 + 1) as f32,
                    (Some(v), None) | (None, Some(v)) => v,
                    (None, None) => data[i],
                };
            }
        }
    }
}

/// Smooth background: medians on a coarse grid (every `step` px, over
/// ±`half` px), interpolated between grid points. NaN where nothing is near.
fn coarse_median(d: &[f32], w: usize, h: usize, step: usize, half: usize) -> Vec<f32> {
    let grid = |n: usize| {
        let mut g: Vec<usize> = (0..n).step_by(step).collect();
        if *g.last().unwrap() != n - 1 {
            g.push(n - 1);
        }
        g
    };
    let (gx, gy) = (grid(w), grid(h));
    let nx = gx.len();
    let mut g = vec![f32::NAN; nx * gy.len()];
    let mut buf = Vec::new();
    for (j, &cy) in gy.iter().enumerate() {
        for (i, &cx) in gx.iter().enumerate() {
            buf.clear();
            for y in cy.saturating_sub(half)..(cy + half + 1).min(h) {
                buf.extend(
                    d[y * w + cx.saturating_sub(half)..y * w + (cx + half + 1).min(w)]
                        .iter()
                        .copied()
                        .filter(|v| !v.is_nan()),
                );
            }
            if buf.len() >= 3 {
                g[j * nx + i] = median(&mut buf);
            }
        }
    }
    let cell = |p: usize, gs: &[usize]| {
        let k = (p / step).min(gs.len() - 2);
        (k, (p - gs[k]) as f32 / (gs[k + 1] - gs[k]) as f32)
    };
    let mut out = vec![f32::NAN; w * h];
    for y in 0..h {
        let (j, ty) = cell(y, &gy);
        for x in 0..w {
            let (i, tx) = cell(x, &gx);
            let (mut s, mut ws) = (0., 0.);
            for (v, wt) in [
                (g[j * nx + i], (1. - tx) * (1. - ty)),
                (g[j * nx + i + 1], tx * (1. - ty)),
                (g[(j + 1) * nx + i], (1. - tx) * ty),
                (g[(j + 1) * nx + i + 1], tx * ty),
            ] {
                if !v.is_nan() {
                    s += v * wt;
                    ws += wt;
                }
            }
            if ws > 1e-6 {
                out[y * w + x] = s / ws;
            }
        }
    }
    out
}

/// How many spreads of the surroundings a spot's peak must stand out.
const BLOB_SIGMA: f32 = 5.;
/// A spot's extent is measured down to this fraction of the peak limit, so
/// the top of a big soft feature doesn't count as a small spot.
const BLOB_EDGE: f32 = 0.3;

/// Removes small sharp spots: connected regions no more than `max_px`
/// across that stand out strongly from a local median background are
/// filled in smoothly from around them. `data` should be line-levelled.
fn remove_blobs(data: &mut [f32], w: usize, h: usize, max_px: f32) {
    if w < 3 || h < 3 {
        return;
    }
    let max = (max_px.round() as usize).max(1);
    let r = (max + 1) / 2;
    // The median window is several times the largest spot, so a spot
    // doesn't move it.
    let bg = coarse_median(data, w, h, r.max(2), 2 * r + 2);
    let res: Vec<f32> = data.iter().zip(&bg).map(|(d, b)| d - b).collect();
    let mut abs: Vec<f32> = res.iter().filter(|v| !v.is_nan()).map(|v| v.abs()).collect();
    if abs.is_empty() {
        return;
    }
    let spread = 1.4826 * median(&mut abs);
    if !(spread > 0.) {
        return;
    }
    let hi = BLOB_SIGMA * spread;
    let lo = BLOB_EDGE * hi;
    let mut seen = vec![false; w * h];
    let mut mark = vec![false; w * h];
    let (mut comp, mut stack) = (Vec::new(), Vec::new());
    for start in 0..w * h {
        if seen[start] || !(res[start].abs() > lo) {
            continue;
        }
        let up = res[start] > 0.;
        seen[start] = true;
        stack.push(start);
        comp.clear();
        let (mut x0, mut x1, mut y0, mut y1, mut peak) = (w, 0, h, 0, 0f32);
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (x, y) = (i % w, i / w);
            (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            peak = peak.max(res[i].abs());
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    let j = yy * w + xx;
                    if !seen[j] && res[j].abs() > lo && (res[j] > 0.) == up {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        if peak > hi && x1 - x0 < max && y1 - y0 < max {
            comp.iter().for_each(|&i| mark[i] = true);
        }
    }
    if !mark.contains(&true) {
        return;
    }
    // One pixel more for the spot's soft edge.
    let core = mark.clone();
    for i in (0..w * h).filter(|&i| core[i]) {
        let (x, y) = (i % w, i / w);
        for yy in y.saturating_sub(1)..(y + 2).min(h) {
            for xx in x.saturating_sub(1)..(x + 2).min(w) {
                mark[yy * w + xx] = true;
            }
        }
    }
    let fill: Vec<usize> =
        (0..w * h).filter(|&i| mark[i] && !data[i].is_nan() && !bg[i].is_nan()).collect();
    for &i in &fill {
        data[i] = bg[i];
    }
    // Relax towards a smooth fill that meets the pixels around the spot.
    for _ in 0..(3 * max).max(8) {
        for &i in &fill {
            let (x, y) = (i % w, i / w);
            let (mut s, mut n) = (0., 0);
            for j in [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ]
            .into_iter()
            .flatten()
            {
                if !data[j].is_nan() {
                    s += data[j];
                    n += 1;
                }
            }
            if n > 0 {
                data[i] = s / n as f32;
            }
        }
    }
}

/// Normalised Gaussian blur: missing pixels don't count and stay missing.
pub(crate) fn gaussian_blur(data: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (3. * sigma).ceil().max(1.) as isize;
    let kernel: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2. * sigma * sigma)).exp())
        .collect();
    let mut num: Vec<f32> = data.iter().map(|v| if v.is_nan() { 0. } else { *v }).collect();
    let mut den: Vec<f32> = data.iter().map(|v| if v.is_nan() { 0. } else { 1. }).collect();
    let mut tmp_n = vec![0.; w * h];
    let mut tmp_d = vec![0.; w * h];
    // Along the lines.
    for y in 0..h {
        for x in 0..w {
            let (mut sn, mut sd) = (0., 0.);
            for (k, kv) in kernel.iter().enumerate() {
                let xx = x as isize + k as isize - r;
                if xx < 0 || xx >= w as isize {
                    continue;
                }
                let i = y * w + xx as usize;
                sn += kv * num[i];
                sd += kv * den[i];
            }
            tmp_n[y * w + x] = sn;
            tmp_d[y * w + x] = sd;
        }
    }
    // Across the lines.
    for y in 0..h {
        for x in 0..w {
            let (mut sn, mut sd) = (0., 0.);
            for (k, kv) in kernel.iter().enumerate() {
                let yy = y as isize + k as isize - r;
                if yy < 0 || yy >= h as isize {
                    continue;
                }
                let i = yy as usize * w + x;
                sn += kv * tmp_n[i];
                sd += kv * tmp_d[i];
            }
            num[y * w + x] = sn;
            den[y * w + x] = sd;
        }
    }
    data.iter()
        .zip(num.iter().zip(&den))
        .map(|(v, (n, d))| if v.is_nan() || *d <= 0. { f32::NAN } else { n / d })
        .collect()
}

/// Position in -1..=1 across `n` pixels.
fn norm(i: usize, n: usize) -> f64 {
    if n < 2 {
        0.
    } else {
        2. * i as f64 / (n - 1) as f64 - 1.
    }
}

/// Least-squares fit; terms the data can't determine (e.g. y terms with a
/// single scanned line) get zero weight instead of failing.
fn solve(ata: DMatrix<f64>, atb: DVector<f64>) -> Option<DVector<f64>> {
    let svd = ata.svd(true, true);
    let max = svd.singular_values.max();
    if !(max > 0.) {
        return None;
    }
    svd.solve(&atb, max * 1e-10).ok()
}

/// Fits z = a + bx + cy + dx² + exy + fy² to the whole image and subtracts it.
fn subtract_parabola(data: &mut [f32], w: usize, h: usize) {
    let basis = |x: f64, y: f64| [1., x, y, x * x, x * y, y * y];
    let mut ata = DMatrix::<f64>::zeros(6, 6);
    let mut atb = DVector::<f64>::zeros(6);
    for y in 0..h {
        let ny = norm(y, h);
        for x in 0..w {
            let z = data[y * w + x];
            if z.is_nan() {
                continue;
            }
            let b = basis(norm(x, w), ny);
            for i in 0..6 {
                atb[i] += b[i] * z as f64;
                for j in 0..6 {
                    ata[(i, j)] += b[i] * b[j];
                }
            }
        }
    }
    let Some(c) = solve(ata, atb) else { return };
    for y in 0..h {
        let ny = norm(y, h);
        for x in 0..w {
            let b = basis(norm(x, w), ny);
            let fit: f64 = (0..6).map(|i| c[i] * b[i]).sum();
            data[y * w + x] -= fit as f32;
        }
    }
}

/// Fits z = a + bx + cx² to each scan line and subtracts it.
fn subtract_line_parabolas(data: &mut [f32], w: usize, h: usize) {
    for line in data.chunks_mut(w).take(h) {
        let mut ata = DMatrix::<f64>::zeros(3, 3);
        let mut atb = DVector::<f64>::zeros(3);
        for (x, z) in line.iter().enumerate() {
            if z.is_nan() {
                continue;
            }
            let nx = norm(x, w);
            let b = [1., nx, nx * nx];
            for i in 0..3 {
                atb[i] += b[i] * *z as f64;
                for j in 0..3 {
                    ata[(i, j)] += b[i] * b[j];
                }
            }
        }
        let Some(c) = solve(ata, atb) else { continue };
        for (x, z) in line.iter_mut().enumerate() {
            let nx = norm(x, w);
            *z -= (c[0] + c[1] * nx + c[2] * nx * nx) as f32;
        }
    }
}

/// Colour palette for scan images, shared by all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ColorMap {
    #[default]
    Gray,
    Gold,
    Viridis,
    Inferno,
    BlueWhiteRed,
}

impl ColorMap {
    pub const ALL: [ColorMap; 5] = [
        ColorMap::Gray,
        ColorMap::Gold,
        ColorMap::Viridis,
        ColorMap::Inferno,
        ColorMap::BlueWhiteRed,
    ];
    pub fn label(self) -> &'static str {
        match self {
            ColorMap::Gray => "Gray",
            ColorMap::Gold => "Gold",
            ColorMap::Viridis => "Viridis",
            ColorMap::Inferno => "Inferno",
            ColorMap::BlueWhiteRed => "Blue–white–red",
        }
    }
    fn stops(self) -> &'static [[u8; 3]] {
        match self {
            ColorMap::Gray => &[[0, 0, 0], [255, 255, 255]],
            // Black → brown → orange → yellow → white.
            ColorMap::Gold => &[
                [0, 0, 0],
                [96, 32, 0],
                [192, 96, 16],
                [240, 176, 64],
                [255, 240, 176],
                [255, 255, 255],
            ],
            ColorMap::Viridis => &[
                [68, 1, 84],
                [71, 44, 122],
                [59, 81, 139],
                [44, 113, 142],
                [33, 144, 141],
                [39, 173, 129],
                [92, 200, 99],
                [170, 220, 50],
                [253, 231, 37],
            ],
            ColorMap::Inferno => &[
                [0, 0, 4],
                [31, 12, 72],
                [85, 15, 109],
                [136, 34, 106],
                [186, 54, 85],
                [227, 89, 51],
                [249, 140, 10],
                [249, 201, 50],
                [252, 255, 164],
            ],
            ColorMap::BlueWhiteRed => &[[59, 76, 192], [221, 221, 221], [180, 4, 38]],
        }
    }
    pub fn color(self, t: f32) -> Color32 {
        let stops = self.stops();
        let pos = t.clamp(0., 1.) * (stops.len() - 1) as f32;
        let i = (pos.floor() as usize).min(stops.len() - 2);
        let f = pos - i as f32;
        let [a, b] = [stops[i], stops[i + 1]];
        let mix = |k: usize| (a[k] as f32 + (b[k] as f32 - a[k] as f32) * f).round() as u8;
        Color32::from_rgb(mix(0), mix(1), mix(2))
    }
    pub fn table(self, invert: bool) -> Box<[Color32; COLOR_MAP_SIZE]> {
        let mut table = Box::new([Color32::BLACK; COLOR_MAP_SIZE]);
        for (i, c) in table.iter_mut().enumerate() {
            let t = i as f32 / (COLOR_MAP_SIZE - 1) as f32;
            *c = self.color(if invert { 1. - t } else { t });
        }
        table
    }
}

/// Palette and invert choice; saved in the UI preferences and used for every
/// scan image.
/// One row (label | control) for a two-column [`egui::Grid`].
pub fn color_scale_ui(ui: &mut Ui) {
    let prefs = crate::theme::prefs();
    let (mut map, mut invert) = (prefs.color_map, prefs.color_invert);
    ui.label("Colours").on_hover_text(
        "Colour scale for all scan images. Blue and red still mark values clipped below and above the range.",
    );
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("scan_color_map")
            .selected_text(map.label())
            .show_ui(ui, |ui| {
                for m in ColorMap::ALL {
                    ui.horizontal(|ui| {
                        color_bar(ui, m, invert);
                        ui.selectable_value(&mut map, m, m.label());
                    });
                }
            });
        ui.checkbox(&mut invert, "Invert");
    });
    ui.end_row();
    if (map, invert) != (prefs.color_map, prefs.color_invert) {
        crate::theme::update_prefs(|p| {
            p.color_map = map;
            p.color_invert = invert;
        });
    }
}

/// Small preview of a palette.
fn color_bar(ui: &mut Ui, map: ColorMap, invert: bool) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(48., 12.), egui::Sense::hover());
    let n = 24;
    for i in 0..n {
        let t = (i as f32 + 0.5) / n as f32;
        let x0 = rect.left() + rect.width() * i as f32 / n as f32;
        let x1 = rect.left() + rect.width() * (i + 1) as f32 / n as f32;
        ui.painter().rect_filled(
            egui::Rect::from_x_y_ranges(x0..=x1 + 0.5, rect.y_range()),
            0.,
            map.color(if invert { 1. - t } else { t }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(w: usize, h: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f32> {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| f(norm(x, w), norm(y, h)) as f32)
            .collect()
    }

    #[test]
    fn parabola_removes_quadratic_surface() {
        let (w, h) = (40, 30);
        let mut d = surface(w, h, |x, y| 3. + 2. * x - y + 5. * x * x + 0.5 * x * y - 4. * y * y);
        d[17] = f32::NAN;
        subtract_parabola(&mut d, w, h);
        assert!(d[17].is_nan());
        assert!(d.iter().filter(|v| !v.is_nan()).all(|v| v.abs() < 1e-4), "{d:?}");
    }

    #[test]
    fn parabola_handles_one_scanned_line() {
        let (w, h) = (20, 10);
        let mut d = vec![f32::NAN; w * h];
        for x in 0..w {
            let nx = norm(x, w);
            d[x] = (1. + nx * nx) as f32;
        }
        subtract_parabola(&mut d, w, h);
        assert!(d[..w].iter().all(|v| v.abs() < 1e-4));
        assert!(d[w..].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn line_parabola_removes_per_line_bow() {
        let (w, h) = (32, 8);
        let mut d = surface(w, h, |x, y| 10. * y + (1. + y) * x * x - x);
        subtract_line_parabolas(&mut d, w, h);
        assert!(d.iter().all(|v| v.abs() < 1e-4));
    }

    #[test]
    fn blur_keeps_flat_image_and_holes() {
        let (w, h) = (16, 16);
        let mut d = vec![2.5f32; w * h];
        d[40] = f32::NAN;
        let b = gaussian_blur(&d, w, h, 2.);
        assert!(b[40].is_nan());
        assert!(b.iter().filter(|v| !v.is_nan()).all(|v| (v - 2.5).abs() < 1e-5));
    }

    /// Repeatable noise in -0.5..0.5.
    fn noise(i: usize) -> f32 {
        // splitmix64
        let mut z = (i as u64).wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    /// Largest difference between `a` and `b`, and where (x, y).
    fn worst(a: &[f32], b: &[f32], w: usize) -> (f32, usize, usize) {
        let (at, e) = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .enumerate()
            .fold((0, 0.), |m, (i, e)| if e > m.1 { (i, e) } else { m });
        (e, at % w, at / w)
    }

    fn run(d: &[f32], w: usize, h: usize, filters: Filters) -> Vec<f32> {
        process(d, [w as u32, h as u32], Background::Raw, filters).unwrap()
    }

    const LINES: Filters = Filters {
        smooth: false,
        smooth_px: 1.,
        sharpen: false,
        sharpen_amount: 1.,
        lines: true,
        lines_threshold: 3.,
        blobs: false,
        blob_px: 6.,
    };
    const BLOBS: Filters = Filters { lines: false, blobs: true, ..LINES };

    #[test]
    fn streaks_are_removed_and_original_kept() {
        let (w, h) = (64, 48);
        let clean: Vec<f32> = (0..w * h)
            .map(|i| 0.1 * (i / w) as f32 + 0.05 * noise(i))
            .collect();
        let mut d = clean.clone();
        for x in 10..50 {
            d[12 * w + x] += 3.; // one-line tip change
            d[20 * w + x] -= 2.; // two-line streak
            d[21 * w + x] -= 2.;
            for y in 30..33 {
                d[y * w + x] += 1.; // three-line band
            }
        }
        // A noisy streak (feedback oscillation).
        for x in 5..40 {
            d[40 * w + x] += 1. + 0.5 * noise(x + 999);
        }
        let raw = d.clone();
        let out = run(&d, w, h, LINES);
        assert_eq!(d, raw, "input must not be modified");
        let (e, x, y) = worst(&out, &clean, w);
        assert!(e < 0.2, "worst error {e} at x {x} y {y}");
    }

    #[test]
    fn streak_filter_keeps_clean_images() {
        let (w, h) = (64, 64);
        let bump = |x: f32, y: f32| 2. * (-((x - 30.).powi(2) + (y - 30.).powi(2)) / 18.).exp();
        // A round bump, a step edge along the lines, and atom rows.
        let cases: [&dyn Fn(usize, usize) -> f32; 3] = [
            &|x, y| bump(x as f32, y as f32),
            &|_, y| if y < 25 { 0. } else { 1.5 },
            &|x, y| {
                (y as f32 * std::f32::consts::TAU / 5.).cos()
                    + 0.3 * (x as f32 * std::f32::consts::TAU / 7.).cos()
            },
        ];
        for (n, f) in cases.iter().enumerate() {
            let d: Vec<f32> = (0..w * h).map(|i| f(i % w, i / w) + 0.05 * noise(i)).collect();
            let (e, x, y) = worst(&run(&d, w, h, LINES), &d, w);
            assert!(e < 0.3, "case {n}: worst change {e} at x {x} y {y}");
        }
    }

    #[test]
    fn streaks_fixed_while_scanning() {
        let (w, h) = (64, 32);
        let mut d: Vec<f32> = (0..w * h).map(|i| 0.05 * noise(i)).collect();
        for x in 10..50 {
            d[10 * w + x] += 2.;
        }
        d[20 * w..].fill(f32::NAN);
        let out = run(&d, w, h, LINES);
        assert!(out[20 * w..].iter().all(|v| v.is_nan()));
        let (e, x, y) = worst(&out[..20 * w], &vec![0.; 20 * w], w);
        assert!(e < 0.1, "worst {e} at x {x} y {y}");
    }

    #[test]
    fn blobs_are_removed_features_kept() {
        let (w, h) = (64, 64);
        let bump = |x: f32, y: f32| 2. * (-((x - 40.).powi(2) + (y - 40.).powi(2)) / 18.).exp();
        let clean: Vec<f32> = (0..w * h)
            .map(|i| bump((i % w) as f32, (i / w) as f32) + 0.05 * noise(i))
            .collect();
        let mut d = clean.clone();
        for (cx, cy, s) in [(10, 10, 1.5f32), (20, 50, -1.), (50, 12, 3.)] {
            for y in cy..cy + 3 {
                for x in cx..cx + 4 {
                    d[y * w + x] += s;
                }
            }
        }
        d[30 * w + 5] += 4.; // single-pixel spike
        let out = run(&d, w, h, BLOBS);
        let (e, x, y) = worst(&out, &clean, w);
        assert!(e < 0.25, "worst error {e} at x {x} y {y}");
    }

    #[test]
    fn nothing_to_do_returns_none() {
        assert!(process(&[1.; 4], [2, 2], Background::Plane, Filters::default()).is_none());
    }

    #[test]
    fn old_fit_numbers_still_load() {
        for (n, b) in [
            (0u8, Background::Raw),
            (1, Background::Plane),
            (2, Background::LineMean),
            (3, Background::LineLinear),
        ] {
            assert_eq!(Background::try_from(n).unwrap(), b);
            assert_eq!(FitType::try_from(n).unwrap(), b.gpu_fit());
        }
    }
}
