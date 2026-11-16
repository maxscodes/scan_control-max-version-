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
    /// Remove horizontal streaks (tip changes, blemishes).
    pub deglitch: bool,
    /// How far (in noise levels) a streak must stand out to be removed.
    pub deglitch_threshold: f32,
}

impl Default for Filters {
    fn default() -> Self {
        Self {
            smooth: false,
            smooth_px: 1.,
            sharpen: false,
            sharpen_amount: 1.,
            deglitch: false,
            deglitch_threshold: 4.,
        }
    }
}

impl Filters {
    pub const MAX_SMOOTH_PX: f32 = 5.;
    pub fn to_array(self) -> [f32; 6] {
        [
            self.smooth as u8 as f32,
            self.smooth_px,
            self.sharpen as u8 as f32,
            self.sharpen_amount,
            self.deglitch as u8 as f32,
            self.deglitch_threshold,
        ]
    }
    pub fn from_array(a: [f32; 6]) -> Self {
        Self {
            smooth: a[0] != 0.,
            smooth_px: a[1].clamp(0.3, Self::MAX_SMOOTH_PX),
            sharpen: a[2] != 0.,
            sharpen_amount: a[3].clamp(0.1, 5.),
            deglitch: a[4] != 0.,
            deglitch_threshold: a[5].clamp(1.5, 20.),
        }
    }
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
    fn any(&self) -> bool {
        self.smooth || self.sharpen || self.deglitch
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
        "Fix glitches",
        &mut filters.deglitch,
        DragValue::new(&mut filters.deglitch_threshold).range(1.5..=20.).speed(0.05),
        "Removes horizontal streaks from tip changes and blemishes: stretches of a scan \
         line that jump away from the lines above and below are filled in from those \
         lines. Lower values remove more. Only the drawn image changes; the scan data \
         and .sxm files are never modified.",
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
    if filters.deglitch {
        remove_line_glitches(&mut out, w, h, filters.deglitch_threshold);
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

/// Removes horizontal streaks (scars): stretches of one or two scan lines
/// that jump away from the lines above and below by more than `threshold`
/// noise levels are replaced by interpolating between those lines. Features
/// that span several lines (atoms, steps) don't stand out from their
/// neighbouring lines and are kept.
fn remove_line_glitches(data: &mut [f32], w: usize, h: usize, threshold: f32) {
    if h < 3 || w < 3 {
        return;
    }
    let min_run = (w / 16).max(6);
    for _pass in 0..2 {
        // Height expected at (x, rows y0..y1) from the lines around it.
        let around = |d: &[f32], x: usize, y0: usize, y1: usize| -> Option<f32> {
            let above = (y0 > 0).then(|| d[(y0 - 1) * w + x]).filter(|v| !v.is_nan());
            let below = (y1 < h).then(|| d[y1 * w + x]).filter(|v| !v.is_nan());
            match (above, below) {
                (Some(a), Some(b)) => Some(0.5 * (a + b)),
                (a, b) => a.or(b),
            }
        };
        let both = |d: &[f32], x: usize, y0: usize, y1: usize| {
            y0 > 0 && y1 < h && !d[(y0 - 1) * w + x].is_nan() && !d[y1 * w + x].is_nan()
        };
        let dev: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                // Both neighbouring lines are needed: with one, a slope
                // looks like a streak.
                if data[i].is_nan() || !both(data, x, y, y + 1) {
                    return f32::NAN;
                }
                data[i] - around(data, x, y, y + 1).unwrap()
            })
            .collect();
        let mut abs: Vec<f32> = dev.iter().filter(|v| !v.is_nan()).map(|v| v.abs()).collect();
        if abs.is_empty() {
            return;
        }
        let mid = abs.len() / 2;
        let median = *abs.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
        let limit = threshold * 1.4826 * median;
        if !(limit > 0.) {
            return;
        }
        // +1 / -1: the pixel stands out up / down from the lines around it.
        let mut loose = vec![0i8; w * h];
        for i in 0..w * h {
            if dev[i].abs() > limit {
                loose[i] = dev[i].signum() as i8;
            }
        }
        // A streak is flat along the line; a real bump curves the same way
        // along the line as across it, so it doesn't count.
        // How far each pixel stands out across the lines.
        let mut across: Vec<f32> = dev.iter().map(|v| if v.is_nan() { 0. } else { v.abs() }).collect();
        // Streaks two lines thick.
        for y in 0..h - 1 {
            for x in 0..w {
                let (i, j) = (y * w + x, (y + 1) * w + x);
                if data[i].is_nan() || data[j].is_nan() || !both(data, x, y, y + 2) {
                    continue;
                }
                let e = around(data, x, y, y + 2).unwrap();
                let (da, db) = (data[i] - e, data[j] - e);
                if da.abs() > limit && db.abs() > limit && da.signum() == db.signum() {
                    for (k, d) in [(i, da), (j, db)] {
                        if loose[k] == 0 || across[k] < d.abs() {
                            loose[k] = d.signum() as i8;
                            across[k] = d.abs();
                        }
                    }
                }
            }
        }
        let mut mark = loose.clone();
        for i in 0..w * h {
            let x = i % w;
            if mark[i] == 0 || x == 0 || x == w - 1 {
                continue;
            }
            let (l, r) = (data[i - 1], data[i + 1]);
            if l.is_nan() || r.is_nan() {
                continue;
            }
            let along = data[i] - 0.5 * (l + r);
            if along.abs() * 2. > across[i] {
                mark[i] = 0;
            }
        }
        // Keep only horizontal runs: a streak follows the scan line. Single
        // pixels that miss the test (noise) don't break a run, and a run
        // is grown to its ends, where the step makes the line curve.
        for (row, lrow) in mark.chunks_mut(w).zip(loose.chunks(w)) {
            for x in 1..w - 1 {
                if row[x] == 0 && row[x - 1] != 0 && row[x - 1] == row[x + 1] && lrow[x] == row[x - 1]
                {
                    row[x] = row[x - 1];
                }
            }
            let mut x = 0;
            while x < w {
                let s = row[x];
                let start = x;
                while x < w && row[x] == s {
                    x += 1;
                }
                if s == 0 {
                    continue;
                }
                if x - start < min_run {
                    row[start..x].fill(0);
                    continue;
                }
                let mut a = start;
                while a > 0 && lrow[a - 1] == s {
                    a -= 1;
                    row[a] = s;
                }
                let mut b = x;
                while b < w && lrow[b] == s {
                    row[b] = s;
                    b += 1;
                }
                x = b;
            }
        }
        if mark.iter().all(|m| *m == 0) {
            return;
        }
        // Fill each marked stretch of a column from the rows around it.
        for x in 0..w {
            let mut y = 0;
            while y < h {
                if mark[y * w + x] == 0 {
                    y += 1;
                    continue;
                }
                let y0 = y;
                while y < h && mark[y * w + x] != 0 {
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
                        (Some(a), Some(b)) => {
                            let t = (k + 1) as f32 / (y - y0 + 1) as f32;
                            a + (b - a) * t
                        }
                        (Some(v), None) | (None, Some(v)) => v,
                        (None, None) => data[i],
                    };
                }
            }
        }
    }
}

/// Normalised Gaussian blur: missing pixels don't count and stay missing.
fn gaussian_blur(data: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
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

    #[test]
    fn glitch_lines_are_removed_and_original_kept() {
        let (w, h) = (64, 32);
        let clean: Vec<f32> = (0..w * h)
            .map(|i| 0.1 * (i / w) as f32 + 0.05 * noise(i))
            .collect();
        let mut d = clean.clone();
        for x in 10..50 {
            d[12 * w + x] += 3.; // one-line tip change
            d[20 * w + x] -= 2.; // two-line streak
            d[21 * w + x] -= 2.;
        }
        let raw = d.clone();
        let filters = Filters { deglitch: true, ..Default::default() };
        let out = process(&d, [w as u32, h as u32], Background::Raw, filters).unwrap();
        assert_eq!(d, raw, "input must not be modified");
        let (at, worst) = out
            .iter()
            .zip(&clean)
            .map(|(a, b)| (a - b).abs())
            .enumerate()
            .fold((0, 0.), |m, (i, e)| if e > m.1 { (i, e) } else { m });
        assert!(worst < 0.2, "worst error {worst} at x {} y {}", at % w, at / w);
    }

    #[test]
    fn glitch_filter_keeps_round_features() {
        let (w, h) = (64, 64);
        let bump = |x: f32, y: f32| 2. * (-((x - 30.).powi(2) + (y - 30.).powi(2)) / 18.).exp();
        let d: Vec<f32> = (0..w * h)
            .map(|i| bump((i % w) as f32, (i / w) as f32) + 0.05 * noise(i))
            .collect();
        let mut out = d.clone();
        remove_line_glitches(&mut out, w, h, 4.);
        let worst = out.iter().zip(&d).map(|(a, b)| (a - b).abs()).fold(0., f32::max);
        assert!(worst < 0.3, "worst change {worst}");
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
