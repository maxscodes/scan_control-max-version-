//! Two overlapping scans laid over each other, the second one dragged by
//! hand until the features line up: checking (and correcting) how far a
//! coarse move went, in calibration runs and survey grids.

use egui::{CollapsingHeader, Color32, Stroke, Ui};
use glam::DVec2;

use crate::{
    connection::nanonis::coarse_cal::CalFrame,
    scan_view::processing::{self, Background, Filters},
};

/// View state: the hand alignment and how the scans are drawn.
pub struct PairView {
    /// Content shift of the after-scan, pixels (`image_shift` convention).
    pub shift: DVec2,
    tex: [egui::TextureHandle; 2],
    /// How strongly the after-scan is drawn over the before-scan.
    opacity: f32,
    /// Flip between the two instead of blending.
    blink: bool,
    /// Magnification: 1 = the image fits the view.
    zoom: f32,
    /// Where the view is looking, image pixels from the centre.
    pan: egui::Vec2,
    /// Processing for display only (the match uses the raw scans).
    background: Background,
    filters: Filters,
    /// Contrast: the share of pixels clipped at each end, %.
    clip_pct: f32,
}

impl PairView {
    /// A view of `a` (before) and `b` (after) starting at `shift`; zoom and
    /// processing are taken from `like` (the view being replaced), if any.
    pub fn new(ctx: &egui::Context, a: &CalFrame, b: &CalFrame, shift: DVec2, like: Option<&PairView>) -> Self {
        let (background, filters, clip_pct, zoom, pan, opacity, blink) = match like {
            Some(v) => (v.background, v.filters, v.clip_pct, v.zoom, v.pan, v.opacity, v.blink),
            None => (Background::LineMean, Filters::default(), 1., 1., egui::Vec2::ZERO, 0.5, false),
        };
        let mut v = Self {
            shift,
            tex: [blank(ctx, "pair_a"), blank(ctx, "pair_b")],
            opacity,
            blink,
            zoom,
            pan,
            background,
            filters,
            clip_pct,
        };
        v.retexture(ctx, a, b);
        v
    }

    fn retexture(&mut self, ctx: &egui::Context, a: &CalFrame, b: &CalFrame) {
        let (bg, f, c) = (self.background, self.filters, self.clip_pct);
        self.tex = [scan_texture(ctx, "pair_a", a, bg, f, c), scan_texture(ctx, "pair_b", b, bg, f, c)];
    }

    /// The controls and the picture, filling the space left except
    /// `reserve` points below it. `auto`: the automatic match (blue outline).
    pub fn ui(&mut self, ui: &mut Ui, a: &CalFrame, b: &CalFrame, auto: Option<DVec2>, reserve: f32) {
        let ctx = ui.ctx().clone();
        let [w, h] = a.data.size;
        ui.label(
            "Grey: scan before the move. Orange: scan after it. Drag the orange scan \
             (or use the arrow keys over the picture, Shift for \u{bc} px) until the \
             features line up. With Blink on, it is right when nothing jumps.",
        );
        ui.weak(format!("Before: {}  \u{b7}  After: {}", a.source, b.source));
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.blink, "Blink");
            ui.add_enabled(
                !self.blink,
                egui::Slider::new(&mut self.opacity, 0.0..=1.0).text("after scan").show_value(false),
            );
            ui.add(
                egui::Slider::new(&mut self.zoom, 0.5..=16.)
                    .logarithmic(true)
                    .text("zoom")
                    .custom_formatter(|v, _| format!("{v:.1}\u{d7}")),
            );
            if ui.button("Fit").clicked() {
                self.zoom = 1.;
                self.pan = egui::Vec2::ZERO;
            }
        });
        let mut redraw = false;
        CollapsingHeader::new("Display processing")
            .id_salt("pair_view_processing")
            .default_open(false)
            .show(ui, |ui| {
                ui.weak("Only changes what you see here; the scans and the automatic match are untouched.");
                redraw |= processing::processing_ui(ui, "pair_view_proc", &mut self.background, &mut self.filters);
                redraw |= ui
                    .add(egui::Slider::new(&mut self.clip_pct, 0.0..=10.).text("contrast clip %"))
                    .changed();
            });
        if redraw {
            self.retexture(&ctx, a, b);
        }
        ui.weak("Drag: move the after-scan \u{b7} scroll: zoom \u{b7} right- or middle-drag: pan");
        // The picture fills the rest of the window.
        let avail = ui.available_size() - egui::vec2(0., reserve);
        let view_size = egui::vec2(avail.x.max(200.), avail.y.max(200.));
        let (rect, resp) = ui.allocate_exact_size(view_size, egui::Sense::drag());
        let fit = (view_size.x / w.max(1) as f32).min(view_size.y / h.max(1) as f32);
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0. {
                // Zoom about the pointer.
                let old = self.zoom;
                self.zoom = (self.zoom * (scroll / 200.).exp()).clamp(0.5, 16.);
                if let Some(p) = resp.hover_pos() {
                    let rel = (p - rect.center()) / (fit * old) + self.pan;
                    self.pan = rel - (rel - self.pan) * old / self.zoom;
                }
            }
        }
        // Screen points per image pixel.
        let px = fit * self.zoom;
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            self.shift -= DVec2::new(d.x as f64, d.y as f64) / px as f64;
        }
        if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            self.pan -= resp.drag_delta() / px;
        }
        if resp.hovered() {
            let (l, r, u, dn, fine) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                    i.key_pressed(egui::Key::ArrowUp),
                    i.key_pressed(egui::Key::ArrowDown),
                    i.modifiers.shift,
                )
            });
            let s = if fine { 0.25 } else { 1. };
            // Moving the after-scan left means a larger content shift.
            self.shift += DVec2::new(s * (l as i32 - r as i32) as f64, s * (u as i32 - dn as i32) as f64);
        }
        let size = egui::vec2(w as f32, h as f32) * px;
        let ra = egui::Rect::from_center_size(rect.center() - self.pan * px, size);
        let at = |s: DVec2| ra.translate(-egui::vec2(s.x as f32, s.y as f32) * px);
        let rb = at(self.shift);
        let uv = egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(1., 1.));
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0., Color32::from_gray(15));
        let orange = |a: f32| Color32::from_rgba_unmultiplied(255, 170, 70, (a * 255.) as u8);
        if self.blink {
            let show_b = (ui.input(|i| i.time) * 2.).floor() as i64 % 2 == 1;
            if show_b {
                painter.image(self.tex[1].id(), rb, uv, orange(1.));
            } else {
                painter.image(self.tex[0].id(), ra, uv, Color32::WHITE);
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        } else {
            painter.image(self.tex[0].id(), ra, uv, Color32::WHITE);
            painter.image(self.tex[1].id(), rb, uv, orange(self.opacity));
        }
        painter.rect_stroke(ra, 0., Stroke::new(1., Color32::GRAY), egui::StrokeKind::Outside);
        painter.rect_stroke(rb, 0., Stroke::new(1., orange(1.)), egui::StrokeKind::Outside);
        if let Some(auto) = auto {
            painter.rect_stroke(at(auto), 0., Stroke::new(1., Color32::from_rgb(80, 200, 255)), egui::StrokeKind::Outside);
        }
    }
}

fn blank(ctx: &egui::Context, name: &str) -> egui::TextureHandle {
    ctx.load_texture(name, egui::ColorImage::from_gray([1, 1], &[0]), egui::TextureOptions::NEAREST)
}

/// Grey image of a scan with display processing (on a copy), contrast
/// clipping `clip_pct` % at each end.
pub fn scan_texture(
    ctx: &egui::Context,
    name: &str,
    f: &CalFrame,
    background: Background,
    filters: Filters,
    clip_pct: f32,
) -> egui::TextureHandle {
    let [w, h] = f.data.size;
    let (w, h) = (w as usize, h as usize);
    let mut v = processing::process(&f.data.data, f.data.size, background, filters)
        .unwrap_or_else(|| f.data.data.to_vec());
    // Line and plane fits run on the GPU for live images; do them here.
    match background {
        Background::LineMean | Background::LineLinear => {
            for j in 0..h {
                let row = &mut v[j * w..(j + 1) * w];
                let pts: Vec<(f64, f64)> = row
                    .iter()
                    .enumerate()
                    .filter(|(_, x)| x.is_finite())
                    .map(|(i, &x)| (i as f64, x as f64))
                    .collect();
                if pts.is_empty() {
                    continue;
                }
                let n = pts.len() as f64;
                let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
                let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
                let slope = if background == Background::LineLinear {
                    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
                    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
                    if sxx > 0. { sxy / sxx } else { 0. }
                } else {
                    0.
                };
                for (i, x) in row.iter_mut().enumerate() {
                    *x -= (my + slope * (i as f64 - mx)) as f32;
                }
            }
        }
        Background::Plane => {
            // Least-squares plane z = a + b·x + c·y.
            let mut s = [[0f64; 3]; 3];
            let mut r = [0f64; 3];
            let mut n = 0usize;
            for j in 0..h {
                for i in 0..w {
                    let z = v[j * w + i];
                    if !z.is_finite() {
                        continue;
                    }
                    let q = [1., i as f64, j as f64];
                    for a in 0..3 {
                        r[a] += q[a] * z as f64;
                        for b in 0..3 {
                            s[a][b] += q[a] * q[b];
                        }
                    }
                    n += 1;
                }
            }
            let m = glam::DMat3::from_cols_array_2d(&s);
            if n > 3 && m.determinant().abs() > 1e-9 {
                let c = m.inverse() * glam::DVec3::from_array(r);
                for j in 0..h {
                    for i in 0..w {
                        v[j * w + i] -= (c.x + c.y * i as f64 + c.z * j as f64) as f32;
                    }
                }
            }
        }
        _ => {}
    }
    let clip = (clip_pct as f64 / 100.).clamp(0., 0.45);
    let mut sorted: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    let pick = |q: f64| sorted.get(((sorted.len().max(1) - 1) as f64 * q) as usize).copied().unwrap_or(0.);
    let (lo, hi) = (pick(clip), pick(1. - clip));
    let span = (hi - lo).max(f32::MIN_POSITIVE);
    let bytes: Vec<u8> = v
        .iter()
        .map(|&x| if x.is_finite() { ((x - lo) / span).clamp(0., 1.) * 255. } else { 0. } as u8)
        .collect();
    ctx.load_texture(name, egui::ColorImage::from_gray([w, h], &bytes), egui::TextureOptions::NEAREST)
}
