//! One figure and one table holding every curve of a series, saved next to
//! the per-curve .dat files so the batch stays together.

use std::fmt::Write as _;

use egui::{ecolor::Hsva, Color32};

use super::model::{didv_channel_names, IvCurve};

/// Same colour for a curve in every plot, well separated between neighbours.
pub fn curve_color(index: usize) -> Color32 {
    let h = (index as f32 * 0.618_034).fract();
    Hsva::new(h, 0.85, 0.65, 1.0).into()
}

fn hex(c: Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// About `n` round tick values covering `lo..hi`.
fn ticks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let span = (hi - lo).abs();
    if !span.is_finite() || span == 0. {
        return vec![lo];
    }
    let raw = span / n as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1., 2., 5., 10.]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(10. * mag);
    let first = (lo / step).ceil() as i64;
    let last = (hi / step).floor() as i64;
    (first..=last).map(|k| k as f64 * step).collect()
}

fn fmt_tick(v: f64) -> String {
    if v == 0. {
        return "0".into();
    }
    let a = v.abs();
    if (1e-3..1e5).contains(&a) {
        let s = format!("{v:.4}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    } else {
        format!("{v:.1e}")
    }
}

/// One panel: all curves on the same axes.
struct Panel<'a> {
    title: String,
    y_label: &'a str,
    /// (curve index, points)
    series: Vec<(usize, Vec<[f64; 2]>)>,
}

const PANEL_W: f64 = 560.;
const PANEL_H: f64 = 400.;
const MARGIN_L: f64 = 80.;
const MARGIN_R: f64 = 20.;
const MARGIN_T: f64 = 60.;
const MARGIN_B: f64 = 60.;
const LEGEND_W: f64 = 190.;

fn draw_panel(s: &mut String, p: &Panel, x0: f64) {
    let (px, py) = (x0 + MARGIN_L, MARGIN_T);
    let all = p.series.iter().flat_map(|(_, pts)| pts.iter());
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for [x, y] in all {
        if x.is_finite() && y.is_finite() {
            xmin = xmin.min(*x);
            xmax = xmax.max(*x);
            ymin = ymin.min(*y);
            ymax = ymax.max(*y);
        }
    }
    if xmin > xmax {
        (xmin, xmax, ymin, ymax) = (0., 1., 0., 1.);
    }
    if xmax == xmin {
        xmax = xmin + 1.;
    }
    if ymax == ymin {
        ymax = ymin + 1.;
    }
    let pad = (ymax - ymin) * 0.05;
    let (ymin, ymax) = (ymin - pad, ymax + pad);
    let sx = |x: f64| px + (x - xmin) / (xmax - xmin) * PANEL_W;
    let sy = |y: f64| py + PANEL_H - (y - ymin) / (ymax - ymin) * PANEL_H;

    let _ = writeln!(
        s,
        r#"<text x="{:.1}" y="{:.1}" font-size="16" font-weight="bold" text-anchor="middle">{}</text>"#,
        px + PANEL_W / 2.,
        py - 20.,
        esc(&p.title)
    );
    // Grid and ticks.
    for t in ticks(xmin, xmax, 10) {
        let x = sx(t);
        let _ = writeln!(s, r##"<line x1="{x:.1}" y1="{py:.1}" x2="{x:.1}" y2="{:.1}" stroke="#e4e4e4"/>"##, py + PANEL_H);
        let _ = writeln!(
            s,
            r#"<text x="{x:.1}" y="{:.1}" font-size="12" text-anchor="middle">{}</text>"#,
            py + PANEL_H + 18.,
            fmt_tick(t)
        );
    }
    for t in ticks(ymin, ymax, 8) {
        let y = sy(t);
        let _ = writeln!(s, r##"<line x1="{px:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="#e4e4e4"/>"##, px + PANEL_W);
        let _ = writeln!(
            s,
            r#"<text x="{:.1}" y="{:.1}" font-size="12" text-anchor="end">{}</text>"#,
            px - 6.,
            y + 4.,
            fmt_tick(t)
        );
    }
    if (ymin..=ymax).contains(&0.) {
        let y = sy(0.);
        let _ = writeln!(s, r##"<line x1="{px:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="#999999"/>"##, px + PANEL_W);
    }
    let _ = writeln!(
        s,
        r#"<rect x="{px:.1}" y="{py:.1}" width="{PANEL_W}" height="{PANEL_H}" fill="none" stroke="black"/>"#
    );
    let _ = writeln!(
        s,
        r#"<text x="{:.1}" y="{:.1}" font-size="14" text-anchor="middle">Bias (V)</text>"#,
        px + PANEL_W / 2.,
        py + PANEL_H + 42.
    );
    let (lx, ly) = (x0 + 22., py + PANEL_H / 2.);
    let _ = writeln!(
        s,
        r#"<text x="{lx:.1}" y="{ly:.1}" font-size="14" text-anchor="middle" transform="rotate(-90 {lx:.1} {ly:.1})">{}</text>"#,
        esc(p.y_label)
    );
    // Curves, clipped to the panel.
    let clip = format!("clip{}", x0 as i64);
    let _ = writeln!(
        s,
        r#"<clipPath id="{clip}"><rect x="{px:.1}" y="{py:.1}" width="{PANEL_W}" height="{PANEL_H}"/></clipPath><g clip-path="url(#{clip})">"#
    );
    for (index, pts) in &p.series {
        let d: Vec<String> = pts
            .iter()
            .filter(|[x, y]| x.is_finite() && y.is_finite())
            .map(|[x, y]| format!("{:.2},{:.2}", sx(*x), sy(*y)))
            .collect();
        if d.len() < 2 {
            continue;
        }
        let _ = writeln!(
            s,
            r#"<polyline fill="none" stroke="{}" stroke-width="2.5" stroke-linejoin="round" points="{}"/>"#,
            hex(curve_color(*index)),
            d.join(" ")
        );
    }
    let _ = writeln!(s, "</g>");
}

/// Which plots are shown (and saved): I(V) always, plus these.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlotChoice {
    pub log_i: bool,
    pub didv: bool,
    pub log_didv: bool,
    /// `None`: automatic (lock-in Y first).
    pub didv_channel: Option<String>,
}
/// The I(V) window's choice, also used when a sequence saves a series.
pub type SharedPlotChoice = std::sync::Arc<parking_lot::Mutex<PlotChoice>>;

impl PlotChoice {
    /// The dI/dV channel to plot for these curves.
    pub fn channel(&self, curves: &[IvCurve]) -> Option<String> {
        let names = didv_channel_names(curves);
        self.didv_channel
            .clone()
            .filter(|c| names.contains(c))
            .or_else(|| super::model::auto_didv_channel(&names).cloned())
    }
}

/// `log10 |y|`, dropping zeros.
fn log_points(bias: &[f64], y: &[f64]) -> Vec<[f64; 2]> {
    bias.iter().zip(y).filter(|(_, y)| y.abs() > 0.).map(|(&v, &y)| [v, y.abs().log10()]).collect()
}

/// An SVG with every plot chosen in `choice` side by side (as in the I(V)
/// window), every curve on the same axes, the starting bias and setpoint,
/// and a legend beside the plots in series order.
pub fn combined_svg(title: &str, curves: &[IvCurve], choice: &PlotChoice) -> String {
    let channel = choice.channel(curves);
    let mut curves: Vec<&IvCurve> = curves.iter().collect();
    curves.sort_by_key(|c| c.index);
    let mut panels = vec![Panel {
        title: "I(V)".into(),
        y_label: "Current (pA)",
        series: curves
            .iter()
            .map(|c| (c.index, c.bias_v.iter().zip(&c.current_pa).map(|(&v, &i)| [v, i]).collect()))
            .collect(),
    }];
    if choice.log_i {
        panels.push(Panel {
            title: "log |I|".into(),
            y_label: "log₁₀ |I| (pA)",
            series: curves.iter().map(|c| (c.index, log_points(&c.bias_v, &c.current_pa))).collect(),
        });
    }
    if let Some(ch) = channel.as_deref() {
        let didv = |log: bool| -> Vec<(usize, Vec<[f64; 2]>)> {
            curves
                .iter()
                .filter_map(|c| {
                    let d = c.didv(ch)?;
                    Some((
                        c.index,
                        if log {
                            log_points(&c.bias_v, d)
                        } else {
                            c.bias_v.iter().zip(d).map(|(&v, &y)| [v, y]).collect()
                        },
                    ))
                })
                .collect()
        };
        if choice.didv {
            let series = didv(false);
            if !series.is_empty() {
                panels.push(Panel { title: format!("dI/dV: {ch}"), y_label: "Lock-in (pA)", series });
            }
        }
        if choice.log_didv {
            let series = didv(true);
            if !series.is_empty() {
                panels.push(Panel {
                    title: format!("log dI/dV: {ch}"),
                    y_label: "log₁₀ |lock-in| (pA)",
                    series,
                });
            }
        }
    }

    let panel_w = MARGIN_L + PANEL_W + MARGIN_R;
    let width = panel_w * panels.len() as f64 + LEGEND_W;
    let legend_h = 40. + 20. * curves.len() as f64;
    let height = (MARGIN_T + PANEL_H + MARGIN_B + 30.).max(legend_h + MARGIN_T);
    let mut s = String::new();
    let _ = writeln!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" viewBox="0 0 {width:.0} {height:.0}" font-family="Arial, Helvetica, sans-serif">"#
    );
    let _ = writeln!(s, r#"<rect width="100%" height="100%" fill="white"/>"#);
    let _ = writeln!(
        s,
        r##"<text x="12" y="22" font-size="15" fill="#444444">{} — {} curves</text>"##,
        esc(title),
        curves.len()
    );
    if let Some(c) = curves.first() {
        // Starting conditions, top-left inside the I(V) plot (as in the window).
        let _ = writeln!(
            s,
            r##"<text x="{:.1}" y="{:.1}" font-size="13" font-weight="bold" fill="#222222">{}</text>"##,
            MARGIN_L + 8.,
            MARGIN_T + 18.,
            esc(&c.start_conditions())
        );
    }
    for (k, p) in panels.iter().enumerate() {
        draw_panel(&mut s, p, panel_w * k as f64);
    }
    // Legend outside the plots.
    let lx = panel_w * panels.len() as f64 + 4.;
    let _ = writeln!(s, r#"<text x="{lx:.1}" y="{MARGIN_T:.1}" font-size="13" font-weight="bold">Curves</text>"#);
    for (k, c) in curves.iter().enumerate() {
        let y = MARGIN_T + 20. + 20. * k as f64;
        let _ = writeln!(
            s,
            r#"<line x1="{lx:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="3"/>"#,
            y - 4.,
            lx + 24.,
            y - 4.,
            hex(curve_color(c.index))
        );
        let _ = writeln!(
            s,
            r#"<text x="{:.1}" y="{y:.1}" font-size="12">#{} {}</text>"#,
            lx + 30.,
            c.index,
            esc(&c.label())
        );
    }
    let _ = writeln!(s, "</svg>");
    s
}

/// Save a series into a new folder `<parent>/<prefix>_BATCH_<time>`: copies
/// of the .dat files Nanonis saved (its own format, for its viewer), plus all
/// curves together as `..._all_N_curves.jpg/.svg` (one plot) and `.dat` (one
/// table). Without Nanonis files, our own per-curve export .dat is written
/// instead. Returns the folder and a short message.
pub fn save_batch(
    parent: &std::path::Path,
    prefix: &str,
    curves: &[IvCurve],
    choice: &PlotChoice,
    nanonis_files: &[std::path::PathBuf],
) -> Result<(std::path::PathBuf, String), String> {
    let stamp = super::model::utc_timestamp().replace(['-', ':', 'Z'], "").replace('T', "_");
    let batch = format!("{prefix}_BATCH_{stamp}");
    let dir = parent.join(&batch);
    // Named after the batch so they stay recognisable if moved out of the folder.
    let together = format!("{batch}_all_{}_curves", curves.len());
    let result = std::fs::create_dir_all(&dir)
        .and_then(|_| {
            if nanonis_files.is_empty() {
                return super::model::save_curves(&dir, prefix, curves);
            }
            for f in nanonis_files {
                if let Some(name) = f.file_name() {
                    std::fs::copy(f, dir.join(name))?;
                }
            }
            Ok(nanonis_files.len())
        })
        .and_then(|n| {
            let svg = combined_svg(&batch, curves, choice);
            std::fs::write(dir.join(format!("{together}.svg")), &svg)?;
            let jpg = svg_to_jpeg(&svg, 2.).map_err(std::io::Error::other)?;
            std::fs::write(dir.join(format!("{together}.jpg")), jpg)?;
            std::fs::write(dir.join(format!("{together}.dat")), combined_dat(&batch, curves))?;
            Ok(n)
        });
    match result {
        Ok(n) => {
            tracing::info!("saved {n} I(V) curves to {}", dir.display());
            let what = if nanonis_files.is_empty() {
                format!(
                    "Nanonis didn't save its own files, so {n} curves were written in our export \
                     format (the Nanonis viewer can't open these): turn on saving in the Nanonis \
                     Bias Spectroscopy module."
                )
            } else {
                format!("Copied the {n} Nanonis .dat files.")
            };
            let msg = format!(
                "{what} Saved {together} (.jpg and .svg: all in one plot; .dat: one table) to {}",
                dir.display()
            );
            Ok((dir, msg))
        }
        Err(e) => Err(format!("Could not save curves to {}: {e}", dir.display())),
    }
}

/// Render an SVG (from [`combined_svg`]) to JPEG bytes at `scale`× its size,
/// on white. Text uses the system fonts.
pub fn svg_to_jpeg(svg: &str, scale: f32) -> Result<Vec<u8>, String> {
    use resvg::{tiny_skia, usvg};
    let mut opt = usvg::Options::default();
    opt.fontdb_mut().load_system_fonts();
    let tree = usvg::Tree::from_str(svg, &opt).map_err(|e| format!("bad SVG: {e}"))?;
    let size = tree.size();
    let (w, h) = ((size.width() * scale).ceil() as u32, (size.height() * scale).ceil() as u32);
    let mut pixmap = tiny_skia::Pixmap::new(w, h).ok_or("empty image")?;
    pixmap.fill(tiny_skia::Color::WHITE);
    resvg::render(&tree, tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    // Opaque (white background), so premultiplied RGBA is plain RGB.
    let rgb: Vec<u8> = pixmap.data().chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 92)
        .encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("JPEG encoding failed: {e}"))?;
    Ok(out)
}

/// Every curve in one tab-separated table: for each curve its bias, current
/// and dI/dV columns, side by side (curves may differ in length).
pub fn combined_dat(title: &str, curves: &[IvCurve]) -> String {
    let channels = didv_channel_names(curves);
    let mut curves: Vec<&IvCurve> = curves.iter().collect();
    curves.sort_by_key(|c| c.index);
    let mut s = String::new();
    let _ = writeln!(s, "# All curves of I(V) series {title}");
    let _ = writeln!(s, "# Curves: {}", curves.len());
    if let Some(c) = curves.first() {
        let _ = writeln!(s, "# {}", c.start_conditions());
    }
    for c in &curves {
        let _ = writeln!(s, "# Curve {}: {} ({})", c.index, c.label(), c.timestamp);
    }
    let mut header = Vec::new();
    for c in &curves {
        let tag = format!("[{} Z={:+.1}pm]", c.index, c.z_offset_pm);
        header.push(format!("Bias_V {tag}"));
        header.push(format!("Current_pA {tag}"));
        for ch in &channels {
            header.push(format!("{ch}_pA {tag}"));
        }
    }
    let _ = writeln!(s, "{}", header.join("\t"));
    let rows = curves.iter().map(|c| c.bias_v.len()).max().unwrap_or(0);
    for i in 0..rows {
        let mut row = Vec::new();
        for c in &curves {
            let cell = |v: Option<&f64>| v.map_or(String::new(), |x| format!("{x:.6e}"));
            row.push(cell(c.bias_v.get(i)));
            row.push(cell(c.current_pa.get(i)));
            for ch in &channels {
                row.push(cell(c.didv(ch).and_then(|d| d.get(i))));
            }
        }
        let _ = writeln!(s, "{}", row.join("\t"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(index: usize, z: f64) -> IvCurve {
        IvCurve {
            index,
            z_offset_pm: z,
            bias_v: vec![-1., 0., 1.],
            current_pa: vec![-2., 0., 2. + index as f64],
            didv_pa: vec![("LI Demod 1 Y (A)".into(), vec![1., 2., 3.])],
            timestamp: "2026-11-13T00:00:00Z".into(),
            start_bias_v: -2.,
            start_setpoint_pa: 50.,
        }
    }

    #[test]
    fn svg_has_the_chosen_plots_and_start_conditions() {
        let curves = vec![curve(1, -10.), curve(0, 0.)];
        let all = PlotChoice { log_i: true, didv: true, log_didv: true, didv_channel: None };
        let svg = combined_svg("test", &curves, &all);
        // 4 panels × 2 curves.
        assert_eq!(svg.matches("<polyline").count(), 8);
        assert!(svg.contains("log |I|"));
        assert!(svg.contains("dI/dV: LI Demod 1 Y (A)"));
        assert!(svg.contains("log dI/dV: LI Demod 1 Y (A)"));
        assert!(svg.contains("Start: bias -2.000 V, setpoint 50.0 pA"));
        // Legend in series order.
        assert!(svg.find("#0 ").unwrap() < svg.find("#1 ").unwrap());
        // Only I(V).
        let svg = combined_svg("test", &curves, &PlotChoice::default());
        assert_eq!(svg.matches("<polyline").count(), 2);
        // dI/dV only.
        let d = PlotChoice { didv: true, ..Default::default() };
        assert_eq!(combined_svg("test", &curves, &d).matches("<polyline").count(), 4);
    }

    #[test]
    fn dat_has_columns_for_every_curve() {
        let curves = vec![curve(0, 0.), curve(1, -10.)];
        let dat = combined_dat("test", &curves);
        let header = dat.lines().find(|l| !l.starts_with('#')).unwrap();
        assert_eq!(header.split('\t').count(), 6);
        assert_eq!(dat.lines().filter(|l| !l.starts_with('#')).count(), 4);
    }

    #[test]
    fn jpeg_renders() {
        let choice = PlotChoice { didv: true, ..Default::default() };
        let svg = combined_svg("test", &[curve(0, 0.), curve(1, -10.)], &choice);
        let jpg = svg_to_jpeg(&svg, 2.).unwrap();
        let img = image::load_from_memory(&jpg).unwrap();
        assert!(img.width() > 1000);
    }

    #[test]
    fn ticks_are_round() {
        assert_eq!(ticks(-1., 1., 4), vec![-1., -0.5, 0., 0.5, 1.]);
    }
}
