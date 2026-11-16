//! How far apart two overlapping scans are (image registration), e.g. the
//! same area imaged before and after a coarse move.
//!
//! Normalised cross-correlation: an exhaustive search on downsampled images
//! finds the rough shift, a local search at full resolution refines it, and a
//! parabola through the peak gives sub-pixel precision. Scan rows are levelled
//! first (STM line offsets) and unscanned (NaN) pixels are ignored.

/// Shift of the content of one image relative to another, in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shift {
    /// `b(x, y) ≈ a(x − dx, y − dy)`: +dx = content moved right (higher
    /// column), +dy = content moved down (higher row).
    pub dx: f64,
    pub dy: f64,
    /// Correlation at the best shift (1 = identical). Below
    /// [`GOOD_SCORE`] the match is doubtful.
    pub score: f64,
    /// Share of the image the two overlap in at that shift.
    pub overlap: f64,
}

/// Matches scoring below this are not trusted.
pub const GOOD_SCORE: f64 = 0.5;
/// Shifts leaving less than this share of the image in common are not tried.
const MIN_OVERLAP: f64 = 0.25;
/// Longest side of the downsampled images used for the rough search.
const COARSE_SIZE: usize = 64;

/// Each row minus its own mean (removes line-to-line offsets).
fn level_rows(d: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = d.to_vec();
    for r in 0..h {
        let row = &mut out[r * w..(r + 1) * w];
        let (mut s, mut n) = (0f64, 0usize);
        for v in row.iter().filter(|v| v.is_finite()) {
            s += *v as f64;
            n += 1;
        }
        if n == 0 {
            continue;
        }
        let m = (s / n as f64) as f32;
        for v in row.iter_mut() {
            *v -= m;
        }
    }
    out
}

/// Box-average by `f` (NaN pixels ignored; all-NaN blocks stay NaN).
fn downsample(d: &[f32], w: usize, h: usize, f: usize) -> (Vec<f32>, usize, usize) {
    let (ws, hs) = (w / f, h / f);
    let mut out = vec![f32::NAN; ws * hs];
    for y in 0..hs {
        for x in 0..ws {
            let (mut s, mut n) = (0f64, 0usize);
            for yy in y * f..(y + 1) * f {
                for xx in x * f..(x + 1) * f {
                    let v = d[yy * w + xx];
                    if v.is_finite() {
                        s += v as f64;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                out[y * ws + x] = (s / n as f64) as f32;
            }
        }
    }
    (out, ws, hs)
}

/// Correlation of `b` with `a` moved by (dx, dy), over the pixels both
/// have, and the share of the image that is.
fn ncc(a: &[f32], b: &[f32], w: usize, h: usize, dx: i64, dy: i64) -> (f64, f64) {
    let (w, h) = (w as i64, h as i64);
    let (x0, x1) = (dx.max(0), (w + dx).min(w));
    let (y0, y1) = (dy.max(0), (h + dy).min(h));
    if x1 <= x0 || y1 <= y0 {
        return (0., 0.);
    }
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for y in y0..y1 {
        let rb = (y * w) as usize;
        let ra = ((y - dy) * w) as usize;
        for x in x0..x1 {
            let vb = b[rb + x as usize];
            let va = a[ra + (x - dx) as usize];
            if va.is_finite() && vb.is_finite() {
                let (va, vb) = (va as f64, vb as f64);
                n += 1.;
                sa += va;
                sb += vb;
                saa += va * va;
                sbb += vb * vb;
                sab += va * vb;
            }
        }
    }
    let overlap = n / (w * h) as f64;
    if n < 16. {
        return (0., overlap);
    }
    let caa = saa - sa * sa / n;
    let cbb = sbb - sb * sb / n;
    if caa <= 0. || cbb <= 0. {
        return (0., overlap);
    }
    ((sab - sa * sb / n) / (caa * cbb).sqrt(), overlap)
}

/// Peak offset (−0.5..0.5) of a parabola through three samples.
fn parabola(l: f64, c: f64, r: f64) -> f64 {
    let d = l - 2. * c + r;
    if d.abs() < 1e-12 {
        0.
    } else {
        (0.5 * (l - r) / d).clamp(-0.5, 0.5)
    }
}

/// Shift of `b` relative to `a` (both `w`×`h`, row-major). `None` if the
/// images are too small or have no usable overlap.
pub fn image_shift(a: &[f32], b: &[f32], w: usize, h: usize) -> Option<Shift> {
    if w < 8 || h < 8 || a.len() != w * h || b.len() != w * h {
        return None;
    }
    let a = level_rows(a, w, h);
    let b = level_rows(b, w, h);

    // Rough: every shift on small images. Small overlaps correlate by
    // chance more easily, so they need a clearly better match to win.
    let f = w.max(h).div_ceil(COARSE_SIZE).max(1);
    let (sa, ws, hs) = downsample(&a, w, h, f);
    let (sb, _, _) = downsample(&b, w, h, f);
    let mut best: Option<(f64, i64, i64)> = None;
    let (mx, my) = (ws as i64, hs as i64);
    for dy in -my + 1..my {
        for dx in -mx + 1..mx {
            let frac = (1. - dx.abs() as f64 / mx as f64) * (1. - dy.abs() as f64 / my as f64);
            if frac < MIN_OVERLAP {
                continue;
            }
            let (c, overlap) = ncc(&sa, &sb, ws, hs, dx, dy);
            let rank = c * overlap.sqrt();
            if best.is_none_or(|b| rank > b.0) {
                best = Some((rank, dx, dy));
            }
        }
    }
    let (_, cx, cy) = best?;

    // Fine: full resolution around the rough shift.
    let r = f as i64 + 1;
    let (cx, cy) = (cx * f as i64, cy * f as i64);
    let mut fine: Option<(f64, f64, i64, i64)> = None;
    for dy in cy - r..=cy + r {
        for dx in cx - r..=cx + r {
            let (c, overlap) = ncc(&a, &b, w, h, dx, dy);
            if overlap >= MIN_OVERLAP * 0.8 && fine.is_none_or(|b| c > b.0) {
                fine = Some((c, overlap, dx, dy));
            }
        }
    }
    let (score, overlap, dx, dy) = fine?;
    let at = |x: i64, y: i64| ncc(&a, &b, w, h, x, y).0;
    let sub_x = parabola(at(dx - 1, dy), score, at(dx + 1, dy));
    let sub_y = parabola(at(dx, dy - 1), score, at(dx, dy + 1));
    Some(Shift { dx: dx as f64 + sub_x, dy: dy as f64 + sub_y, score, overlap })
}

/// Left-right slope of the image, lightly smoothed. Each tile is levelled
/// line by line on its own, so the same line differs by a constant between
/// two tiles; the slope along the line doesn't, and neither does a tilt
/// along the scan direction.
fn slope_x(d: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut g = vec![f32::NAN; w * h];
    for y in 0..h {
        for x in 1..w - 1 {
            g[y * w + x] = d[y * w + x + 1] - d[y * w + x - 1];
        }
    }
    // 3×3 average of the finite neighbours (noise in a single difference).
    let mut out = vec![f32::NAN; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut s, mut n) = (0f32, 0);
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    let v = g[yy * w + xx];
                    if v.is_finite() {
                        s += v;
                        n += 1;
                    }
                }
            }
            if n > 0 && g[y * w + x].is_finite() {
                out[y * w + x] = s / n as f32;
            }
        }
    }
    out
}

/// Like [`image_shift`], but only around an expected shift `guess`
/// (pixels, same convention) within `radius` pixels, and accepting small
/// overlaps down to `min_overlap` (share of the image) — for neighbouring
/// tiles that share only a strip.
pub fn shift_near(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    guess: (f64, f64),
    radius: f64,
    min_overlap: f64,
) -> Option<Shift> {
    if w < 8 || h < 8 || a.len() != w * h || b.len() != w * h {
        return None;
    }
    let a = slope_x(a, w, h);
    let b = slope_x(b, w, h);
    // Rough: downsampled so the search stays quick on large scans.
    let f = w.max(h).div_ceil(128).max(1);
    let (sa, ws, hs) = downsample(&a, w, h, f);
    let (sb, _, _) = downsample(&b, w, h, f);
    let (gx, gy) = ((guess.0 / f as f64).round() as i64, (guess.1 / f as f64).round() as i64);
    let r = (radius / f as f64).ceil() as i64;
    let mut best: Option<(f64, i64, i64)> = None;
    for dy in gy - r..=gy + r {
        for dx in gx - r..=gx + r {
            let (c, overlap) = ncc(&sa, &sb, ws, hs, dx, dy);
            if overlap < min_overlap {
                continue;
            }
            let rank = c * overlap.sqrt();
            if best.is_none_or(|b| rank > b.0) {
                best = Some((rank, dx, dy));
            }
        }
    }
    let (_, cx, cy) = best?;
    let rf = f as i64 + 1;
    let (cx, cy) = (cx * f as i64, cy * f as i64);
    let mut fine: Option<(f64, f64, i64, i64)> = None;
    for dy in cy - rf..=cy + rf {
        for dx in cx - rf..=cx + rf {
            let (c, overlap) = ncc(&a, &b, w, h, dx, dy);
            if overlap >= min_overlap * 0.8 && fine.is_none_or(|b| c > b.0) {
                fine = Some((c, overlap, dx, dy));
            }
        }
    }
    let (score, overlap, dx, dy) = fine?;
    let at = |x: i64, y: i64| ncc(&a, &b, w, h, x, y).0;
    let sub_x = parabola(at(dx - 1, dy), score, at(dx + 1, dy));
    let sub_y = parabola(at(dx, dy - 1), score, at(dx, dy + 1));
    Some(Shift { dx: dx as f64 + sub_x, dy: dy as f64 + sub_y, score, overlap })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth random-looking surface, sampled at (x, y) in pixels.
    fn surface(x: f64, y: f64) -> f32 {
        let mut v = 0.;
        for k in 1..7 {
            let k = k as f64;
            v += (x * 0.11 * k + (y * 0.07 * k).sin() * 3.).sin() / k
                + (y * 0.13 * k + (x * 0.05 * k).cos() * 2.).cos() / k;
        }
        v as f32
    }

    /// `a` and a copy whose content moved by (sx, sy) pixels.
    fn pair(w: usize, h: usize, sx: f64, sy: f64) -> (Vec<f32>, Vec<f32>) {
        let mut a = vec![0.; w * h];
        let mut b = vec![0.; w * h];
        for y in 0..h {
            for x in 0..w {
                a[y * w + x] = surface(x as f64, y as f64);
                b[y * w + x] = surface(x as f64 - sx, y as f64 - sy);
            }
        }
        (a, b)
    }

    #[test]
    fn finds_whole_pixel_shifts() {
        for (sx, sy) in [(0., 0.), (40., 0.), (-35., 12.), (0., -50.), (60., 45.)] {
            let (a, b) = pair(160, 140, sx, sy);
            let s = image_shift(&a, &b, 160, 140).unwrap();
            assert!((s.dx - sx).abs() < 0.3 && (s.dy - sy).abs() < 0.3, "{sx},{sy}: {s:?}");
            assert!(s.score > 0.95, "{s:?}");
        }
    }

    #[test]
    fn finds_sub_pixel_shifts() {
        let (a, b) = pair(128, 128, 20.4, -7.7);
        let s = image_shift(&a, &b, 128, 128).unwrap();
        assert!((s.dx - 20.4).abs() < 0.25 && (s.dy + 7.7).abs() < 0.25, "{s:?}");
    }

    #[test]
    fn ignores_line_offsets_and_unscanned_rows() {
        let (w, h) = (128, 128);
        let (a, mut b) = pair(w, h, 25., 18.);
        for y in 0..h {
            for x in 0..w {
                // A different offset on every scan line, and the last rows not scanned yet.
                b[y * w + x] += (y % 7) as f32 * 3.;
                if y >= 110 {
                    b[y * w + x] = f32::NAN;
                }
            }
        }
        let s = image_shift(&a, &b, w, h).unwrap();
        assert!((s.dx - 25.).abs() < 0.3 && (s.dy - 18.).abs() < 0.3, "{s:?}");
    }

    #[test]
    fn finds_shifts_with_a_narrow_overlap() {
        // Tiles sharing a ~12 % strip; the guess is 9 px off.
        let (w, h) = (128, 128);
        let (a, b) = pair(w, h, -113., 4.);
        let s = shift_near(&a, &b, w, h, (-104., 0.), 25., 0.05).unwrap();
        assert!((s.dx + 113.).abs() < 0.3 && (s.dy - 4.).abs() < 0.3, "{s:?}");
        assert!(s.score > 0.9, "{s:?}");
    }

    #[test]
    fn unrelated_images_score_low() {
        let (w, h) = (96, 96);
        let a: Vec<f32> = (0..w * h).map(|i| surface((i % w) as f64, (i / w) as f64)).collect();
        // A different pattern, not a shifted copy.
        let b: Vec<f32> = (0..w * h).map(|i| (((i * 2654435761) % 1000) as f32) / 1000.).collect();
        let s = image_shift(&a, &b, w, h).unwrap();
        assert!(s.score < GOOD_SCORE, "{s:?}");
    }

    /// Real STM data: consecutive scans of the same frame (same size, offset
    /// and range) should match with a small shift (drift only).
    /// `SXM_DIR=dir cargo test --release --bin scan_control register_real_scans -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn register_real_scans() {
        let dir = std::env::var("SXM_DIR").unwrap_or(r"F:\Data\LT SPM\2026\November".into());
        let files = crate::analysis::sxm_match::list_sxm(std::path::Path::new(&dir));
        let key = |p: &std::path::Path| {
            let f = crate::analysis::sxm_match::load_frame(p, Some("Z")).ok()?;
            let sxm = sxmfile::SXM::parse_file(p).ok()?;
            Some((f, sxm.get_scan_center().ok()?, sxm.get_scan_range().ok()?))
        };
        let (mut n, mut good) = (0, 0);
        for pair in files.windows(2) {
            let (Some((a, ca, ra)), Some((b, cb, rb))) = (key(&pair[0]), key(&pair[1])) else { continue };
            if a.size != b.size || ca != cb || ra != rb {
                continue;
            }
            let (w, h) = (a.size[0] as usize, a.size[1] as usize);
            let t = std::time::Instant::now();
            let Some(s) = image_shift(&a.forward, &b.forward, w, h) else { continue };
            n += 1;
            good += (s.score >= GOOD_SCORE) as usize;
            println!(
                "{} -> {}: shift ({:+.1}, {:+.1}) px, score {:.2}, {:.0} ms",
                a.name, b.name, s.dx, s.dy, s.score, t.elapsed().as_secs_f64() * 1e3
            );
            if n >= 25 {
                break;
            }
        }
        println!("{good} of {n} pairs matched well");
        assert!(n > 0);
    }
}
