//! One picture of several stamps (e.g. the tiles of a survey) laid out as
//! they are on the map, for saving.

use glam::{DAffine2, DVec2};

use crate::connection::FrameData;

/// A stamp to draw: its image and where it lies on the map.
pub struct Piece {
    pub transform: DAffine2,
    pub data: FrameData,
}

/// Each line minus its own straight-line fit (STM line levelling), so the
/// tiles share a common contrast.
fn level_lines(d: &FrameData) -> Vec<f32> {
    let [w, h] = d.size;
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![f32::NAN; w * h];
    for j in 0..h {
        let row = &d.data[j * w..(j + 1) * w];
        let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (i, &v) in row.iter().enumerate() {
            if v.is_finite() {
                let (x, y) = (i as f64, v as f64);
                n += 1.;
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
        }
        if n < 2. {
            continue;
        }
        let den = n * sxx - sx * sx;
        let slope = if den.abs() > 0. { (n * sxy - sx * sy) / den } else { 0. };
        let mean = sy / n;
        let xm = sx / n;
        for (i, &v) in row.iter().enumerate() {
            if v.is_finite() {
                out[j * w + i] = (v as f64 - mean - slope * (i as f64 - xm)) as f32;
            }
        }
    }
    out
}

/// Draw `pieces` (topmost first) into one image at the finest pixel size
/// among them, at most `max_side` pixels on a side. Empty parts are
/// transparent.
pub fn render(pieces: &[Piece], max_side: u32) -> Option<image::RgbaImage> {
    let pieces: Vec<&Piece> = pieces
        .iter()
        .filter(|p| p.data.size[0] > 0 && p.data.size[1] > 0 && p.transform.matrix2.determinant().abs() > 0.)
        .collect();
    if pieces.is_empty() {
        return None;
    }
    let mut lo = DVec2::INFINITY;
    let mut hi = DVec2::NEG_INFINITY;
    for p in &pieces {
        for c in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
            let w = p.transform.transform_point2(DVec2::new(c.0, c.1));
            lo = lo.min(w);
            hi = hi.max(w);
        }
    }
    let mut px = pieces
        .iter()
        .map(|p| {
            let m = p.transform.matrix2;
            (m.x_axis.length() / p.data.size[0] as f64).min(m.y_axis.length() / p.data.size[1] as f64)
        })
        .fold(f64::INFINITY, f64::min);
    let span = hi - lo;
    let longest = (span.x.max(span.y) / px).ceil();
    if longest > max_side as f64 {
        px *= longest / max_side as f64;
    }
    let (w, h) = ((span.x / px).ceil().max(1.) as u32, (span.y / px).ceil().max(1.) as u32);

    let levelled: Vec<Vec<f32>> = pieces.iter().map(|p| level_lines(&p.data)).collect();
    // Shared contrast: 1–99 % of all values.
    let mut all: Vec<f32> = levelled.iter().flatten().copied().filter(|v| v.is_finite()).collect();
    if all.is_empty() {
        return None;
    }
    all.sort_by(f32::total_cmp);
    let pick = |q: f64| all[((all.len() - 1) as f64 * q) as usize];
    let (vlo, vhi) = (pick(0.01), pick(0.99));
    let span_v = (vhi - vlo).max(f32::MIN_POSITIVE);
    let inv: Vec<DAffine2> = pieces.iter().map(|p| p.transform.inverse()).collect();

    Some(image::RgbaImage::from_fn(w, h, |x, y| {
        // Image row 0 is the top of the map (largest y).
        let p = DVec2::new(lo.x + (x as f64 + 0.5) * px, hi.y - (y as f64 + 0.5) * px);
        for (k, piece) in pieces.iter().enumerate() {
            let l = inv[k].transform_point2(p);
            if l.x.abs() > 0.5 || l.y.abs() > 0.5 {
                continue;
            }
            let [pw, ph] = piece.data.size;
            let i = (((l.x + 0.5) * pw as f64) as u32).min(pw - 1) as usize;
            let j = (((l.y + 0.5) * ph as f64) as u32).min(ph - 1) as usize;
            let v = levelled[k][j * pw as usize + i];
            if v.is_finite() {
                let g = (((v - vlo) / span_v).clamp(0., 1.) * 255.) as u8;
                return image::Rgba([g, g, g, 255]);
            }
        }
        image::Rgba([0, 0, 0, 0])
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn piece(at: DVec2, value: f32) -> Piece {
        let n = 10;
        // A ramp along each line (levelled away) plus a line-to-line pattern.
        let data: Vec<f32> = (0..n * n).map(|k| value + (k % n) as f32 * 0.1 + ((k / n) % 2) as f32).collect();
        Piece {
            transform: DAffine2::from_scale_angle_translation(DVec2::new(100., -100.), 0., at),
            data: FrameData { size: [n as u32, n as u32], data: Arc::new(data.into_boxed_slice()) },
        }
    }

    #[test]
    fn lays_tiles_out_as_on_the_map() {
        // Two 100 nm tiles side by side with a 20 nm gap; 10 nm pixels.
        let img = render(&[piece(DVec2::ZERO, 0.), piece(DVec2::new(120., 0.), 5.)], 4000).unwrap();
        assert_eq!((img.width(), img.height()), (22, 10));
        // Inside the tiles opaque, in the gap transparent.
        assert_eq!(img.get_pixel(2, 5)[3], 255);
        assert_eq!(img.get_pixel(11, 5)[3], 0);
        assert_eq!(img.get_pixel(20, 5)[3], 255);
        // Capped size keeps the shape.
        let small = render(&[piece(DVec2::ZERO, 0.), piece(DVec2::new(120., 0.), 5.)], 11).unwrap();
        assert_eq!((small.width(), small.height()), (11, 5));
    }
}
