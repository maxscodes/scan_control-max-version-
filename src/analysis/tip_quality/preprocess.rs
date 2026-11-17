//! Getting a scan ready for the tip-quality measures: crop to the scanned
//! rows, shrink large frames, level, and make the derived images.

/// Longest side analysed; larger frames are averaged down.
pub const MAX_SIDE: usize = 256;
/// Smallest frame (after cropping) worth judging.
pub const MIN_SIDE: usize = 32;

/// A levelled, possibly shrunk frame.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub w: usize,
    pub h: usize,
    /// Height, levelled (plane and line offsets removed), outliers clipped;
    /// unscanned pixels are 0.
    pub img: Vec<f32>,
    /// Pixels that hold data.
    pub mask: Vec<bool>,
    /// Size of one (analysed) pixel, nm (x along a line, y between lines).
    pub px_nm: [f64; 2],
    /// Averaging factor applied (1 = full resolution).
    pub factor: usize,
    /// Share of the original frame that was scanned.
    pub valid: f64,
}

/// Crop, shrink and level `data` (`size` = [w, h], NaN = unscanned).
/// `frame_nm` is the frame's width and height in nm. `None` if too little
/// was scanned to judge.
pub fn prepare(data: &[f32], size: [u32; 2], frame_nm: [f64; 2]) -> Option<Prepared> {
    let (w0, h0) = (size[0] as usize, size[1] as usize);
    if w0 * h0 == 0 || data.len() != w0 * h0 {
        return None;
    }
    // Rows with at least half their pixels.
    let good_row = |r: usize| data[r * w0..(r + 1) * w0].iter().filter(|v| v.is_finite()).count() * 2 >= w0;
    let rows: Vec<usize> = (0..h0).filter(|&r| good_row(r)).collect();
    let (&r0, &r1) = (rows.first()?, rows.last()?);
    let valid = data.iter().filter(|v| v.is_finite()).count() as f64 / data.len() as f64;
    let hc = r1 - r0 + 1;
    if w0 < MIN_SIDE || hc < MIN_SIDE {
        return None;
    }
    let f = w0.max(hc).div_ceil(MAX_SIDE).max(1);
    let (w, h) = (w0 / f, hc / f);
    if w < MIN_SIDE / 2 || h < MIN_SIDE / 2 {
        return None;
    }
    // NaN-aware box average.
    let mut img = vec![0f32; w * h];
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut s, mut n) = (0f64, 0usize);
            for yy in 0..f {
                let row = &data[(r0 + y * f + yy) * w0..];
                for xx in 0..f {
                    let v = row[x * f + xx];
                    if v.is_finite() {
                        s += v as f64;
                        n += 1;
                    }
                }
            }
            if n * 2 >= f * f {
                img[y * w + x] = (s / n as f64) as f32;
                mask[y * w + x] = true;
            }
        }
    }
    level(&mut img, &mask, w, h);
    clip(&mut img, &mask, 0.001);
    let px_nm = [frame_nm[0] / w0 as f64 * f as f64, frame_nm[1] / h0 as f64 * f as f64];
    Some(Prepared { w, h, img, mask, px_nm, factor: f, valid })
}

/// Remove a plane, then each line's offset (median of the line, robust to
/// steps), in place. Unscanned pixels are set to 0.
pub fn level(img: &mut [f32], mask: &[bool], w: usize, h: usize) {
    // Least-squares plane.
    let (mut sxx, mut sxy, mut syy, mut sx, mut sy, mut n) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let (mut sxz, mut syz, mut sz) = (0f64, 0f64, 0f64);
    for y in 0..h {
        for x in 0..w {
            if !mask[y * w + x] {
                continue;
            }
            let (xf, yf, z) = (x as f64, y as f64, img[y * w + x] as f64);
            sxx += xf * xf;
            sxy += xf * yf;
            syy += yf * yf;
            sx += xf;
            sy += yf;
            n += 1.;
            sxz += xf * z;
            syz += yf * z;
            sz += z;
        }
    }
    if n >= 3. {
        let m = nalgebra::Matrix3::new(sxx, sxy, sx, sxy, syy, sy, sx, sy, n);
        if let Some(inv) = m.try_inverse() {
            let p = inv * nalgebra::Vector3::new(sxz, syz, sz);
            for y in 0..h {
                for x in 0..w {
                    img[y * w + x] -= (p[0] * x as f64 + p[1] * y as f64 + p[2]) as f32;
                }
            }
        }
    }
    // Line offsets.
    let mut buf = Vec::with_capacity(w);
    for y in 0..h {
        buf.clear();
        buf.extend((0..w).filter(|&x| mask[y * w + x]).map(|x| img[y * w + x]));
        if buf.is_empty() {
            continue;
        }
        let med = median(&mut buf);
        for x in 0..w {
            img[y * w + x] = if mask[y * w + x] { img[y * w + x] - med } else { 0. };
        }
    }
}

/// Clip the extreme `q` share at each end (spikes, crashes), keeping
/// real features: a robust spread would be tiny on flat terraces.
pub fn clip(img: &mut [f32], mask: &[bool], q: f32) {
    let mut v: Vec<f32> = img.iter().zip(mask).filter(|(_, m)| **m).map(|(v, _)| *v).collect();
    if v.len() < 100 {
        return;
    }
    let n = v.len();
    let k = ((n as f32 * q) as usize).min(n / 2);
    let lo = *v.select_nth_unstable_by(k, f32::total_cmp).1;
    let hi = *v.select_nth_unstable_by(n - 1 - k, f32::total_cmp).1;
    for (x, m) in img.iter_mut().zip(mask) {
        if *m {
            *x = x.clamp(lo, hi);
        }
    }
}

pub fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.;
    }
    let k = v.len() / 2;
    *v.select_nth_unstable_by(k, f32::total_cmp).1
}

/// Separable Gaussian blur (σ in pixels), edges clamped.
pub fn blur(img: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.3 {
        return img.to_vec();
    }
    let r = (3. * sigma).ceil() as i64;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2. * sigma * sigma)).exp()).collect();
    let ks: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / ks).collect();
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.;
            for (j, kv) in k.iter().enumerate() {
                let xx = (x as i64 + j as i64 - r).clamp(0, w as i64 - 1) as usize;
                s += kv * img[y * w + xx];
            }
            tmp[y * w + x] = s;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.;
            for (j, kv) in k.iter().enumerate() {
                let yy = (y as i64 + j as i64 - r).clamp(0, h as i64 - 1) as usize;
                s += kv * tmp[yy * w + x];
            }
            out[y * w + x] = s;
        }
    }
    out
}

/// High-pass: the image minus a blur at `sigma` pixels (masked pixels 0).
pub fn high_pass(p: &Prepared, sigma: f32) -> Vec<f32> {
    let b = blur(&p.img, p.w, p.h, sigma);
    p.img.iter().zip(&b).zip(&p.mask).map(|((v, bv), m)| if *m { v - bv } else { 0. }).collect()
}

/// Gradient magnitude of the lightly smoothed image: step edges become
/// ridges (a doubled step, two parallel ridges).
pub fn gradient(p: &Prepared) -> Vec<f32> {
    let (w, h) = (p.w, p.h);
    let s = blur(&p.img, w, h, 1.0);
    let mut g = vec![0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            if !(p.mask[i] && p.mask[i - 1] && p.mask[i + 1] && p.mask[i - w] && p.mask[i + w]) {
                continue;
            }
            let gx = 0.5 * (s[i + 1] - s[i - 1]);
            let gy = 0.5 * (s[i + w] - s[i - w]);
            g[i] = (gx * gx + gy * gy).sqrt();
        }
    }
    // Remove the mean so flat areas don't correlate with everything.
    let n = p.mask.iter().filter(|m| **m).count().max(1) as f32;
    let mean: f32 = g.iter().zip(&p.mask).filter(|(_, m)| **m).map(|(v, _)| *v).sum::<f32>() / n;
    for (v, m) in g.iter_mut().zip(&p.mask) {
        *v = if *m { *v - mean } else { 0. };
    }
    g
}
