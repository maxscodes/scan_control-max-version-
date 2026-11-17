//! Autocorrelation and ghost peaks.
//!
//! With two or more apexes every feature is imaged again, shifted by the
//! vector between them. The image's autocorrelation then has a peak at
//! that vector (and its negative). A crystal lattice also gives peaks, but
//! they repeat: a lattice vector v comes with 2v, while a ghost d does not
//! come with 2d (two apexes give one copy, not a row of them).

use glam::DVec2;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

/// In-place radix-2 FFT (`a.len()` a power of two).
fn fft(a: &mut [C], inverse: bool) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2. * std::f64::consts::PI / len as f64 * if inverse { 1. } else { -1. };
        let wl = C { re: ang.cos(), im: ang.sin() };
        for i in (0..n).step_by(len) {
            let mut w = C { re: 1., im: 0. };
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = a[i + k + len / 2].mul(w);
                a[i + k] = C { re: u.re + v.re, im: u.im + v.im };
                a[i + k + len / 2] = C { re: u.re - v.re, im: u.im - v.im };
                w = w.mul(wl);
            }
        }
        len <<= 1;
    }
    if inverse {
        for v in a.iter_mut() {
            v.re /= n as f64;
            v.im /= n as f64;
        }
    }
}

fn fft2(a: &mut [C], nx: usize, ny: usize, inverse: bool) {
    for row in a.chunks_mut(nx) {
        fft(row, inverse);
    }
    let mut col = vec![C::default(); ny];
    for x in 0..nx {
        for y in 0..ny {
            col[y] = a[y * nx + x];
        }
        fft(&mut col, inverse);
        for y in 0..ny {
            a[y * nx + x] = col[y];
        }
    }
}

/// Normalised autocorrelation of a (windowed) image, for lags
/// |dx| < w/2, |dy| < h/2. `get(dx, dy)` is 1 at zero lag.
pub struct Acf {
    nx: usize,
    ny: usize,
    v: Vec<f64>,
    pub w: usize,
    pub h: usize,
}
impl Acf {
    pub fn new(img: &[f32], w: usize, h: usize) -> Option<Self> {
        // Pad to avoid wrap-around for the lags we look at.
        let nx = (w + w / 2).next_power_of_two();
        let ny = (h + h / 2).next_power_of_two();
        let mut a = vec![C::default(); nx * ny];
        let mean = img.iter().map(|v| *v as f64).sum::<f64>() / img.len().max(1) as f64;
        // Tukey window (flat middle, edges tapered over 15 %): the frame
        // edges don't make false correlations, and most features count.
        let hann = |i: usize, n: usize| {
            let t = (i as f64 + 0.5) / n as f64;
            let a = 0.15;
            let e = t.min(1. - t);
            if e >= a { 1. } else { 0.5 - 0.5 * (std::f64::consts::PI * e / a).cos() }
        };
        let wx: Vec<f64> = (0..w).map(|x| hann(x, w)).collect();
        let wy: Vec<f64> = (0..h).map(|y| hann(y, h)).collect();
        for y in 0..h {
            for x in 0..w {
                a[y * nx + x].re = (img[y * w + x] as f64 - mean) * wx[x] * wy[y];
            }
        }
        fft2(&mut a, nx, ny, false);
        let mut pw: Vec<f64> = a.iter().map(|c| c.re * c.re + c.im * c.im).collect();
        remove_lattice(&mut pw, nx, ny);
        for (c, p) in a.iter_mut().zip(&pw) {
            *c = C { re: *p, im: 0. };
        }
        fft2(&mut a, nx, ny, true);
        let zero = a[0].re;
        if !(zero > 0.) {
            return None;
        }
        // Correct for the shrinking overlap of the window at larger lags.
        let wacf = |win: &[f64], d: usize| -> f64 {
            win.iter().zip(&win[d.min(win.len())..]).map(|(a, b)| a * b).sum::<f64>()
        };
        let w0x = wacf(&wx, 0);
        let w0y = wacf(&wy, 0);
        let ox: Vec<f64> = (0..nx).map(|d| { let dd = if d <= nx / 2 { d } else { nx - d }; wacf(&wx, dd) / w0x }).collect();
        let oy: Vec<f64> = (0..ny).map(|d| { let dd = if d <= ny / 2 { d } else { ny - d }; wacf(&wy, dd) / w0y }).collect();
        let v = a
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let (x, y) = (i % nx, i / nx);
                let o = (ox[x] * oy[y]).max(0.5);
                c.re / zero / o
            })
            .collect();
        Some(Self { nx, ny, v, w, h })
    }
    /// Value at lag (dx, dy) (0 outside the usable range).
    pub fn get(&self, dx: i64, dy: i64) -> f64 {
        if dx.unsigned_abs() as usize >= self.w / 2 || dy.unsigned_abs() as usize >= self.h / 2 {
            return 0.;
        }
        let x = dx.rem_euclid(self.nx as i64) as usize;
        let y = dy.rem_euclid(self.ny as i64) as usize;
        self.v[y * self.nx + x]
    }
    /// Bilinear value at a fractional lag.
    pub fn at(&self, d: DVec2) -> f64 {
        let (x0, y0) = (d.x.floor(), d.y.floor());
        let (fx, fy) = (d.x - x0, d.y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        self.get(x0, y0) * (1. - fx) * (1. - fy)
            + self.get(x0 + 1, y0) * fx * (1. - fy)
            + self.get(x0, y0 + 1) * (1. - fx) * fy
            + self.get(x0 + 1, y0 + 1) * fx * fy
    }
    /// Radius (px) where the central peak has fallen below `level` (median
    /// over directions, so a ridge from a straight edge doesn't widen it).
    pub fn lobe_radius(&self, level: f64) -> f64 {
        let max_r = (self.w.min(self.h) / 4).max(2);
        for r in 1..max_r {
            let n = 16;
            let mut v: Vec<f64> = (0..n)
                .map(|k| {
                    let a = std::f64::consts::PI * k as f64 / n as f64;
                    self.at(DVec2::new(a.cos(), a.sin()) * r as f64)
                })
                .collect();
            v.sort_by(f64::total_cmp);
            let m = v[n / 2];
            if m < level {
                return r as f64;
            }
        }
        max_r as f64
    }
    /// Median of the ACF on a ring of radius `r`.
    pub fn ring_median(&self, r: f64) -> f64 {
        let n = ((2. * std::f64::consts::PI * r) as usize).clamp(16, 256);
        let mut v: Vec<f64> = (0..n)
            .map(|k| {
                let a = 2. * std::f64::consts::PI * k as f64 / n as f64;
                self.at(DVec2::new(a.cos(), a.sin()) * r)
            })
            .collect();
        v.sort_by(f64::total_cmp);
        v[n / 2]
    }
}

/// Clip sharp spikes in the power spectrum (a crystal lattice, periodic
/// noise) to the level around them. A ghost copy doesn't make spikes: it
/// multiplies the whole spectrum by smooth fringes, which stay.
fn remove_lattice(p: &mut [f64], nx: usize, ny: usize) {
    let src = p.to_vec();
    let ring: Vec<(i64, i64)> = (0..16)
        .map(|k| {
            let a = 2. * std::f64::consts::PI * k as f64 / 16.;
            ((3.5 * a.cos()).round() as i64, (3.5 * a.sin()).round() as i64)
        })
        .collect();
    let mut around = [0f64; 16];
    for y in 0..ny {
        for x in 0..nx {
            // Frequencies near zero are the large-scale shape, not a lattice.
            let fx = if x <= nx / 2 { x } else { nx - x };
            let fy = if y <= ny / 2 { y } else { ny - y };
            if fx * fx + fy * fy <= 9 {
                continue;
            }
            for (k, (dx, dy)) in ring.iter().enumerate() {
                let xx = (x as i64 + dx).rem_euclid(nx as i64) as usize;
                let yy = (y as i64 + dy).rem_euclid(ny as i64) as usize;
                around[k] = src[yy * nx + xx];
            }
            around.sort_by(f64::total_cmp);
            let level = around[8];
            if src[y * nx + x] > 6. * level {
                p[y * nx + x] = level;
            }
        }
    }
}

/// A candidate ghost: an off-centre ACF peak that a lattice doesn't
/// explain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    /// Lag in (analysed) pixels, with dy ≥ 0 (the peak at −d is the same).
    pub d: DVec2,
    /// ACF value at d.
    pub value: f64,
    /// Above the ring around it, and not repeated at 2d: 0..1.
    pub strength: f64,
    /// Height above the ACF's noise, in standard deviations.
    pub z: f64,
}

thread_local! {
    /// Print every candidate (tests).
    pub static DEBUG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Off-centre peaks, strongest first (at most `max`).
pub fn ghost_peaks(acf: &Acf, max: usize) -> Vec<Peak> {
    let r0 = acf.lobe_radius(0.3).max(1.5);
    let r_min = (1.6 * r0).max(2.5);
    // Ghosts are close copies: searching out to 15 % of the frame keeps
    // 2d inside the measured lags for the lattice check.
    let r_max = (0.15 * acf.w.min(acf.h) as f64).max(r_min + 4.);
    let (rx, ry) = ((acf.w / 2) as i64 - 1, (acf.h / 2) as i64 - 1);
    // Noise of the ACF away from the centre: a ghost must stand well
    // clear of it (random features correlate by chance).
    let noise = {
        let mut v = Vec::new();
        for dy in (0..=ry).step_by(2) {
            for dx in (-rx..=rx).step_by(2) {
                let r = ((dx * dx + dy * dy) as f64).sqrt();
                if r >= r_min && r <= 2. * r_max && dy.abs() > 1 {
                    v.push(acf.get(dx, dy));
                }
            }
        }
        if v.len() < 16 {
            return Vec::new();
        }
        v.sort_by(f64::total_cmp);
        let med = v[v.len() / 2];
        let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
        dev.sort_by(f64::total_cmp);
        (1.4826 * dev[dev.len() / 2]).max(1e-4)
    };
    let mut out: Vec<Peak> = Vec::new();
    // Line-to-line noise leaves a ridge along the fast scan axis (dy = 0):
    // judge candidates there against the ridge, not the ring.
    let ridge = |dx: f64| -> f64 {
        let mut v: Vec<f64> = ((dx.abs() * 0.5) as i64..=(dx.abs() * 2.).min(rx as f64) as i64)
            .map(|x| acf.get(x, 0))
            .collect();
        if v.is_empty() {
            return 0.;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    for dy in 0..=ry {
        for dx in -rx..=rx {
            if dy == 0 && dx <= 0 {
                continue;
            }
            let r = ((dx * dx + dy * dy) as f64).sqrt();
            if r < r_min || r > r_max {
                continue;
            }
            let v = acf.get(dx, dy);
            if v <= 0.05 {
                continue;
            }
            // Local maximum in a 5×5 neighbourhood.
            let mut is_max = true;
            'n: for ny in -2..=2i64 {
                for nx in -2..=2i64 {
                    if (nx, ny) != (0, 0) && acf.get(dx + nx, dy + ny) > v {
                        is_max = false;
                        break 'n;
                    }
                }
            }
            if !is_max {
                continue;
            }
            // Sub-pixel position from parabolas through the peak.
            let par = |l: f64, c: f64, r: f64| {
                let den = l - 2. * c + r;
                if den.abs() < 1e-12 { 0. } else { (0.5 * (l - r) / den).clamp(-0.5, 0.5) }
            };
            let sx = par(acf.get(dx - 1, dy), v, acf.get(dx + 1, dy));
            let sy = par(acf.get(dx, dy - 1), v, acf.get(dx, dy + 1));
            let d = DVec2::new(dx as f64 + sx, dy as f64 + sy);
            let base = if dy.abs() <= 1 { ridge(dx as f64).max(acf.ring_median(r)) } else { acf.ring_median(r) };
            let lift = ((v - base) / (1. - base).max(1e-6)).clamp(0., 1.);
            // A lattice repeats at 2d (and the lobe repeats too).
            let twice = acf.at(d * 2.).max(0.);
            let repeat = (twice / v).clamp(0., 1.);
            // A line (a single step edge) gives a ridge through the centre:
            // a point on it also has the ridge at half the lag. A ghost
            // doesn't.
            // Only checked where the lag is clear of the central lobe and of
            // the peak's own width, else their tails look like a ridge.
            let mut on_line: Vec<f64> = [1. / 3., 0.5, 2. / 3.]
                .iter()
                .filter(|&&f| f * r > r_min + r0 && (1. - f) * r > 2. * r0)
                .map(|&f| (acf.at(d * f) - acf.ring_median(f * r)).max(0.) / v)
                .collect();
            on_line.sort_by(f64::total_cmp);
            let line = on_line.get(on_line.len() / 2).copied().unwrap_or(0.);
            let repeat = repeat.max(line.clamp(0., 1.));
            // How far above the noise: below 3σ it counts for nothing,
            // from 7σ fully.
            let z = (v - base) / noise;
            let sig = ((z - 3.) / 4.).clamp(0., 1.);
            let strength = lift * (1. - repeat) * sig;
            if DEBUG.with(|d| d.get()) && z > 2.5 {
                eprintln!(
                    "  cand ({:.1},{:.1}) v {v:.3} base {base:.3} lift {lift:.2} twice {twice:.2} line {line:.2} z {z:.1} -> {strength:.3}",
                    d.x, d.y
                );
            }
            if strength > 0.01 {
                out.push(Peak { d, value: v, strength, z });
            }
        }
    }
    out.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    // Keep distinct peaks.
    let mut kept: Vec<Peak> = Vec::new();
    for p in out {
        if kept.iter().all(|k| (k.d - p.d).length() > 2.0 && (k.d + p.d).length() > 2.0) {
            kept.push(p);
            if kept.len() >= max {
                break;
            }
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip() {
        let mut a: Vec<C> = (0..16).map(|i| C { re: (i as f64).sin(), im: 0. }).collect();
        let orig = a.clone();
        fft(&mut a, false);
        fft(&mut a, true);
        for (x, y) in a.iter().zip(&orig) {
            assert!((x.re - y.re).abs() < 1e-9 && x.im.abs() < 1e-9);
        }
    }
}
