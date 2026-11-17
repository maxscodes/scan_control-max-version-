//! Tip-health measures other than ghost copies.

use super::{
    autocorr::Acf,
    preprocess::{blur, Prepared},
};

/// Tip changes and streaks: share of scan lines that differ sharply from
/// the lines around them, mapped to 0..1 (10 % of lines or more = 1).
pub fn instability(p: &Prepared) -> f32 {
    let (w, h) = (p.w, p.h);
    if h < 5 {
        return 0.;
    }
    let mut anomaly = Vec::with_capacity(h);
    for y in 1..h - 1 {
        let (mut s, mut n) = (0f64, 0usize);
        for x in 0..w {
            let (a, b, c) = ((y - 1) * w + x, y * w + x, (y + 1) * w + x);
            if p.mask[a] && p.mask[b] && p.mask[c] {
                let d = p.img[b] as f64 - 0.5 * (p.img[a] + p.img[c]) as f64;
                s += d * d;
                n += 1;
            }
        }
        if n * 2 >= w {
            anomaly.push((s / n as f64).sqrt());
        }
    }
    if anomaly.len() < 3 {
        return 0.;
    }
    let mut sorted = anomaly.clone();
    sorted.sort_by(f64::total_cmp);
    // Typical line: the lower quartile (streak lines and their
    // neighbours can be half of all lines).
    let typical = sorted[sorted.len() / 4].max(1e-30);
    let bad = anomaly.iter().filter(|a| **a > 4. * typical).count() as f64 / anomaly.len() as f64;
    (bad / 0.1).clamp(0., 1.) as f32
}

/// How well the forward and backward images agree (normalised
/// correlation at the best small shift), 0..1. `None` without a usable
/// backward image.
pub fn fb_agreement(fwd: &Prepared, bwd: &Prepared) -> Option<f32> {
    if fwd.w != bwd.w || fwd.h != bwd.h {
        return None;
    }
    let to_nan = |p: &Prepared| -> Vec<f32> {
        p.img.iter().zip(&p.mask).map(|(v, m)| if *m { *v } else { f32::NAN }).collect()
    };
    let s = crate::analysis::register::image_shift(&to_nan(fwd), &to_nan(bwd), fwd.w, fwd.h)?;
    // Forward and backward only differ by a few pixels of lag/drift.
    if s.dx.abs() > 0.1 * fwd.w as f64 || s.dy.abs() > 0.1 * fwd.h as f64 {
        return Some(0.);
    }
    Some(s.score.clamp(0., 1.) as f32)
}

/// Bluntness on atomic-scale frames: how wide the narrowest features are.
/// The central peak of the autocorrelation of the high-passed image gives
/// the typical feature size; a sharp tip resolves features of a few
/// ångström. `None` when the pixels are too coarse to tell.
pub fn bluntness(p: &Prepared, hp_acf: &Acf) -> Option<f32> {
    let px = p.px_nm[0].min(p.px_nm[1]);
    if !(px > 0. && px < 0.12) {
        return None;
    }
    let r_nm = hp_acf.lobe_radius(0.5) * px;
    Some(((r_nm - 0.35) / 0.6).clamp(0., 1.) as f32)
}

/// Signal over pixel noise: features finer than the frame but coarser
/// than a pixel, against pixel-to-pixel noise.
pub fn signal_to_noise(p: &Prepared) -> f64 {
    let s1 = blur(&p.img, p.w, p.h, 1.0);
    let mut sig = 0f64;
    let mut noise = 0f64;
    let mut n = 0usize;
    for i in 0..p.img.len() {
        if p.mask[i] {
            sig += (s1[i] as f64).powi(2);
            noise += (p.img[i] - s1[i]) as f64 * (p.img[i] - s1[i]) as f64;
            n += 1;
        }
    }
    if n == 0 || noise <= 0. {
        return f64::INFINITY;
    }
    (sig / noise).sqrt()
}
