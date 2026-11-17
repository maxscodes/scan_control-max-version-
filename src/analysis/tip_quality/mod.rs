//! Tip quality from a scan image.
//!
//! STM tips get damaged and image with more than one apex: every feature
//! then shows up again, shifted by the vector between the apexes (doubled
//! dangling bonds or defects at the atomic scale, doubled step edges on
//! larger frames). This module scores that, and other signs of a poor tip:
//!
//! - **multi_tip**: ghost copies, found as autocorrelation peaks that a
//!   lattice doesn't explain, in three versions of the image: fine detail
//!   (atoms, defects), larger features, and the gradient (step edges);
//! - **instability**: tip changes and streaks;
//! - **fb_agreement**: forward and backward images disagreeing;
//! - **blunt**: features wider than a sharp tip gives (atomic scale only).
//!
//! Detection only: nothing here touches the instrument. Thresholds live in
//! [`TipParams`]; every score is kept so a learned model can use them later.

pub mod autocorr;
pub mod features;
pub mod preprocess;

use glam::{DAffine2, DVec2};
use serde::{Deserialize, Serialize};

use autocorr::{ghost_peaks, Acf, Peak};
use preprocess::{gradient, high_pass, prepare, Prepared};

/// The judgement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    Good,
    Suspect,
    Poor,
    /// Can't judge (reason).
    Unsure(String),
}
impl Verdict {
    pub fn label(&self) -> &str {
        match self {
            Verdict::Good => "Good",
            Verdict::Suspect => "Suspect",
            Verdict::Poor => "Poor",
            Verdict::Unsure(_) => "Unsure",
        }
    }
}

/// A ghost copy: everything is imaged again, shifted by this offset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ghost {
    /// Offset in the original image's pixels (+x along a line, +y to later
    /// lines). The copy can be either way (±).
    pub offset_px: [f64; 2],
    /// Same in nm, image axes.
    pub offset_nm: [f64; 2],
    /// 0..1.
    pub strength: f64,
    /// Which images showed it (bit 0 fine detail, 1 larger features, 2
    /// edges).
    pub seen_in: u8,
}
impl Ghost {
    pub fn length_nm(&self) -> f64 {
        DVec2::from_array(self.offset_nm).length()
    }
}

/// Result of [`analyse`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TipReport {
    pub verdict: Verdict,
    /// 0..1: how clearly the image shows ghost copies.
    pub multi_tip: f32,
    /// 0..1: tip changes and streaks.
    pub instability: f32,
    /// 0..1: forward/backward agreement (`None`: no backward image).
    pub fb_agreement: Option<f32>,
    /// 0..1: bluntness (`None`: frame too coarse to tell).
    pub blunt: Option<f32>,
    /// Ghost offsets, strongest first.
    pub ghosts: Vec<Ghost>,
    /// Apexes the ghosts suggest (1 = single tip).
    pub apexes: u8,
    /// Pixel size analysed (nm) and share of the frame scanned.
    pub px_nm: f64,
    pub scanned: f64,
    pub snr: f64,
    /// Short reasons for the verdict.
    pub notes: Vec<String>,
}
impl TipReport {
    /// Fixed-length feature vector for learning (see [`FEATURE_NAMES`]).
    pub fn features(&self) -> [f32; 9] {
        let g = |k: usize| self.ghosts.get(k).map_or(0., |g| g.strength as f32);
        [
            self.multi_tip,
            g(0),
            g(1),
            self.ghosts.first().map_or(0., |g| g.length_nm() as f32),
            self.instability,
            self.fb_agreement.unwrap_or(-1.),
            self.blunt.unwrap_or(-1.),
            self.px_nm as f32,
            self.snr.min(100.) as f32,
        ]
    }
    pub fn summary(&self) -> String {
        let mut s = format!("{} \u{b7} multi-tip {:.2}", self.verdict.label(), self.multi_tip);
        if let Some(g) = self.ghosts.first() {
            s += &format!(" (copy at {:.2} nm)", g.length_nm());
        }
        s
    }
}
pub const FEATURE_NAMES: [&str; 9] =
    ["multi_tip", "ghost1", "ghost2", "ghost1_nm", "instability", "fb_agreement", "blunt", "px_nm", "snr"];

/// Thresholds for the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TipParams {
    pub suspect_multi: f32,
    pub poor_multi: f32,
    pub suspect_instability: f32,
    pub poor_instability: f32,
    /// Forward/backward agreement below this is suspect.
    pub min_fb: f32,
    pub suspect_blunt: f32,
    /// Below this signal-to-noise the image is too featureless to judge.
    pub min_snr: f32,
}
impl Default for TipParams {
    fn default() -> Self {
        Self {
            suspect_multi: 0.15,
            poor_multi: 0.25,
            suspect_instability: 0.35,
            poor_instability: 0.75,
            min_fb: 0.35,
            suspect_blunt: 0.7,
            min_snr: 0.8,
        }
    }
}
impl TipParams {
    pub fn path() -> std::path::PathBuf {
        crate::app::CONFIG_DIR.config_local_dir().join("tip_params.json")
    }
    pub fn load() -> Self {
        std::fs::read_to_string(Self::path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }
    pub fn save(&self) {
        if let Ok(j) = serde_json::to_string_pretty(self) {
            if let Err(e) = std::fs::write(Self::path(), j) {
                tracing::error!("could not save the tip-check settings: {e}");
            }
        }
    }
    /// The verdict for given scores (shared with re-judging stored reports).
    pub fn judge(&self, r: &TipReport) -> (Verdict, Vec<String>) {
        let mut notes = Vec::new();
        // Too featureless for ghosts, but tip changes still show.
        if r.snr < self.min_snr as f64 {
            if r.instability >= self.poor_instability {
                return (Verdict::Poor, vec!["unstable: many tip changes or streaks".into()]);
            }
            if r.instability >= self.suspect_instability {
                return (Verdict::Suspect, vec!["some tip changes or streaks".into()]);
            }
            return (Verdict::Unsure("too featureless to judge".into()), notes);
        }
        let mut level = 0;
        if r.multi_tip >= self.poor_multi {
            level = 2;
            notes.push(format!("ghost copies ({} apexes)", r.apexes.max(2)));
        } else if r.multi_tip >= self.suspect_multi {
            level = 1;
            notes.push("possible ghost copy".into());
        }
        if r.instability >= self.poor_instability {
            level = 2;
            notes.push("unstable: many tip changes or streaks".into());
        } else if r.instability >= self.suspect_instability {
            level = level.max(1);
            notes.push("some tip changes or streaks".into());
        }
        if let Some(fb) = r.fb_agreement {
            if fb < self.min_fb {
                level = level.max(1);
                notes.push("forward and backward disagree".into());
            }
        }
        if let Some(b) = r.blunt {
            if b >= self.suspect_blunt {
                level = level.max(1);
                notes.push("features blurred: blunt tip?".into());
            }
        }
        let v = match level {
            0 => Verdict::Good,
            1 => Verdict::Suspect,
            _ => Verdict::Poor,
        };
        (v, notes)
    }
}

/// Judge a frame. `forward`/`backward` are row-major `size` = [w, h] with
/// NaN for unscanned pixels; `frame` maps the unit quad to nm (as a live
/// image transform), for the pixel size.
pub fn analyse(
    forward: &[f32],
    backward: Option<&[f32]>,
    size: [u32; 2],
    frame: DAffine2,
    params: &TipParams,
) -> TipReport {
    let frame_nm = [frame.matrix2.x_axis.length(), frame.matrix2.y_axis.length()];
    let unsure = |why: &str| TipReport {
        verdict: Verdict::Unsure(why.into()),
        multi_tip: 0.,
        instability: 0.,
        fb_agreement: None,
        blunt: None,
        ghosts: Vec::new(),
        apexes: 1,
        px_nm: frame_nm[0] / size[0].max(1) as f64,
        scanned: 0.,
        snr: 0.,
        notes: Vec::new(),
    };
    let Some(p) = prepare(forward, size, frame_nm) else {
        return unsure("too little scanned");
    };
    let snr = features::signal_to_noise(&p);
    // Fine detail (atoms, defects), larger features, and edges.
    let coarse = (p.w.min(p.h) as f32 / 16.).max(4.);
    let maps = [high_pass(&p, 3.), high_pass(&p, coarse), gradient(&p)];
    let mut found: Vec<(usize, Peak)> = Vec::new();
    let mut hp_acf = None;
    for (k, m) in maps.iter().enumerate() {
        let Some(acf) = Acf::new(m, p.w, p.h) else { continue };
        for pk in ghost_peaks(&acf, 4) {
            found.push((k, pk));
        }
        if k == 0 {
            hp_acf = Some(acf);
        }
    }
    let ghosts = merge_ghosts(&found, &p);
    let multi_tip = ghosts.first().map_or(0., |g| g.strength) as f32;
    // Two apexes give one ghost pair; three give three (a, b, a − b).
    let strong = ghosts.iter().filter(|g| g.strength >= 0.5 * multi_tip as f64 && g.strength > 0.1).count();
    let apexes = match strong {
        0 => 1,
        1 | 2 => 2,
        _ => 3,
    };
    let instability = features::instability(&p);
    let fb_agreement = backward
        .filter(|b| b.len() == forward.len())
        .and_then(|b| prepare(b, size, frame_nm))
        .and_then(|bp| features::fb_agreement(&p, &bp));
    let blunt = hp_acf.as_ref().and_then(|a| features::bluntness(&p, a));
    let mut r = TipReport {
        verdict: Verdict::Good,
        multi_tip,
        instability,
        fb_agreement,
        blunt,
        ghosts,
        apexes,
        px_nm: p.px_nm[0] / p.factor as f64,
        scanned: p.valid,
        snr,
        notes: Vec::new(),
    };
    let (v, notes) = params.judge(&r);
    r.verdict = v;
    r.notes = notes;
    r
}

/// Combine the peaks from the three images: the same offset (either
/// sign) seen in more than one image counts more.
fn merge_ghosts(found: &[(usize, Peak)], p: &Prepared) -> Vec<Ghost> {
    let mut groups: Vec<(DVec2, f64, u8, Vec<f64>)> = Vec::new();
    for (k, pk) in found {
        let same = groups.iter_mut().find(|g| (g.0 - pk.d).length() < 1.5 || (g.0 + pk.d).length() < 1.5);
        match same {
            Some(g) => {
                g.2 |= 1 << k;
                g.3.push(pk.strength);
                if pk.strength > g.1 {
                    g.1 = pk.strength;
                    g.0 = pk.d;
                }
            }
            None => groups.push((pk.d, pk.strength, 1 << k, vec![pk.strength])),
        }
    }
    let f = p.factor as f64;
    let mut ghosts: Vec<Ghost> = groups
        .into_iter()
        .map(|(d, best, seen, all)| {
            // Agreement between images adds up to half the others' strength.
            let others: f64 = all.iter().sum::<f64>() - best;
            let strength = (best + 0.5 * others).min(1.);
            Ghost {
                offset_px: [d.x * f, d.y * f],
                offset_nm: [d.x * p.px_nm[0], d.y * p.px_nm[1]],
                strength,
                seen_in: seen,
            }
        })
        .collect();
    ghosts.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    ghosts.truncate(4);
    ghosts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random numbers in [0, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A surface (height, nm-ish units): optional lattice, point defects,
    /// a step edge. `f(x, y)` in pixels.
    fn surface(lattice: bool, defects: usize, step: bool, seed: u64) -> impl Fn(f64, f64) -> f64 {
        let mut rng = Rng(seed);
        let spots: Vec<(f64, f64, f64)> =
            (0..defects).map(|_| (rng.next() * 256., rng.next() * 256., 0.5 + rng.next())).collect();
        move |x: f64, y: f64| {
            let mut z = 0.;
            if lattice {
                z += 0.15 * ((x * 2. * std::f64::consts::PI / 7.).cos() + (y * 2. * std::f64::consts::PI / 9.).cos());
            }
            for (sx, sy, a) in &spots {
                let d2 = (x - sx).powi(2) + (y - sy).powi(2);
                z += a * (-d2 / (2. * 1.6 * 1.6)).exp();
            }
            if step {
                // A smooth edge, about 2 px wide.
                z += 1. + ((x + 0.3 * y - 140.) / 1.5).tanh();
            }
            z
        }
    }

    /// Image a surface with apexes at `offsets` (px): the tip follows the
    /// highest apex, so the image is the max over shifted copies.
    fn image(f: &dyn Fn(f64, f64) -> f64, offsets: &[(f64, f64, f64)], n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed);
        let mut out = vec![0f32; n * n];
        for y in 0..n {
            for x in 0..n {
                let z = offsets
                    .iter()
                    .map(|(dx, dy, dz)| f(x as f64 - dx, y as f64 - dy) - dz)
                    .fold(f64::NEG_INFINITY, f64::max);
                out[y * n + x] = (z + 0.03 * (rng.next() - 0.5)) as f32;
            }
        }
        out
    }

    fn frame(n: usize) -> DAffine2 {
        // 0.2 nm pixels.
        DAffine2::from_scale(DVec2::new(n as f64 * 0.2, -(n as f64) * 0.2))
    }

    fn run(img: &[f32], n: usize) -> TipReport {
        analyse(img, None, [n as u32, n as u32], frame(n), &TipParams::default())
    }

    /// `cargo test --release tip_quality::tests::print_all -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn print_all() {
        let d = surface(false, 60, false, 1);
        let l = surface(true, 0, false, 3);
        let ld = surface(true, 40, false, 3);
        let st = surface(false, 0, true, 5);
        let t = surface(false, 80, false, 7);
        let cases: Vec<(&str, Vec<f32>)> = vec![
            ("defects single", image(&d, &[(0., 0., 0.)], 200, 2)),
            ("defects double", image(&d, &[(0., 0., 0.), (9., 5., 0.1)], 200, 2)),
            ("lattice single", image(&l, &[(0., 0., 0.)], 200, 4)),
            ("lattice+def single", image(&ld, &[(0., 0., 0.)], 200, 4)),
            ("lattice+def double", image(&ld, &[(0., 0., 0.), (-11., 6., 0.05)], 200, 4)),
            ("step single", image(&st, &[(0., 0., 0.)], 200, 6)),
            ("step double", image(&st, &[(0., 0., 0.), (-12., 0., 0.3)], 200, 6)),
            ("triple", image(&t, &[(0., 0., 0.), (10., 3., 0.05), (-2., 11., 0.1)], 200, 8)),
        ];
        for (name, img) in cases {
            autocorr::DEBUG.with(|d| d.set(std::env::var("TIPQ_DEBUG").is_ok_and(|v| v == name)));
            let r = run(&img, 200);
            autocorr::DEBUG.with(|d| d.set(false));
            let g: Vec<String> = r.ghosts.iter().map(|g| format!("({:.1},{:.1})={:.2}/{}", g.offset_px[0], g.offset_px[1], g.strength, g.seen_in)).collect();
            println!("{name:20} {:8} multi {:.2} inst {:.2} snr {:.1} apexes {} {}", r.verdict.label(), r.multi_tip, r.instability, r.snr, r.apexes, g.join(" "));
        }
    }

    #[test]
    fn single_tip_defects_is_good() {
        let s = surface(false, 60, false, 1);
        let r = run(&image(&s, &[(0., 0., 0.)], 200, 2), 200);
        assert!(r.multi_tip < 0.2, "{r:#?}");
        assert_eq!(r.verdict, Verdict::Good, "{r:#?}");
    }

    #[test]
    fn double_tip_defects_found_with_offset() {
        let s = surface(false, 60, false, 1);
        let r = run(&image(&s, &[(0., 0., 0.), (9., 5., 0.1)], 200, 2), 200);
        assert!(r.multi_tip >= 0.35, "{r:#?}");
        assert_eq!(r.verdict, Verdict::Poor, "{r:#?}");
        let g = DVec2::from_array(r.ghosts[0].offset_px);
        let want = DVec2::new(9., 5.);
        assert!((g - want).length() < 1.0 || (g + want).length() < 1.0, "{g:?}");
    }

    #[test]
    fn lattice_alone_is_not_flagged() {
        let s = surface(true, 0, false, 3);
        let r = run(&image(&s, &[(0., 0., 0.)], 200, 4), 200);
        assert!(r.multi_tip < 0.2, "{r:#?}");
    }

    #[test]
    fn lattice_with_double_tip_is_flagged() {
        let s = surface(true, 40, false, 3);
        let single = run(&image(&s, &[(0., 0., 0.)], 200, 4), 200);
        let double = run(&image(&s, &[(0., 0., 0.), (-11., 6., 0.05)], 200, 4), 200);
        assert!(double.multi_tip > single.multi_tip + 0.15, "{single:#?}\n{double:#?}");
        assert!(double.verdict != Verdict::Good, "{double:#?}");
    }

    #[test]
    fn doubled_step_edge_is_flagged() {
        let s = surface(false, 0, true, 5);
        let single = run(&image(&s, &[(0., 0., 0.)], 200, 6), 200);
        let double = run(&image(&s, &[(0., 0., 0.), (-12., 0., 0.3)], 200, 6), 200);
        // A straight edge only fixes the offset across it: check the score.
        assert!(double.multi_tip > single.multi_tip + 0.1, "{single:#?}\n{double:#?}");
        assert_eq!(single.verdict, Verdict::Good, "{single:#?}");
    }

    #[test]
    fn triple_tip_counts_apexes() {
        let s = surface(false, 80, false, 7);
        let r = run(&image(&s, &[(0., 0., 0.), (10., 3., 0.05), (-2., 11., 0.1)], 200, 8), 200);
        assert!(r.verdict == Verdict::Poor, "{r:#?}");
        assert!(r.apexes >= 3, "{r:#?}");
    }

    #[test]
    fn streaks_raise_instability() {
        let s = surface(false, 60, false, 1);
        let mut img = image(&s, &[(0., 0., 0.)], 200, 2);
        for y in (0..200).step_by(6) {
            for x in 0..200 {
                img[y * 200 + x] += 1.5 * (((x / 7) % 2) as f32 - 0.5);
            }
        }
        let r = run(&img, 200);
        assert!(r.instability > 0.5, "{r:#?}");
    }

    #[test]
    fn partial_frame_is_unsure_or_judged() {
        let s = surface(false, 60, false, 1);
        let mut img = image(&s, &[(0., 0., 0.)], 200, 2);
        for v in img[200 * 10..].iter_mut() {
            *v = f32::NAN;
        }
        let r = run(&img, 200);
        assert!(matches!(r.verdict, Verdict::Unsure(_)), "{r:#?}");
    }
}
