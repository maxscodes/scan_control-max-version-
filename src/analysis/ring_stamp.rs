//! Ring Assist result stored with a stamped image.
//!
//! Positions and directions are kept in the stamp's own frame coordinates
//! (quad space, −0.5..0.5 on each axis), so the marker follows the stamp if
//! it is moved, and the result can be saved in the project file.

use egui::{Align2, Color32, FontId, Id, Stroke, Ui};
use glam::{DAffine2, DVec2};

use crate::{
    analysis::ring_center::{self, RingParams, UnsureReason, Verdict},
    scan_view::ScanViewCtx,
    utils::vec_interop::IntoEgui as _,
};

pub const ARC_COLOR: Color32 = Color32::from_rgb(0x3c, 0xd0, 0x5a);
pub const ARM_COLOR: Color32 = Color32::from_rgb(0x3c, 0xb4, 0xff);
pub const UNSURE_COLOR: Color32 = Color32::from_rgb(0xff, 0x50, 0x50);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StampVerdict {
    Arc,
    ArmEnd,
    Unsure(UnsureReason),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingStamp {
    pub verdict: StampVerdict,
    /// Unit move direction in quad space (zero when unsure).
    pub dir_quad: DVec2,
    /// Fitted ring centre in quad space, for arc verdicts.
    pub center_quad: Option<DVec2>,
    pub radius_um: f64,
    pub inlier_fraction: f64,
    /// Fraction of the frame that had been scanned when stamped.
    pub valid_fraction: f64,
}

impl RingStamp {
    pub fn to_array(&self) -> [f64; 10] {
        let (kind, reason) = match self.verdict {
            StampVerdict::Arc => (0., 0.),
            StampVerdict::ArmEnd => (1., 0.),
            StampVerdict::Unsure(r) => (
                2.,
                match r {
                    UnsureReason::NoEdges => 0.,
                    UnsureReason::NotRingLike => 1.,
                    UnsureReason::OnlyLocalFeatures => 2.,
                },
            ),
        };
        let (has_c, c) = match self.center_quad {
            Some(c) => (1., c),
            None => (0., DVec2::ZERO),
        };
        [
            kind,
            reason,
            self.dir_quad.x,
            self.dir_quad.y,
            has_c,
            c.x,
            c.y,
            self.radius_um,
            self.inlier_fraction,
            self.valid_fraction,
        ]
    }
    pub fn from_array(a: [f64; 10]) -> Self {
        let verdict = match a[0] as u8 {
            0 => StampVerdict::Arc,
            1 => StampVerdict::ArmEnd,
            _ => StampVerdict::Unsure(match a[1] as u8 {
                0 => UnsureReason::NoEdges,
                1 => UnsureReason::NotRingLike,
                _ => UnsureReason::OnlyLocalFeatures,
            }),
        };
        Self {
            verdict,
            dir_quad: DVec2::new(a[2], a[3]),
            center_quad: (a[4] != 0.).then(|| DVec2::new(a[5], a[6])),
            radius_um: a[7],
            inlier_fraction: a[8],
            valid_fraction: a[9],
        }
    }
    pub fn summary(&self) -> String {
        match self.verdict {
            StampVerdict::Arc => "On ring arcs".into(),
            StampVerdict::ArmEnd => format!(
                "Arm end: move {}",
                if self.dir_quad.x < 0. { "left" } else { "right" }
            ),
            StampVerdict::Unsure(r) => format!("Unsure: {}", r.describe()),
        }
    }
    pub fn color(&self) -> Color32 {
        match self.verdict {
            StampVerdict::Arc => ARC_COLOR,
            StampVerdict::ArmEnd => ARM_COLOR,
            StampVerdict::Unsure(_) => UNSURE_COLOR,
        }
    }
}

/// The user's label for a stamp, used to train Ring Assist.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RingLabel {
    /// The model's direction was right (stored as that direction, quad space).
    Good(DVec2),
    /// The model was wrong; the tip should go this way (unit, quad space).
    Direction(DVec2),
    /// No move wanted here (centre, electrodes).
    Stop,
}

impl RingLabel {
    pub fn to_array(&self) -> [f64; 3] {
        match *self {
            RingLabel::Good(d) => [0., d.x, d.y],
            RingLabel::Direction(d) => [1., d.x, d.y],
            RingLabel::Stop => [2., 0., 0.],
        }
    }
    pub fn from_array(a: [f64; 3]) -> Self {
        let d = DVec2::new(a[1], a[2]);
        match a[0] as u8 {
            0 => RingLabel::Good(d),
            1 => RingLabel::Direction(d),
            _ => RingLabel::Stop,
        }
    }
    pub fn describe(&self) -> &'static str {
        match self {
            RingLabel::Good(_) => "good",
            RingLabel::Direction(_) => "corrected",
            RingLabel::Stop => "should stop",
        }
    }
}

/// Convert a quad-space direction to frame pixel axes (+x right, +y down rows).
pub fn quad_to_px_dir(d: DVec2, size: [u32; 2]) -> DVec2 {
    DVec2::new(d.x * size[0] as f64, d.y * size[1] as f64).normalize_or_zero()
}

/// A quad-to-world transform with the right scale for analysing a frame of
/// `size` pixels at `nm_per_px` (position and orientation don't matter).
pub fn frame_for(size: [u32; 2], nm_per_px: f64) -> DAffine2 {
    DAffine2::from_scale(DVec2::new(
        size[0] as f64 * nm_per_px,
        -(size[1] as f64) * nm_per_px,
    ))
}

/// Run the ring estimator on one frame. `frame` maps quad space to world nm
/// (only its scale is used here, to convert radii to µm).
pub fn analyse_frame(
    data: &[f32],
    size: [u32; 2],
    frame: DAffine2,
    params: &RingParams,
    min_radius_um: f64,
) -> Option<RingStamp> {
    let (w, h) = (size[0] as usize, size[1] as usize);
    if w < 8 || h < 8 || data.len() != w * h {
        return None;
    }
    let valid_fraction =
        data.iter().filter(|v| v.is_finite()).count() as f64 / data.len() as f64;
    if valid_fraction < 0.3 {
        return None;
    }
    let nm_per_px = frame.matrix2.x_axis.length() / w as f64;
    let mut params = *params;
    params.min_ring_radius_px = Some(min_radius_um * 1e3 / nm_per_px);
    let pre = ring_center::preprocess(data, w, h, &params);
    let res =
        ring_center::estimate_preprocessed(&pre.img, pre.width, pre.height, pre.scale, &params);
    let verdict = ring_center::decide(&res, pre.scale, &params);

    let s = pre.scale;
    let to_quad = |p: DVec2| {
        let input = (p + 0.5) * s - 0.5;
        DVec2::new((input.x + 0.5) / w as f64 - 0.5, (input.y + 0.5) / h as f64 - 0.5)
    };
    let dir_to_quad =
        |d: DVec2| DVec2::new(d.x / w as f64, d.y / h as f64).normalize_or_zero();
    let (radius_um, inlier_fraction, center_quad) = match &res {
        Ok(a) => (
            a.est.radius_px * nm_per_px * s * 1e-3,
            a.est.inlier_fraction,
            matches!(verdict, Verdict::Arc { .. }).then(|| to_quad(a.est.center_px)),
        ),
        Err(_) => (0., 0., None),
    };
    let (verdict, dir_quad) = match verdict {
        Verdict::Arc { dir } => (StampVerdict::Arc, dir_to_quad(dir)),
        Verdict::ArmEnd { dir } => (StampVerdict::ArmEnd, dir_to_quad(dir)),
        Verdict::Unsure(r) => (StampVerdict::Unsure(r), DVec2::ZERO),
    };
    Some(RingStamp {
        verdict,
        dir_quad,
        center_quad,
        radius_um,
        inlier_fraction,
        valid_fraction,
    })
}

pub const LABEL_COLOR: Color32 = Color32::WHITE;

/// Draw a stamp's ring result and/or the user's label on the map.
pub fn draw(ui: &Ui, stamp_transform: DAffine2, r: Option<&RingStamp>, label: Option<&RingLabel>) {
    let Some(ctx) = ui.data(|map| map.get_temp::<ScanViewCtx>(Id::new(()))) else {
        return;
    };
    let w2s = ctx.world2egui();
    let to_screen = |p: DVec2| w2s.transform_point2(p).to_egui_pos2();
    let painter = ui.painter();
    let c = stamp_transform.translation;
    let width = stamp_transform.matrix2.x_axis.length();
    let p = to_screen(c);
    match label {
        Some(RingLabel::Good(_)) => {
            painter.text(
                p + egui::vec2(-10., -6.),
                Align2::RIGHT_BOTTOM,
                "\u{2714}",
                FontId::proportional(16.),
                ARC_COLOR,
            );
        }
        Some(RingLabel::Direction(d)) => {
            let dir = stamp_transform.transform_vector2(*d).normalize_or_zero();
            let b = to_screen(c + dir * width * 0.4);
            let stroke = Stroke::new(2., LABEL_COLOR);
            painter.extend(egui::Shape::dashed_line(&[p, b], stroke, 6., 4.));
            let tip = (b - p).normalized() * 10.;
            painter.arrow(b - tip, tip, stroke);
        }
        Some(RingLabel::Stop) => {
            painter.rect_filled(
                egui::Rect::from_center_size(p + egui::vec2(-14., -14.), egui::vec2(9., 9.)),
                1.,
                LABEL_COLOR,
            );
        }
        None => {}
    }
    let Some(r) = r else { return };
    let color = r.color();
    match r.verdict {
        StampVerdict::Unsure(_) => {
            let k = 7.;
            let stroke = Stroke::new(2.5, color);
            painter.line_segment([p + egui::vec2(-k, -k), p + egui::vec2(k, k)], stroke);
            painter.line_segment([p + egui::vec2(-k, k), p + egui::vec2(k, -k)], stroke);
        }
        _ => {
            let dir = stamp_transform.transform_vector2(r.dir_quad).normalize_or_zero();
            let b = to_screen(c + dir * width * 0.3);
            painter.arrow(p, b - p, Stroke::new(2.5, color));
        }
    }
    painter.text(
        p + egui::vec2(8., 8.),
        Align2::LEFT_TOP,
        r.summary(),
        FontId::proportional(11.),
        color,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_round_trip() {
        for verdict in [
            StampVerdict::Arc,
            StampVerdict::ArmEnd,
            StampVerdict::Unsure(UnsureReason::NoEdges),
            StampVerdict::Unsure(UnsureReason::NotRingLike),
            StampVerdict::Unsure(UnsureReason::OnlyLocalFeatures),
        ] {
            let r = RingStamp {
                verdict,
                dir_quad: DVec2::new(0.6, -0.8),
                center_quad: matches!(verdict, StampVerdict::Arc).then(|| DVec2::new(1.5, -2.)),
                radius_um: 3.2,
                inlier_fraction: 0.4,
                valid_fraction: 1.,
            };
            assert_eq!(RingStamp::from_array(r.to_array()), r);
        }
    }

    #[test]
    fn synthetic_ring_points_to_centre() {
        // Concentric rings centred up-left of a 96 px frame, 3 µm wide.
        let (w, h) = (96usize, 96usize);
        let centre = DVec2::new(-60., -40.);
        let data: Vec<f32> = (0..w * h)
            .map(|i| {
                let p = DVec2::new((i % w) as f64, (i / w) as f64);
                let r = (p - centre).length();
                if (r / 12.).floor() as i64 % 2 == 0 { 1. } else { 0. }
            })
            .collect();
        let frame = DAffine2::from_scale(DVec2::new(3000., -3000.));
        let params = RingParams { target_px: None, ..RingParams::default() };
        let r = analyse_frame(&data, [w as u32, h as u32], frame, &params, 0.5).unwrap();
        assert_eq!(r.verdict, StampVerdict::Arc);
        // In quad space the centre is up-left (−x, −y, with row 0 at quad −0.5).
        assert!(r.dir_quad.x < 0. && r.dir_quad.y < 0., "{r:?}");
    }
}
