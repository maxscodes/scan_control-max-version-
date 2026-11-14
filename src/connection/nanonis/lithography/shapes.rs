//! Basic shapes drawn directly in the lithography window, used alongside (or
//! instead of) a GDS. Coordinates are map (world) nm, the same space the GDS
//! is placed in, so shapes stay on the sample like the GDS does. Angles are
//! degrees, counter-clockwise on the map.

use std::f64::consts::TAU;

use glam::{DMat2, DVec2};
use serde::{Deserialize, Serialize};

/// Segments per full turn for ellipses and arcs.
const CIRCLE_SEGMENTS: f64 = 96.;
/// Samples per control-point span of a smooth curve.
const CURVE_SAMPLES: usize = 16;
/// Joins sharper than this (miter length / half width) are bevelled.
const MITER_LIMIT: f64 = 4.;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Shape {
    pub name: String,
    pub enabled: bool,
    pub kind: ShapeKind,
}
impl Default for Shape {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: true,
            kind: ShapeKind::Rect { center: [0., 0.], size: [10., 10.], angle: 0. },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShapeKind {
    /// Filled rectangle.
    Rect { center: [f64; 2], size: [f64; 2], angle: f64 },
    /// Filled ellipse (circle when both radii match).
    Ellipse { center: [f64; 2], radii: [f64; 2], angle: f64 },
    /// Arc from `start` to `end` degrees (counter-clockwise when `end > start`).
    /// Width 0 writes the arc as a single line; otherwise a filled band.
    Arc { center: [f64; 2], radius: f64, start: f64, end: f64, width: f64 },
    /// Filled polygon.
    Polygon { points: Vec<[f64; 2]> },
    /// Open line through the points; `smooth` draws a curve through them.
    /// Width 0 writes it as a single line; otherwise a filled band.
    Line { points: Vec<[f64; 2]>, smooth: bool, width: f64 },
}

/// What a shape contributes to the lithography path.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// Area to raster (or outline in boundary mode).
    Fill(Vec<DVec2>),
    /// Line written directly along its length.
    Stroke(Vec<DVec2>),
}

/// A draggable point of a shape on the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handle {
    /// The whole shape.
    Body,
    Center,
    /// Rectangle corner: sets the size around the centre.
    Corner,
    RadiusX,
    RadiusY,
    ArcStart,
    ArcEnd,
    Vertex(usize),
}

fn v(p: [f64; 2]) -> DVec2 {
    DVec2::from_array(p)
}
fn rot(deg: f64) -> DMat2 {
    DMat2::from_angle(deg.to_radians())
}
fn polar(c: DVec2, r: f64, deg: f64) -> DVec2 {
    let a = deg.to_radians();
    c + r * DVec2::new(a.cos(), a.sin())
}
fn angle_deg(c: DVec2, p: DVec2) -> f64 {
    let d = p - c;
    d.y.atan2(d.x).to_degrees()
}

impl ShapeKind {
    pub fn label(&self) -> &'static str {
        match self {
            ShapeKind::Rect { .. } => "Rectangle",
            ShapeKind::Ellipse { .. } => "Ellipse",
            ShapeKind::Arc { .. } => "Arc",
            ShapeKind::Polygon { .. } => "Polygon",
            ShapeKind::Line { smooth: true, .. } => "Curve",
            ShapeKind::Line { .. } => "Line",
        }
    }
}

impl Shape {
    /// Default shapes of each kind, sized to a frame `size` nm wide at `c`.
    pub fn new_rect(c: DVec2, size: f64) -> Self {
        Self::of(ShapeKind::Rect { center: c.to_array(), size: [0.3 * size, 0.2 * size], angle: 0. })
    }
    pub fn new_ellipse(c: DVec2, size: f64) -> Self {
        Self::of(ShapeKind::Ellipse { center: c.to_array(), radii: [0.15 * size, 0.15 * size], angle: 0. })
    }
    pub fn new_arc(c: DVec2, size: f64) -> Self {
        Self::of(ShapeKind::Arc { center: c.to_array(), radius: 0.25 * size, start: 0., end: 180., width: 0. })
    }
    pub fn new_polygon(c: DVec2, size: f64) -> Self {
        let r = 0.2 * size;
        let points = (0..3).map(|k| polar(c, r, 90. + 120. * k as f64).to_array()).collect();
        Self::of(ShapeKind::Polygon { points })
    }
    pub fn new_line(c: DVec2, size: f64) -> Self {
        let d = DVec2::new(0.3 * size, 0.);
        Self::of(ShapeKind::Line { points: vec![(c - d).to_array(), (c + d).to_array()], smooth: false, width: 0. })
    }
    pub fn new_curve(c: DVec2, size: f64) -> Self {
        let s = 0.3 * size;
        let points = [(-1., 0.), (-0.33, 0.5), (0.33, -0.5), (1., 0.)]
            .map(|(x, y)| (c + DVec2::new(x * s, y * s)).to_array())
            .to_vec();
        Self::of(ShapeKind::Line { points, smooth: true, width: 0. })
    }
    fn of(kind: ShapeKind) -> Self {
        Self { kind, ..Default::default() }
    }

    /// Short description for the shape list.
    pub fn summary(&self) -> String {
        match &self.kind {
            ShapeKind::Rect { size, angle, .. } => format!("{:.1}×{:.1} nm, {angle:.0}°", size[0], size[1]),
            ShapeKind::Ellipse { radii, .. } if radii[0] == radii[1] => format!("r {:.1} nm", radii[0]),
            ShapeKind::Ellipse { radii, .. } => format!("r {:.1}×{:.1} nm", radii[0], radii[1]),
            ShapeKind::Arc { radius, start, end, width, .. } => {
                format!("r {radius:.1} nm, {start:.0}°→{end:.0}°{}", width_text(*width))
            }
            ShapeKind::Polygon { points } => format!("{} points", points.len()),
            ShapeKind::Line { points, width, .. } => format!("{} points{}", points.len(), width_text(*width)),
        }
    }

    /// The area or line this shape writes, in map nm. `None` if degenerate.
    pub fn geometry(&self) -> Option<Geometry> {
        match &self.kind {
            ShapeKind::Rect { center, size, angle } => {
                let (c, h, r) = (v(*center), v(*size) / 2., rot(*angle));
                if h.x <= 0. || h.y <= 0. {
                    return None;
                }
                let corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)];
                Some(Geometry::Fill(corners.iter().map(|&(x, y)| c + r * (h * DVec2::new(x, y))).collect()))
            }
            ShapeKind::Ellipse { center, radii, angle } => {
                let (c, rr, r) = (v(*center), v(*radii), rot(*angle));
                if rr.x <= 0. || rr.y <= 0. {
                    return None;
                }
                let n = CIRCLE_SEGMENTS as usize;
                Some(Geometry::Fill(
                    (0..n)
                        .map(|k| {
                            let a = TAU * k as f64 / n as f64;
                            c + r * (rr * DVec2::new(a.cos(), a.sin()))
                        })
                        .collect(),
                ))
            }
            ShapeKind::Arc { center, radius, start, end, width } => {
                let c = v(*center);
                let sweep = end - start;
                if *radius <= 0. || sweep == 0. {
                    return None;
                }
                let n = ((sweep.abs() / 360. * CIRCLE_SEGMENTS).ceil() as usize).max(4);
                let at = |r: f64, k: usize| polar(c, r, start + sweep * k as f64 / n as f64);
                if *width <= 0. {
                    return Some(Geometry::Stroke((0..=n).map(|k| at(*radius, k)).collect()));
                }
                let (outer, inner) = (radius + width / 2., (radius - width / 2.).max(0.));
                let mut pts: Vec<DVec2> = (0..=n).map(|k| at(outer, k)).collect();
                pts.extend((0..=n).rev().map(|k| at(inner, k)));
                Some(Geometry::Fill(pts))
            }
            ShapeKind::Polygon { points } => {
                (points.len() >= 3).then(|| Geometry::Fill(points.iter().copied().map(v).collect()))
            }
            ShapeKind::Line { points, smooth, width } => {
                let pts = dedup(&points.iter().copied().map(v).collect::<Vec<_>>());
                if pts.len() < 2 {
                    return None;
                }
                let pts = if *smooth { dedup(&catmull_rom(&pts)) } else { pts };
                if *width <= 0. {
                    Some(Geometry::Stroke(pts))
                } else {
                    Some(Geometry::Fill(outline(&pts, width / 2.)))
                }
            }
        }
    }

    /// Draggable points (map nm), the centre first where there is one.
    pub fn handles(&self) -> Vec<(Handle, DVec2)> {
        match &self.kind {
            ShapeKind::Rect { center, size, angle } => {
                let c = v(*center);
                vec![(Handle::Center, c), (Handle::Corner, c + rot(*angle) * (v(*size) / 2.))]
            }
            ShapeKind::Ellipse { center, radii, angle } => {
                let (c, r) = (v(*center), rot(*angle));
                vec![
                    (Handle::Center, c),
                    (Handle::RadiusX, c + r * DVec2::new(radii[0], 0.)),
                    (Handle::RadiusY, c + r * DVec2::new(0., radii[1])),
                ]
            }
            ShapeKind::Arc { center, radius, start, end, .. } => {
                let c = v(*center);
                vec![
                    (Handle::Center, c),
                    (Handle::ArcStart, polar(c, *radius, *start)),
                    (Handle::ArcEnd, polar(c, *radius, *end)),
                ]
            }
            ShapeKind::Polygon { points } | ShapeKind::Line { points, .. } => {
                points.iter().enumerate().map(|(k, p)| (Handle::Vertex(k), v(*p))).collect()
            }
        }
    }

    /// Move `handle` to `at` (map nm); `delta` is the pointer movement, used
    /// to move the whole shape.
    pub fn drag(&mut self, handle: Handle, at: DVec2, delta: DVec2) {
        match (&mut self.kind, handle) {
            (_, Handle::Body) => self.translate(delta),
            (_, Handle::Center) => self.translate(at - self.handles()[0].1),
            (ShapeKind::Rect { center, size, angle }, Handle::Corner) => {
                let local = rot(-*angle) * (at - v(*center));
                *size = (local.abs() * 2.).max(DVec2::splat(0.01)).to_array();
            }
            (ShapeKind::Ellipse { center, radii, .. }, Handle::RadiusX) => {
                radii[0] = v(*center).distance(at).max(0.01);
            }
            (ShapeKind::Ellipse { center, radii, .. }, Handle::RadiusY) => {
                radii[1] = v(*center).distance(at).max(0.01);
            }
            (ShapeKind::Arc { center, radius, start, end, .. }, Handle::ArcStart) => {
                let c = v(*center);
                *radius = c.distance(at).max(0.01);
                let sweep = *end - *start;
                *start = angle_deg(c, at);
                *end = *start + sweep;
            }
            (ShapeKind::Arc { center, start, end, .. }, Handle::ArcEnd) => {
                let raw = angle_deg(v(*center), at);
                // Keep the arc's direction; the sweep wraps within one turn.
                *end = if *end >= *start {
                    *start + (raw - *start).rem_euclid(360.)
                } else {
                    *start - (*start - raw).rem_euclid(360.)
                };
            }
            (ShapeKind::Polygon { points } | ShapeKind::Line { points, .. }, Handle::Vertex(k)) => {
                if let Some(p) = points.get_mut(k) {
                    *p = at.to_array();
                }
            }
            _ => {}
        }
    }

    pub fn translate(&mut self, d: DVec2) {
        let mv = |p: &mut [f64; 2]| *p = (v(*p) + d).to_array();
        match &mut self.kind {
            ShapeKind::Rect { center, .. } | ShapeKind::Ellipse { center, .. } | ShapeKind::Arc { center, .. } => {
                mv(center)
            }
            ShapeKind::Polygon { points } | ShapeKind::Line { points, .. } => points.iter_mut().for_each(mv),
        }
    }

    /// The editable points of a polygon or line.
    pub fn points_mut(&mut self) -> Option<&mut Vec<[f64; 2]>> {
        match &mut self.kind {
            ShapeKind::Polygon { points } | ShapeKind::Line { points, .. } => Some(points),
            _ => None,
        }
    }
    fn min_points(&self) -> usize {
        match self.kind {
            ShapeKind::Polygon { .. } => 3,
            _ => 2,
        }
    }
    /// Remove point `k` unless the shape would become degenerate.
    pub fn remove_point(&mut self, k: usize) -> bool {
        let min = self.min_points();
        match self.points_mut() {
            Some(points) if points.len() > min && k < points.len() => {
                points.remove(k);
                true
            }
            _ => false,
        }
    }
    /// Add a point at `at`: inserted into the edge nearest to it if that edge
    /// is within `tol` nm, otherwise appended at the end.
    pub fn add_point(&mut self, at: DVec2, tol: f64) {
        let closed = matches!(self.kind, ShapeKind::Polygon { .. });
        let Some(points) = self.points_mut() else { return };
        let n = points.len();
        let edges = if closed { n } else { n.saturating_sub(1) };
        let nearest = (0..edges)
            .map(|i| (i, dist_to_segment(at, v(points[i]), v(points[(i + 1) % n]))))
            .filter(|(_, d)| *d <= tol)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        match nearest {
            Some((i, _)) => points.insert(i + 1, at.to_array()),
            None => points.push(at.to_array()),
        }
    }

    /// Whether `p` (map nm) is on the shape: inside its area, or within
    /// `tol` nm of its outline or line.
    pub fn hit(&self, p: DVec2, tol: f64) -> bool {
        match self.geometry() {
            Some(Geometry::Fill(poly)) => inside(p, &poly) || dist_to_polyline(p, &poly, true) <= tol,
            Some(Geometry::Stroke(line)) => dist_to_polyline(p, &line, false) <= tol,
            None => self.handles().iter().any(|(_, h)| h.distance(p) <= tol),
        }
    }
}

fn width_text(w: f64) -> String {
    if w > 0. {
        format!(", {w:.1} nm wide")
    } else {
        ", single line".into()
    }
}

fn dedup(pts: &[DVec2]) -> Vec<DVec2> {
    let mut out: Vec<DVec2> = Vec::with_capacity(pts.len());
    for &p in pts {
        if out.last().is_none_or(|l| l.distance(p) > 1e-9) {
            out.push(p);
        }
    }
    out
}

/// Uniform Catmull-Rom curve through all points (ends held).
fn catmull_rom(pts: &[DVec2]) -> Vec<DVec2> {
    let n = pts.len();
    let at = |i: isize| pts[i.clamp(0, n as isize - 1) as usize];
    let mut out = Vec::with_capacity((n - 1) * CURVE_SAMPLES + 1);
    for i in 0..n - 1 {
        let (p0, p1, p2, p3) = (at(i as isize - 1), at(i as isize), at(i as isize + 1), at(i as isize + 2));
        for s in 0..CURVE_SAMPLES {
            let t = s as f64 / CURVE_SAMPLES as f64;
            let (t2, t3) = (t * t, t * t * t);
            out.push(
                0.5 * (2. * p1
                    + (p2 - p0) * t
                    + (2. * p0 - 5. * p1 + 4. * p2 - p3) * t2
                    + (3. * p1 - p0 - 3. * p2 + p3) * t3),
            );
        }
    }
    out.push(pts[n - 1]);
    out
}

/// Closed outline of a band `hw` either side of the polyline: mitred joins
/// (bevelled when sharp), flat ends.
fn outline(pts: &[DVec2], hw: f64) -> Vec<DVec2> {
    let n = pts.len();
    let normal = |i: usize| (pts[i + 1] - pts[i]).normalize().perp();
    let side = |sign: f64| -> Vec<DVec2> {
        let mut out = Vec::with_capacity(n + 4);
        out.push(pts[0] + normal(0) * hw * sign);
        for i in 1..n - 1 {
            let (n1, n2) = (normal(i - 1) * sign, normal(i) * sign);
            let m = n1 + n2;
            let cos = if m.length() > 1e-9 { m.normalize().dot(n1) } else { 0. };
            if cos > 1. / MITER_LIMIT {
                out.push(pts[i] + m.normalize() * (hw / cos));
            } else {
                out.push(pts[i] + n1 * hw);
                out.push(pts[i] + n2 * hw);
            }
        }
        out.push(pts[n - 1] + normal(n - 2) * hw * sign);
        out
    };
    let mut poly = side(1.);
    poly.extend(side(-1.).into_iter().rev());
    poly
}

fn inside(p: DVec2, poly: &[DVec2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
    }
    inside
}

fn dist_to_segment(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let ab = b - a;
    let t = if ab.length_squared() > 0. { ((p - a).dot(ab) / ab.length_squared()).clamp(0., 1.) } else { 0. };
    p.distance(a + ab * t)
}

fn dist_to_polyline(p: DVec2, pts: &[DVec2], closed: bool) -> f64 {
    let n = pts.len();
    let edges = if closed { n } else { n.saturating_sub(1) };
    (0..edges)
        .map(|i| dist_to_segment(p, pts[i], pts[(i + 1) % n]))
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(poly: &[DVec2]) -> f64 {
        let n = poly.len();
        (0..n).map(|i| poly[i].perp_dot(poly[(i + 1) % n])).sum::<f64>().abs() / 2.
    }

    #[test]
    fn rect_corners_and_corner_drag() {
        let mut s = Shape::of(ShapeKind::Rect { center: [10., 0.], size: [4., 2.], angle: 0. });
        let Some(Geometry::Fill(p)) = s.geometry() else { panic!() };
        assert!((area(&p) - 8.).abs() < 1e-9);
        s.drag(Handle::Corner, DVec2::new(13., 2.), DVec2::ZERO);
        assert_eq!(s.kind, ShapeKind::Rect { center: [10., 0.], size: [6., 4.], angle: 0. });
        s.drag(Handle::Body, DVec2::ZERO, DVec2::new(1., 1.));
        assert!(s.hit(DVec2::new(11., 1.), 0.));
    }

    #[test]
    fn circle_area_is_close() {
        let s = Shape::of(ShapeKind::Ellipse { center: [0., 0.], radii: [10., 10.], angle: 0. });
        let Some(Geometry::Fill(p)) = s.geometry() else { panic!() };
        let exact = std::f64::consts::PI * 100.;
        assert!((area(&p) - exact).abs() / exact < 0.01);
    }

    #[test]
    fn thin_arc_is_a_stroke_and_wide_arc_a_band() {
        let mut s = Shape::of(ShapeKind::Arc { center: [0., 0.], radius: 10., start: 0., end: 90., width: 0. });
        let Some(Geometry::Stroke(line)) = s.geometry() else { panic!() };
        assert!(line[0].distance(DVec2::new(10., 0.)) < 1e-9);
        assert!(line.last().unwrap().distance(DVec2::new(0., 10.)) < 1e-9);
        if let ShapeKind::Arc { width, .. } = &mut s.kind {
            *width = 2.;
        }
        let Some(Geometry::Fill(band)) = s.geometry() else { panic!() };
        // Quarter annulus 9..11 nm.
        let exact = std::f64::consts::PI * (121. - 81.) / 4.;
        assert!((area(&band) - exact).abs() / exact < 0.01);
    }

    #[test]
    fn arc_end_keeps_direction() {
        let mut s = Shape::of(ShapeKind::Arc { center: [0., 0.], radius: 1., start: 0., end: 90., width: 0. });
        s.drag(Handle::ArcEnd, DVec2::new(0., -1.), DVec2::ZERO);
        assert!(matches!(s.kind, ShapeKind::Arc { end, .. } if (end - 270.).abs() < 1e-9));
    }

    #[test]
    fn wide_straight_line_is_a_rectangle() {
        let s = Shape::of(ShapeKind::Line { points: vec![[0., 0.], [10., 0.]], smooth: false, width: 2. });
        let Some(Geometry::Fill(p)) = s.geometry() else { panic!() };
        assert!((area(&p) - 20.).abs() < 1e-9);
        // A right-angle bend gets a mitred corner: 10×2 + 10×2 band.
        let s = Shape::of(ShapeKind::Line { points: vec![[0., 0.], [10., 0.], [10., 10.]], smooth: false, width: 2. });
        let Some(Geometry::Fill(p)) = s.geometry() else { panic!() };
        assert!((area(&p) - 40.).abs() < 1e-9);
    }

    #[test]
    fn curve_passes_through_its_points() {
        let s = Shape::new_curve(DVec2::ZERO, 100.);
        let ShapeKind::Line { points, .. } = &s.kind else { panic!() };
        let Some(Geometry::Stroke(line)) = s.geometry() else { panic!() };
        for p in points {
            assert!(line.iter().any(|q| q.distance(v(*p)) < 1e-9));
        }
    }

    #[test]
    fn points_insert_on_the_nearest_edge() {
        let mut s = Shape::new_line(DVec2::ZERO, 100.); // (-30,0) → (30,0)
        s.add_point(DVec2::new(0., 1.), 2.);
        s.add_point(DVec2::new(50., 0.), 2.);
        let ShapeKind::Line { points, .. } = &s.kind else { panic!() };
        assert_eq!(points, &vec![[-30., 0.], [0., 1.], [30., 0.], [50., 0.]]);
        assert!(s.remove_point(0));
        assert!(s.remove_point(0));
        assert!(!s.remove_point(0), "a line keeps two points");
    }

    #[test]
    fn json_round_trip() {
        let shapes = vec![Shape::new_rect(DVec2::ZERO, 100.), Shape::new_curve(DVec2::ONE, 50.)];
        let s = serde_json::to_string(&shapes).unwrap();
        assert_eq!(serde_json::from_str::<Vec<Shape>>(&s).unwrap(), shapes);
    }
}
