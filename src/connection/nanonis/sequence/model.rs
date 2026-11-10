use std::fmt;

use glam::{DAffine2, DVec2};
use nanonis_tcp::{
    commands::scan::{FrameGetResponse, FrameSetArgs},
    ScanDir,
};
use serde::{Deserialize, Serialize};

/// A planned series of scans. Each entry is one image; it expands into a
/// flat list of [`Step`]s which is what the runner actually executes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    pub entries: Vec<ScanEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScanEntry {
    pub name: String,
    pub enabled: bool,
    /// Bias in V. `None` leaves the bias unchanged.
    pub bias: Option<f32>,
    /// Z-controller setpoint (tunnelling current) in A. `None` leaves it unchanged.
    pub current: Option<f32>,
    /// Scan frame changes. Every part left unset keeps Nanonis's value.
    pub frame: FrameSpec,
    /// Pixels per line and number of lines. `None` keeps the current buffer.
    pub resolution: Option<[u32; 2]>,
    /// Scan speed. `None` keeps the current speed.
    pub speed: Option<Speed>,
    /// Coarse move performed before anything else in this entry.
    pub coarse: Option<CoarseSpec>,
    /// Wait after the coarse move / approach, in seconds.
    pub settle_s: f64,
    /// Wait after setting frame/bias/current, right before the scan, in seconds.
    pub wait_before_s: f64,
    /// Wait after the scan (and stamp), in seconds.
    pub wait_after_s: f64,
    pub dir: Dir,
    /// Stamp the finished image into the layer list.
    pub stamp: bool,
}
impl Default for ScanEntry {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: true,
            bias: None,
            current: None,
            frame: FrameSpec::default(),
            resolution: None,
            speed: None,
            coarse: None,
            settle_s: 0.,
            wait_before_s: 0.,
            wait_after_s: 0.,
            dir: Dir::Alternate,
            stamp: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dir {
    Up,
    Down,
    /// Opposite of the previous scan (resolved by [`compile`]).
    Alternate,
}
impl From<Dir> for ScanDir {
    fn from(d: Dir) -> Self {
        match d {
            Dir::Up => ScanDir::Up,
            // Alternate is resolved before it reaches the runner.
            Dir::Down | Dir::Alternate => ScanDir::Down,
        }
    }
}
impl From<ScanDir> for Dir {
    fn from(d: ScanDir) -> Self {
        match d {
            ScanDir::Up => Dir::Up,
            ScanDir::Down => Dir::Down,
        }
    }
}
impl Dir {
    pub fn arrow(self) -> &'static str {
        match self {
            Dir::Up => "↑",
            Dir::Down => "↓",
            Dir::Alternate => "⇅",
        }
    }
    pub fn flip(self) -> Self {
        match self {
            Dir::Up => Dir::Down,
            Dir::Down | Dir::Alternate => Dir::Up,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoarseSpec {
    /// Steps along the X and Y motor axes.
    pub steps: [i32; 2],
    pub withdraw: bool,
    pub approach: bool,
}
impl Default for CoarseSpec {
    fn default() -> Self {
        Self {
            steps: [0, 0],
            withdraw: true,
            approach: true,
        }
    }
}

/// Where to put the frame centre.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum Position {
    /// Leave the frame where it is (including manual moves made meanwhile).
    #[default]
    Keep,
    /// Absolute centre in nm.
    At([f64; 2]),
    /// Move the centre by this many nm from wherever it is.
    Shift([f64; 2]),
}

/// Partial frame change: unset parts keep the current Nanonis frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FrameSpec {
    pub position: Position,
    /// Width and height in nm.
    pub size: Option<[f64; 2]>,
    /// Degrees, positive = clockwise.
    pub angle: Option<f64>,
}
impl FrameSpec {
    pub fn is_keep(&self) -> bool {
        self.position == Position::Keep && self.size.is_none() && self.angle.is_none()
    }
    /// Fix the whole frame to `f`.
    pub fn capture(f: Frame) -> Self {
        Self {
            position: Position::At([f.center_x, f.center_y]),
            size: Some([f.width, f.height]),
            angle: Some(f.angle),
        }
    }
    pub fn apply(&self, mut f: Frame) -> Frame {
        match self.position {
            Position::Keep => {}
            Position::At([x, y]) => (f.center_x, f.center_y) = (x, y),
            Position::Shift([dx, dy]) => {
                f.center_x += dx;
                f.center_y += dy;
            }
        }
        if let Some([w, h]) = self.size {
            (f.width, f.height) = (w, h);
        }
        if let Some(a) = self.angle {
            f.angle = a;
        }
        f
    }
}
impl fmt::Display for FrameSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        match self.position {
            Position::Keep => {}
            Position::At([x, y]) => parts.push(format!("at ({x:.0}, {y:.0}) nm")),
            Position::Shift([dx, dy]) => parts.push(format!("shift {dx:+.0}, {dy:+.0} nm")),
        }
        if let Some([w, h]) = self.size {
            parts.push(format!("{w:.0}×{h:.0} nm"));
        }
        if let Some(a) = self.angle {
            parts.push(format!("∠{a:.0}°"));
        }
        if parts.is_empty() {
            write!(f, "keep")
        } else {
            write!(f, "{}", parts.join(" · "))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Speed {
    /// Forward time per line in s.
    LineTime(f64),
    /// Forward tip speed in nm/s.
    Linear(f64),
}
impl fmt::Display for Speed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Speed::LineTime(t) => write!(f, "{t} s/line"),
            Speed::Linear(v) => write!(f, "{v} nm/s"),
        }
    }
}

/// Scan frame in Nanonis piezo coordinates: nm and degrees
/// (positive angle = clockwise, as in Scan.FrameSet).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    pub angle: f64,
}
impl Frame {
    /// Inverse of [`Frame::to_transform`]; mirrors `frame_from_transform` in
    /// the fast status worker.
    pub fn from_transform(transform: DAffine2) -> Self {
        let (scale, mut angle, trans) = transform.to_scale_angle_translation();
        if scale.x < 0. {
            angle += std::f64::consts::PI;
        }
        Self {
            center_x: trans.x,
            center_y: trans.y,
            width: scale.x.abs(),
            height: scale.y.abs(),
            angle: normalize_degrees(-angle.to_degrees()),
        }
    }
    /// The image transform for this frame (same convention as
    /// `transform_from_frame` in the fast status worker).
    pub fn to_transform(&self) -> DAffine2 {
        DAffine2::from_scale_angle_translation(
            DVec2::new(self.width, -self.height),
            -self.angle.to_radians(),
            DVec2::new(self.center_x, self.center_y),
        )
    }
    pub fn from_get(r: &FrameGetResponse) -> Self {
        Self {
            center_x: r.center_x as f64 * 1e9,
            center_y: r.center_y as f64 * 1e9,
            width: r.width as f64 * 1e9,
            height: r.height as f64 * 1e9,
            angle: r.angle as f64,
        }
    }
    pub fn to_args(&self) -> FrameSetArgs {
        FrameSetArgs {
            center_x: (self.center_x * 1e-9) as f32,
            center_y: (self.center_y * 1e-9) as f32,
            width: (self.width * 1e-9) as f32,
            height: (self.height * 1e-9) as f32,
            angle: self.angle as f32,
        }
    }
}

/// Wrap an angle into (-180, 180].
fn normalize_degrees(a: f64) -> f64 {
    let r = a.rem_euclid(360.);
    if r > 180. {
        r - 360.
    } else {
        r
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Step {
    Withdraw,
    CoarseMove { steps: [i32; 2] },
    Approach,
    Wait(f64),
    SetFrame(FrameSpec),
    SetResolution([u32; 2]),
    SetSpeed(Speed),
    SetBias(f32),
    SetCurrent(f32),
    Scan(Dir),
    Stamp,
}
impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Step::Withdraw => write!(f, "Withdraw"),
            Step::CoarseMove { steps } => write!(f, "Coarse move {:+}, {:+}", steps[0], steps[1]),
            Step::Approach => write!(f, "Approach (Z-ctrl on)"),
            Step::Wait(s) => write!(f, "Wait {s} s"),
            Step::SetFrame(spec) => write!(f, "Frame: {spec}"),
            Step::SetResolution([px, lines]) => write!(f, "Resolution {px}×{lines}"),
            Step::SetSpeed(speed) => write!(f, "Speed {speed}"),
            Step::SetBias(v) => write!(f, "Set bias {} V", v),
            Step::SetCurrent(i) => write!(f, "Set current {} pA", i * 1e12),
            Step::Scan(d) => write!(f, "Scan {} (wait for end)", d.arrow()),
            Step::Stamp => write!(f, "Stamp image"),
        }
    }
}

/// Expand one entry into its flat step list. Order:
/// `[Withdraw] → [CoarseMove] → [Approach] → [Wait settle] → [SetFrame] →
/// [SetResolution] → [SetSpeed] → [SetBias] → [SetCurrent] → [Wait before] →
/// Scan → [Stamp] → [Wait after]`
///
/// The scan direction may still be [`Dir::Alternate`]; [`compile`] resolves it.
pub fn expand(entry: &ScanEntry) -> Vec<Step> {
    let mut steps = Vec::new();
    if let Some(coarse) = entry.coarse {
        if coarse.withdraw {
            steps.push(Step::Withdraw);
        }
        if coarse.steps != [0, 0] {
            steps.push(Step::CoarseMove {
                steps: coarse.steps,
            });
        }
        if coarse.approach {
            steps.push(Step::Approach);
        }
    }
    if entry.settle_s > 0. {
        steps.push(Step::Wait(entry.settle_s));
    }
    if !entry.frame.is_keep() {
        steps.push(Step::SetFrame(entry.frame));
    }
    if let Some(res) = entry.resolution {
        steps.push(Step::SetResolution(res));
    }
    // After the frame: the speed conversion uses the frame width.
    if let Some(speed) = entry.speed {
        steps.push(Step::SetSpeed(speed));
    }
    if let Some(bias) = entry.bias {
        steps.push(Step::SetBias(bias));
    }
    if let Some(current) = entry.current {
        steps.push(Step::SetCurrent(current));
    }
    if entry.wait_before_s > 0. {
        steps.push(Step::Wait(entry.wait_before_s));
    }
    steps.push(Step::Scan(entry.dir));
    if entry.stamp {
        steps.push(Step::Stamp);
    }
    if entry.wait_after_s > 0. {
        steps.push(Step::Wait(entry.wait_after_s));
    }
    steps
}

/// Expand every enabled entry from `from_entry` on, tagging each step with
/// its entry index. `last_dir` is the direction of the scan before the first
/// one; [`Dir::Alternate`] scans go opposite to the scan before them.
pub fn compile(seq: &Sequence, from_entry: usize, last_dir: Dir) -> Vec<(usize, Step)> {
    let mut last = last_dir;
    let mut out = Vec::new();
    for (i, e) in seq.entries.iter().enumerate().skip(from_entry) {
        if !e.enabled {
            continue;
        }
        for mut step in expand(e) {
            if let Step::Scan(d) = &mut step {
                if *d == Dir::Alternate {
                    *d = last.flip();
                }
                last = *d;
            }
            out.push((i, step));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_entry_is_scan_and_stamp() {
        let e = ScanEntry::default();
        assert_eq!(expand(&e), vec![Step::Scan(Dir::Alternate), Step::Stamp]);
    }

    #[test]
    fn full_entry_order() {
        let frame = FrameSpec {
            position: Position::At([1., 2.]),
            size: Some([100., 100.]),
            angle: None,
        };
        let e = ScanEntry {
            bias: Some(-1.),
            current: Some(50e-12),
            frame,
            resolution: Some([256, 256]),
            speed: Some(Speed::LineTime(1.)),
            coarse: Some(CoarseSpec {
                steps: [20, 0],
                withdraw: true,
                approach: true,
            }),
            settle_s: 120.,
            wait_before_s: 5.,
            wait_after_s: 1.,
            dir: Dir::Up,
            ..Default::default()
        };
        assert_eq!(
            expand(&e),
            vec![
                Step::Withdraw,
                Step::CoarseMove { steps: [20, 0] },
                Step::Approach,
                Step::Wait(120.),
                Step::SetFrame(frame),
                Step::SetResolution([256, 256]),
                Step::SetSpeed(Speed::LineTime(1.)),
                Step::SetBias(-1.),
                Step::SetCurrent(50e-12),
                Step::Wait(5.),
                Step::Scan(Dir::Up),
                Step::Stamp,
                Step::Wait(1.),
            ]
        );
    }

    #[test]
    fn compile_skips_disabled() {
        let seq = Sequence {
            entries: vec![
                ScanEntry::default(),
                ScanEntry {
                    enabled: false,
                    ..Default::default()
                },
                ScanEntry {
                    stamp: false,
                    ..Default::default()
                },
            ],
        };
        let c = compile(&seq, 0, Dir::Down);
        assert_eq!(
            c,
            vec![
                (0, Step::Scan(Dir::Up)),
                (0, Step::Stamp),
                (2, Step::Scan(Dir::Down)),
            ]
        );
    }

    #[test]
    fn alternate_follows_previous_scan() {
        let seq = Sequence {
            entries: vec![
                ScanEntry::default(),
                ScanEntry {
                    dir: Dir::Down,
                    ..Default::default()
                },
                ScanEntry::default(),
                ScanEntry::default(),
            ],
        };
        let dirs: Vec<_> = compile(&seq, 0, Dir::Down)
            .into_iter()
            .filter_map(|(_, s)| match s {
                Step::Scan(d) => Some(d),
                _ => None,
            })
            .collect();
        assert_eq!(dirs, vec![Dir::Up, Dir::Down, Dir::Up, Dir::Down]);
        // Starting later only includes later entries.
        assert_eq!(compile(&seq, 2, Dir::Up)[0], (2, Step::Scan(Dir::Down)));
    }

    #[test]
    fn frame_spec_apply() {
        let f = Frame {
            center_x: 10.,
            center_y: 20.,
            width: 100.,
            height: 100.,
            angle: 0.,
        };
        assert_eq!(FrameSpec::default().apply(f), f);
        let g = FrameSpec {
            position: Position::Shift([5., -5.]),
            size: Some([50., 60.]),
            angle: None,
        }
        .apply(f);
        assert_eq!(
            (g.center_x, g.center_y, g.width, g.height),
            (15., 15., 50., 60.)
        );
        assert_eq!(FrameSpec::capture(f).apply(g), f);
    }

    #[test]
    fn json_round_trip() {
        let seq = Sequence {
            entries: vec![ScanEntry {
                name: "a".into(),
                bias: Some(0.5),
                coarse: Some(CoarseSpec::default()),
                frame: FrameSpec {
                    position: Position::Shift([1., 2.]),
                    size: Some([10., 20.]),
                    angle: Some(15.),
                },
                speed: Some(Speed::Linear(20.)),
                resolution: Some([128, 64]),
                ..Default::default()
            }],
        };
        let s = serde_json::to_string(&seq).unwrap();
        assert_eq!(serde_json::from_str::<Sequence>(&s).unwrap(), seq);
    }

    #[test]
    fn frame_transform_round_trip() {
        let f = Frame {
            center_x: 12.,
            center_y: -7.,
            width: 100.,
            height: 50.,
            angle: 30.,
        };
        let g = Frame::from_transform(f.to_transform());
        for (a, b) in [
            (f.center_x, g.center_x),
            (f.center_y, g.center_y),
            (f.width, g.width),
            (f.height, g.height),
            (f.angle, g.angle),
        ] {
            assert!((a - b).abs() < 1e-9, "{f:?} vs {g:?}");
        }
    }
}
