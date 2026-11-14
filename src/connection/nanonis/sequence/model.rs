use std::fmt;

use glam::{DAffine2, DVec2};
use nanonis_tcp::{
    commands::scan::{FrameGetResponse, FrameSetArgs},
    ScanDir,
};
use serde::{Deserialize, Serialize};

use crate::connection::nanonis::spectroscopy::IvSeriesSettings;

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
    /// Take the scan. Off: the entry only sets things up, moves the tip and
    /// runs spectroscopy.
    pub scan: bool,
    /// Tip positions visited after the scan, each with an optional I(V) series.
    pub tip: TipPlan,
}

/// Fine tip moves (Follow Me) after an entry's scan, with an optional I(V)
/// series at each position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TipPlan {
    /// Tip positions in piezo nm (same coordinates as the frame centre).
    pub points: Vec<[f64; 2]>,
    /// Run an I(V) series at each position; off just moves the tip there.
    pub spectroscopy: bool,
    /// Series settings. Each position gets its own base name, `<base>_P01`…
    pub iv: IvSeriesSettings,
    /// Wait after the tip arrives at each position, in s.
    pub settle_s: f64,
}
impl Default for TipPlan {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            spectroscopy: true,
            iv: IvSeriesSettings::default(),
            settle_s: 1.,
        }
    }
}
impl TipPlan {
    /// Base name of the series at position `k` (0-based) of entry `name`.
    pub fn point_base_name(&self, entry_name: &str, k: usize) -> String {
        let base = self.iv.base_name.trim();
        let name = entry_name.trim().replace(char::is_whitespace, "_");
        if name.is_empty() {
            format!("{base}_P{:02}", k + 1)
        } else {
            format!("{base}_{name}_P{:02}", k + 1)
        }
    }
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
            scan: true,
            tip: TipPlan::default(),
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

#[derive(Debug, Clone, PartialEq)]
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
    /// Follow Me: move the tip to this piezo position (nm) and wait until it
    /// arrives.
    MoveTip([f64; 2]),
    /// An I(V) series where the tip is; saved automatically.
    Spectroscopy(IvSeriesSettings),
    /// Wait for the scan already running in Nanonis (started by hand) to end,
    /// without starting one. `include`: the scan is part of the sequence, so
    /// Pause/Abort act on it; otherwise the sequence only queues behind it
    /// and Abort leaves it running. Done at once if no scan is running.
    AwaitScan { dir: Dir, include: bool },
}

/// Entry index of the steps that belong to the scan that was already
/// running when the sequence started (see [`with_current_scan`]).
pub const CURRENT_SCAN: usize = usize::MAX;

/// What to do with a scan that is running when the sequence starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrentScan {
    /// Start the sequence once it ends.
    After,
    /// Make it the first scan of the sequence (stamped, pausable).
    Include,
}

/// Put the handling of the running scan (direction `dir`) in front of
/// `steps`. Compile `steps` with `dir` as the last direction so alternating
/// entries follow on from it.
pub fn with_current_scan(
    steps: Vec<(usize, Step)>,
    mode: CurrentScan,
    dir: Dir,
    stamp: bool,
) -> Vec<(usize, Step)> {
    let include = mode == CurrentScan::Include;
    let mut out = vec![(CURRENT_SCAN, Step::AwaitScan { dir, include })];
    if include && stamp {
        out.push((CURRENT_SCAN, Step::Stamp));
    }
    out.extend(steps);
    out
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
            Step::MoveTip([x, y]) => write!(f, "Tip to ({x:.1}, {y:.1}) nm"),
            Step::Spectroscopy(iv) => write!(
                f,
                "I(V) series {}: {} curves, {:+} pm/curve",
                iv.base_name, iv.num_curves, iv.z_step_pm
            ),
            Step::AwaitScan { dir, include: true } => {
                write!(f, "Current scan {} (wait for end)", dir.arrow())
            }
            Step::AwaitScan { include: false, .. } => write!(f, "Wait for the current scan to end"),
        }
    }
}

/// Expand one entry into its flat step list. Order:
/// `[Withdraw] → [CoarseMove] → [Approach] → [Wait settle] → [SetFrame] →
/// [SetResolution] → [SetSpeed] → [SetBias] → [SetCurrent] → [Wait before] →
/// [Scan → [Stamp]] → per tip position: MoveTip → [Wait settle] →
/// [Spectroscopy] → [Wait after]`
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
    if entry.scan {
        steps.push(Step::Scan(entry.dir));
        if entry.stamp {
            steps.push(Step::Stamp);
        }
    }
    let tip = &entry.tip;
    for (k, &p) in tip.points.iter().enumerate() {
        steps.push(Step::MoveTip(p));
        if tip.settle_s > 0. {
            steps.push(Step::Wait(tip.settle_s));
        }
        if tip.spectroscopy {
            let mut iv = tip.iv.clone();
            iv.base_name = tip.point_base_name(&entry.name, k);
            steps.push(Step::Spectroscopy(iv));
        }
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
    fn tip_points_follow_the_scan() {
        let mut e = ScanEntry {
            name: "row 1".into(),
            wait_after_s: 2.,
            ..Default::default()
        };
        e.tip.points = vec![[1., 2.], [3., 4.]];
        e.tip.settle_s = 0.5;
        e.tip.iv.base_name = "IV".into();
        let steps = expand(&e);
        let mut iv1 = e.tip.iv.clone();
        iv1.base_name = "IV_row_1_P01".into();
        let mut iv2 = e.tip.iv.clone();
        iv2.base_name = "IV_row_1_P02".into();
        assert_eq!(
            steps,
            vec![
                Step::Scan(Dir::Alternate),
                Step::Stamp,
                Step::MoveTip([1., 2.]),
                Step::Wait(0.5),
                Step::Spectroscopy(iv1),
                Step::MoveTip([3., 4.]),
                Step::Wait(0.5),
                Step::Spectroscopy(iv2),
                Step::Wait(2.),
            ]
        );
        // No scan, tip moves only.
        e.scan = false;
        e.tip.spectroscopy = false;
        e.tip.settle_s = 0.;
        e.wait_after_s = 0.;
        assert_eq!(expand(&e), vec![Step::MoveTip([1., 2.]), Step::MoveTip([3., 4.])]);
    }

    #[test]
    fn old_files_load_with_a_scan_and_no_tip_moves() {
        let seq: Sequence = serde_json::from_str(r#"{"entries":[{"name":"a"}]}"#).unwrap();
        assert!(seq.entries[0].scan);
        assert!(seq.entries[0].tip.points.is_empty());
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
    fn current_scan_goes_first() {
        let seq = Sequence {
            entries: vec![ScanEntry::default()],
        };
        let steps = with_current_scan(compile(&seq, 0, Dir::Up), CurrentScan::Include, Dir::Up, true);
        assert_eq!(
            steps,
            vec![
                (CURRENT_SCAN, Step::AwaitScan { dir: Dir::Up, include: true }),
                (CURRENT_SCAN, Step::Stamp),
                (0, Step::Scan(Dir::Down)),
                (0, Step::Stamp),
            ]
        );
        let steps = with_current_scan(compile(&seq, 0, Dir::Up), CurrentScan::After, Dir::Up, true);
        assert_eq!(steps[0], (CURRENT_SCAN, Step::AwaitScan { dir: Dir::Up, include: false }));
        assert_eq!(steps[1], (0, Step::Scan(Dir::Down)));
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
