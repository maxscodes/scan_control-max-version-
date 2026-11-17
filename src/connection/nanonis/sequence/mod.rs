//! Automated scan sequences: plan a list of scan entries (bias, current,
//! frame, coarse moves, waits) and run them unattended on a dedicated worker.

mod checkpoint;
mod editor;
mod eta;
mod grid_track;
mod model;
mod overlay;
mod runner;

pub use editor::{SequenceAction, SequenceState};
pub(crate) use eta::fmt_duration;
pub(crate) use model::{CoarseSpec, Dir, Frame, FrameSpec, Position, ScanEntry, ScanStart, Sequence, Speed};
#[cfg(test)]
pub(crate) use model::{compile, Step};
pub use runner::SequenceWorker;
