//! Automated scan sequences: plan a list of scan entries (bias, current,
//! frame, coarse moves, waits) and run them unattended on a dedicated worker.

mod editor;
mod model;
mod overlay;
mod runner;

pub use editor::{SequenceAction, SequenceState};
pub(crate) use model::Frame;
pub use runner::SequenceWorker;
