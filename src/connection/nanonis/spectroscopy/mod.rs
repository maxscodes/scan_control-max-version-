//! I(V) spectroscopy: a series of bias sweeps at stepped tip heights, run on
//! its own worker, plotted live and exportable as .dat files. The series is
//! also run by the sequence runner at tip positions between scans.

mod combined;
mod model;
mod window;
mod worker;

pub(crate) use combined::{save_batch, SharedPlotChoice};
pub(crate) use model::{IvCurve, IvSeriesSettings};
pub use window::SpectroscopyState;
pub(crate) use worker::{nanonis_saved_files, run_iv_series, SeriesHost, SeriesOutcome};
pub use worker::SpectroscopyWorker;
