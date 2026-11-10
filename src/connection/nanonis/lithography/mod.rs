//! GDS-guided hydrogen-desorption lithography: plan a tip path over the GDS
//! polygons inside the scan frame and execute it with Follow Me moves.
//! Ported from the Python lithography GUI v1.18.

mod path;
mod window;
mod worker;

pub use window::LithographyState;
pub use worker::LithographyWorker;
