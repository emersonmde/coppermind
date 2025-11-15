//! App shell components: AppBar, StatusStrip, MetricsPane, Footer
//!
//! These components form the persistent UI framework around the main content area.

mod appbar;
mod footer;
mod metrics_pane;
mod status_strip;

pub use appbar::{AppBar, View};
pub use footer::Footer;
pub use metrics_pane::MetricsPane;
pub use status_strip::StatusStrip;
