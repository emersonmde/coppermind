//! App shell components: AppBar, MetricsPane, Footer
//!
//! These components form the persistent UI framework around the main content area.

mod appbar;
mod footer;
mod metrics_pane;

pub use appbar::{AppBar, View};
pub use footer::Footer;
pub use metrics_pane::MetricsPane;
