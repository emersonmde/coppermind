//! Utility modules for common patterns.
//!
//! This module provides reusable utilities that eliminate boilerplate and
//! improve code quality across the codebase.

pub mod error_ext;
pub mod formatting;
pub mod signal_ext;

// Re-export commonly used items
pub use error_ext::ResultExt;
pub use formatting::{format_duration, format_timestamp};
pub use signal_ext::SignalExt;
