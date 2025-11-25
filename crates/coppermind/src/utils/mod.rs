//! Utility modules for common patterns.
//!
//! This module provides reusable utilities that eliminate boilerplate and
//! improve code quality across the codebase.

pub mod error_ext;
pub mod signal_ext;

// Re-export commonly used items
pub use error_ext::ResultExt;
pub use signal_ext::SignalExt;
