//! Hybrid search engine combining vector and keyword search.
//!
//! This module re-exports the search functionality from `coppermind_core`
//! and adds app-specific error conversions.
//!
//! See [`coppermind_core::search`] for the full implementation details.

// Re-export everything from core
pub use coppermind_core::search::*;

// App-specific error conversions (error types live in app crate)
use crate::error::{EmbeddingError, FileProcessingError};

impl From<EmbeddingError> for SearchError {
    fn from(e: EmbeddingError) -> Self {
        SearchError::EmbeddingError(e.to_string())
    }
}

impl From<FileProcessingError> for SearchError {
    fn from(e: FileProcessingError) -> Self {
        SearchError::IndexError(e.to_string())
    }
}
