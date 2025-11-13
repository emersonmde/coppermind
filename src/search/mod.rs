// Public API for hybrid search
pub mod types;

// Internal modules (will be implemented in phases)
mod engine;
mod fusion;
mod keyword;
mod vector;

// Re-export main types (public API)
#[allow(unused_imports)]
pub use types::{DocId, Document, DocumentMetadata, DocumentRecord, SearchError, SearchResult};

// Re-export search engine (will implement later)
pub use engine::HybridSearchEngine;
