use serde::{Deserialize, Serialize};

/// Unique document identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocId(u64);

impl DocId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    pub fn from_u64(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl Default for DocId {
    fn default() -> Self {
        Self::new()
    }
}

/// Document with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub text: String,
    pub metadata: DocumentMetadata,
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub filename: Option<String>,
    pub source: Option<String>,
    pub created_at: u64, // Unix timestamp
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        Self {
            filename: None,
            source: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

/// Stored document record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: DocId,
    pub text: String,
    pub metadata: DocumentMetadata,
}

/// Search result with score
#[derive(Debug, Clone)]
#[allow(dead_code)] // Public API fields
pub struct SearchResult {
    pub doc_id: DocId,
    pub score: f32,                 // RRF fused score
    pub vector_score: Option<f32>,  // Semantic similarity score
    pub keyword_score: Option<f32>, // BM25 score
    pub text: String,
    pub metadata: DocumentMetadata,
}

/// Error types for search operations
#[derive(Debug)]
#[allow(dead_code)] // Public API variants
pub enum SearchError {
    StorageError(String),
    EmbeddingError(String),
    IndexError(String),
    NotFound,
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::StorageError(e) => write!(f, "Storage error: {}", e),
            SearchError::EmbeddingError(e) => write!(f, "Embedding error: {}", e),
            SearchError::IndexError(e) => write!(f, "Index error: {}", e),
            SearchError::NotFound => write!(f, "Not found"),
        }
    }
}

impl std::error::Error for SearchError {}
