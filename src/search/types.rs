use serde::{Deserialize, Serialize};

/// Returns the current Unix timestamp (seconds since UNIX_EPOCH).
///
/// Platform-specific implementation:
/// - **Web**: Uses `instant::SystemTime` (WASM-compatible)
/// - **Desktop**: Uses `std::time::SystemTime`
///
/// If the system time is before UNIX_EPOCH (extremely unlikely),
/// returns 0 instead of panicking.
#[cfg(target_arch = "wasm32")]
pub fn get_current_timestamp() -> u64 {
    instant::SystemTime::now()
        .duration_since(instant::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Returns the current Unix timestamp (seconds since UNIX_EPOCH).
///
/// Platform-specific implementation:
/// - **Web**: Uses `instant::SystemTime` (WASM-compatible)
/// - **Desktop**: Uses `std::time::SystemTime`
///
/// If the system time is before UNIX_EPOCH (extremely unlikely),
/// returns 0 instead of panicking.
#[cfg(not(target_arch = "wasm32"))]
pub fn get_current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Unique document identifier.
///
/// IDs are generated atomically to ensure uniqueness across threads.
/// Use `DocId::new()` to generate a new unique ID.
///
/// # Examples
///
/// ```ignore
/// let id = DocId::new();  // Generates unique ID (e.g., 0)
/// let another = DocId::new();  // Generates next ID (e.g., 1)
/// assert_ne!(id, another);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocId(u64);

impl DocId {
    /// Generates a new unique document ID.
    ///
    /// Uses an atomic counter to ensure IDs are unique across threads.
    ///
    /// Note: Default trait is intentionally NOT implemented because it would
    /// be misleading - calling default() multiple times would generate different
    /// IDs, violating the semantic expectation that default() returns the same value.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Creates a DocId from a raw u64 value.
    ///
    /// Useful for deserialization or testing. Be careful not to create
    /// duplicate IDs when using this method.
    pub fn from_u64(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw u64 value of this ID.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Document with text content and metadata.
///
/// Represents a document to be indexed for search. The text field contains
/// the searchable content, while metadata provides additional context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document text content (will be tokenized for search)
    pub text: String,
    /// Associated metadata (filename, source, timestamp)
    pub metadata: DocumentMetadata,
}

/// Document metadata.
///
/// Contains optional filename, source information, and creation timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Original filename if available
    pub filename: Option<String>,
    /// Source path or URL
    pub source: Option<String>,
    /// Unix timestamp (seconds since UNIX_EPOCH)
    pub created_at: u64,
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        Self {
            filename: None,
            source: None,
            created_at: get_current_timestamp(),
        }
    }
}

/// Stored document record with assigned ID.
///
/// Internal representation of a document after it's been indexed.
/// Contains the assigned DocId plus the original document and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    /// Unique document identifier
    pub id: DocId,
    /// Document text content
    pub text: String,
    /// Document metadata
    pub metadata: DocumentMetadata,
}

/// Search result with relevance scores.
///
/// Returned by the hybrid search engine, containing both the document
/// and relevance scores from different ranking algorithms.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document identifier
    pub doc_id: DocId,
    /// Final fused relevance score (from RRF algorithm)
    pub score: f32,
    /// Semantic similarity score (from vector search, if available)
    pub vector_score: Option<f32>,
    /// Keyword match score (from BM25, if available)
    pub keyword_score: Option<f32>,
    /// Document text content
    pub text: String,
    /// Document metadata
    pub metadata: DocumentMetadata,
}

/// Error types for search operations.
#[derive(Debug, Clone)]
pub enum SearchError {
    /// Storage backend error
    StorageError(String),
    /// Embedding generation error
    EmbeddingError(String),
    /// Index construction or query error
    IndexError(String),
    /// Document not found in index
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

/// Convert EmbeddingError to SearchError
impl From<crate::error::EmbeddingError> for SearchError {
    fn from(err: crate::error::EmbeddingError) -> Self {
        SearchError::EmbeddingError(err.to_string())
    }
}

/// Convert FileProcessingError to SearchError
impl From<crate::error::FileProcessingError> for SearchError {
    fn from(err: crate::error::FileProcessingError) -> Self {
        SearchError::IndexError(err.to_string())
    }
}
