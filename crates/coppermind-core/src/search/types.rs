use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Returns the current Unix timestamp (seconds since UNIX_EPOCH).
///
/// Uses `instant::SystemTime` which provides cross-platform timing
/// (works on both WASM and native platforms).
///
/// If the system time is before UNIX_EPOCH (extremely unlikely),
/// returns 0 instead of panicking.
pub fn get_current_timestamp() -> u64 {
    instant::SystemTime::now()
        .duration_since(instant::SystemTime::UNIX_EPOCH)
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
///
/// Represents a single chunk from the search index.
#[derive(Debug, Clone, PartialEq)]
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

/// File-level search result aggregating multiple chunks from the same source.
///
/// Used for displaying search results at the file level (like Google shows pages)
/// rather than individual chunks (paragraphs). This provides a cleaner UX while
/// preserving access to individual chunk scores and content.
#[derive(Debug, Clone, PartialEq)]
pub struct FileSearchResult {
    /// Source file path or URL (from metadata.source)
    pub file_path: String,
    /// File display name (extracted from path)
    pub file_name: String,
    /// Best chunk's RRF score (represents file relevance)
    pub score: f32,
    /// Best chunk's vector score (if available)
    pub vector_score: Option<f32>,
    /// Best chunk's keyword score (if available)
    pub keyword_score: Option<f32>,
    /// All chunks from this file, sorted by score (descending)
    pub chunks: Vec<SearchResult>,
    /// Timestamp from first indexed chunk
    pub created_at: u64,
}

/// Error types for search operations.
#[derive(Debug, Clone, Error)]
pub enum SearchError {
    /// Storage backend error
    #[error("Storage error: {0}")]
    StorageError(String),
    /// Embedding generation error
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
    /// Index construction or query error
    #[error("Index error: {0}")]
    IndexError(String),
    /// Document not found in index
    #[error("Not found")]
    NotFound,
    /// Vector dimension mismatch (expected vs actual)
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected embedding dimension
        expected: usize,
        /// Actual embedding dimension received
        actual: usize,
    },
}

/// Validates that an embedding has the expected dimension.
///
/// Returns `Ok(())` if dimensions match, or `Err(SearchError::DimensionMismatch)` otherwise.
///
/// # Examples
///
/// ```ignore
/// use coppermind_core::search::types::validate_dimension;
///
/// let embedding = vec![1.0, 2.0, 3.0];
/// validate_dimension(3, embedding.len())?; // Ok
/// validate_dimension(5, embedding.len())?; // Err(DimensionMismatch)
/// ```
pub fn validate_dimension(expected: usize, actual: usize) -> Result<(), SearchError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SearchError::DimensionMismatch { expected, actual })
    }
}

// Note: From<EmbeddingError> and From<FileProcessingError> impls are in the app crate
// since those error types haven't been migrated to core yet.

/// Convert String to SearchError for platform::run_blocking compatibility
impl From<String> for SearchError {
    fn from(s: String) -> Self {
        SearchError::IndexError(s)
    }
}
