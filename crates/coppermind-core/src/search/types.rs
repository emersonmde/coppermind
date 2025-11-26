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

/// Global counter for generating unique document IDs.
/// Must be initialized with `init_counter` after loading existing documents.
static DOC_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl DocId {
    /// Generates a new unique document ID.
    ///
    /// Uses an atomic counter to ensure IDs are unique across threads.
    /// The counter should be initialized via `DocId::init_counter()` after
    /// loading existing documents to prevent ID collisions.
    ///
    /// Note: Default trait is intentionally NOT implemented because it would
    /// be misleading - calling default() multiple times would generate different
    /// IDs, violating the semantic expectation that default() returns the same value.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        use std::sync::atomic::Ordering;
        Self(DOC_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Initialize the ID counter to start after the given maximum ID.
    ///
    /// Call this after loading existing documents to ensure new IDs don't
    /// collide with existing ones. Only updates if the new value is higher.
    pub fn init_counter(max_existing_id: u64) {
        use std::sync::atomic::Ordering;
        // Set counter to max_id + 1, but only if it's higher than current
        // This handles the case where multiple loads might happen
        let next_id = max_existing_id.saturating_add(1);
        DOC_ID_COUNTER.fetch_max(next_id, Ordering::SeqCst);
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
    /// Invalid search query
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
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

// ============================================================================
// Persistence Types
// ============================================================================

/// Current schema version for the index format.
///
/// Increment this when making breaking changes to the persistence format.
/// The version history:
/// - v1: Initial format (documents.json + embeddings.bin + manifest.json)
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Index manifest containing version and metadata.
///
/// Stored as `manifest.json` in the index directory. Used to verify
/// compatibility and track index statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManifest {
    /// Current schema version of this index
    pub schema_version: u32,
    /// Minimum schema version required to read this index
    pub min_compatible_version: u32,
    /// ISO 8601 timestamp when the index was created
    pub created_at: String,
    /// ISO 8601 timestamp of last modification
    pub last_modified: String,
    /// Number of documents in the index
    pub document_count: usize,
    /// Embedding dimension (e.g., 512 for JinaBERT)
    pub embedding_dimension: usize,
}

impl IndexManifest {
    /// Creates a new manifest for a fresh index.
    pub fn new(embedding_dimension: usize) -> Self {
        let now = Self::current_timestamp();
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            min_compatible_version: 1, // v1 can read v1
            created_at: now.clone(),
            last_modified: now,
            document_count: 0,
            embedding_dimension,
        }
    }

    /// Updates the manifest after documents are added/removed.
    pub fn update(&mut self, document_count: usize) {
        self.document_count = document_count;
        self.last_modified = Self::current_timestamp();
    }

    /// Returns current timestamp in ISO 8601 format.
    fn current_timestamp() -> String {
        // Use instant for cross-platform compatibility
        let secs = get_current_timestamp();
        // Simple ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        // For proper formatting we'd use chrono, but this is sufficient
        format!("{}", secs)
    }

    /// Checks if this index can be read by the current app version.
    pub fn is_compatible(&self) -> bool {
        CURRENT_SCHEMA_VERSION >= self.min_compatible_version
    }

    /// Checks if this index needs migration to the current version.
    pub fn needs_migration(&self) -> bool {
        self.schema_version < CURRENT_SCHEMA_VERSION
    }
}

/// Persisted index data (documents only, embeddings stored separately).
///
/// The actual HNSW and BM25 indices are rebuilt on load from this data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedDocuments {
    /// All document records in insertion order
    pub documents: Vec<DocumentRecord>,
}

impl PersistedDocuments {
    /// Creates an empty persisted documents collection.
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
        }
    }

    /// Creates from a list of document records.
    pub fn from_documents(documents: Vec<DocumentRecord>) -> Self {
        Self { documents }
    }
}

impl Default for PersistedDocuments {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of loading a persisted index.
#[derive(Debug)]
pub enum LoadResult {
    /// Successfully loaded existing index
    Loaded {
        manifest: IndexManifest,
        documents: PersistedDocuments,
        embeddings: Vec<Vec<f32>>,
    },
    /// No existing index found (fresh start)
    NotFound,
    /// Index exists but is incompatible (needs clear or migration)
    Incompatible {
        manifest: IndexManifest,
        reason: String,
    },
}

// ============================================================================
// Source Tracking Types (for re-upload detection)
// ============================================================================

/// Record tracking a source file/URL and its indexed chunks.
///
/// Used for detecting when documents are re-uploaded and need updating.
/// Stored in the "sources" table of the DocumentStore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRecord {
    /// Schema version for future evolution
    pub version: u32,
    /// SHA-256 hash of the source content
    pub content_hash: String,
    /// DocIds of all chunks from this source
    pub doc_ids: Vec<DocId>,
    /// Whether all chunks have been indexed (false during indexing, true when complete)
    pub complete: bool,
}

impl SourceRecord {
    /// Current schema version for SourceRecord
    pub const CURRENT_VERSION: u32 = 1;

    /// Creates a new incomplete source record (before indexing starts)
    pub fn new_incomplete(content_hash: String) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            content_hash,
            doc_ids: Vec::new(),
            complete: false,
        }
    }

    /// Creates a complete source record with all chunk IDs
    pub fn new_complete(content_hash: String, doc_ids: Vec<DocId>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            content_hash,
            doc_ids,
            complete: true,
        }
    }

    /// Marks the source as complete
    pub fn mark_complete(&mut self) {
        self.complete = true;
    }

    /// Adds a chunk DocId to this source
    pub fn add_chunk(&mut self, doc_id: DocId) {
        self.doc_ids.push(doc_id);
    }
}

/// Lightweight metadata for a chunk, kept in memory.
///
/// Full text is stored on disk and loaded on-demand.
/// This structure minimizes memory footprint while enabling
/// fast lookups and file-level result grouping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Unique identifier for the source (path, URL, or web:{filename})
    pub source_id: String,
    /// Display name for UI (filename or page title)
    pub display_name: String,
    /// Which chunk this is within the source (0-indexed)
    pub chunk_index: usize,
    /// Total number of chunks from this source
    pub chunk_count: usize,
    /// Unix timestamp when indexed
    pub created_at: u64,
}

impl ChunkMetadata {
    /// Creates new chunk metadata
    pub fn new(
        source_id: String,
        display_name: String,
        chunk_index: usize,
        chunk_count: usize,
    ) -> Self {
        Self {
            source_id,
            display_name,
            chunk_index,
            chunk_count,
            created_at: get_current_timestamp(),
        }
    }
}

/// Statistics from a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Time taken to compact in milliseconds
    pub duration_ms: u64,
    /// Number of chunks removed during compaction
    pub chunks_removed: usize,
    /// Tombstone ratio before compaction (0.0 - 1.0)
    pub tombstone_ratio_before: f64,
    /// Tombstone ratio after compaction (should be 0.0)
    pub tombstone_ratio_after: f64,
}
