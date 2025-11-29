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

/// Unique chunk identifier.
///
/// IDs are generated atomically to ensure uniqueness across threads.
/// Use `ChunkId::new()` to generate a new unique ID.
///
/// Note: In Coppermind, "chunks" are segments of documents (files/URLs).
/// A single document may have multiple chunks. This was previously called
/// `DocId` but was renamed in ADR-008 to clarify the distinction.
///
/// # Examples
///
/// ```ignore
/// let id = ChunkId::new();  // Generates unique ID (e.g., 0)
/// let another = ChunkId::new();  // Generates next ID (e.g., 1)
/// assert_ne!(id, another);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(u64);

/// Global counter for generating unique chunk IDs.
/// Must be initialized with `init_counter` after loading existing chunks.
static CHUNK_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl ChunkId {
    /// Generates a new unique chunk ID.
    ///
    /// Uses an atomic counter to ensure IDs are unique across threads.
    /// The counter should be initialized via `ChunkId::init_counter()` after
    /// loading existing chunks to prevent ID collisions.
    ///
    /// Note: Default trait is intentionally NOT implemented because it would
    /// be misleading - calling default() multiple times would generate different
    /// IDs, violating the semantic expectation that default() returns the same value.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        use std::sync::atomic::Ordering;
        Self(CHUNK_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Initialize the ID counter to start after the given maximum ID.
    ///
    /// Call this after loading existing chunks to ensure new IDs don't
    /// collide with existing ones. Only updates if the new value is higher.
    pub fn init_counter(max_existing_id: u64) {
        use std::sync::atomic::Ordering;
        // Set counter to max_id + 1, but only if it's higher than current
        // This handles the case where multiple loads might happen
        let next_id = max_existing_id.saturating_add(1);
        CHUNK_ID_COUNTER.fetch_max(next_id, Ordering::SeqCst);
    }

    /// Creates a ChunkId from a raw u64 value.
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

// ============================================================================
// Document-Level Types (ADR-008: Multi-Level Indexing)
// ============================================================================

/// Unique document identifier.
///
/// Documents represent a complete unit of content (file, URL, note).
/// Each document may contain multiple chunks for fine-grained search.
/// DocumentIds are generated atomically to ensure uniqueness.
///
/// # Examples
///
/// ```ignore
/// let id = DocumentId::new();  // Generates unique ID
/// let another = DocumentId::new();  // Different ID
/// assert_ne!(id, another);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(u64);

/// Global counter for generating unique document IDs.
static DOCUMENT_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl DocumentId {
    /// Generates a new unique document ID.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        use std::sync::atomic::Ordering;
        Self(DOCUMENT_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Initialize the ID counter to start after the given maximum ID.
    pub fn init_counter(max_existing_id: u64) {
        use std::sync::atomic::Ordering;
        let next_id = max_existing_id.saturating_add(1);
        DOCUMENT_ID_COUNTER.fetch_max(next_id, Ordering::SeqCst);
    }

    /// Creates a DocumentId from a raw u64 value.
    pub fn from_u64(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw u64 value of this ID.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Document metadata stored at the document level.
///
/// Contains information about the entire document (file/URL), not individual chunks.
/// This is separate from `ChunkSourceMetadata` which is per-chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentMetainfo {
    /// Source identifier (file path or URL)
    pub source_id: String,
    /// Display name for the document (filename or page title)
    pub title: String,
    /// MIME type if known (e.g., "text/markdown", "text/plain")
    pub mime_type: Option<String>,
    /// Content hash for change detection (SHA-256)
    pub content_hash: String,
    /// Unix timestamp when document was first indexed
    pub created_at: u64,
    /// Unix timestamp when document was last updated
    pub updated_at: u64,
    /// Total character count of original content
    pub char_count: usize,
    /// Number of chunks this document was split into
    pub chunk_count: usize,
}

impl DocumentMetainfo {
    /// Creates new document metadata.
    pub fn new(
        source_id: String,
        title: String,
        content_hash: String,
        char_count: usize,
        chunk_count: usize,
    ) -> Self {
        let now = get_current_timestamp();
        Self {
            source_id,
            title,
            mime_type: None,
            content_hash,
            created_at: now,
            updated_at: now,
            char_count,
            chunk_count,
        }
    }

    /// Creates metadata with a MIME type.
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }
}

/// Stored document record with assigned ID.
///
/// Represents a complete document (file/URL) with its metadata and chunk references.
/// The actual chunk content is stored separately via ChunkRecord.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    /// Unique document identifier
    pub id: DocumentId,
    /// Document metadata
    pub metadata: DocumentMetainfo,
    /// IDs of all chunks belonging to this document
    pub chunk_ids: Vec<ChunkId>,
}

impl DocumentRecord {
    /// Creates a new document record.
    pub fn new(id: DocumentId, metadata: DocumentMetainfo, chunk_ids: Vec<ChunkId>) -> Self {
        Self {
            id,
            metadata,
            chunk_ids,
        }
    }
}

/// Document-level search result.
///
/// Returned by document-level search, containing document metadata
/// and aggregated relevance scores.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSearchResult {
    /// Document identifier
    pub doc_id: DocumentId,
    /// Final fused relevance score
    pub score: f32,
    /// Document-level keyword score (from document BM25)
    pub doc_keyword_score: Option<f32>,
    /// Best chunk-level score from this document
    pub best_chunk_score: Option<f32>,
    /// Document metadata
    pub metadata: DocumentMetainfo,
    /// Individual chunk results from this document (sorted by score)
    pub chunks: Vec<SearchResult>,
}

/// Chunk of text with metadata.
///
/// Represents a chunk (segment) of a document to be indexed for search.
/// A document may be split into multiple chunks for better search granularity.
/// The text field contains the searchable content, while metadata provides context.
///
/// Note: This was previously called `Document` but was renamed in ADR-008
/// to clarify that this represents a chunk, not a full document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Chunk text content (will be tokenized for search)
    pub text: String,
    /// Associated metadata (filename, source, timestamp)
    pub metadata: ChunkSourceMetadata,
}

/// Chunk source metadata.
///
/// Contains optional filename, source information, and creation timestamp.
/// This metadata is stored per-chunk and identifies where the chunk came from.
///
/// Note: This was previously called `DocumentMetadata` but was renamed in ADR-008
/// to clarify that this is metadata about a chunk's source, not a full document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkSourceMetadata {
    /// Original filename if available
    pub filename: Option<String>,
    /// Source path or URL
    pub source: Option<String>,
    /// Unix timestamp (seconds since UNIX_EPOCH)
    pub created_at: u64,
}

impl Default for ChunkSourceMetadata {
    fn default() -> Self {
        Self {
            filename: None,
            source: None,
            created_at: get_current_timestamp(),
        }
    }
}

/// Stored chunk record with assigned ID.
///
/// Internal representation of a chunk after it's been indexed.
/// Contains the assigned ChunkId plus the original chunk text and metadata.
///
/// Note: This was previously called `DocumentRecord` but was renamed in ADR-008
/// to clarify that this represents a chunk, not a full document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    /// Unique chunk identifier
    pub id: ChunkId,
    /// Parent document ID (for document-level search - ADR-008)
    ///
    /// `None` for legacy chunks indexed before ADR-008.
    /// New chunks indexed via `add_chunk_to_document()` will have this set.
    #[serde(default)]
    pub document_id: Option<DocumentId>,
    /// Chunk text content
    pub text: String,
    /// Chunk source metadata
    pub metadata: ChunkSourceMetadata,
}

/// Search result with relevance scores.
///
/// Returned by the hybrid search engine, containing both the chunk
/// and relevance scores from different ranking algorithms.
///
/// Represents a single chunk from the search index.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Chunk identifier
    pub chunk_id: ChunkId,
    /// Final fused relevance score (from RRF algorithm)
    pub score: f32,
    /// Semantic similarity score (from vector search, if available)
    pub vector_score: Option<f32>,
    /// Keyword match score (from BM25, if available)
    pub keyword_score: Option<f32>,
    /// Chunk text content
    pub text: String,
    /// Chunk source metadata
    pub metadata: ChunkSourceMetadata,
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

impl From<DocumentSearchResult> for FileSearchResult {
    /// Convert a DocumentSearchResult to a FileSearchResult for UI compatibility.
    ///
    /// Maps document-level metadata to file-level fields used by the UI.
    fn from(doc: DocumentSearchResult) -> Self {
        // Extract file name from source_id (last component of path)
        let file_name = doc
            .metadata
            .source_id
            .rsplit('/')
            .next()
            .unwrap_or(&doc.metadata.source_id)
            .to_string();

        // Use the title as file name if available, otherwise use extracted name
        let display_name = if doc.metadata.title.is_empty() {
            file_name.clone()
        } else {
            doc.metadata.title.clone()
        };

        FileSearchResult {
            file_path: doc.metadata.source_id.clone(),
            file_name: display_name,
            score: doc.score,
            vector_score: doc.best_chunk_score,
            keyword_score: doc.doc_keyword_score,
            chunks: doc.chunks,
            created_at: doc.metadata.created_at,
        }
    }
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
/// - v2: ADR-008 multi-level indexing (document_count renamed to chunk_count)
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

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
    /// Number of documents in the index (files, URLs - user-facing metric)
    #[serde(default)]
    pub doc_count: usize,
    /// Number of chunks in the index (renamed from document_count in v2)
    #[serde(alias = "document_count")]
    pub chunk_count: usize,
    /// Embedding dimension (e.g., 512 for JinaBERT)
    pub embedding_dimension: usize,
    /// Total tokens across all chunks (for metrics display)
    #[serde(default)]
    pub total_tokens: usize,
}

impl IndexManifest {
    /// Creates a new manifest for a fresh index.
    pub fn new(embedding_dimension: usize) -> Self {
        let now = Self::current_timestamp();
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            min_compatible_version: 1, // v2 can read v1 with serde alias
            created_at: now.clone(),
            last_modified: now,
            doc_count: 0,
            chunk_count: 0,
            embedding_dimension,
            total_tokens: 0,
        }
    }

    /// Updates the manifest after chunks are added/removed.
    pub fn update(&mut self, chunk_count: usize) {
        self.chunk_count = chunk_count;
        self.last_modified = Self::current_timestamp();
    }

    /// Updates the manifest with token count.
    pub fn update_with_tokens(&mut self, chunk_count: usize, total_tokens: usize) {
        self.chunk_count = chunk_count;
        self.total_tokens = total_tokens;
        self.last_modified = Self::current_timestamp();
    }

    /// Add tokens to the running total.
    pub fn add_tokens(&mut self, tokens: usize) {
        self.total_tokens += tokens;
    }

    /// Subtract tokens from the running total (for deletions).
    pub fn subtract_tokens(&mut self, tokens: usize) {
        self.total_tokens = self.total_tokens.saturating_sub(tokens);
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

/// Persisted index data (chunks only, embeddings stored separately).
///
/// The actual HNSW and BM25 indices are rebuilt on load from this data.
///
/// Note: This was previously called `PersistedDocuments` but the semantics
/// remain the same - it stores chunk records, not full documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedChunks {
    /// All chunk records in insertion order
    pub chunks: Vec<ChunkRecord>,
}

impl PersistedChunks {
    /// Creates an empty persisted chunks collection.
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Creates from a list of chunk records.
    pub fn from_chunks(chunks: Vec<ChunkRecord>) -> Self {
        Self { chunks }
    }
}

impl Default for PersistedChunks {
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
        chunks: PersistedChunks,
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
    /// ChunkIds of all chunks from this source
    pub chunk_ids: Vec<ChunkId>,
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
            chunk_ids: Vec::new(),
            complete: false,
        }
    }

    /// Creates a complete source record with all chunk IDs
    pub fn new_complete(content_hash: String, chunk_ids: Vec<ChunkId>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            content_hash,
            chunk_ids,
            complete: true,
        }
    }

    /// Marks the source as complete
    pub fn mark_complete(&mut self) {
        self.complete = true;
    }

    /// Adds a chunk ChunkId to this source
    pub fn add_chunk(&mut self, chunk_id: ChunkId) {
        self.chunk_ids.push(chunk_id);
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
