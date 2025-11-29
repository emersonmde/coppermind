//! Hybrid search engine combining vector (semantic) and keyword (BM25) search.
//!
//! This module provides the [`HybridSearchEngine`] which orchestrates:
//! - HNSW-based vector search for semantic similarity
//! - BM25 keyword search for exact term matching
//! - Reciprocal Rank Fusion (RRF) for combining results
//!
//! # Architecture
//!
//! The engine maintains both in-memory indices for fast search and persistent storage
//! for durability:
//!
//! - **Hot data (in memory)**: HNSW index, BM25 index, lightweight metadata
//! - **Cold data (in store)**: Full document text, embeddings (loaded on startup)
//!
//! # Two-Level Indexing (ADR-008)
//!
//! The engine supports both chunk-level and document-level indexing:
//!
//! - **Chunk-level**: Traditional approach where each chunk is indexed independently.
//!   Use [`add_chunk`](HybridSearchEngine::add_chunk) and [`search`](HybridSearchEngine::search).
//!
//! - **Document-level**: New approach where chunks are associated with parent documents.
//!   Use [`create_document`](HybridSearchEngine::create_document),
//!   [`add_chunk_to_document`](HybridSearchEngine::add_chunk_to_document), and
//!   [`search_documents`](HybridSearchEngine::search_documents).
//!
//! Document-level indexing provides proper IDF statistics for BM25 by indexing
//! full document text rather than individual chunks.

mod compaction;
mod document;
mod source_tracking;

#[cfg(test)]
mod tests;

use super::document_keyword::DocumentKeywordEngine;
use super::fusion::{reciprocal_rank_fusion, RRF_K};
use super::keyword::KeywordSearchEngine;
use super::types::{
    validate_dimension, Chunk, ChunkId, ChunkRecord, DocumentId, IndexManifest, SearchError,
    SearchResult,
};
use super::vector::VectorSearchEngine;
use crate::storage::{DocumentStore, StoreError};
use std::collections::HashMap;
use tracing::{debug, info, instrument, warn};

/// Hybrid search engine combining vector (semantic) and keyword (BM25) search.
///
/// This engine uses a `DocumentStore` for persistent storage of documents and embeddings,
/// while keeping indices in memory for fast search.
///
/// # Architecture
///
/// - **Hot data (in memory)**: HNSW index, BM25 index, lightweight metadata
/// - **Cold data (in store)**: Full document text, embeddings (loaded on startup)
///
/// # Storage
///
/// Uses the `DocumentStore` trait which provides O(log n) random access:
/// - Desktop: RedbDocumentStore (B-tree)
/// - Web: IndexedDbDocumentStore (IndexedDB)
///
/// # Example
///
/// ```ignore
/// use coppermind_core::search::HybridSearchEngine;
/// use coppermind_core::storage::InMemoryDocumentStore;
///
/// let store = InMemoryDocumentStore::new();
/// let mut engine = HybridSearchEngine::try_load_or_new(store, 512).await?;
///
/// // Add a chunk
/// let chunk = Chunk { text: "Hello world".to_string(), metadata: Default::default() };
/// engine.add_chunk(chunk, embedding).await?;
///
/// // Search
/// let results = engine.search(&query_embedding, "hello", 10).await?;
/// ```
pub struct HybridSearchEngine<S: DocumentStore> {
    /// Vector search engine (semantic similarity) - chunk-level
    pub(crate) vector_engine: VectorSearchEngine,
    /// Keyword search engine (BM25) - chunk-level
    pub(crate) keyword_engine: KeywordSearchEngine,
    /// Document-level keyword search engine (BM25) - ADR-008
    /// Indexes full document text for proper IDF statistics
    pub(crate) document_keyword_engine: DocumentKeywordEngine,
    /// Document store for persistence (documents, embeddings, sources, metadata)
    pub(crate) store: S,
    /// Embedding dimension (e.g., 512 for JinaBERT)
    pub(crate) embedding_dim: usize,
    /// Index manifest for versioning and metrics
    pub(crate) manifest: IndexManifest,
}

impl<S: DocumentStore> HybridSearchEngine<S> {
    /// Create a new empty hybrid search engine (no persistence load).
    ///
    /// Use [`try_load_or_new`](Self::try_load_or_new) to create an engine that loads existing data.
    ///
    /// # Arguments
    /// * `store` - Document store for persisting indexes
    /// * `embedding_dim` - Dimensionality of embeddings (must match the model)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = InMemoryDocumentStore::new();
    /// let engine = HybridSearchEngine::new(store, 512).await?;
    /// ```
    pub async fn new(store: S, embedding_dim: usize) -> Result<Self, SearchError> {
        Ok(Self {
            vector_engine: VectorSearchEngine::new(embedding_dim),
            keyword_engine: KeywordSearchEngine::new(),
            document_keyword_engine: DocumentKeywordEngine::new(),
            store,
            embedding_dim,
            manifest: IndexManifest::new(embedding_dim),
        })
    }

    /// Create a hybrid search engine, loading existing data if available.
    ///
    /// This is the preferred constructor for production use.
    ///
    /// # Returns
    /// - If no existing index: creates empty engine
    /// - If index exists: loads and rebuilds indices from store
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = RedbDocumentStore::open("./data")?;
    /// let engine = HybridSearchEngine::try_load_or_new(store, 512).await?;
    /// ```
    pub async fn try_load_or_new(store: S, embedding_dim: usize) -> Result<Self, SearchError> {
        // Check if there's existing data by counting chunks
        let chunk_count = store
            .chunk_count()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        if chunk_count == 0 {
            info!("No existing index found, creating new engine");
            Self::new(store, embedding_dim).await
        } else {
            info!("Loading existing index with {} chunks", chunk_count);
            Self::rebuild_from_store(store, embedding_dim).await
        }
    }

    /// Rebuild indices from data in the store.
    ///
    /// This is called internally by [`try_load_or_new`](Self::try_load_or_new) when
    /// existing data is detected in the store.
    async fn rebuild_from_store(store: S, embedding_dim: usize) -> Result<Self, SearchError> {
        use instant::Instant;

        let total_start = Instant::now();

        let mut engine = Self {
            vector_engine: VectorSearchEngine::new(embedding_dim),
            keyword_engine: KeywordSearchEngine::new(),
            document_keyword_engine: DocumentKeywordEngine::new(),
            store,
            embedding_dim,
            manifest: IndexManifest::new(embedding_dim),
        };

        // Load all embeddings from store
        let load_start = Instant::now();
        let embeddings = engine
            .store
            .iter_embeddings()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;
        let load_elapsed = load_start.elapsed();
        debug!(
            "Loaded {} embeddings from store in {:?}",
            embeddings.len(),
            load_elapsed
        );

        // Batch load all chunks for BM25 (separate I/O from CPU work)
        let chunks_start = Instant::now();
        let chunk_ids: Vec<_> = embeddings.iter().map(|(id, _)| *id).collect();
        let chunks = engine
            .store
            .get_chunks_batch(&chunk_ids)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;
        // Build lookup map by chunk_id
        let chunk_map: std::collections::HashMap<ChunkId, &ChunkRecord> = chunk_ids
            .iter()
            .zip(chunks.iter())
            .map(|(id, chunk)| (*id, chunk))
            .collect();
        let chunks_elapsed = chunks_start.elapsed();
        debug!(
            "Loaded {} chunks from store in {:?}",
            chunks.len(),
            chunks_elapsed
        );

        // Track maximum ID to initialize the counter
        let mut max_id: u64 = 0;

        // Prepare validated data for index building
        let mut valid_embeddings: Vec<(ChunkId, Vec<f32>)> = Vec::new();
        let mut valid_texts: Vec<(ChunkId, String)> = Vec::new();

        for (chunk_id, embedding) in embeddings {
            max_id = max_id.max(chunk_id.as_u64());

            if embedding.len() != embedding_dim {
                warn!(
                    "Skipping chunk {} with wrong dimension: expected {}, got {}",
                    chunk_id.as_u64(),
                    embedding_dim,
                    embedding.len()
                );
                continue;
            }

            if let Some(record) = chunk_map.get(&chunk_id) {
                valid_embeddings.push((chunk_id, embedding));
                valid_texts.push((chunk_id, record.text.clone()));
            } else {
                warn!(
                    "Embedding for chunk {} has no chunk record, skipping",
                    chunk_id.as_u64()
                );
            }
        }

        let chunk_count = valid_embeddings.len();

        // Build indices - parallel on native, sequential on WASM
        #[cfg(not(target_arch = "wasm32"))]
        let (vector_engine, keyword_engine, hnsw_elapsed, bm25_elapsed) = {
            std::thread::scope(|s| {
                // Spawn HNSW index builder
                let hnsw_handle = s.spawn(|| {
                    let start = Instant::now();
                    let mut ve = VectorSearchEngine::new(embedding_dim);
                    for (chunk_id, embedding) in &valid_embeddings {
                        let _ = ve.add_chunk(*chunk_id, embedding.clone());
                    }
                    let elapsed = start.elapsed();
                    debug!(
                        "Built HNSW index: {} vectors in {:?} ({:.2}ms/vector)",
                        valid_embeddings.len(),
                        elapsed,
                        elapsed.as_secs_f64() * 1000.0 / valid_embeddings.len().max(1) as f64
                    );
                    (ve, elapsed)
                });

                // Spawn BM25 index builder
                let bm25_handle = s.spawn(|| {
                    let start = Instant::now();
                    let mut ke = KeywordSearchEngine::new();
                    for (chunk_id, text) in &valid_texts {
                        ke.add_chunk(*chunk_id, text.clone());
                    }
                    let elapsed = start.elapsed();
                    debug!(
                        "Built BM25 index: {} chunks in {:?} ({:.2}ms/chunk)",
                        valid_texts.len(),
                        elapsed,
                        elapsed.as_secs_f64() * 1000.0 / valid_texts.len().max(1) as f64
                    );
                    (ke, elapsed)
                });

                // Wait for both to complete
                let (ve, hnsw_elapsed) = hnsw_handle.join().expect("HNSW thread panicked");
                let (ke, bm25_elapsed) = bm25_handle.join().expect("BM25 thread panicked");

                (ve, ke, hnsw_elapsed, bm25_elapsed)
            })
        };

        // WASM: Build sequentially (no threading support)
        #[cfg(target_arch = "wasm32")]
        let (vector_engine, keyword_engine, hnsw_elapsed, bm25_elapsed) = {
            // Build HNSW index
            let hnsw_start = Instant::now();
            let mut vector_engine = VectorSearchEngine::new(embedding_dim);
            for (chunk_id, embedding) in &valid_embeddings {
                let _ = vector_engine.add_chunk(*chunk_id, embedding.clone());
            }
            let hnsw_elapsed = hnsw_start.elapsed();
            debug!(
                "Built HNSW index: {} vectors in {:?} ({:.2}ms/vector)",
                valid_embeddings.len(),
                hnsw_elapsed,
                hnsw_elapsed.as_secs_f64() * 1000.0 / valid_embeddings.len().max(1) as f64
            );

            // Build BM25 index
            let bm25_start = Instant::now();
            let mut keyword_engine = KeywordSearchEngine::new();
            for (chunk_id, text) in &valid_texts {
                keyword_engine.add_chunk(*chunk_id, text.clone());
            }
            let bm25_elapsed = bm25_start.elapsed();
            debug!(
                "Built BM25 index: {} chunks in {:?} ({:.2}ms/chunk)",
                valid_texts.len(),
                bm25_elapsed,
                bm25_elapsed.as_secs_f64() * 1000.0 / valid_texts.len().max(1) as f64
            );

            (vector_engine, keyword_engine, hnsw_elapsed, bm25_elapsed)
        };

        // Replace engine's indices with the newly built ones
        engine.vector_engine = vector_engine;
        engine.keyword_engine = keyword_engine;

        // Clear any stale tombstones from storage (from previous sessions)
        let _ = engine
            .store
            .put_tombstones(&std::collections::HashSet::new())
            .await;

        // Clean up incomplete sources from previous session crashes
        let cleanup_start = Instant::now();
        let sources = engine
            .store
            .list_sources()
            .await
            .unwrap_or_else(|_| Vec::new());

        for source_id in sources {
            if let Ok(Some(record)) = engine.store.get_source(&source_id).await {
                if !record.complete {
                    warn!(
                        "Cleaning up incomplete source '{}' from previous session ({} partial chunks)",
                        source_id,
                        record.chunk_ids.len()
                    );
                    // Delete the partial chunks and embeddings
                    for chunk_id in &record.chunk_ids {
                        let _ = engine.store.delete_chunk(*chunk_id).await;
                        let _ = engine.store.delete_embedding(*chunk_id).await;
                    }
                    // Delete the source record
                    let _ = engine.store.delete_source(&source_id).await;
                }
            }
        }
        let cleanup_elapsed = cleanup_start.elapsed();
        debug!("Source cleanup completed in {:?}", cleanup_elapsed);

        // Initialize ChunkId counter to continue after the highest loaded ID
        ChunkId::init_counter(max_id);

        // Load document records and rebuild document-level BM25 (ADR-008)
        let doc_bm25_start = Instant::now();
        let doc_ids = engine
            .store
            .iter_document_ids()
            .await
            .unwrap_or_else(|_| Vec::new());

        let mut max_doc_id: u64 = 0;
        if !doc_ids.is_empty() {
            let docs = engine
                .store
                .get_documents_batch(&doc_ids)
                .await
                .unwrap_or_else(|_| Vec::new());

            // Find max document ID for counter initialization
            for doc in &docs {
                max_doc_id = max_doc_id.max(doc.id.as_u64());
            }

            debug!(
                "Loaded {} document records (document BM25 will be populated on indexing)",
                docs.len()
            );
        }

        // Initialize DocumentId counter
        DocumentId::init_counter(max_doc_id);

        let doc_bm25_elapsed = doc_bm25_start.elapsed();

        // Update manifest with document and chunk counts
        engine.manifest.doc_count = doc_ids.len();
        engine.manifest.update(chunk_count);

        let total_elapsed = total_start.elapsed();
        info!(
            "Rebuilt indices: {} chunks in {:?} (embeddings: {:?}, chunks: {:?}, HNSW: {:?}, BM25: {:?}, cleanup: {:?}, doc_bm25: {:?})",
            chunk_count,
            total_elapsed,
            load_elapsed,
            chunks_elapsed,
            hnsw_elapsed,
            bm25_elapsed,
            cleanup_elapsed,
            doc_bm25_elapsed
        );

        Ok(engine)
    }

    /// Save any pending changes to storage.
    ///
    /// With DocumentStore, document and embedding writes happen immediately.
    /// This method is kept for API compatibility but is essentially a no-op.
    pub async fn save(&mut self) -> Result<(), SearchError> {
        info!("Saved index state: {} chunks", self.manifest.chunk_count);
        Ok(())
    }

    /// Check if there are unsaved changes.
    ///
    /// With DocumentStore, all writes happen immediately, so this always
    /// returns false. Kept for API compatibility.
    pub fn is_dirty(&self) -> bool {
        false
    }

    /// Get a reference to the document store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get the current manifest.
    pub fn manifest(&self) -> &IndexManifest {
        &self.manifest
    }

    /// Add a chunk to the index.
    ///
    /// The chunk is immediately persisted to the store and added to in-memory indices.
    ///
    /// # Arguments
    /// * `chunk` - Chunk containing text and metadata
    /// * `embedding` - Pre-computed embedding vector for the chunk
    ///
    /// # Returns
    /// The assigned ChunkId
    ///
    /// # Errors
    /// Returns [`SearchError::DimensionMismatch`] if embedding dimension doesn't match.
    /// Returns [`SearchError::StorageError`] if persistence fails.
    #[must_use = "Chunk ID should be stored or errors handled"]
    #[instrument(skip_all, fields(text_len = chunk.text.len()))]
    pub async fn add_chunk(
        &mut self,
        chunk: Chunk,
        embedding: Vec<f32>,
    ) -> Result<ChunkId, SearchError> {
        self.add_chunk_with_tokens(chunk, embedding, 0).await
    }

    /// Add a chunk to the index with token count for metrics tracking.
    ///
    /// The chunk is immediately persisted to the store.
    ///
    /// # Arguments
    /// * `chunk` - Chunk containing text and metadata
    /// * `embedding` - Pre-computed embedding vector for the chunk
    /// * `token_count` - Number of tokens in this chunk (for metrics)
    ///
    /// # Returns
    /// The assigned ChunkId
    #[must_use = "Chunk ID should be stored or errors handled"]
    #[instrument(skip_all, fields(text_len = chunk.text.len(), token_count))]
    pub async fn add_chunk_with_tokens(
        &mut self,
        chunk: Chunk,
        embedding: Vec<f32>,
        token_count: usize,
    ) -> Result<ChunkId, SearchError> {
        // Validate embedding dimension
        validate_dimension(self.embedding_dim, embedding.len())?;

        // Generate unique ID
        let chunk_id = ChunkId::new();

        // Create chunk record (no document_id for legacy add_chunk API)
        let record = ChunkRecord {
            id: chunk_id,
            document_id: None,
            text: chunk.text.clone(),
            metadata: chunk.metadata,
        };

        // Persist to store first (fail fast if storage fails)
        self.store
            .put_chunk(chunk_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        self.store
            .put_embedding(chunk_id, &embedding)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Add to in-memory indices
        self.vector_engine.add_chunk(chunk_id, embedding)?;
        self.keyword_engine.add_chunk(chunk_id, record.text);

        // Update manifest with token count
        self.manifest.update(self.manifest.chunk_count + 1);
        self.manifest.add_tokens(token_count);

        Ok(chunk_id)
    }

    /// Add a chunk without rebuilding vector index (for batch operations).
    ///
    /// Call [`rebuild_vector_index`](Self::rebuild_vector_index) once after all chunks are added.
    ///
    /// # Note
    /// With the current hnsw_rs implementation, this is functionally equivalent to
    /// [`add_chunk`](Self::add_chunk) since hnsw_rs supports incremental updates.
    #[must_use = "Chunk ID should be stored or errors handled"]
    #[instrument(skip_all, fields(text_len = chunk.text.len()))]
    pub async fn add_chunk_deferred(
        &mut self,
        chunk: Chunk,
        embedding: Vec<f32>,
    ) -> Result<ChunkId, SearchError> {
        self.add_chunk_deferred_with_tokens(chunk, embedding, 0)
            .await
    }

    /// Add a chunk without rebuilding vector index, with token count for metrics.
    ///
    /// Call [`rebuild_vector_index`](Self::rebuild_vector_index) once after all chunks are added.
    #[must_use = "Chunk ID should be stored or errors handled"]
    #[instrument(skip_all, fields(text_len = chunk.text.len(), token_count))]
    pub async fn add_chunk_deferred_with_tokens(
        &mut self,
        chunk: Chunk,
        embedding: Vec<f32>,
        token_count: usize,
    ) -> Result<ChunkId, SearchError> {
        // Validate embedding dimension
        validate_dimension(self.embedding_dim, embedding.len())?;

        // Generate unique ID
        let chunk_id = ChunkId::new();

        // Create chunk record (no document_id for legacy deferred add API)
        let record = ChunkRecord {
            id: chunk_id,
            document_id: None,
            text: chunk.text.clone(),
            metadata: chunk.metadata,
        };

        // Persist to store first
        self.store
            .put_chunk(chunk_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        self.store
            .put_embedding(chunk_id, &embedding)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Add to in-memory indices (deferred rebuild for vector)
        self.vector_engine.add_chunk_deferred(chunk_id, embedding)?;
        self.keyword_engine.add_chunk(chunk_id, record.text);

        // Update manifest with token count
        self.manifest.update(self.manifest.chunk_count + 1);
        self.manifest.add_tokens(token_count);

        Ok(chunk_id)
    }

    /// Rebuild the vector search index after batch operations.
    ///
    /// # Note
    /// With hnsw_rs, this is a no-op since the index supports incremental updates.
    pub async fn rebuild_vector_index(&mut self) -> Result<(), SearchError> {
        Ok(())
    }

    /// Perform hybrid search combining vector and keyword search.
    ///
    /// Results are ranked using Reciprocal Rank Fusion (RRF) which combines
    /// the rankings from both vector similarity and BM25 keyword matching.
    ///
    /// # Arguments
    /// * `query_embedding` - Pre-computed embedding for the query text
    /// * `query_text` - The query text for keyword matching
    /// * `k` - Number of results to return (must be > 0)
    ///
    /// # Returns
    /// A vector of [`SearchResult`] sorted by fused score (descending).
    ///
    /// # Errors
    /// Returns [`SearchError::InvalidQuery`] if:
    /// - `query_text` is empty or whitespace-only
    /// - `k` is 0
    ///
    /// Returns [`SearchError::DimensionMismatch`] if `query_embedding` has wrong dimension.
    #[must_use = "Search results should be used or errors handled"]
    pub async fn search(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Validate query parameters
        if query_text.trim().is_empty() {
            return Err(SearchError::InvalidQuery(
                "Query text cannot be empty".to_string(),
            ));
        }
        if k == 0 {
            return Err(SearchError::InvalidQuery(
                "Number of results (k) must be greater than 0".to_string(),
            ));
        }

        // Validate query embedding dimension
        validate_dimension(self.embedding_dim, query_embedding.len())?;

        // Get vector search results (semantic similarity)
        let vector_results = self.vector_engine.search(query_embedding, k * 2)?;
        info!("📊 Vector search found {} results", vector_results.len());

        // Get keyword search results (BM25)
        let keyword_results = self.keyword_engine.search(query_text, k * 2);
        info!("📊 Keyword search found {} results", keyword_results.len());

        // Fuse results using Reciprocal Rank Fusion (RRF)
        info!("🔀 Applying Reciprocal Rank Fusion (RRF)...");

        let fused_results = reciprocal_rank_fusion(&vector_results, &keyword_results, RRF_K);

        // Build maps of individual scores for lookup
        let vector_scores: HashMap<ChunkId, f32> = vector_results.into_iter().collect();
        let keyword_scores: HashMap<ChunkId, f32> = keyword_results.into_iter().collect();

        // Fetch chunk details from store and build SearchResults
        let mut search_results = Vec::with_capacity(k);

        for (chunk_id, score) in fused_results.into_iter().take(k) {
            // Fetch chunk from store
            match self.store.get_chunk(chunk_id).await {
                Ok(Some(record)) => {
                    search_results.push(SearchResult {
                        chunk_id,
                        score,
                        vector_score: vector_scores.get(&chunk_id).copied(),
                        keyword_score: keyword_scores.get(&chunk_id).copied(),
                        text: record.text,
                        metadata: record.metadata,
                    });
                }
                Ok(None) => {
                    // Chunk not found in store - skip (may have been deleted)
                    warn!("Chunk {} not found in store, skipping", chunk_id.as_u64());
                }
                Err(e) => {
                    warn!("Error fetching chunk {}: {}", chunk_id.as_u64(), e);
                }
            }
        }

        Ok(search_results)
    }

    /// Search with two-level RRF fusion (ADR-008).
    ///
    /// Implements the multi-level search architecture:
    /// 1. Chunk-level HNSW vector search for semantic similarity
    /// 2. Lift chunk results to document IDs via O(1) lookup
    /// 3. Document-level BM25 keyword search
    /// 4. RRF fusion at document level (combining vector and keyword rankings)
    /// 5. Return DocumentSearchResult with nested chunk results
    ///
    /// # Arguments
    /// * `query_embedding` - Pre-computed embedding for the query
    /// * `query_text` - Query text for keyword matching
    /// * `k` - Maximum number of documents to return
    ///
    /// # Returns
    /// A vector of [`DocumentSearchResult`](super::types::DocumentSearchResult) sorted by
    /// fused score, each containing document-level metadata and nested chunk results.
    ///
    /// # Note
    /// Only chunks indexed via [`add_chunk_to_document`](Self::add_chunk_to_document)
    /// are included in document-level search. Legacy chunks without a document_id
    /// are skipped (use regular [`search`](Self::search) for those).
    #[must_use = "Search results should be used or errors handled"]
    pub async fn search_documents(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
    ) -> Result<Vec<super::types::DocumentSearchResult>, SearchError> {
        use super::types::DocumentSearchResult;
        use std::collections::HashSet;

        // Validate query parameters
        if query_text.trim().is_empty() {
            return Err(SearchError::InvalidQuery(
                "Query text cannot be empty".to_string(),
            ));
        }
        if k == 0 {
            return Err(SearchError::InvalidQuery(
                "Number of results (k) must be greater than 0".to_string(),
            ));
        }

        validate_dimension(self.embedding_dim, query_embedding.len())?;

        // Step 1: Chunk-level HNSW vector search
        let vector_results = self.vector_engine.search(query_embedding, k * 4)?;
        debug!("Vector search found {} chunk results", vector_results.len());

        // Step 2: Lift chunks to documents using document_id from ChunkRecord (O(1) lookup)
        let mut doc_chunks: HashMap<DocumentId, Vec<(ChunkId, f32)>> = HashMap::new();
        let mut seen_doc_ids: HashSet<DocumentId> = HashSet::new();

        for (chunk_id, score) in &vector_results {
            // Use document_id from ChunkRecord for O(1) lookup
            match self.store.get_chunk(*chunk_id).await {
                Ok(Some(chunk_record)) => {
                    if let Some(doc_id) = chunk_record.document_id {
                        doc_chunks
                            .entry(doc_id)
                            .or_default()
                            .push((*chunk_id, *score));
                        seen_doc_ids.insert(doc_id);
                    }
                    // Legacy chunks without document_id are skipped in document-level search
                }
                Ok(None) => {
                    // Chunk in vector index but not in store - may be tombstoned
                    debug!(chunk_id = chunk_id.as_u64(), "Chunk not found in store");
                }
                Err(e) => {
                    // Storage error during chunk lookup - log at warn level
                    warn!(chunk_id = chunk_id.as_u64(), error = %e, "Storage error fetching chunk");
                }
            }
        }

        // Step 3: Document-level BM25 keyword search
        let doc_keyword_results = self.document_keyword_engine.search(query_text, k * 2);
        debug!(
            "Document keyword search found {} results",
            doc_keyword_results.len()
        );

        // Add documents from keyword search to seen set
        for (doc_id, _) in &doc_keyword_results {
            seen_doc_ids.insert(*doc_id);
        }

        // Step 4: Document-level RRF fusion
        // For vector ranking, use the best chunk score for each document
        let doc_vector_rankings: Vec<(DocumentId, f32)> = doc_chunks
            .iter()
            .map(|(doc_id, chunks)| {
                let best_score = chunks
                    .iter()
                    .map(|(_, s)| *s)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                (*doc_id, best_score)
            })
            .collect();

        // Sort by score for RRF (higher is better)
        let mut sorted_vector: Vec<_> = doc_vector_rankings;
        sorted_vector.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut sorted_keyword: Vec<_> = doc_keyword_results.clone();
        sorted_keyword.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply RRF
        let fused_doc_results = reciprocal_rank_fusion(&sorted_vector, &sorted_keyword, RRF_K);

        // Build score lookup maps
        let keyword_doc_scores: HashMap<DocumentId, f32> = sorted_keyword.into_iter().collect();

        // Step 5: Build DocumentSearchResult
        let mut results: Vec<DocumentSearchResult> = Vec::with_capacity(k);

        for (doc_id, score) in fused_doc_results.into_iter().take(k) {
            // Get document metadata
            let doc_record = match self.store.get_document(doc_id).await {
                Ok(Some(record)) => record,
                Ok(None) => {
                    // Document referenced by chunks but not in store - data inconsistency
                    warn!(
                        doc_id = doc_id.as_u64(),
                        "Document not found in store during search"
                    );
                    continue;
                }
                Err(e) => {
                    // Storage error during document lookup
                    warn!(doc_id = doc_id.as_u64(), error = %e, "Storage error fetching document");
                    continue;
                }
            };

            // Get chunk search results for this document
            let mut chunks = Vec::new();
            if let Some(chunk_scores) = doc_chunks.get(&doc_id) {
                for (chunk_id, chunk_score) in chunk_scores {
                    if let Ok(Some(chunk_record)) = self.store.get_chunk(*chunk_id).await {
                        chunks.push(SearchResult {
                            chunk_id: *chunk_id,
                            score: *chunk_score,
                            vector_score: Some(*chunk_score),
                            keyword_score: None,
                            text: chunk_record.text,
                            metadata: chunk_record.metadata,
                        });
                    }
                }
            }

            // Sort chunks by score descending
            chunks.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_chunk_score = chunks.first().map(|c| c.score);

            results.push(DocumentSearchResult {
                doc_id,
                score,
                doc_keyword_score: keyword_doc_scores.get(&doc_id).copied(),
                best_chunk_score,
                metadata: doc_record.metadata,
                chunks,
            });
        }

        info!(
            "Document search returned {} results (from {} candidate docs)",
            results.len(),
            seen_doc_ids.len()
        );

        Ok(results)
    }

    /// Get the number of indexed chunks.
    ///
    /// This returns the total count of chunks in the index, regardless of
    /// whether they were added via [`add_chunk`](Self::add_chunk) or
    /// [`add_chunk_to_document`](Self::add_chunk_to_document).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.manifest.chunk_count
    }

    /// Check if the index is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.manifest.chunk_count == 0
    }

    /// Get detailed index metrics.
    ///
    /// # Returns
    /// A tuple of (doc_count, total_chunks, total_tokens, avg_tokens_per_chunk).
    pub async fn get_index_metrics(&self) -> Result<(usize, usize, usize, f64), SearchError> {
        Ok(self.get_index_metrics_sync())
    }

    /// Get detailed index metrics (synchronous version).
    ///
    /// # Returns
    /// A tuple of (doc_count, total_chunks, total_tokens, avg_tokens_per_chunk).
    /// - `doc_count`: Number of documents (files, URLs) - user-facing metric
    /// - `total_chunks`: Number of chunks (used internally for HNSW)
    /// - `total_tokens`: Total tokens processed
    /// - `avg_tokens_per_chunk`: Average tokens per chunk
    pub fn get_index_metrics_sync(&self) -> (usize, usize, usize, f64) {
        let doc_count = self.manifest.doc_count;
        let total_chunks = self.manifest.chunk_count;
        let total_tokens = self.manifest.total_tokens;
        let avg_tokens = if total_chunks > 0 {
            total_tokens as f64 / total_chunks as f64
        } else {
            0.0
        };
        (doc_count, total_chunks, total_tokens, avg_tokens)
    }

    /// Get vector index size (number of embeddings).
    pub fn vector_index_len(&self) -> usize {
        self.vector_engine.len()
    }

    /// Get keyword index size.
    pub fn keyword_index_len(&self) -> usize {
        self.keyword_engine.len()
    }

    /// Get a chunk by ID.
    ///
    /// # Returns
    /// `Ok(Some(record))` if the chunk exists, `Ok(None)` if not found.
    pub async fn get_chunk(&self, chunk_id: &ChunkId) -> Result<Option<ChunkRecord>, SearchError> {
        self.store
            .get_chunk(*chunk_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))
    }

    /// Clear all documents from both memory and persistent storage.
    ///
    /// This removes all chunks, documents, embeddings, and source records.
    /// The engine is reset to an empty state.
    pub async fn clear_all(&mut self) -> Result<(), SearchError> {
        // Clear persistent storage
        self.store
            .clear()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Clear in-memory indices
        self.vector_engine = VectorSearchEngine::new(self.embedding_dim);
        self.keyword_engine = KeywordSearchEngine::new();
        self.document_keyword_engine = DocumentKeywordEngine::new();
        self.manifest = IndexManifest::new(self.embedding_dim);

        info!("Cleared all index data (memory and storage)");
        Ok(())
    }
}

// Conversion from StoreError to SearchError
impl From<StoreError> for SearchError {
    fn from(e: StoreError) -> Self {
        SearchError::StorageError(e.to_string())
    }
}
