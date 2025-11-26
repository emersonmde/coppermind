// HybridSearchEngine - combines vector and keyword search

use super::fusion::{reciprocal_rank_fusion, RRF_K};
use super::keyword::KeywordSearchEngine;
#[cfg(test)]
use super::types::DocumentMetadata;
use super::types::{
    validate_dimension, DocId, Document, DocumentRecord, IndexManifest, SearchError, SearchResult,
};
use super::vector::VectorSearchEngine;
use crate::storage::{DocumentStore, StoreError};
use std::collections::HashMap;
use tracing::{debug, info, instrument, warn};

/// Maximum characters to show in debug dump text preview.
const DEBUG_TEXT_PREVIEW_LEN: usize = 100;

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
/// Uses the new `DocumentStore` trait which provides O(log n) random access:
/// - Desktop: RedbDocumentStore (B-tree)
/// - Web: IndexedDbDocumentStore (IndexedDB)
pub struct HybridSearchEngine<S: DocumentStore> {
    /// Vector search engine (semantic similarity)
    vector_engine: VectorSearchEngine,
    /// Keyword search engine (BM25)
    keyword_engine: KeywordSearchEngine,
    /// Document store for persistence (documents, embeddings, sources, metadata)
    store: S,
    /// Embedding dimension (e.g., 512 for JinaBERT)
    embedding_dim: usize,
    /// Index manifest for versioning
    manifest: IndexManifest,
}

impl<S: DocumentStore> HybridSearchEngine<S> {
    /// Create a new empty hybrid search engine (no persistence load).
    ///
    /// Use `try_load_or_new()` to create an engine that loads existing data.
    ///
    /// # Arguments
    /// * `store` - Document store for persisting indexes
    /// * `embedding_dim` - Dimensionality of embeddings (must match the model)
    pub async fn new(store: S, embedding_dim: usize) -> Result<Self, SearchError> {
        Ok(Self {
            vector_engine: VectorSearchEngine::new(embedding_dim),
            keyword_engine: KeywordSearchEngine::new(),
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
    pub async fn try_load_or_new(store: S, embedding_dim: usize) -> Result<Self, SearchError> {
        // Check if there's existing data by counting documents
        let doc_count = store
            .document_count()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        if doc_count == 0 {
            info!("No existing index found, creating new engine");
            Self::new(store, embedding_dim).await
        } else {
            info!("Loading existing index with {} documents", doc_count);
            Self::rebuild_from_store(store, embedding_dim).await
        }
    }

    /// Rebuild indices from data in the store.
    async fn rebuild_from_store(store: S, embedding_dim: usize) -> Result<Self, SearchError> {
        let mut engine = Self {
            vector_engine: VectorSearchEngine::new(embedding_dim),
            keyword_engine: KeywordSearchEngine::new(),
            store,
            embedding_dim,
            manifest: IndexManifest::new(embedding_dim),
        };

        // Load all embeddings from store and rebuild indices
        let embeddings = engine
            .store
            .iter_embeddings()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Track maximum ID to initialize the counter
        let mut max_id: u64 = 0;
        let mut doc_count = 0;

        for (doc_id, embedding) in embeddings {
            // Track max ID
            max_id = max_id.max(doc_id.as_u64());

            // Validate embedding dimension
            if embedding.len() != embedding_dim {
                warn!(
                    "Skipping doc {} with wrong dimension: expected {}, got {}",
                    doc_id.as_u64(),
                    embedding_dim,
                    embedding.len()
                );
                continue;
            }

            // Get document text for BM25
            let doc_record = engine
                .store
                .get_document(doc_id)
                .await
                .map_err(|e| SearchError::StorageError(e.to_string()))?;

            if let Some(record) = doc_record {
                // Add to vector index
                engine.vector_engine.add_document(doc_id, embedding)?;

                // Add to keyword index
                engine.keyword_engine.add_document(doc_id, record.text);

                doc_count += 1;
            } else {
                warn!(
                    "Embedding for doc {} has no document record, skipping",
                    doc_id.as_u64()
                );
            }
        }

        // Note: We don't load tombstones on reload because:
        // 1. Deleted documents/embeddings are removed from storage immediately
        // 2. HNSW indices aren't stable across rebuilds (indices change)
        // 3. The rebuilt index only contains live documents
        //
        // Tombstones are only meaningful within a session to filter search results
        // until compaction runs or the app restarts.

        // Clear any stale tombstones from storage (from previous sessions)
        let _ = engine
            .store
            .put_tombstones(&std::collections::HashSet::new())
            .await;

        // Initialize DocId counter to continue after the highest loaded ID
        DocId::init_counter(max_id);

        // Update manifest
        engine.manifest.update(doc_count);

        info!(
            "Rebuilt indices: {} documents, {} vectors",
            doc_count,
            engine.vector_engine.len()
        );

        Ok(engine)
    }

    /// Save any pending changes to storage.
    ///
    /// With DocumentStore, document and embedding writes happen immediately.
    /// Tombstones are session-local (not persisted) since deleted docs are
    /// removed from storage and HNSW indices aren't stable across rebuilds.
    pub async fn save(&mut self) -> Result<(), SearchError> {
        // Documents and embeddings are already persisted on add/delete.
        // Tombstones are session-local only - on restart, the index is
        // rebuilt from storage (which excludes deleted documents).
        info!(
            "Saved index state: {} documents",
            self.manifest.document_count
        );
        Ok(())
    }

    /// Check if there are unsaved changes.
    ///
    /// With DocumentStore, all writes happen immediately, so this always
    /// returns false. Kept for API compatibility.
    pub fn is_dirty(&self) -> bool {
        // All changes are persisted immediately with DocumentStore
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

    /// Add a document to the index.
    ///
    /// The document is immediately persisted to the store.
    ///
    /// # Arguments
    /// * `doc` - Document containing text and metadata
    /// * `embedding` - Pre-computed embedding vector for the document
    ///
    /// Returns the assigned DocId
    #[must_use = "Document ID should be stored or errors handled"]
    #[instrument(skip_all, fields(text_len = doc.text.len()))]
    pub async fn add_document(
        &mut self,
        doc: Document,
        embedding: Vec<f32>,
    ) -> Result<DocId, SearchError> {
        // Validate embedding dimension
        validate_dimension(self.embedding_dim, embedding.len())?;

        // Generate unique ID
        let doc_id = DocId::new();

        // Create document record
        let record = DocumentRecord {
            id: doc_id,
            text: doc.text.clone(),
            metadata: doc.metadata,
        };

        // Persist to store first (fail fast if storage fails)
        self.store
            .put_document(doc_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        self.store
            .put_embedding(doc_id, &embedding)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Add to in-memory indices
        self.vector_engine.add_document(doc_id, embedding)?;
        self.keyword_engine.add_document(doc_id, record.text);

        // Update manifest
        self.manifest.update(self.manifest.document_count + 1);

        Ok(doc_id)
    }

    /// Add a document without rebuilding vector index (for batch operations).
    ///
    /// Call rebuild_vector_index() once after all documents are added.
    #[must_use = "Document ID should be stored or errors handled"]
    #[instrument(skip_all, fields(text_len = doc.text.len()))]
    pub async fn add_document_deferred(
        &mut self,
        doc: Document,
        embedding: Vec<f32>,
    ) -> Result<DocId, SearchError> {
        // Validate embedding dimension
        validate_dimension(self.embedding_dim, embedding.len())?;

        // Generate unique ID
        let doc_id = DocId::new();

        // Create document record
        let record = DocumentRecord {
            id: doc_id,
            text: doc.text.clone(),
            metadata: doc.metadata,
        };

        // Persist to store first
        self.store
            .put_document(doc_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        self.store
            .put_embedding(doc_id, &embedding)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Add to in-memory indices (deferred rebuild for vector)
        self.vector_engine
            .add_document_deferred(doc_id, embedding)?;
        self.keyword_engine.add_document(doc_id, record.text);

        // Update manifest
        self.manifest.update(self.manifest.document_count + 1);

        Ok(doc_id)
    }

    /// Rebuild the vector search index after batch operations.
    ///
    /// NOTE: With hnsw_rs, this is a no-op since the index supports incremental updates.
    pub async fn rebuild_vector_index(&mut self) -> Result<(), SearchError> {
        // No-op: hnsw_rs supports incremental insertion
        Ok(())
    }

    /// Perform hybrid search combining vector and keyword search.
    ///
    /// # Arguments
    /// * `query_embedding` - Pre-computed embedding for the query text
    /// * `query_text` - The query text for keyword matching
    /// * `k` - Number of results to return (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidQuery` if:
    /// - `query_text` is empty or whitespace-only
    /// - `k` is 0
    ///
    /// Returns `SearchError::DimensionMismatch` if `query_embedding` has wrong dimension.
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
        let vector_scores: HashMap<DocId, f32> = vector_results.into_iter().collect();
        let keyword_scores: HashMap<DocId, f32> = keyword_results.into_iter().collect();

        // Fetch document details from store and build SearchResults
        let mut search_results = Vec::with_capacity(k);

        for (doc_id, score) in fused_results.into_iter().take(k) {
            // Fetch document from store
            match self.store.get_document(doc_id).await {
                Ok(Some(record)) => {
                    search_results.push(SearchResult {
                        doc_id,
                        score,
                        vector_score: vector_scores.get(&doc_id).copied(),
                        keyword_score: keyword_scores.get(&doc_id).copied(),
                        text: record.text,
                        metadata: record.metadata,
                    });
                }
                Ok(None) => {
                    // Document not found in store - skip (may have been deleted)
                    warn!("Document {} not found in store, skipping", doc_id.as_u64());
                }
                Err(e) => {
                    warn!("Error fetching document {}: {}", doc_id.as_u64(), e);
                }
            }
        }

        Ok(search_results)
    }

    /// Get the number of indexed documents.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.manifest.document_count
    }

    /// Check if the index is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.manifest.document_count == 0
    }

    /// Get detailed index metrics.
    ///
    /// Returns (total_chunks, total_tokens, avg_tokens_per_chunk).
    ///
    /// Note: This requires loading documents from store, so it's more expensive
    /// than with the in-memory implementation.
    pub async fn get_index_metrics(&self) -> Result<(usize, usize, f64), SearchError> {
        let total_chunks = self.manifest.document_count;

        if total_chunks == 0 {
            return Ok((0, 0, 0.0));
        }

        // For now, return approximate metrics without loading all documents
        // A full implementation would iterate all documents
        Ok((total_chunks, 0, 0.0))
    }

    /// Get detailed index metrics (synchronous version for backwards compatibility).
    ///
    /// Note: Returns (total_chunks, 0, 0.0) since we can't load docs synchronously.
    pub fn get_index_metrics_sync(&self) -> (usize, usize, f64) {
        let total_chunks = self.manifest.document_count;
        (total_chunks, 0, 0.0)
    }

    /// Get vector index size (number of embeddings).
    pub fn vector_index_len(&self) -> usize {
        self.vector_engine.len()
    }

    /// Get keyword index size.
    pub fn keyword_index_len(&self) -> usize {
        self.keyword_engine.len()
    }

    /// Get a document by ID.
    pub async fn get_document(
        &self,
        doc_id: &DocId,
    ) -> Result<Option<DocumentRecord>, SearchError> {
        self.store
            .get_document(*doc_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))
    }

    /// Clear all documents from both memory and persistent storage.
    pub async fn clear_all(&mut self) -> Result<(), SearchError> {
        // Clear persistent storage
        self.store
            .clear()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Clear in-memory indices
        self.vector_engine = VectorSearchEngine::new(self.embedding_dim);
        self.keyword_engine = KeywordSearchEngine::new();
        self.manifest = IndexManifest::new(self.embedding_dim);

        info!("Cleared all index data (memory and storage)");
        Ok(())
    }

    // =========================================================================
    // Source Tracking (for re-upload detection)
    // =========================================================================

    /// Get a source record by its source_id.
    ///
    /// source_id is platform-specific:
    /// - Desktop: full file path (e.g., "/Users/matt/docs/README.md")
    /// - Web: "web:{filename}" (e.g., "web:README.md")
    /// - Crawler: full URL (e.g., "https://example.com/docs/intro")
    ///
    /// Returns `Ok(None)` if the source doesn't exist.
    pub async fn get_source(
        &self,
        source_id: &str,
    ) -> Result<Option<super::types::SourceRecord>, SearchError> {
        self.store
            .get_source(source_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))
    }

    /// Register a new source before indexing begins.
    ///
    /// Creates an incomplete source record with the given content hash.
    /// Call `add_doc_to_source` for each chunk, then `complete_source` when done.
    ///
    /// # Arguments
    /// * `source_id` - Unique identifier for the source (path, URL, etc.)
    /// * `content_hash` - SHA-256 hash of the source content
    pub async fn register_source(
        &self,
        source_id: &str,
        content_hash: String,
    ) -> Result<(), SearchError> {
        let record = super::types::SourceRecord::new_incomplete(content_hash);
        self.store
            .put_source(source_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;
        info!("Registered source: {}", source_id);
        Ok(())
    }

    /// Add a document ID to a source's chunk list.
    ///
    /// Call this after adding each chunk from a source.
    pub async fn add_doc_to_source(
        &self,
        source_id: &str,
        doc_id: DocId,
    ) -> Result<(), SearchError> {
        let mut record = self
            .store
            .get_source(source_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?
            .ok_or(SearchError::NotFound)?;

        record.doc_ids.push(doc_id);

        self.store
            .put_source(source_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Mark a source as completely indexed.
    ///
    /// Call this after all chunks from the source have been added.
    pub async fn complete_source(&self, source_id: &str) -> Result<(), SearchError> {
        let mut record = self
            .store
            .get_source(source_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?
            .ok_or(SearchError::NotFound)?;

        record.mark_complete();

        self.store
            .put_source(source_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        info!(
            "Completed source: {} ({} chunks)",
            source_id,
            record.doc_ids.len()
        );
        Ok(())
    }

    /// Delete all documents from a source (soft delete via tombstones).
    ///
    /// This marks all chunks from the source as tombstoned in the vector index
    /// and deletes the documents/embeddings from storage. The source record
    /// is also deleted.
    ///
    /// # Returns
    /// Number of chunks tombstoned.
    pub async fn delete_source(&mut self, source_id: &str) -> Result<usize, SearchError> {
        // Get the source record
        let record = match self.store.get_source(source_id).await {
            Ok(Some(r)) => r,
            Ok(None) => {
                warn!("Source not found for deletion: {}", source_id);
                return Ok(0);
            }
            Err(e) => return Err(SearchError::StorageError(e.to_string())),
        };

        let chunk_count = record.doc_ids.len();

        // Tombstone each document in the vector index
        for doc_id in &record.doc_ids {
            // Find the index position for this doc_id
            if let Some(idx) = self.vector_engine.find_index(*doc_id) {
                self.vector_engine.mark_tombstone(idx);
            }

            // Delete from storage (documents and embeddings)
            let _ = self.store.delete_document(*doc_id).await;
            let _ = self.store.delete_embedding(*doc_id).await;
        }

        // Delete the source record
        self.store
            .delete_source(source_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Note: Tombstones are session-local only (not persisted).
        // Documents/embeddings are deleted from storage immediately above,
        // so on restart the index rebuilds without them.

        // Update manifest
        self.manifest.document_count = self.manifest.document_count.saturating_sub(chunk_count);

        info!(
            "Deleted source: {} ({} chunks tombstoned)",
            source_id, chunk_count
        );
        debug!(
            "Tombstone count after deletion: {}/{}",
            self.vector_engine.tombstone_count(),
            self.vector_engine.len()
        );
        Ok(chunk_count)
    }

    /// List all tracked source IDs.
    pub async fn list_sources(&self) -> Result<Vec<String>, SearchError> {
        self.store
            .list_sources()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))
    }

    /// Check if a source has changed by comparing content hashes.
    ///
    /// # Returns
    /// - `Ok(true)` if the source is new or has changed
    /// - `Ok(false)` if the source exists with the same hash
    pub async fn source_needs_update(
        &self,
        source_id: &str,
        new_content_hash: &str,
    ) -> Result<bool, SearchError> {
        match self.get_source(source_id).await? {
            Some(record) => Ok(record.content_hash != new_content_hash),
            None => Ok(true), // New source
        }
    }

    // =========================================================================
    // Compaction (reclaim space from tombstoned entries)
    // =========================================================================

    /// Check if the vector index needs compaction.
    ///
    /// Returns `true` if tombstone ratio exceeds 30% threshold.
    pub fn needs_compaction(&self) -> bool {
        self.vector_engine.needs_compaction()
    }

    /// Get compaction statistics.
    ///
    /// Returns (tombstone_count, total_count, ratio).
    pub fn compaction_stats(&self) -> (usize, usize, f32) {
        let tombstone_count = self.vector_engine.tombstone_count();
        let total_count = self.vector_engine.len();
        let ratio = self.vector_engine.tombstone_ratio();
        (tombstone_count, total_count, ratio)
    }

    /// Run compaction to remove tombstoned entries from the vector index.
    ///
    /// This rebuilds the HNSW index from scratch, loading embeddings from
    /// the document store for all live (non-tombstoned) entries.
    ///
    /// # Returns
    /// The number of entries in the compacted index.
    ///
    /// # Note
    /// This is an expensive operation that:
    /// 1. Loads all live embeddings from storage
    /// 2. Rebuilds the entire HNSW graph
    /// 3. Clears tombstones from storage
    ///
    /// Search continues to work during compaction (on old index until swap).
    #[instrument(skip(self), fields(before_size = self.vector_engine.len()))]
    pub async fn compact(&mut self) -> Result<usize, SearchError> {
        let (tombstone_count, total_count, ratio) = self.compaction_stats();

        if tombstone_count == 0 {
            info!("Compaction skipped: no tombstones");
            return Ok(total_count);
        }

        info!(
            "Starting compaction: {} tombstones / {} total ({:.1}%)",
            tombstone_count,
            total_count,
            ratio * 100.0
        );

        // Get all live entries (non-tombstoned)
        let live_entries = self.vector_engine.get_live_entries();

        // Load embeddings from storage for live entries
        let mut entries_with_embeddings = Vec::with_capacity(live_entries.len());

        for (_idx, doc_id) in live_entries {
            match self.store.get_embedding(doc_id).await {
                Ok(Some(embedding)) => {
                    entries_with_embeddings.push((doc_id, embedding));
                }
                Ok(None) => {
                    warn!(
                        "Embedding not found for doc_id {} during compaction, skipping",
                        doc_id.as_u64()
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to load embedding for doc_id {} during compaction: {}",
                        doc_id.as_u64(),
                        e
                    );
                }
            }
        }

        // Rebuild the vector index with only live entries
        // This also clears the in-memory tombstones
        let compacted_count = self.vector_engine.compact(entries_with_embeddings)?;

        info!(
            "Compaction complete: {} → {} entries ({} removed)",
            total_count,
            compacted_count,
            total_count - compacted_count
        );

        Ok(compacted_count)
    }

    /// Conditionally run compaction if needed.
    ///
    /// Only compacts if tombstone ratio exceeds 30% threshold.
    ///
    /// # Returns
    /// - `Ok(Some(count))` if compaction was performed, with the new entry count
    /// - `Ok(None)` if compaction was not needed
    pub async fn compact_if_needed(&mut self) -> Result<Option<usize>, SearchError> {
        if self.needs_compaction() {
            let count = self.compact().await?;
            Ok(Some(count))
        } else {
            Ok(None)
        }
    }

    /// Debug dump of the index state.
    pub fn debug_dump(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Search Index Debug Dump ===\n");
        output.push_str(&format!(
            "Total documents: {}\n",
            self.manifest.document_count
        ));
        output.push_str(&format!("Embedding dimension: {}\n", self.embedding_dim));
        output.push_str(&format!(
            "Vector index size: {}\n",
            self.vector_engine.len()
        ));
        output.push_str(&format!(
            "Tombstones: {}\n",
            self.vector_engine.get_tombstones().len()
        ));
        output.push_str("\n=== End Debug Dump ===\n");
        output
    }

    /// Debug dump with document details (async version).
    pub async fn debug_dump_full(&self) -> Result<String, SearchError> {
        let mut output = String::new();
        output.push_str("=== Search Index Debug Dump ===\n");
        output.push_str(&format!(
            "Total documents: {}\n",
            self.manifest.document_count
        ));
        output.push_str(&format!("Embedding dimension: {}\n", self.embedding_dim));
        output.push_str(&format!(
            "Vector index size: {}\n",
            self.vector_engine.len()
        ));
        output.push_str(&format!(
            "Tombstones: {}\n",
            self.vector_engine.get_tombstones().len()
        ));
        output.push('\n');

        // Load a sample of documents
        let embeddings = self
            .store
            .iter_embeddings()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        if embeddings.is_empty() {
            output.push_str("(empty index)\n");
        } else {
            output.push_str("Documents (first 10):\n");
            for (idx, (doc_id, _)) in embeddings.iter().take(10).enumerate() {
                if let Ok(Some(doc)) = self.store.get_document(*doc_id).await {
                    output.push_str(&format!("\n[{}] DocId: {}\n", idx + 1, doc_id.as_u64()));
                    output.push_str(&format!("  Filename: {:?}\n", doc.metadata.filename));
                    output.push_str(&format!("  Source: {:?}\n", doc.metadata.source));
                    if doc.text.len() > DEBUG_TEXT_PREVIEW_LEN {
                        output.push_str(&format!(
                            "  Text: {}...\n",
                            &doc.text[..DEBUG_TEXT_PREVIEW_LEN]
                        ));
                    } else {
                        output.push_str(&format!("  Text: {}\n", &doc.text));
                    }
                }
            }
        }

        output.push_str("\n=== End Debug Dump ===\n");
        Ok(output)
    }
}

// Conversion from StoreError to SearchError
impl From<StoreError> for SearchError {
    fn from(e: StoreError) -> Self {
        SearchError::StorageError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryDocumentStore;

    #[tokio::test]
    async fn test_hybrid_search_engine() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add documents
        let doc1 = Document {
            text: "machine learning algorithms".to_string(),
            metadata: DocumentMetadata {
                filename: Some("doc1.txt".to_string()),
                source: None,
                created_at: 0,
            },
        };

        let doc2 = Document {
            text: "deep neural networks".to_string(),
            metadata: DocumentMetadata {
                filename: Some("doc2.txt".to_string()),
                source: None,
                created_at: 1,
            },
        };

        // Dummy embeddings (in practice, these come from JinaBERT)
        engine
            .add_document(doc1, vec![1.0, 0.0, 0.0])
            .await
            .unwrap();
        engine
            .add_document(doc2, vec![0.9, 0.1, 0.0])
            .await
            .unwrap();

        // Search
        let results = engine
            .search(&[1.0, 0.0, 0.0], "machine learning", 2)
            .await
            .unwrap();

        assert!(results.len() <= 2);
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_add_document_dimension_mismatch() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let doc = Document {
            text: "test".to_string(),
            metadata: DocumentMetadata {
                filename: Some("test.txt".to_string()),
                source: None,
                created_at: 0,
            },
        };

        // Try to add document with wrong embedding dimension
        let result = engine.add_document(doc, vec![1.0, 0.0]).await; // 2D instead of 3D

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(SearchError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[tokio::test]
    async fn test_search_empty_index() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Search empty index
        let results = engine.search(&[1.0, 0.0, 0.0], "query", 10).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_dimension_mismatch() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let doc = Document {
            text: "test document".to_string(),
            metadata: DocumentMetadata::default(),
        };

        engine.add_document(doc, vec![1.0, 0.0, 0.0]).await.unwrap();

        // Try to search with wrong dimension
        let result = engine.search(&[1.0, 0.0], "query", 10).await; // 2D instead of 3D

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(SearchError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[tokio::test]
    async fn test_batch_add_deferred() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add multiple documents without rebuilding index each time
        for i in 0..5 {
            let doc = Document {
                text: format!("document {}", i),
                metadata: DocumentMetadata {
                    filename: Some(format!("doc{}.txt", i)),
                    source: None,
                    created_at: i as u64,
                },
            };

            engine
                .add_document_deferred(doc, vec![i as f32, 0.0, 0.0])
                .await
                .unwrap();
        }

        // Rebuild index once
        engine.rebuild_vector_index().await.unwrap();

        // Verify all documents are indexed
        assert_eq!(engine.len(), 5);
        assert_eq!(engine.vector_index_len(), 5);

        // Search should work after rebuild
        let results = engine
            .search(&[2.0, 0.0, 0.0], "document 2", 3)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_get_index_metrics() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Empty index
        let (chunks, tokens, avg) = engine.get_index_metrics().await.unwrap();
        assert_eq!(chunks, 0);
        assert_eq!(tokens, 0);
        assert_eq!(avg, 0.0);

        // Add documents
        let doc1 = Document {
            text: "one two three".to_string(),
            metadata: DocumentMetadata::default(),
        };
        let doc2 = Document {
            text: "four five".to_string(),
            metadata: DocumentMetadata::default(),
        };

        engine
            .add_document(doc1, vec![1.0, 0.0, 0.0])
            .await
            .unwrap();
        engine
            .add_document(doc2, vec![0.0, 1.0, 0.0])
            .await
            .unwrap();

        let (chunks, _, _) = engine.get_index_metrics().await.unwrap();
        assert_eq!(chunks, 2);
    }

    #[tokio::test]
    async fn test_vector_and_keyword_index_sync() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add documents
        for i in 0..3 {
            let doc = Document {
                text: format!("document {}", i),
                metadata: DocumentMetadata::default(),
            };
            engine
                .add_document(doc, vec![i as f32, 0.0, 0.0])
                .await
                .unwrap();
        }

        // Both indexes should have same count
        assert_eq!(engine.len(), 3);
        assert_eq!(engine.vector_index_len(), 3);
        assert_eq!(engine.keyword_index_len(), 3);
    }

    #[tokio::test]
    async fn test_clear_index() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add documents
        let doc = Document {
            text: "test document".to_string(),
            metadata: DocumentMetadata::default(),
        };
        engine.add_document(doc, vec![1.0, 0.0, 0.0]).await.unwrap();

        assert_eq!(engine.len(), 1);

        // Clear
        engine.clear_all().await.unwrap();

        assert_eq!(engine.len(), 0);
        assert!(engine.is_empty());
        assert_eq!(engine.vector_index_len(), 0);

        // Should be able to add documents after clear
        let doc2 = Document {
            text: "new document".to_string(),
            metadata: DocumentMetadata::default(),
        };
        let result = engine.add_document(doc2, vec![1.0, 0.0, 0.0]).await;
        assert!(result.is_ok());
        assert_eq!(engine.len(), 1);
    }

    #[tokio::test]
    async fn test_get_document() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let doc = Document {
            text: "test document".to_string(),
            metadata: DocumentMetadata {
                filename: Some("test.txt".to_string()),
                source: Some("manual".to_string()),
                created_at: 12345,
            },
        };

        let doc_id = engine
            .add_document(doc.clone(), vec![1.0, 0.0, 0.0])
            .await
            .unwrap();

        // Get document by ID
        let retrieved = engine.get_document(&doc_id).await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, doc_id);
        assert_eq!(retrieved.text, doc.text);
        assert_eq!(retrieved.metadata.filename, doc.metadata.filename);
        assert_eq!(retrieved.metadata.source, doc.metadata.source);
        assert_eq!(retrieved.metadata.created_at, doc.metadata.created_at);
    }

    #[tokio::test]
    async fn test_debug_dump() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Empty index
        let dump = engine.debug_dump();
        assert!(dump.contains("Total documents: 0"));

        // Add document
        let doc = Document {
            text: "This is a test document with some content".to_string(),
            metadata: DocumentMetadata {
                filename: Some("test.txt".to_string()),
                source: Some("test".to_string()),
                created_at: 123,
            },
        };
        engine.add_document(doc, vec![1.0, 0.0, 0.0]).await.unwrap();

        let dump = engine.debug_dump();
        assert!(dump.contains("Total documents: 1"));
    }

    #[tokio::test]
    async fn test_search_returns_top_k() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add 10 documents
        for i in 0..10 {
            let doc = Document {
                text: format!("document number {}", i),
                metadata: DocumentMetadata::default(),
            };
            engine
                .add_document(doc, vec![i as f32, 0.0, 0.0])
                .await
                .unwrap();
        }

        // Request top 3
        let results = engine
            .search(&[5.0, 0.0, 0.0], "document", 3)
            .await
            .unwrap();

        // Should return at most 3 results
        assert!(results.len() <= 3);

        // Verify results have scores
        for result in &results {
            assert!(result.score > 0.0);
        }
    }

    #[tokio::test]
    async fn test_search_result_structure() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let doc = Document {
            text: "semantic search test".to_string(),
            metadata: DocumentMetadata {
                filename: Some("search.txt".to_string()),
                source: None,
                created_at: 999,
            },
        };

        engine
            .add_document(doc.clone(), vec![1.0, 0.5, 0.2])
            .await
            .unwrap();

        let results = engine
            .search(&[1.0, 0.5, 0.2], "semantic", 1)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];

        // Verify SearchResult structure
        assert!(result.score > 0.0); // RRF fused score
        assert!(result.vector_score.is_some()); // Vector search score
        assert!(result.keyword_score.is_some()); // BM25 score
        assert_eq!(result.text, doc.text);
        assert_eq!(result.metadata.filename, doc.metadata.filename);
        assert_eq!(result.metadata.created_at, doc.metadata.created_at);
    }

    #[tokio::test]
    async fn test_search_empty_query_text() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let doc = Document {
            text: "test document".to_string(),
            metadata: DocumentMetadata::default(),
        };
        engine.add_document(doc, vec![1.0, 0.0, 0.0]).await.unwrap();

        // Empty query text should return InvalidQuery error
        let result = engine.search(&[1.0, 0.0, 0.0], "", 10).await;
        assert!(matches!(result, Err(SearchError::InvalidQuery(_))));

        // Whitespace-only query should also fail
        let result = engine.search(&[1.0, 0.0, 0.0], "   \t\n  ", 10).await;
        assert!(matches!(result, Err(SearchError::InvalidQuery(_))));
    }

    #[tokio::test]
    async fn test_search_zero_k() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let doc = Document {
            text: "test document".to_string(),
            metadata: DocumentMetadata::default(),
        };
        engine.add_document(doc, vec![1.0, 0.0, 0.0]).await.unwrap();

        // k=0 should return InvalidQuery error
        let result = engine.search(&[1.0, 0.0, 0.0], "test", 0).await;
        assert!(matches!(result, Err(SearchError::InvalidQuery(_))));
    }

    #[tokio::test]
    async fn test_persistence_reload() {
        use std::sync::Arc;

        let store = Arc::new(InMemoryDocumentStore::new());

        // Create engine and add document
        let doc_id = {
            let mut engine = HybridSearchEngine::new(Arc::clone(&store), 3)
                .await
                .unwrap();

            let doc = Document {
                text: "persistent document".to_string(),
                metadata: DocumentMetadata {
                    filename: Some("persist.txt".to_string()),
                    source: None,
                    created_at: 42,
                },
            };

            let doc_id = engine.add_document(doc, vec![1.0, 2.0, 3.0]).await.unwrap();
            engine.save().await.unwrap();
            doc_id
        };

        // Reload from store (simulating app restart with same underlying store)
        let mut engine = HybridSearchEngine::try_load_or_new(Arc::clone(&store), 3)
            .await
            .unwrap();

        // Verify document is present
        assert_eq!(engine.len(), 1);
        let retrieved = engine.get_document(&doc_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().text, "persistent document");

        // Search should work
        let results = engine
            .search(&[1.0, 2.0, 3.0], "persistent", 1)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    // =========================================================================
    // Source Tracking Tests
    // =========================================================================

    #[tokio::test]
    async fn test_source_registration_and_lookup() {
        let store = InMemoryDocumentStore::new();
        let engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let source_id = "/Users/test/README.md";
        let content_hash = "abc123def456".to_string();

        // Source shouldn't exist initially
        assert!(engine.get_source(source_id).await.unwrap().is_none());

        // Register source
        engine
            .register_source(source_id, content_hash.clone())
            .await
            .unwrap();

        // Source should exist and be incomplete
        let record = engine.get_source(source_id).await.unwrap().unwrap();
        assert_eq!(record.content_hash, content_hash);
        assert!(!record.complete);
        assert!(record.doc_ids.is_empty());
    }

    #[tokio::test]
    async fn test_source_document_tracking() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let source_id = "web:test.txt";
        let content_hash = "hash123".to_string();

        // Register source
        engine
            .register_source(source_id, content_hash)
            .await
            .unwrap();

        // Add documents and track them
        let doc1 = Document {
            text: "First chunk".to_string(),
            metadata: DocumentMetadata {
                filename: Some("test.txt (chunk 1)".to_string()),
                source: Some(source_id.to_string()),
                created_at: 100,
            },
        };
        let doc_id1 = engine
            .add_document(doc1, vec![1.0, 0.0, 0.0])
            .await
            .unwrap();
        engine.add_doc_to_source(source_id, doc_id1).await.unwrap();

        let doc2 = Document {
            text: "Second chunk".to_string(),
            metadata: DocumentMetadata {
                filename: Some("test.txt (chunk 2)".to_string()),
                source: Some(source_id.to_string()),
                created_at: 100,
            },
        };
        let doc_id2 = engine
            .add_document(doc2, vec![0.0, 1.0, 0.0])
            .await
            .unwrap();
        engine.add_doc_to_source(source_id, doc_id2).await.unwrap();

        // Complete the source
        engine.complete_source(source_id).await.unwrap();

        // Verify source record
        let record = engine.get_source(source_id).await.unwrap().unwrap();
        assert!(record.complete);
        assert_eq!(record.doc_ids.len(), 2);
        assert!(record.doc_ids.contains(&doc_id1));
        assert!(record.doc_ids.contains(&doc_id2));
    }

    #[tokio::test]
    async fn test_source_needs_update() {
        let store = InMemoryDocumentStore::new();
        let engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let source_id = "/path/to/file.md";
        let hash_v1 = "version1hash".to_string();
        let hash_v2 = "version2hash".to_string();

        // New source needs update
        assert!(engine
            .source_needs_update(source_id, &hash_v1)
            .await
            .unwrap());

        // Register source
        engine
            .register_source(source_id, hash_v1.clone())
            .await
            .unwrap();

        // Same hash - no update needed
        assert!(!engine
            .source_needs_update(source_id, &hash_v1)
            .await
            .unwrap());

        // Different hash - update needed
        assert!(engine
            .source_needs_update(source_id, &hash_v2)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_delete_source() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        let source_id = "web:delete-test.txt";

        // Register and add documents
        engine
            .register_source(source_id, "hash".to_string())
            .await
            .unwrap();

        let doc1 = Document {
            text: "Chunk to delete".to_string(),
            metadata: DocumentMetadata {
                filename: Some("delete-test.txt".to_string()),
                source: Some(source_id.to_string()),
                created_at: 0,
            },
        };
        let doc_id1 = engine
            .add_document(doc1, vec![1.0, 0.0, 0.0])
            .await
            .unwrap();
        engine.add_doc_to_source(source_id, doc_id1).await.unwrap();

        let doc2 = Document {
            text: "Another chunk".to_string(),
            metadata: DocumentMetadata {
                filename: Some("delete-test.txt".to_string()),
                source: Some(source_id.to_string()),
                created_at: 0,
            },
        };
        let doc_id2 = engine
            .add_document(doc2, vec![0.0, 1.0, 0.0])
            .await
            .unwrap();
        engine.add_doc_to_source(source_id, doc_id2).await.unwrap();

        engine.complete_source(source_id).await.unwrap();

        // Verify documents are searchable
        assert_eq!(engine.len(), 2);

        // Delete the source
        let deleted = engine.delete_source(source_id).await.unwrap();
        assert_eq!(deleted, 2);

        // Source should be gone
        assert!(engine.get_source(source_id).await.unwrap().is_none());

        // Documents should be tombstoned (excluded from search)
        let results = engine
            .search(&[1.0, 0.0, 0.0], "chunk delete", 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_list_sources() {
        let store = InMemoryDocumentStore::new();
        let engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Initially empty
        let sources = engine.list_sources().await.unwrap();
        assert!(sources.is_empty());

        // Add some sources
        engine
            .register_source("/path/a.md", "hash1".to_string())
            .await
            .unwrap();
        engine
            .register_source("/path/b.md", "hash2".to_string())
            .await
            .unwrap();
        engine
            .register_source("web:c.txt", "hash3".to_string())
            .await
            .unwrap();

        // List should contain all sources
        let sources = engine.list_sources().await.unwrap();
        assert_eq!(sources.len(), 3);
        assert!(sources.contains(&"/path/a.md".to_string()));
        assert!(sources.contains(&"/path/b.md".to_string()));
        assert!(sources.contains(&"web:c.txt".to_string()));
    }

    // =========================================================================
    // Compaction Tests
    // =========================================================================

    #[tokio::test]
    async fn test_compaction_stats() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add 5 documents
        for i in 0..5 {
            let doc = Document {
                text: format!("document {}", i),
                metadata: DocumentMetadata::default(),
            };
            engine
                .add_document(doc, vec![i as f32, 0.0, 0.0])
                .await
                .unwrap();
        }

        // Initially no tombstones
        let (tombstone_count, total_count, ratio) = engine.compaction_stats();
        assert_eq!(tombstone_count, 0);
        assert_eq!(total_count, 5);
        assert_eq!(ratio, 0.0);
        assert!(!engine.needs_compaction());
    }

    #[tokio::test]
    async fn test_compaction_after_delete() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add source with 5 documents
        let source_id = "test-source";
        engine
            .register_source(source_id, "hash1".to_string())
            .await
            .unwrap();

        for i in 0..5 {
            let doc = Document {
                text: format!("document {}", i),
                metadata: DocumentMetadata {
                    filename: Some(format!("doc{}.txt", i)),
                    source: Some(source_id.to_string()),
                    created_at: i as u64,
                },
            };
            let doc_id = engine
                .add_document(doc, vec![i as f32, 0.0, 0.0])
                .await
                .unwrap();
            engine.add_doc_to_source(source_id, doc_id).await.unwrap();
        }
        engine.complete_source(source_id).await.unwrap();

        // Delete the source (creates tombstones)
        engine.delete_source(source_id).await.unwrap();

        // Now we have 100% tombstones
        let (tombstone_count, total_count, ratio) = engine.compaction_stats();
        assert_eq!(tombstone_count, 5);
        assert_eq!(total_count, 5);
        assert!((ratio - 1.0).abs() < 0.01);
        assert!(engine.needs_compaction());

        // Run compaction
        let compacted_count = engine.compact().await.unwrap();
        assert_eq!(compacted_count, 0); // All entries were tombstoned

        // After compaction
        let (tombstone_count, total_count, _) = engine.compaction_stats();
        assert_eq!(tombstone_count, 0);
        assert_eq!(total_count, 0);
        assert!(!engine.needs_compaction());
    }

    #[tokio::test]
    async fn test_compact_if_needed() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add 10 documents
        let source_id = "test-source";
        engine
            .register_source(source_id, "hash1".to_string())
            .await
            .unwrap();

        for i in 0..10 {
            let doc = Document {
                text: format!("document {}", i),
                metadata: DocumentMetadata {
                    filename: Some(format!("doc{}.txt", i)),
                    source: Some(source_id.to_string()),
                    created_at: i as u64,
                },
            };
            let doc_id = engine
                .add_document(doc, vec![i as f32, 0.0, 0.0])
                .await
                .unwrap();
            engine.add_doc_to_source(source_id, doc_id).await.unwrap();
        }
        engine.complete_source(source_id).await.unwrap();

        // No compaction needed yet
        let result = engine.compact_if_needed().await.unwrap();
        assert!(result.is_none());

        // Delete source (creates tombstones)
        engine.delete_source(source_id).await.unwrap();

        // Now compaction is needed (100% tombstones > 30% threshold)
        let result = engine.compact_if_needed().await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_compaction_preserves_live_entries() {
        let store = InMemoryDocumentStore::new();
        let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

        // Add source1 with 3 documents
        let source1 = "source1";
        engine
            .register_source(source1, "hash1".to_string())
            .await
            .unwrap();

        for i in 0..3 {
            let doc = Document {
                text: format!("source1 document {}", i),
                metadata: DocumentMetadata {
                    filename: Some(format!("s1-doc{}.txt", i)),
                    source: Some(source1.to_string()),
                    created_at: i as u64,
                },
            };
            let doc_id = engine
                .add_document(doc, vec![i as f32, 0.0, 0.0])
                .await
                .unwrap();
            engine.add_doc_to_source(source1, doc_id).await.unwrap();
        }
        engine.complete_source(source1).await.unwrap();

        // Add source2 with 2 documents
        let source2 = "source2";
        engine
            .register_source(source2, "hash2".to_string())
            .await
            .unwrap();

        for i in 0..2 {
            let doc = Document {
                text: format!("source2 document {}", i),
                metadata: DocumentMetadata {
                    filename: Some(format!("s2-doc{}.txt", i)),
                    source: Some(source2.to_string()),
                    created_at: (i + 100) as u64,
                },
            };
            let doc_id = engine
                .add_document(doc, vec![(i + 10) as f32, 0.0, 0.0])
                .await
                .unwrap();
            engine.add_doc_to_source(source2, doc_id).await.unwrap();
        }
        engine.complete_source(source2).await.unwrap();

        // Delete source1 (tombstones 3 of 5 = 60% > 30% threshold)
        engine.delete_source(source1).await.unwrap();

        assert!(engine.needs_compaction());

        // Compact
        let compacted_count = engine.compact().await.unwrap();
        assert_eq!(compacted_count, 2); // Only source2's documents remain

        // source2 documents should still be searchable
        let results = engine
            .search(&[10.0, 0.0, 0.0], "source2", 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }
}
