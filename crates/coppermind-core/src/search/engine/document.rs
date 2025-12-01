//! Document-level indexing for ADR-008 Multi-Level Indexing.
//!
//! This module provides methods for indexing at the document level, enabling
//! proper IDF (Inverse Document Frequency) statistics for BM25 keyword search.
//!
//! # Usage
//!
//! ```ignore
//! // Create document first
//! let doc_id = engine.create_document(source_id, title, hash, full_text).await?;
//!
//! // Add chunks to the document
//! for (chunk, embedding) in chunks {
//!     engine.add_chunk_to_document(doc_id, chunk, embedding).await?;
//! }
//!
//! // Finalize to update chunk count
//! engine.finalize_document(doc_id).await?;
//! ```
//!
//! # Why Document-Level Indexing?
//!
//! See ADR-008 for details. In brief: chunk-level BM25 dilutes IDF statistics
//! because a term appearing in 5 chunks of one document is treated as appearing
//! in 5 "documents", artificially lowering its discriminative value.

use super::HybridSearchEngine;
use crate::search::types::{
    validate_dimension, Chunk, ChunkId, ChunkRecord, DocumentId, DocumentMetainfo, DocumentRecord,
    SearchError,
};
use crate::storage::DocumentStore;
use tracing::{info, instrument};

/// Maximum characters to show in debug dump text preview.
const DEBUG_TEXT_PREVIEW_LEN: usize = 100;

impl<S: DocumentStore> HybridSearchEngine<S> {
    /// Create a new document and return its ID.
    ///
    /// Call this before adding chunks. The document record is persisted immediately.
    ///
    /// # Arguments
    /// * `source_id` - Unique identifier for the source (file path, URL)
    /// * `title` - Display name for the document
    /// * `content_hash` - SHA-256 hash of the document content
    /// * `full_text` - Full text of the document (for document-level BM25)
    ///
    /// # Returns
    /// The assigned DocumentId
    #[instrument(skip_all, fields(source_id = %source_id))]
    pub async fn create_document(
        &mut self,
        source_id: &str,
        title: &str,
        content_hash: &str,
        full_text: &str,
    ) -> Result<DocumentId, SearchError> {
        let doc_id = DocumentId::new();

        let metadata = DocumentMetainfo::new(
            source_id.to_string(),
            title.to_string(),
            content_hash.to_string(),
            full_text.len(),
            0, // chunk_count will be updated in finalize_document
        );

        let record = DocumentRecord::new(doc_id, metadata, Vec::new());

        // Persist document record
        self.store
            .put_document(doc_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Add to document-level BM25 immediately
        self.document_keyword_engine
            .add_document(doc_id, full_text.to_string());

        // Update manifest document count
        self.manifest.doc_count += 1;

        info!(
            "Created document {} for source: {} ({} chars)",
            doc_id.as_u64(),
            source_id,
            full_text.len()
        );

        Ok(doc_id)
    }

    /// Add a chunk belonging to a document.
    ///
    /// Adds the chunk to HNSW and chunk-level BM25 indexes, and associates
    /// it with the parent document. The chunk's `document_id` field is set
    /// for O(1) document lookups in `search_documents()`.
    ///
    /// # Arguments
    /// * `document_id` - Parent document ID (from create_document)
    /// * `chunk` - Chunk containing text and metadata
    /// * `embedding` - Pre-computed embedding vector
    ///
    /// # Returns
    /// The assigned ChunkId
    ///
    /// # Note
    /// This method does not track token counts. Use [`Self::add_chunk_to_document_with_tokens`]
    /// if you need to track token counts in the index manifest.
    #[instrument(skip_all, fields(doc_id = document_id.as_u64()))]
    pub async fn add_chunk_to_document(
        &mut self,
        document_id: DocumentId,
        chunk: Chunk,
        embedding: Vec<f32>,
    ) -> Result<ChunkId, SearchError> {
        self.add_chunk_to_document_with_tokens(document_id, chunk, embedding, 0)
            .await
    }

    /// Add a chunk belonging to a document with token count tracking.
    ///
    /// Like [`Self::add_chunk_to_document`] but also tracks token count in the index manifest
    /// for metrics display and persistence.
    ///
    /// # Arguments
    /// * `document_id` - Parent document ID (from create_document)
    /// * `chunk` - Chunk containing text and metadata
    /// * `embedding` - Pre-computed embedding vector
    /// * `token_count` - Number of tokens in this chunk (for metrics)
    ///
    /// # Returns
    /// The assigned ChunkId
    #[instrument(skip_all, fields(doc_id = document_id.as_u64(), tokens = token_count))]
    pub async fn add_chunk_to_document_with_tokens(
        &mut self,
        document_id: DocumentId,
        chunk: Chunk,
        embedding: Vec<f32>,
        token_count: usize,
    ) -> Result<ChunkId, SearchError> {
        // Validate embedding dimension
        validate_dimension(self.embedding_dim, embedding.len())?;

        // Generate unique ID
        let chunk_id = ChunkId::new();

        // Create chunk record WITH document_id for O(1) lookup
        let record = ChunkRecord {
            id: chunk_id,
            document_id: Some(document_id),
            text: chunk.text.clone(),
            metadata: chunk.metadata,
        };

        // Persist to store
        self.store
            .put_chunk(chunk_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        self.store
            .put_embedding(chunk_id, &embedding)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Add to in-memory indices with separate timing
        use crate::metrics::global_metrics;
        use instant::Instant;

        let hnsw_start = Instant::now();
        self.vector_engine.add_chunk(chunk_id, embedding)?;
        let hnsw_ms = hnsw_start.elapsed().as_secs_f64() * 1000.0;
        global_metrics().record_hnsw_indexing(hnsw_ms);

        let bm25_start = Instant::now();
        self.keyword_engine.add_chunk(chunk_id, record.text);
        let bm25_ms = bm25_start.elapsed().as_secs_f64() * 1000.0;
        global_metrics().record_bm25_indexing(bm25_ms);

        // Update manifest
        self.manifest.chunk_count += 1;
        if token_count > 0 {
            self.manifest.add_tokens(token_count);
        }

        // Update document record with chunk ID
        if let Ok(Some(mut doc_record)) = self.store.get_document(document_id).await {
            doc_record.chunk_ids.push(chunk_id);
            self.store
                .put_document(document_id, &doc_record)
                .await
                .map_err(|e| SearchError::StorageError(e.to_string()))?;
        }

        Ok(chunk_id)
    }

    /// Finalize document indexing.
    ///
    /// Updates the document record with final chunk count. Call this after
    /// all chunks have been added via add_chunk_to_document.
    ///
    /// # Arguments
    /// * `document_id` - Document ID to finalize
    #[instrument(skip_all, fields(doc_id = document_id.as_u64()))]
    pub async fn finalize_document(&mut self, document_id: DocumentId) -> Result<(), SearchError> {
        if let Ok(Some(mut record)) = self.store.get_document(document_id).await {
            let chunk_count = record.chunk_ids.len();
            record.metadata.chunk_count = chunk_count;

            self.store
                .put_document(document_id, &record)
                .await
                .map_err(|e| SearchError::StorageError(e.to_string()))?;

            info!(
                "Finalized document {} with {} chunks",
                document_id.as_u64(),
                chunk_count
            );
        }

        Ok(())
    }

    /// Get a document record by ID.
    ///
    /// Returns `Ok(None)` if the document doesn't exist.
    pub async fn get_document(
        &self,
        document_id: DocumentId,
    ) -> Result<Option<DocumentRecord>, SearchError> {
        self.store
            .get_document(document_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))
    }

    /// Get the number of indexed documents (full documents, not chunks).
    ///
    /// This counts documents created via `create_document()`, not legacy
    /// chunks indexed via `add_chunk()`.
    pub async fn document_count(&self) -> Result<usize, SearchError> {
        self.store
            .document_count()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))
    }

    /// Get the size of the document-level keyword index.
    ///
    /// This is the number of documents indexed in the document-level BM25 engine,
    /// used for proper IDF statistics.
    pub fn document_keyword_index_len(&self) -> usize {
        self.document_keyword_engine.len()
    }

    /// Debug dump of the index state.
    ///
    /// Returns a human-readable string with index statistics.
    /// For a more detailed dump including document samples, use `debug_dump_full()`.
    pub fn debug_dump(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Search Index Debug Dump ===\n");
        output.push_str(&format!("Total chunks: {}\n", self.manifest.chunk_count));
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
    ///
    /// Returns a human-readable string with index statistics and samples
    /// of the first 10 chunks.
    pub async fn debug_dump_full(&self) -> Result<String, SearchError> {
        let mut output = String::new();
        output.push_str("=== Search Index Debug Dump ===\n");
        output.push_str(&format!("Total chunks: {}\n", self.manifest.chunk_count));
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

        // Load a sample of chunks
        let embeddings = self
            .store
            .iter_embeddings()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        if embeddings.is_empty() {
            output.push_str("(empty index)\n");
        } else {
            output.push_str("Chunks (first 10):\n");
            for (idx, (chunk_id, _)) in embeddings.iter().take(10).enumerate() {
                if let Ok(Some(chunk)) = self.store.get_chunk(*chunk_id).await {
                    output.push_str(&format!("\n[{}] ChunkId: {}\n", idx + 1, chunk_id.as_u64()));
                    output.push_str(&format!("  Filename: {:?}\n", chunk.metadata.filename));
                    output.push_str(&format!("  Source: {:?}\n", chunk.metadata.source));
                    if chunk.text.len() > DEBUG_TEXT_PREVIEW_LEN {
                        output.push_str(&format!(
                            "  Text: {}...\n",
                            &chunk.text[..DEBUG_TEXT_PREVIEW_LEN]
                        ));
                    } else {
                        output.push_str(&format!("  Text: {}\n", &chunk.text));
                    }
                }
            }
        }

        output.push_str("\n=== End Debug Dump ===\n");
        Ok(output)
    }
}
