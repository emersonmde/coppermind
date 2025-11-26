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
use tracing::{info, instrument, warn};

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

        // Load tombstones from store
        let tombstones = engine
            .store
            .get_tombstones()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Apply tombstones to vector engine
        for idx in tombstones {
            engine.vector_engine.mark_tombstone(idx);
        }

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
    /// With DocumentStore, writes happen immediately during add_document(),
    /// so this mainly ensures tombstones are persisted.
    pub async fn save(&mut self) -> Result<(), SearchError> {
        // Save tombstones
        let tombstones = self.vector_engine.get_tombstones();
        self.store
            .put_tombstones(&tombstones)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        info!(
            "Saved index state: {} documents",
            self.manifest.document_count
        );
        Ok(())
    }

    /// Check if there are unsaved changes.
    ///
    /// With DocumentStore, documents are saved immediately on add,
    /// but tombstones may need saving.
    pub fn is_dirty(&self) -> bool {
        // Tombstones may have changed since last save
        !self.vector_engine.get_tombstones().is_empty()
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
}
