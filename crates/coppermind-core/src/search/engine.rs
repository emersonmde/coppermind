// HybridSearchEngine - combines vector and keyword search

use super::fusion::{reciprocal_rank_fusion, RRF_K};
use super::keyword::KeywordSearchEngine;
#[cfg(test)]
use super::types::DocumentMetadata;
use super::types::{
    validate_dimension, DocId, Document, DocumentRecord, IndexManifest, LoadResult,
    PersistedDocuments, SearchError, SearchResult, CURRENT_SCHEMA_VERSION,
};
use super::vector::VectorSearchEngine;
use crate::storage::{StorageBackend, StorageError};
use std::collections::HashMap;
use tracing::{info, instrument, warn};

/// Maximum characters to show in debug dump text preview.
const DEBUG_TEXT_PREVIEW_LEN: usize = 100;

/// Storage keys for persisted data.
const MANIFEST_KEY: &str = "index/manifest.json";
const DOCUMENTS_KEY: &str = "index/documents.json";
const EMBEDDINGS_KEY: &str = "index/embeddings.bin";

/// Hybrid search engine combining vector (semantic) and keyword (BM25) search
pub struct HybridSearchEngine<S: StorageBackend> {
    /// Vector search engine (semantic similarity)
    vector_engine: VectorSearchEngine,
    /// Keyword search engine (BM25)
    keyword_engine: KeywordSearchEngine,
    /// Document storage (metadata + text)
    documents: HashMap<DocId, DocumentRecord>,
    /// Embeddings storage (parallel to documents, keyed by DocId)
    /// Kept separate from VectorSearchEngine for persistence
    embeddings: HashMap<DocId, Vec<f32>>,
    /// Storage backend for persistence
    storage: S,
    /// Embedding dimension (e.g., 512 for JinaBERT)
    embedding_dim: usize,
    /// Index manifest for versioning
    manifest: IndexManifest,
    /// Whether there are unsaved changes
    dirty: bool,
}

impl<S: StorageBackend> HybridSearchEngine<S> {
    /// Create a new empty hybrid search engine (no persistence load).
    ///
    /// Use `try_load_or_new()` to create an engine that loads existing data.
    ///
    /// # Arguments
    /// * `storage` - Storage backend for persisting indexes
    /// * `embedding_dim` - Dimensionality of embeddings (must match the model)
    pub async fn new(storage: S, embedding_dim: usize) -> Result<Self, SearchError> {
        Ok(Self {
            vector_engine: VectorSearchEngine::new(embedding_dim),
            keyword_engine: KeywordSearchEngine::new(),
            documents: HashMap::new(),
            embeddings: HashMap::new(),
            storage,
            embedding_dim,
            manifest: IndexManifest::new(embedding_dim),
            dirty: false,
        })
    }

    /// Create a hybrid search engine, loading existing data if available.
    ///
    /// This is the preferred constructor for production use.
    ///
    /// # Returns
    /// - If no existing index: creates empty engine
    /// - If compatible index exists: loads and rebuilds indices
    /// - If incompatible index: returns error (caller should offer to clear)
    pub async fn try_load_or_new(storage: S, embedding_dim: usize) -> Result<Self, SearchError> {
        // Try to load existing index
        match Self::load_from_storage(&storage, embedding_dim).await? {
            LoadResult::NotFound => {
                info!("No existing index found, creating new engine");
                Self::new(storage, embedding_dim).await
            }
            LoadResult::Loaded {
                manifest,
                documents,
                embeddings,
            } => {
                info!(
                    "Loading existing index with {} documents",
                    manifest.document_count
                );
                Self::rebuild_from_data(storage, manifest, documents, embeddings).await
            }
            LoadResult::Incompatible { manifest, reason } => {
                warn!(
                    "Incompatible index version {} (min: {}): {}",
                    manifest.schema_version, manifest.min_compatible_version, reason
                );
                Err(SearchError::StorageError(format!(
                    "Incompatible index version: {}. Please clear storage to continue.",
                    reason
                )))
            }
        }
    }

    /// Load index data from storage without building indices.
    async fn load_from_storage(
        storage: &S,
        expected_dim: usize,
    ) -> Result<LoadResult, SearchError> {
        // Check if manifest exists
        let manifest_data = match storage.load(MANIFEST_KEY).await {
            Ok(data) => data,
            Err(StorageError::NotFound(_)) => return Ok(LoadResult::NotFound),
            Err(e) => return Err(SearchError::StorageError(e.to_string())),
        };

        // Parse manifest
        let manifest: IndexManifest = serde_json::from_slice(&manifest_data)
            .map_err(|e| SearchError::StorageError(format!("Failed to parse manifest: {}", e)))?;

        // Check compatibility
        if !manifest.is_compatible() {
            let reason = format!(
                "Index requires version >= {}, app has version {}",
                manifest.min_compatible_version, CURRENT_SCHEMA_VERSION
            );
            return Ok(LoadResult::Incompatible { manifest, reason });
        }

        // Check dimension match
        if manifest.embedding_dimension != expected_dim {
            let reason = format!(
                "Embedding dimension mismatch: index has {}, expected {}",
                manifest.embedding_dimension, expected_dim
            );
            return Ok(LoadResult::Incompatible { manifest, reason });
        }

        // Load documents
        let documents_data = storage
            .load(DOCUMENTS_KEY)
            .await
            .map_err(|e| SearchError::StorageError(format!("Failed to load documents: {}", e)))?;

        let documents: PersistedDocuments = serde_json::from_slice(&documents_data)
            .map_err(|e| SearchError::StorageError(format!("Failed to parse documents: {}", e)))?;

        // Load embeddings (binary format: doc_count * embedding_dim * f32)
        let embeddings_data = storage
            .load(EMBEDDINGS_KEY)
            .await
            .map_err(|e| SearchError::StorageError(format!("Failed to load embeddings: {}", e)))?;

        let embeddings = Self::deserialize_embeddings(&embeddings_data, expected_dim)?;

        // Verify counts match
        if documents.documents.len() != embeddings.len() {
            return Err(SearchError::StorageError(format!(
                "Document/embedding count mismatch: {} documents, {} embeddings",
                documents.documents.len(),
                embeddings.len()
            )));
        }

        Ok(LoadResult::Loaded {
            manifest,
            documents,
            embeddings,
        })
    }

    /// Rebuild indices from loaded data.
    async fn rebuild_from_data(
        storage: S,
        manifest: IndexManifest,
        documents: PersistedDocuments,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Self, SearchError> {
        let embedding_dim = manifest.embedding_dimension;
        let mut engine = Self {
            vector_engine: VectorSearchEngine::new(embedding_dim),
            keyword_engine: KeywordSearchEngine::new(),
            documents: HashMap::new(),
            embeddings: HashMap::new(),
            storage,
            embedding_dim,
            manifest,
            dirty: false,
        };

        // Track maximum ID to initialize the counter
        let mut max_id: u64 = 0;

        // Rebuild indices from documents and embeddings
        for (doc_record, embedding) in documents.documents.into_iter().zip(embeddings.into_iter()) {
            let doc_id = doc_record.id;

            // Track max ID
            max_id = max_id.max(doc_id.as_u64());

            // Add to vector index
            engine
                .vector_engine
                .add_document(doc_id, embedding.clone())?;

            // Add to keyword index
            engine
                .keyword_engine
                .add_document(doc_id, doc_record.text.clone());

            // Store in maps
            engine.embeddings.insert(doc_id, embedding);
            engine.documents.insert(doc_id, doc_record);
        }

        // Initialize DocId counter to continue after the highest loaded ID
        // This prevents ID collisions when adding new documents
        DocId::init_counter(max_id);

        info!(
            "Rebuilt indices: {} documents, {} vectors",
            engine.documents.len(),
            engine.vector_engine.len()
        );

        Ok(engine)
    }

    /// Save the current index to storage.
    pub async fn save(&mut self) -> Result<(), SearchError> {
        if !self.dirty && self.manifest.document_count == self.documents.len() {
            // No changes to save
            return Ok(());
        }

        // Update manifest
        self.manifest.update(self.documents.len());

        // Serialize manifest
        let manifest_json = serde_json::to_vec_pretty(&self.manifest).map_err(|e| {
            SearchError::StorageError(format!("Failed to serialize manifest: {}", e))
        })?;

        // Serialize documents (maintaining insertion order via doc_ids)
        let doc_records: Vec<DocumentRecord> = self.documents.values().cloned().collect();
        let documents = PersistedDocuments::from_documents(doc_records);
        let documents_json = serde_json::to_vec_pretty(&documents).map_err(|e| {
            SearchError::StorageError(format!("Failed to serialize documents: {}", e))
        })?;

        // Serialize embeddings (binary, same order as documents)
        let embeddings_bin = self.serialize_embeddings(&documents.documents)?;

        // Save all files
        self.storage
            .save(MANIFEST_KEY, &manifest_json)
            .await
            .map_err(|e| SearchError::StorageError(format!("Failed to save manifest: {}", e)))?;

        self.storage
            .save(DOCUMENTS_KEY, &documents_json)
            .await
            .map_err(|e| SearchError::StorageError(format!("Failed to save documents: {}", e)))?;

        self.storage
            .save(EMBEDDINGS_KEY, &embeddings_bin)
            .await
            .map_err(|e| SearchError::StorageError(format!("Failed to save embeddings: {}", e)))?;

        self.dirty = false;
        info!("Saved index: {} documents", self.documents.len());

        Ok(())
    }

    /// Serialize embeddings to binary format (little-endian f32).
    fn serialize_embeddings(&self, documents: &[DocumentRecord]) -> Result<Vec<u8>, SearchError> {
        let mut buffer = Vec::with_capacity(documents.len() * self.embedding_dim * 4);

        for doc in documents {
            let embedding = self.embeddings.get(&doc.id).ok_or_else(|| {
                SearchError::StorageError(format!("Missing embedding for doc {}", doc.id.as_u64()))
            })?;

            for &value in embedding {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
        }

        Ok(buffer)
    }

    /// Deserialize embeddings from binary format.
    fn deserialize_embeddings(data: &[u8], dim: usize) -> Result<Vec<Vec<f32>>, SearchError> {
        let float_size = std::mem::size_of::<f32>();
        let embedding_size = dim * float_size;

        if !data.len().is_multiple_of(embedding_size) {
            return Err(SearchError::StorageError(format!(
                "Invalid embeddings data size: {} bytes not divisible by {} (dim={})",
                data.len(),
                embedding_size,
                dim
            )));
        }

        let count = data.len() / embedding_size;
        let mut embeddings = Vec::with_capacity(count);

        for i in 0..count {
            let start = i * embedding_size;
            let mut embedding = Vec::with_capacity(dim);

            for j in 0..dim {
                let offset = start + j * float_size;
                let bytes: [u8; 4] = data[offset..offset + 4].try_into().map_err(|_| {
                    SearchError::StorageError("Invalid embedding bytes".to_string())
                })?;
                embedding.push(f32::from_le_bytes(bytes));
            }

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Check if there are unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get a reference to the storage backend.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Get the current manifest.
    pub fn manifest(&self) -> &IndexManifest {
        &self.manifest
    }

    /// Add a document to the index
    ///
    /// # Arguments
    /// * `doc` - Document containing text and metadata
    /// * `embedding` - Pre-computed embedding vector for the document
    ///
    /// Returns the assigned DocId that should be stored or errors handled
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

        // Add to vector index
        self.vector_engine.add_document(doc_id, embedding.clone())?;

        // Add to keyword index
        self.keyword_engine.add_document(doc_id, doc.text.clone());

        // Store embedding for persistence
        self.embeddings.insert(doc_id, embedding);

        // Store document record
        let record = DocumentRecord {
            id: doc_id,
            text: doc.text,
            metadata: doc.metadata,
        };
        self.documents.insert(doc_id, record);

        // Mark as dirty for save
        self.dirty = true;

        Ok(doc_id)
    }

    /// Add a document without rebuilding vector index (for batch operations)
    /// Call rebuild_vector_index() once after all documents are added
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

        // Add to vector index (deferred rebuild)
        self.vector_engine
            .add_document_deferred(doc_id, embedding.clone())?;

        // Add to keyword index
        self.keyword_engine.add_document(doc_id, doc.text.clone());

        // Store embedding for persistence
        self.embeddings.insert(doc_id, embedding);

        // Store document record
        let record = DocumentRecord {
            id: doc_id,
            text: doc.text,
            metadata: doc.metadata,
        };
        self.documents.insert(doc_id, record);

        // Mark as dirty for save
        self.dirty = true;

        Ok(doc_id)
    }

    /// Rebuild the vector search index after batch operations
    ///
    /// NOTE: With hnsw_rs, this is a no-op since the index supports incremental updates.
    /// This method is kept for API compatibility but does nothing.
    ///
    /// Unlike the previous instant-distance implementation which required expensive
    /// rebuilds, hnsw_rs maintains the index structure during insertions.
    pub async fn rebuild_vector_index(&mut self) -> Result<(), SearchError> {
        // No-op: hnsw_rs supports incremental insertion
        // The index is already up-to-date
        Ok(())
    }

    /// Perform hybrid search combining vector and keyword search
    ///
    /// # Arguments
    /// * `query_embedding` - Pre-computed embedding for the query text
    /// * `query_text` - The query text for keyword matching
    /// * `k` - Number of results to return (must be > 0)
    ///
    /// Returns ranked search results that should be used or errors handled
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidQuery` if:
    /// - `query_text` is empty or whitespace-only
    /// - `k` is 0
    ///
    /// Returns `SearchError::DimensionMismatch` if `query_embedding` has wrong dimension.
    ///
    /// Note: Takes `&mut self` because the vector search requires mutable access
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

        // Convert to SearchResult with document details and individual scores
        let search_results: Vec<SearchResult> = fused_results
            .into_iter()
            .take(k)
            .filter_map(|(doc_id, score)| {
                self.documents.get(&doc_id).map(|record| SearchResult {
                    doc_id,
                    score,
                    vector_score: vector_scores.get(&doc_id).copied(),
                    keyword_score: keyword_scores.get(&doc_id).copied(),
                    text: record.text.clone(),
                    metadata: record.metadata.clone(),
                })
            })
            .collect();

        Ok(search_results)
    }

    /// Get the number of indexed documents
    #[allow(dead_code)] // Public API
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if the index is empty
    #[allow(dead_code)] // Public API
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Get detailed index metrics
    ///
    /// Returns (total_chunks, total_tokens, avg_tokens_per_chunk)
    pub fn get_index_metrics(&self) -> (usize, usize, f64) {
        let total_chunks = self.documents.len();

        if total_chunks == 0 {
            return (0, 0, 0.0);
        }

        // Count total tokens across all documents
        // Approximate using whitespace tokenization (fast, good enough for metrics)
        let total_tokens: usize = self
            .documents
            .values()
            .map(|doc| doc.text.split_whitespace().count())
            .sum();

        let avg_tokens_per_chunk = if total_chunks > 0 {
            total_tokens as f64 / total_chunks as f64
        } else {
            0.0
        };

        (total_chunks, total_tokens, avg_tokens_per_chunk)
    }

    /// Get vector index size (number of embeddings)
    pub fn vector_index_len(&self) -> usize {
        self.vector_engine.len()
    }

    /// Get keyword index size (number of documents in BM25)
    /// Note: This returns the same count as len() since both indexes share documents
    pub fn keyword_index_len(&self) -> usize {
        // BM25 doesn't expose count, but it has the same documents
        self.documents.len()
    }

    /// Get a document by ID
    #[allow(dead_code)] // Public API
    pub fn get_document(&self, doc_id: &DocId) -> Option<&DocumentRecord> {
        self.documents.get(doc_id)
    }

    /// Clear all documents from the in-memory index (does not clear storage).
    ///
    /// Use `clear_all()` to also clear persistent storage.
    pub fn clear(&mut self) {
        self.documents.clear();
        self.embeddings.clear();
        self.vector_engine = VectorSearchEngine::new(self.embedding_dim);
        self.keyword_engine = KeywordSearchEngine::new();
        self.manifest = IndexManifest::new(self.embedding_dim);
        self.dirty = false;
    }

    /// Clear all documents from both memory and persistent storage.
    ///
    /// This is a full reset - useful for testing or when user wants to start fresh.
    pub async fn clear_all(&mut self) -> Result<(), SearchError> {
        // Clear in-memory state
        self.clear();

        // Clear persistent storage
        self.storage
            .clear()
            .await
            .map_err(|e| SearchError::StorageError(format!("Failed to clear storage: {}", e)))?;

        info!("Cleared all index data (memory and storage)");
        Ok(())
    }

    /// Debug dump of the index state
    pub fn debug_dump(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Search Index Debug Dump ===\n");
        output.push_str(&format!("Total documents: {}\n", self.documents.len()));
        output.push_str(&format!("Embedding dimension: {}\n", self.embedding_dim));
        output.push('\n');

        if self.documents.is_empty() {
            output.push_str("(empty index)\n");
        } else {
            output.push_str("Documents:\n");
            for (idx, (doc_id, doc)) in self.documents.iter().enumerate() {
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

        output.push_str("\n=== End Debug Dump ===\n");
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageError;

    // Mock storage backend for testing
    struct MockStorage;

    #[async_trait::async_trait(?Send)]
    impl StorageBackend for MockStorage {
        async fn save(&self, _key: &str, _data: &[u8]) -> Result<(), StorageError> {
            Ok(())
        }

        async fn load(&self, _key: &str) -> Result<Vec<u8>, StorageError> {
            Err(StorageError::NotFound("test".to_string()))
        }

        async fn exists(&self, _key: &str) -> Result<bool, StorageError> {
            Ok(false)
        }

        async fn delete(&self, _key: &str) -> Result<(), StorageError> {
            Ok(())
        }

        async fn list_keys(&self) -> Result<Vec<String>, StorageError> {
            Ok(vec![])
        }

        async fn clear(&self) -> Result<(), StorageError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_hybrid_search_engine() {
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

        // Search empty index
        let results = engine.search(&[1.0, 0.0, 0.0], "query", 10).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_dimension_mismatch() {
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

        // Empty index
        let (chunks, tokens, avg) = engine.get_index_metrics();
        assert_eq!(chunks, 0);
        assert_eq!(tokens, 0);
        assert_eq!(avg, 0.0);

        // Add documents with known token counts
        let doc1 = Document {
            text: "one two three".to_string(), // 3 tokens
            metadata: DocumentMetadata::default(),
        };
        let doc2 = Document {
            text: "four five".to_string(), // 2 tokens
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

        let (chunks, tokens, avg) = engine.get_index_metrics();
        assert_eq!(chunks, 2);
        assert_eq!(tokens, 5); // 3 + 2
        assert_eq!(avg, 2.5); // 5 / 2
    }

    #[tokio::test]
    async fn test_vector_and_keyword_index_sync() {
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

        // Add documents
        let doc = Document {
            text: "test document".to_string(),
            metadata: DocumentMetadata::default(),
        };
        engine.add_document(doc, vec![1.0, 0.0, 0.0]).await.unwrap();

        assert_eq!(engine.len(), 1);

        // Clear
        engine.clear();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let retrieved = engine.get_document(&doc_id);
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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

        // Empty index
        let dump = engine.debug_dump();
        assert!(dump.contains("Total documents: 0"));
        assert!(dump.contains("(empty index)"));

        // Add document
        let doc = Document {
            text: "This is a test document with some content".to_string(),
            metadata: DocumentMetadata {
                filename: Some("test.txt".to_string()),
                source: Some("test".to_string()),
                created_at: 123,
            },
        };
        let doc_id = engine.add_document(doc, vec![1.0, 0.0, 0.0]).await.unwrap();

        let dump = engine.debug_dump();
        assert!(dump.contains("Total documents: 1"));
        assert!(dump.contains(&format!("DocId: {}", doc_id.as_u64())));
        assert!(dump.contains("test.txt"));
        assert!(dump.contains("test"));
        assert!(dump.contains("This is a test document"));
    }

    #[tokio::test]
    async fn test_search_returns_top_k() {
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

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
        let storage = MockStorage;
        let mut engine = HybridSearchEngine::new(storage, 3).await.unwrap();

        let doc = Document {
            text: "test document".to_string(),
            metadata: DocumentMetadata::default(),
        };
        engine.add_document(doc, vec![1.0, 0.0, 0.0]).await.unwrap();

        // k=0 should return InvalidQuery error
        let result = engine.search(&[1.0, 0.0, 0.0], "test", 0).await;
        assert!(matches!(result, Err(SearchError::InvalidQuery(_))));
    }
}
