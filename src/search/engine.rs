// HybridSearchEngine - combines vector and keyword search

use super::fusion::reciprocal_rank_fusion;
use super::keyword::KeywordSearchEngine;
#[cfg(test)]
use super::types::DocumentMetadata;
use super::types::{DocId, Document, DocumentRecord, SearchError, SearchResult};
use super::vector::VectorSearchEngine;
use crate::storage::StorageBackend;
use std::collections::HashMap;

/// Hybrid search engine combining vector (semantic) and keyword (BM25) search
pub struct HybridSearchEngine<S: StorageBackend> {
    /// Vector search engine (semantic similarity)
    vector_engine: VectorSearchEngine,
    /// Keyword search engine (BM25)
    keyword_engine: KeywordSearchEngine,
    /// Document storage (metadata + text)
    documents: HashMap<DocId, DocumentRecord>,
    /// Storage backend for persistence
    _storage: S,
    /// Embedding dimension (e.g., 512 for JinaBERT)
    embedding_dim: usize,
}

impl<S: StorageBackend> HybridSearchEngine<S> {
    /// Create a new hybrid search engine
    ///
    /// # Arguments
    /// * `storage` - Storage backend for persisting indexes
    /// * `embedding_dim` - Dimensionality of embeddings (must match the model)
    pub async fn new(storage: S, embedding_dim: usize) -> Result<Self, SearchError> {
        Ok(Self {
            vector_engine: VectorSearchEngine::new(embedding_dim),
            keyword_engine: KeywordSearchEngine::new(),
            documents: HashMap::new(),
            _storage: storage,
            embedding_dim,
        })
    }

    /// Add a document to the index
    ///
    /// # Arguments
    /// * `doc` - Document containing text and metadata
    /// * `embedding` - Pre-computed embedding vector for the document
    ///
    /// Returns the assigned DocId
    pub async fn add_document(
        &mut self,
        doc: Document,
        embedding: Vec<f32>,
    ) -> Result<DocId, SearchError> {
        // Validate embedding dimension
        if embedding.len() != self.embedding_dim {
            return Err(SearchError::EmbeddingError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                embedding.len()
            )));
        }

        // Generate unique ID
        let doc_id = DocId::new();

        // Add to vector index
        self.vector_engine.add_document(doc_id, embedding);

        // Add to keyword index
        self.keyword_engine.add_document(doc_id, doc.text.clone());

        // Store document record
        let record = DocumentRecord {
            id: doc_id,
            text: doc.text,
            metadata: doc.metadata,
        };
        self.documents.insert(doc_id, record);

        Ok(doc_id)
    }

    /// Add a document without rebuilding vector index (for batch operations)
    /// Call rebuild_vector_index() once after all documents are added
    pub async fn add_document_deferred(
        &mut self,
        doc: Document,
        embedding: Vec<f32>,
    ) -> Result<DocId, SearchError> {
        // Validate embedding dimension
        if embedding.len() != self.embedding_dim {
            return Err(SearchError::EmbeddingError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                embedding.len()
            )));
        }

        // Generate unique ID
        let doc_id = DocId::new();

        // Add to vector index (deferred rebuild)
        self.vector_engine.add_document_deferred(doc_id, embedding);

        // Add to keyword index
        self.keyword_engine.add_document(doc_id, doc.text.clone());

        // Store document record
        let record = DocumentRecord {
            id: doc_id,
            text: doc.text,
            metadata: doc.metadata,
        };
        self.documents.insert(doc_id, record);

        Ok(doc_id)
    }

    /// Rebuild the vector search index after batch operations
    /// This is CPU-intensive and should be called after all documents are added
    pub async fn rebuild_vector_index(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Desktop: Run in thread pool to avoid blocking UI
            // We need to extract the vector engine, rebuild it, then put it back
            let mut vector_engine = std::mem::replace(
                &mut self.vector_engine,
                VectorSearchEngine::new(self.embedding_dim),
            );

            let rebuilt = tokio::task::spawn_blocking(move || {
                vector_engine.rebuild_index();
                vector_engine
            })
            .await
            .expect("Failed to join rebuild task");

            self.vector_engine = rebuilt;
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.vector_engine.rebuild_index();
        }
    }

    /// Perform hybrid search combining vector and keyword search
    ///
    /// # Arguments
    /// * `query_embedding` - Pre-computed embedding for the query text
    /// * `query_text` - The query text for keyword matching
    /// * `k` - Number of results to return
    ///
    /// Returns ranked search results
    pub async fn search(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Validate query embedding dimension
        if query_embedding.len() != self.embedding_dim {
            return Err(SearchError::EmbeddingError(format!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                query_embedding.len()
            )));
        }

        // Get vector search results (semantic similarity)
        let vector_results = self.vector_engine.search(query_embedding, k * 2);
        #[cfg(target_arch = "wasm32")]
        {
            dioxus::logger::tracing::info!("📊 Vector search (semantic) results:");
            for (i, (doc_id, score)) in vector_results.iter().take(k).enumerate() {
                if let Some(doc) = self.documents.get(doc_id) {
                    dioxus::logger::tracing::info!(
                        "  {}. [Vector: {:.4}] {}",
                        i + 1,
                        score,
                        doc.text
                    );
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!("📊 Vector search (semantic) results:");
            for (i, (doc_id, score)) in vector_results.iter().take(k).enumerate() {
                if let Some(doc) = self.documents.get(doc_id) {
                    println!("  {}. [Vector: {:.4}] {}", i + 1, score, doc.text);
                }
            }
        }

        // Get keyword search results (BM25)
        let keyword_results = self.keyword_engine.search(query_text, k * 2);
        #[cfg(target_arch = "wasm32")]
        {
            dioxus::logger::tracing::info!("📊 Keyword search (BM25) results:");
            for (i, (doc_id, score)) in keyword_results.iter().take(k).enumerate() {
                if let Some(doc) = self.documents.get(doc_id) {
                    dioxus::logger::tracing::info!(
                        "  {}. [BM25: {:.4}] {}",
                        i + 1,
                        score,
                        doc.text
                    );
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!("📊 Keyword search (BM25) results:");
            for (i, (doc_id, score)) in keyword_results.iter().take(k).enumerate() {
                if let Some(doc) = self.documents.get(doc_id) {
                    println!("  {}. [BM25: {:.4}] {}", i + 1, score, doc.text);
                }
            }
        }

        // Fuse results using Reciprocal Rank Fusion (RRF)
        #[cfg(target_arch = "wasm32")]
        dioxus::logger::tracing::info!("🔀 Applying Reciprocal Rank Fusion (RRF)...");
        #[cfg(not(target_arch = "wasm32"))]
        println!("🔀 Applying Reciprocal Rank Fusion (RRF)...");

        let fused_results = reciprocal_rank_fusion(&vector_results, &keyword_results, 60);

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

    /// Get a document by ID
    #[allow(dead_code)] // Public API
    pub fn get_document(&self, doc_id: &DocId) -> Option<&DocumentRecord> {
        self.documents.get(doc_id)
    }

    /// Clear all documents from the index
    #[allow(dead_code)] // Public API
    pub fn clear(&mut self) {
        self.documents.clear();
        self.vector_engine = VectorSearchEngine::new(self.embedding_dim);
        self.keyword_engine = KeywordSearchEngine::new();
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
                if doc.text.len() > 100 {
                    output.push_str(&format!("  Text: {}...\n", &doc.text[..100]));
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
}
