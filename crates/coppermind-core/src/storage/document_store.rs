//! Document store trait for efficient KV-based storage.
//!
//! This module provides the [`DocumentStore`] trait which abstracts over
//! platform-specific storage backends (redb on desktop, IndexedDB on web).
//!
//! DocumentStore provides O(log n) random access to individual documents,
//! embeddings, and source records - essential for scaling to millions of chunks.

use crate::search::types::{DocId, DocumentRecord, SourceRecord};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during document store operations.
#[derive(Debug, Error)]
pub enum StoreError {
    /// Key/document not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// I/O error (filesystem, IndexedDB, etc.)
    #[error("I/O error: {0}")]
    IoError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Database error (redb, IndexedDB)
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Store not initialized
    #[error("Store not initialized")]
    NotInitialized,
}

/// Platform-specific document store with efficient random access.
///
/// This trait provides O(log n) or O(1) access to documents, embeddings, and
/// source tracking data. Each platform implements this differently:
///
/// - **Desktop**: redb (Pure Rust B-tree database)
/// - **Web**: IndexedDB via rexie
///
/// # Design Notes
///
/// - No transaction primitives exposed - IndexedDB auto-commits and doesn't map
///   cleanly to explicit begin/commit. Each operation is self-contained.
/// - Batch operations provided for efficiency when hydrating search results.
/// - Iterator for embeddings enables index rebuilds without loading all into memory.
#[async_trait::async_trait(?Send)]
pub trait DocumentStore {
    // =========================================================================
    // Document Operations
    // =========================================================================

    /// Retrieves a document by ID.
    ///
    /// Returns `Ok(None)` if the document doesn't exist.
    async fn get_document(&self, id: DocId) -> Result<Option<DocumentRecord>, StoreError>;

    /// Stores a document with the given ID.
    ///
    /// Overwrites any existing document with the same ID.
    async fn put_document(&self, id: DocId, doc: &DocumentRecord) -> Result<(), StoreError>;

    /// Deletes a document by ID.
    ///
    /// Returns `Ok(())` even if the document didn't exist.
    async fn delete_document(&self, id: DocId) -> Result<(), StoreError>;

    /// Retrieves multiple documents by ID in a single operation.
    ///
    /// More efficient than multiple `get_document` calls for hydrating search results.
    /// Returns documents in the same order as the input IDs. Missing documents
    /// are skipped (not included in output).
    async fn get_documents_batch(&self, ids: &[DocId]) -> Result<Vec<DocumentRecord>, StoreError>;

    // =========================================================================
    // Embedding Operations
    // =========================================================================

    /// Retrieves an embedding by document ID.
    ///
    /// Returns `Ok(None)` if the embedding doesn't exist.
    async fn get_embedding(&self, id: DocId) -> Result<Option<Vec<f32>>, StoreError>;

    /// Stores an embedding for a document.
    ///
    /// Overwrites any existing embedding for the same ID.
    async fn put_embedding(&self, id: DocId, embedding: &[f32]) -> Result<(), StoreError>;

    /// Deletes an embedding by document ID.
    ///
    /// Returns `Ok(())` even if the embedding didn't exist.
    async fn delete_embedding(&self, id: DocId) -> Result<(), StoreError>;

    /// Returns all embeddings as (DocId, embedding) pairs.
    ///
    /// Used for rebuilding HNSW index on load. Implementations should stream
    /// this data efficiently rather than loading everything into memory at once.
    async fn iter_embeddings(&self) -> Result<Vec<(DocId, Vec<f32>)>, StoreError>;

    // =========================================================================
    // Source Operations (for re-upload detection)
    // =========================================================================

    /// Retrieves a source record by source_id.
    ///
    /// source_id is platform-specific:
    /// - Desktop: full file path (e.g., "/Users/matt/docs/README.md")
    /// - Web: "web:{filename}" (e.g., "web:README.md")
    /// - Crawler: full URL (e.g., "https://example.com/docs/intro")
    ///
    /// Returns `Ok(None)` if the source doesn't exist.
    async fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>, StoreError>;

    /// Stores a source record.
    ///
    /// Overwrites any existing source with the same ID.
    async fn put_source(&self, source_id: &str, record: &SourceRecord) -> Result<(), StoreError>;

    /// Deletes a source record by ID.
    ///
    /// Returns `Ok(())` even if the source didn't exist.
    /// Note: This only deletes the source record, not the associated documents/embeddings.
    async fn delete_source(&self, source_id: &str) -> Result<(), StoreError>;

    /// Returns all source IDs in the store.
    ///
    /// Used for listing indexed sources in the UI.
    async fn list_sources(&self) -> Result<Vec<String>, StoreError>;

    // =========================================================================
    // Metadata Operations
    // =========================================================================

    /// Retrieves the set of tombstoned (soft-deleted) HNSW indices.
    ///
    /// These are internal HNSW index positions that have been marked for deletion
    /// but not yet compacted. Used to filter search results.
    async fn get_tombstones(&self) -> Result<HashSet<usize>, StoreError>;

    /// Stores the tombstone set.
    ///
    /// Called after marking documents as deleted or after compaction.
    async fn put_tombstones(&self, tombstones: &HashSet<usize>) -> Result<(), StoreError>;

    // =========================================================================
    // Utility Operations
    // =========================================================================

    /// Returns the number of documents in the store.
    async fn document_count(&self) -> Result<usize, StoreError>;

    /// Clears all data from the store.
    ///
    /// Used for "Clear Index" functionality.
    async fn clear(&self) -> Result<(), StoreError>;
}

/// In-memory document store for testing.
///
/// This implementation stores everything in HashMaps and doesn't persist
/// anything to disk. Useful for unit tests and development.
#[derive(Default)]
pub struct InMemoryDocumentStore {
    documents: std::sync::RwLock<HashMap<u64, DocumentRecord>>,
    embeddings: std::sync::RwLock<HashMap<u64, Vec<f32>>>,
    sources: std::sync::RwLock<HashMap<String, SourceRecord>>,
    tombstones: std::sync::RwLock<HashSet<usize>>,
}

impl InMemoryDocumentStore {
    /// Creates a new empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait(?Send)]
impl DocumentStore for InMemoryDocumentStore {
    async fn get_document(&self, id: DocId) -> Result<Option<DocumentRecord>, StoreError> {
        let docs = self
            .documents
            .read()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        Ok(docs.get(&id.as_u64()).cloned())
    }

    async fn put_document(&self, id: DocId, doc: &DocumentRecord) -> Result<(), StoreError> {
        let mut docs = self
            .documents
            .write()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        docs.insert(id.as_u64(), doc.clone());
        Ok(())
    }

    async fn delete_document(&self, id: DocId) -> Result<(), StoreError> {
        let mut docs = self
            .documents
            .write()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        docs.remove(&id.as_u64());
        Ok(())
    }

    async fn get_documents_batch(&self, ids: &[DocId]) -> Result<Vec<DocumentRecord>, StoreError> {
        let docs = self
            .documents
            .read()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        Ok(ids
            .iter()
            .filter_map(|id| docs.get(&id.as_u64()).cloned())
            .collect())
    }

    async fn get_embedding(&self, id: DocId) -> Result<Option<Vec<f32>>, StoreError> {
        let embs = self
            .embeddings
            .read()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        Ok(embs.get(&id.as_u64()).cloned())
    }

    async fn put_embedding(&self, id: DocId, embedding: &[f32]) -> Result<(), StoreError> {
        let mut embs = self
            .embeddings
            .write()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        embs.insert(id.as_u64(), embedding.to_vec());
        Ok(())
    }

    async fn delete_embedding(&self, id: DocId) -> Result<(), StoreError> {
        let mut embs = self
            .embeddings
            .write()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        embs.remove(&id.as_u64());
        Ok(())
    }

    async fn iter_embeddings(&self) -> Result<Vec<(DocId, Vec<f32>)>, StoreError> {
        let embs = self
            .embeddings
            .read()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        Ok(embs
            .iter()
            .map(|(&id, emb)| (DocId::from_u64(id), emb.clone()))
            .collect())
    }

    async fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>, StoreError> {
        let sources = self
            .sources
            .read()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        Ok(sources.get(source_id).cloned())
    }

    async fn put_source(&self, source_id: &str, record: &SourceRecord) -> Result<(), StoreError> {
        let mut sources = self
            .sources
            .write()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        sources.insert(source_id.to_string(), record.clone());
        Ok(())
    }

    async fn delete_source(&self, source_id: &str) -> Result<(), StoreError> {
        let mut sources = self
            .sources
            .write()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        sources.remove(source_id);
        Ok(())
    }

    async fn list_sources(&self) -> Result<Vec<String>, StoreError> {
        let sources = self
            .sources
            .read()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        Ok(sources.keys().cloned().collect())
    }

    async fn get_tombstones(&self) -> Result<HashSet<usize>, StoreError> {
        let tombstones = self
            .tombstones
            .read()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        Ok(tombstones.clone())
    }

    async fn put_tombstones(&self, tombstones: &HashSet<usize>) -> Result<(), StoreError> {
        let mut ts = self
            .tombstones
            .write()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        *ts = tombstones.clone();
        Ok(())
    }

    async fn document_count(&self) -> Result<usize, StoreError> {
        let docs = self
            .documents
            .read()
            .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
        Ok(docs.len())
    }

    async fn clear(&self) -> Result<(), StoreError> {
        {
            let mut docs = self
                .documents
                .write()
                .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
            docs.clear();
        }
        {
            let mut embs = self
                .embeddings
                .write()
                .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
            embs.clear();
        }
        {
            let mut sources = self
                .sources
                .write()
                .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
            sources.clear();
        }
        {
            let mut ts = self
                .tombstones
                .write()
                .map_err(|e| StoreError::DatabaseError(format!("Lock poisoned: {}", e)))?;
            ts.clear();
        }
        Ok(())
    }
}

// Blanket implementation for Arc<T> where T: DocumentStore
// This allows sharing a store between multiple engine instances (e.g., in tests)
#[async_trait::async_trait(?Send)]
impl<T: DocumentStore> DocumentStore for std::sync::Arc<T> {
    async fn get_document(&self, id: DocId) -> Result<Option<DocumentRecord>, StoreError> {
        (**self).get_document(id).await
    }

    async fn put_document(&self, id: DocId, doc: &DocumentRecord) -> Result<(), StoreError> {
        (**self).put_document(id, doc).await
    }

    async fn delete_document(&self, id: DocId) -> Result<(), StoreError> {
        (**self).delete_document(id).await
    }

    async fn get_documents_batch(&self, ids: &[DocId]) -> Result<Vec<DocumentRecord>, StoreError> {
        (**self).get_documents_batch(ids).await
    }

    async fn get_embedding(&self, id: DocId) -> Result<Option<Vec<f32>>, StoreError> {
        (**self).get_embedding(id).await
    }

    async fn put_embedding(&self, id: DocId, embedding: &[f32]) -> Result<(), StoreError> {
        (**self).put_embedding(id, embedding).await
    }

    async fn delete_embedding(&self, id: DocId) -> Result<(), StoreError> {
        (**self).delete_embedding(id).await
    }

    async fn iter_embeddings(&self) -> Result<Vec<(DocId, Vec<f32>)>, StoreError> {
        (**self).iter_embeddings().await
    }

    async fn get_source(&self, source_key: &str) -> Result<Option<SourceRecord>, StoreError> {
        (**self).get_source(source_key).await
    }

    async fn put_source(&self, source_key: &str, record: &SourceRecord) -> Result<(), StoreError> {
        (**self).put_source(source_key, record).await
    }

    async fn delete_source(&self, source_key: &str) -> Result<(), StoreError> {
        (**self).delete_source(source_key).await
    }

    async fn list_sources(&self) -> Result<Vec<String>, StoreError> {
        (**self).list_sources().await
    }

    async fn get_tombstones(&self) -> Result<HashSet<usize>, StoreError> {
        (**self).get_tombstones().await
    }

    async fn put_tombstones(&self, tombstones: &HashSet<usize>) -> Result<(), StoreError> {
        (**self).put_tombstones(tombstones).await
    }

    async fn document_count(&self) -> Result<usize, StoreError> {
        (**self).document_count().await
    }

    async fn clear(&self) -> Result<(), StoreError> {
        (**self).clear().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::types::DocumentMetadata;

    fn make_test_doc(id: u64, text: &str) -> DocumentRecord {
        DocumentRecord {
            id: DocId::from_u64(id),
            text: text.to_string(),
            metadata: DocumentMetadata {
                filename: Some("test.txt".to_string()),
                source: Some("/path/to/test.txt".to_string()),
                created_at: 12345,
            },
        }
    }

    #[tokio::test]
    async fn test_document_crud() {
        let store = InMemoryDocumentStore::new();
        let doc = make_test_doc(1, "Hello world");

        // Initially empty
        assert!(store
            .get_document(DocId::from_u64(1))
            .await
            .unwrap()
            .is_none());

        // Put and get
        store.put_document(DocId::from_u64(1), &doc).await.unwrap();
        let retrieved = store
            .get_document(DocId::from_u64(1))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.text, "Hello world");

        // Delete
        store.delete_document(DocId::from_u64(1)).await.unwrap();
        assert!(store
            .get_document(DocId::from_u64(1))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_batch_get() {
        let store = InMemoryDocumentStore::new();

        // Add some documents
        for i in 1..=5 {
            let doc = make_test_doc(i, &format!("Doc {}", i));
            store.put_document(DocId::from_u64(i), &doc).await.unwrap();
        }

        // Batch get (including one that doesn't exist)
        let ids = vec![
            DocId::from_u64(1),
            DocId::from_u64(3),
            DocId::from_u64(99), // doesn't exist
            DocId::from_u64(5),
        ];
        let docs = store.get_documents_batch(&ids).await.unwrap();
        assert_eq!(docs.len(), 3); // Only 3 found
    }

    #[tokio::test]
    async fn test_embedding_operations() {
        let store = InMemoryDocumentStore::new();
        let embedding = vec![1.0, 2.0, 3.0, 4.0];

        // Put and get
        store
            .put_embedding(DocId::from_u64(1), &embedding)
            .await
            .unwrap();
        let retrieved = store
            .get_embedding(DocId::from_u64(1))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retrieved, embedding);

        // Iterate
        store
            .put_embedding(DocId::from_u64(2), &[5.0, 6.0, 7.0, 8.0])
            .await
            .unwrap();
        let all = store.iter_embeddings().await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_source_operations() {
        let store = InMemoryDocumentStore::new();
        let source = SourceRecord::new_complete(
            "abc123".to_string(),
            vec![DocId::from_u64(1), DocId::from_u64(2)],
        );

        // Put and get
        store.put_source("web:README.md", &source).await.unwrap();
        let retrieved = store.get_source("web:README.md").await.unwrap().unwrap();
        assert_eq!(retrieved.content_hash, "abc123");
        assert_eq!(retrieved.doc_ids.len(), 2);

        // List sources
        let sources = store.list_sources().await.unwrap();
        assert_eq!(sources.len(), 1);
        assert!(sources.contains(&"web:README.md".to_string()));

        // Delete
        store.delete_source("web:README.md").await.unwrap();
        assert!(store.get_source("web:README.md").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_tombstones() {
        let store = InMemoryDocumentStore::new();

        // Initially empty
        let ts = store.get_tombstones().await.unwrap();
        assert!(ts.is_empty());

        // Set tombstones
        let mut new_ts = HashSet::new();
        new_ts.insert(1);
        new_ts.insert(5);
        new_ts.insert(10);
        store.put_tombstones(&new_ts).await.unwrap();

        // Verify
        let ts = store.get_tombstones().await.unwrap();
        assert_eq!(ts.len(), 3);
        assert!(ts.contains(&5));
    }

    #[tokio::test]
    async fn test_clear() {
        let store = InMemoryDocumentStore::new();

        // Add data
        store
            .put_document(DocId::from_u64(1), &make_test_doc(1, "test"))
            .await
            .unwrap();
        store
            .put_embedding(DocId::from_u64(1), &[1.0, 2.0])
            .await
            .unwrap();
        store
            .put_source("test", &SourceRecord::new_incomplete("hash".to_string()))
            .await
            .unwrap();

        assert_eq!(store.document_count().await.unwrap(), 1);

        // Clear
        store.clear().await.unwrap();

        assert_eq!(store.document_count().await.unwrap(), 0);
        assert!(store
            .get_embedding(DocId::from_u64(1))
            .await
            .unwrap()
            .is_none());
        assert!(store.list_sources().await.unwrap().is_empty());
    }
}
