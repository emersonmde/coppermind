//! Redb-backed document store for desktop platforms.
//!
//! Uses [redb](https://github.com/cberner/redb) - a pure Rust, ACID-compliant,
//! embedded B-tree database. Provides O(log n) lookups for documents, embeddings,
//! and source records.
//!
//! # Tables
//!
//! - `documents`: DocId (u64) -> DocumentRecord (JSON)
//! - `embeddings`: DocId (u64) -> Vec<f32> (raw bytes, little-endian)
//! - `sources`: source_id (string) -> SourceRecord (JSON)
//! - `metadata`: key (string) -> value (JSON) - stores tombstones, etc.

use super::{DocumentStore, StoreError};
use crate::search::types::{DocId, DocumentRecord, SourceRecord};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

// Table definitions
const DOCUMENTS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("documents");
const EMBEDDINGS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("embeddings");
const SOURCES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("sources");
const METADATA_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("metadata");

// Metadata keys
const TOMBSTONES_KEY: &str = "tombstones";

/// Redb-backed document store for native platforms.
///
/// Provides efficient O(log n) access to documents, embeddings, and source records
/// using redb's B-tree tables. All operations are ACID-compliant.
///
/// # Example
///
/// ```ignore
/// use coppermind_core::storage::RedbDocumentStore;
///
/// let store = RedbDocumentStore::open("./data/index.redb")?;
/// store.put_document(doc_id, &doc).await?;
/// ```
pub struct RedbDocumentStore {
    db: Arc<Database>,
}

impl RedbDocumentStore {
    /// Opens or creates a redb database at the given path.
    ///
    /// Creates the database file and all required tables if they don't exist.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StoreError> {
        let db = Database::create(path.as_ref())
            .map_err(|e| StoreError::DatabaseError(format!("Failed to open database: {}", e)))?;

        // Create tables if they don't exist
        {
            let write_txn = db.begin_write().map_err(|e| {
                StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
            })?;

            // Open (and create if needed) each table
            write_txn.open_table(DOCUMENTS_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to create documents table: {}", e))
            })?;
            write_txn.open_table(EMBEDDINGS_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to create embeddings table: {}", e))
            })?;
            write_txn.open_table(SOURCES_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to create sources table: {}", e))
            })?;
            write_txn.open_table(METADATA_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to create metadata table: {}", e))
            })?;

            write_txn.commit().map_err(|e| {
                StoreError::DatabaseError(format!("Failed to commit table creation: {}", e))
            })?;
        }

        Ok(Self { db: Arc::new(db) })
    }

    /// Returns the path to the database file.
    #[allow(dead_code)]
    pub fn path(&self) -> Option<&Path> {
        // redb Database doesn't expose path, so we can't return it
        None
    }

    /// Serializes a DocumentRecord to JSON bytes.
    fn serialize_document(doc: &DocumentRecord) -> Result<Vec<u8>, StoreError> {
        serde_json::to_vec(doc).map_err(|e| {
            StoreError::SerializationError(format!("Failed to serialize document: {}", e))
        })
    }

    /// Deserializes a DocumentRecord from JSON bytes.
    fn deserialize_document(bytes: &[u8]) -> Result<DocumentRecord, StoreError> {
        serde_json::from_slice(bytes).map_err(|e| {
            StoreError::SerializationError(format!("Failed to deserialize document: {}", e))
        })
    }

    /// Serializes a SourceRecord to JSON bytes.
    fn serialize_source(record: &SourceRecord) -> Result<Vec<u8>, StoreError> {
        serde_json::to_vec(record).map_err(|e| {
            StoreError::SerializationError(format!("Failed to serialize source: {}", e))
        })
    }

    /// Deserializes a SourceRecord from JSON bytes.
    fn deserialize_source(bytes: &[u8]) -> Result<SourceRecord, StoreError> {
        serde_json::from_slice(bytes).map_err(|e| {
            StoreError::SerializationError(format!("Failed to deserialize source: {}", e))
        })
    }

    /// Serializes an embedding to raw bytes (little-endian f32s).
    fn serialize_embedding(embedding: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(embedding.len() * 4);
        for &val in embedding {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Deserializes an embedding from raw bytes.
    fn deserialize_embedding(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Serializes tombstones to JSON bytes.
    fn serialize_tombstones(tombstones: &HashSet<usize>) -> Result<Vec<u8>, StoreError> {
        serde_json::to_vec(tombstones).map_err(|e| {
            StoreError::SerializationError(format!("Failed to serialize tombstones: {}", e))
        })
    }

    /// Deserializes tombstones from JSON bytes.
    fn deserialize_tombstones(bytes: &[u8]) -> Result<HashSet<usize>, StoreError> {
        serde_json::from_slice(bytes).map_err(|e| {
            StoreError::SerializationError(format!("Failed to deserialize tombstones: {}", e))
        })
    }
}

#[async_trait::async_trait(?Send)]
impl DocumentStore for RedbDocumentStore {
    // =========================================================================
    // Document Operations
    // =========================================================================

    async fn get_document(&self, id: DocId) -> Result<Option<DocumentRecord>, StoreError> {
        let read_txn = self.db.begin_read().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(DOCUMENTS_TABLE).map_err(|e| {
            StoreError::DatabaseError(format!("Failed to open documents table: {}", e))
        })?;

        match table.get(id.as_u64()) {
            Ok(Some(guard)) => {
                let bytes = guard.value();
                let doc = Self::deserialize_document(bytes)?;
                Ok(Some(doc))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StoreError::DatabaseError(format!(
                "Failed to get document: {}",
                e
            ))),
        }
    }

    async fn put_document(&self, id: DocId, doc: &DocumentRecord) -> Result<(), StoreError> {
        let bytes = Self::serialize_document(doc)?;

        let write_txn = self.db.begin_write().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(DOCUMENTS_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open documents table: {}", e))
            })?;

            table.insert(id.as_u64(), bytes.as_slice()).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to insert document: {}", e))
            })?;
        }

        write_txn
            .commit()
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit document: {}", e)))?;

        Ok(())
    }

    async fn delete_document(&self, id: DocId) -> Result<(), StoreError> {
        let write_txn = self.db.begin_write().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(DOCUMENTS_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open documents table: {}", e))
            })?;

            // Remove returns Ok(None) if key didn't exist, which is fine
            table.remove(id.as_u64()).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to delete document: {}", e))
            })?;
        }

        write_txn.commit().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to commit document deletion: {}", e))
        })?;

        Ok(())
    }

    async fn get_documents_batch(&self, ids: &[DocId]) -> Result<Vec<DocumentRecord>, StoreError> {
        let read_txn = self.db.begin_read().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(DOCUMENTS_TABLE).map_err(|e| {
            StoreError::DatabaseError(format!("Failed to open documents table: {}", e))
        })?;

        let mut docs = Vec::with_capacity(ids.len());
        for id in ids {
            if let Ok(Some(guard)) = table.get(id.as_u64()) {
                if let Ok(doc) = Self::deserialize_document(guard.value()) {
                    docs.push(doc);
                }
            }
        }

        Ok(docs)
    }

    // =========================================================================
    // Embedding Operations
    // =========================================================================

    async fn get_embedding(&self, id: DocId) -> Result<Option<Vec<f32>>, StoreError> {
        let read_txn = self.db.begin_read().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(EMBEDDINGS_TABLE).map_err(|e| {
            StoreError::DatabaseError(format!("Failed to open embeddings table: {}", e))
        })?;

        match table.get(id.as_u64()) {
            Ok(Some(guard)) => {
                let embedding = Self::deserialize_embedding(guard.value());
                Ok(Some(embedding))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StoreError::DatabaseError(format!(
                "Failed to get embedding: {}",
                e
            ))),
        }
    }

    async fn put_embedding(&self, id: DocId, embedding: &[f32]) -> Result<(), StoreError> {
        let bytes = Self::serialize_embedding(embedding);

        let write_txn = self.db.begin_write().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(EMBEDDINGS_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open embeddings table: {}", e))
            })?;

            table.insert(id.as_u64(), bytes.as_slice()).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to insert embedding: {}", e))
            })?;
        }

        write_txn
            .commit()
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit embedding: {}", e)))?;

        Ok(())
    }

    async fn delete_embedding(&self, id: DocId) -> Result<(), StoreError> {
        let write_txn = self.db.begin_write().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(EMBEDDINGS_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open embeddings table: {}", e))
            })?;

            table.remove(id.as_u64()).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to delete embedding: {}", e))
            })?;
        }

        write_txn.commit().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to commit embedding deletion: {}", e))
        })?;

        Ok(())
    }

    async fn iter_embeddings(&self) -> Result<Vec<(DocId, Vec<f32>)>, StoreError> {
        let read_txn = self.db.begin_read().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(EMBEDDINGS_TABLE).map_err(|e| {
            StoreError::DatabaseError(format!("Failed to open embeddings table: {}", e))
        })?;

        let mut embeddings = Vec::new();
        let iter = table.iter().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to iterate embeddings: {}", e))
        })?;

        for result in iter {
            let (key, value) = result.map_err(|e| {
                StoreError::DatabaseError(format!("Failed to read embedding entry: {}", e))
            })?;
            let doc_id = DocId::from_u64(key.value());
            let embedding = Self::deserialize_embedding(value.value());
            embeddings.push((doc_id, embedding));
        }

        Ok(embeddings)
    }

    // =========================================================================
    // Source Operations
    // =========================================================================

    async fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>, StoreError> {
        let read_txn = self.db.begin_read().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(SOURCES_TABLE).map_err(|e| {
            StoreError::DatabaseError(format!("Failed to open sources table: {}", e))
        })?;

        match table.get(source_id) {
            Ok(Some(guard)) => {
                let record = Self::deserialize_source(guard.value())?;
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StoreError::DatabaseError(format!(
                "Failed to get source: {}",
                e
            ))),
        }
    }

    async fn put_source(&self, source_id: &str, record: &SourceRecord) -> Result<(), StoreError> {
        let bytes = Self::serialize_source(record)?;

        let write_txn = self.db.begin_write().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(SOURCES_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open sources table: {}", e))
            })?;

            table.insert(source_id, bytes.as_slice()).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to insert source: {}", e))
            })?;
        }

        write_txn
            .commit()
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit source: {}", e)))?;

        Ok(())
    }

    async fn delete_source(&self, source_id: &str) -> Result<(), StoreError> {
        let write_txn = self.db.begin_write().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(SOURCES_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open sources table: {}", e))
            })?;

            table.remove(source_id).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to delete source: {}", e))
            })?;
        }

        write_txn.commit().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to commit source deletion: {}", e))
        })?;

        Ok(())
    }

    async fn list_sources(&self) -> Result<Vec<String>, StoreError> {
        let read_txn = self.db.begin_read().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(SOURCES_TABLE).map_err(|e| {
            StoreError::DatabaseError(format!("Failed to open sources table: {}", e))
        })?;

        let mut sources = Vec::new();
        let iter = table
            .iter()
            .map_err(|e| StoreError::DatabaseError(format!("Failed to iterate sources: {}", e)))?;

        for result in iter {
            let (key, _) = result.map_err(|e| {
                StoreError::DatabaseError(format!("Failed to read source entry: {}", e))
            })?;
            sources.push(key.value().to_string());
        }

        Ok(sources)
    }

    // =========================================================================
    // Metadata Operations
    // =========================================================================

    async fn get_tombstones(&self) -> Result<HashSet<usize>, StoreError> {
        let read_txn = self.db.begin_read().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(METADATA_TABLE).map_err(|e| {
            StoreError::DatabaseError(format!("Failed to open metadata table: {}", e))
        })?;

        match table.get(TOMBSTONES_KEY) {
            Ok(Some(guard)) => Self::deserialize_tombstones(guard.value()),
            Ok(None) => Ok(HashSet::new()),
            Err(e) => Err(StoreError::DatabaseError(format!(
                "Failed to get tombstones: {}",
                e
            ))),
        }
    }

    async fn put_tombstones(&self, tombstones: &HashSet<usize>) -> Result<(), StoreError> {
        let bytes = Self::serialize_tombstones(tombstones)?;

        let write_txn = self.db.begin_write().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(METADATA_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open metadata table: {}", e))
            })?;

            table
                .insert(TOMBSTONES_KEY, bytes.as_slice())
                .map_err(|e| {
                    StoreError::DatabaseError(format!("Failed to insert tombstones: {}", e))
                })?;
        }

        write_txn.commit().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to commit tombstones: {}", e))
        })?;

        Ok(())
    }

    // =========================================================================
    // Utility Operations
    // =========================================================================

    async fn document_count(&self) -> Result<usize, StoreError> {
        let read_txn = self.db.begin_read().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(DOCUMENTS_TABLE).map_err(|e| {
            StoreError::DatabaseError(format!("Failed to open documents table: {}", e))
        })?;

        let count = table.len().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to get document count: {}", e))
        })?;

        Ok(count as usize)
    }

    async fn clear(&self) -> Result<(), StoreError> {
        let write_txn = self.db.begin_write().map_err(|e| {
            StoreError::DatabaseError(format!("Failed to begin write transaction: {}", e))
        })?;

        // Clear each table by draining all entries
        {
            let mut table = write_txn.open_table(DOCUMENTS_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open documents table: {}", e))
            })?;
            // Collect keys first to avoid borrowing issues
            let keys: Vec<u64> = table
                .iter()
                .map_err(|e| StoreError::DatabaseError(format!("Failed to iterate: {}", e)))?
                .filter_map(|r| r.ok().map(|(k, _)| k.value()))
                .collect();
            for key in keys {
                let _ = table.remove(key);
            }
        }

        {
            let mut table = write_txn.open_table(EMBEDDINGS_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open embeddings table: {}", e))
            })?;
            let keys: Vec<u64> = table
                .iter()
                .map_err(|e| StoreError::DatabaseError(format!("Failed to iterate: {}", e)))?
                .filter_map(|r| r.ok().map(|(k, _)| k.value()))
                .collect();
            for key in keys {
                let _ = table.remove(key);
            }
        }

        {
            let mut table = write_txn.open_table(SOURCES_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open sources table: {}", e))
            })?;
            let keys: Vec<String> = table
                .iter()
                .map_err(|e| StoreError::DatabaseError(format!("Failed to iterate: {}", e)))?
                .filter_map(|r| r.ok().map(|(k, _)| k.value().to_string()))
                .collect();
            for key in &keys {
                let _ = table.remove(key.as_str());
            }
        }

        {
            let mut table = write_txn.open_table(METADATA_TABLE).map_err(|e| {
                StoreError::DatabaseError(format!("Failed to open metadata table: {}", e))
            })?;
            let keys: Vec<String> = table
                .iter()
                .map_err(|e| StoreError::DatabaseError(format!("Failed to iterate: {}", e)))?
                .filter_map(|r| r.ok().map(|(k, _)| k.value().to_string()))
                .collect();
            for key in &keys {
                let _ = table.remove(key.as_str());
            }
        }

        write_txn
            .commit()
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit clear: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::types::DocumentMetadata;
    use tempfile::TempDir;

    fn create_test_store() -> (RedbDocumentStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.redb");
        let store = RedbDocumentStore::open(&db_path).unwrap();
        (store, temp_dir)
    }

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
        let (store, _temp) = create_test_store();
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
        let (store, _temp) = create_test_store();

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
        let (store, _temp) = create_test_store();
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
        let (store, _temp) = create_test_store();
        let source = SourceRecord::new_complete(
            "abc123".to_string(),
            vec![DocId::from_u64(1), DocId::from_u64(2)],
        );

        // Put and get
        store.put_source("/path/to/file.md", &source).await.unwrap();
        let retrieved = store.get_source("/path/to/file.md").await.unwrap().unwrap();
        assert_eq!(retrieved.content_hash, "abc123");
        assert_eq!(retrieved.doc_ids.len(), 2);

        // List sources
        let sources = store.list_sources().await.unwrap();
        assert_eq!(sources.len(), 1);
        assert!(sources.contains(&"/path/to/file.md".to_string()));

        // Delete
        store.delete_source("/path/to/file.md").await.unwrap();
        assert!(store
            .get_source("/path/to/file.md")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_tombstones() {
        let (store, _temp) = create_test_store();

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
        let (store, _temp) = create_test_store();

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

    #[tokio::test]
    async fn test_persistence_across_reopens() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("persist.redb");

        // Create and populate
        {
            let store = RedbDocumentStore::open(&db_path).unwrap();
            store
                .put_document(DocId::from_u64(42), &make_test_doc(42, "Persisted"))
                .await
                .unwrap();
            store
                .put_embedding(DocId::from_u64(42), &[1.0, 2.0, 3.0])
                .await
                .unwrap();
        }

        // Reopen and verify
        {
            let store = RedbDocumentStore::open(&db_path).unwrap();
            let doc = store
                .get_document(DocId::from_u64(42))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(doc.text, "Persisted");

            let emb = store
                .get_embedding(DocId::from_u64(42))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(emb, vec![1.0, 2.0, 3.0]);
        }
    }
}
