//! IndexedDB-backed document store for web platform.
//!
//! Uses [rexie](https://github.com/devashishdxt/rexie) - a futures-based IndexedDB wrapper.
//! Provides O(1) key lookups for chunks, embeddings, and source records.
//!
//! # Object Stores
//!
//! - `documents`: ChunkId (u64 as string) -> ChunkRecord (JSON) [store name kept for backward compat]
//! - `embeddings`: ChunkId (u64 as string) -> ArrayBuffer (raw f32 bytes)
//! - `sources`: source_id (string) -> SourceRecord (JSON)
//! - `metadata`: key (string) -> value (JSON) - stores tombstones, etc.

use coppermind_core::search::types::{ChunkId, ChunkRecord, SourceRecord};
use coppermind_core::storage::{DocumentStore, StoreError};
use rexie::{ObjectStore, Rexie, TransactionMode};
use std::collections::HashSet;
use wasm_bindgen::JsValue;

// Store names
const DB_NAME: &str = "coppermind";
const DB_VERSION: u32 = 1;
const DOCUMENTS_STORE: &str = "documents";
const EMBEDDINGS_STORE: &str = "embeddings";
const SOURCES_STORE: &str = "sources";
const METADATA_STORE: &str = "metadata";

// Metadata keys
const TOMBSTONES_KEY: &str = "tombstones";

/// IndexedDB-backed document store for web platform.
///
/// Provides efficient O(1) access to chunks, embeddings, and source records
/// using IndexedDB object stores.
///
/// # Example
///
/// ```ignore
/// use coppermind::storage::IndexedDbDocumentStore;
///
/// let store = IndexedDbDocumentStore::open().await?;
/// store.put_chunk(chunk_id, &chunk).await?;
/// ```
pub struct IndexedDbDocumentStore {
    db: Rexie,
}

impl IndexedDbDocumentStore {
    /// Opens or creates the IndexedDB database.
    ///
    /// Creates all required object stores if they don't exist.
    pub async fn open() -> Result<Self, StoreError> {
        let db = Rexie::builder(DB_NAME)
            .version(DB_VERSION)
            .add_object_store(ObjectStore::new(DOCUMENTS_STORE))
            .add_object_store(ObjectStore::new(EMBEDDINGS_STORE))
            .add_object_store(ObjectStore::new(SOURCES_STORE))
            .add_object_store(ObjectStore::new(METADATA_STORE))
            .build()
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to open IndexedDB: {:?}", e)))?;

        Ok(Self { db })
    }

    /// Converts a ChunkId to a string key for IndexedDB.
    fn chunk_id_to_key(id: ChunkId) -> String {
        id.as_u64().to_string()
    }

    /// Converts a string key back to a ChunkId.
    fn key_to_chunk_id(key: &str) -> Option<ChunkId> {
        key.parse::<u64>().ok().map(ChunkId::from_u64)
    }

    /// Serializes a ChunkRecord to a JsValue.
    fn serialize_chunk(chunk: &ChunkRecord) -> Result<JsValue, StoreError> {
        serde_wasm_bindgen::to_value(chunk).map_err(|e| {
            StoreError::SerializationError(format!("Failed to serialize chunk: {}", e))
        })
    }

    /// Deserializes a ChunkRecord from a JsValue.
    fn deserialize_chunk(value: JsValue) -> Result<ChunkRecord, StoreError> {
        serde_wasm_bindgen::from_value(value).map_err(|e| {
            StoreError::SerializationError(format!("Failed to deserialize chunk: {}", e))
        })
    }

    /// Serializes a SourceRecord to a JsValue.
    fn serialize_source(record: &SourceRecord) -> Result<JsValue, StoreError> {
        serde_wasm_bindgen::to_value(record).map_err(|e| {
            StoreError::SerializationError(format!("Failed to serialize source: {}", e))
        })
    }

    /// Deserializes a SourceRecord from a JsValue.
    fn deserialize_source(value: JsValue) -> Result<SourceRecord, StoreError> {
        serde_wasm_bindgen::from_value(value).map_err(|e| {
            StoreError::SerializationError(format!("Failed to deserialize source: {}", e))
        })
    }

    /// Serializes an embedding to a JsValue (Uint8Array of little-endian f32s).
    fn serialize_embedding(embedding: &[f32]) -> JsValue {
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let array = js_sys::Uint8Array::from(bytes.as_slice());
        array.into()
    }

    /// Deserializes an embedding from a JsValue.
    fn deserialize_embedding(value: JsValue) -> Result<Vec<f32>, StoreError> {
        let array = js_sys::Uint8Array::new(&value);
        let bytes = array.to_vec();
        if !bytes.len().is_multiple_of(4) {
            return Err(StoreError::SerializationError(
                "Invalid embedding byte length".to_string(),
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    /// Serializes tombstones to a JsValue.
    fn serialize_tombstones(tombstones: &HashSet<usize>) -> Result<JsValue, StoreError> {
        let vec: Vec<usize> = tombstones.iter().copied().collect();
        serde_wasm_bindgen::to_value(&vec).map_err(|e| {
            StoreError::SerializationError(format!("Failed to serialize tombstones: {}", e))
        })
    }

    /// Deserializes tombstones from a JsValue.
    fn deserialize_tombstones(value: JsValue) -> Result<HashSet<usize>, StoreError> {
        let vec: Vec<usize> = serde_wasm_bindgen::from_value(value).map_err(|e| {
            StoreError::SerializationError(format!("Failed to deserialize tombstones: {}", e))
        })?;
        Ok(vec.into_iter().collect())
    }
}

#[async_trait::async_trait(?Send)]
impl DocumentStore for IndexedDbDocumentStore {
    // =========================================================================
    // Chunk Operations
    // =========================================================================

    async fn get_chunk(&self, id: ChunkId) -> Result<Option<ChunkRecord>, StoreError> {
        let tx = self
            .db
            .transaction(&[DOCUMENTS_STORE], TransactionMode::ReadOnly)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(DOCUMENTS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(&Self::chunk_id_to_key(id));
        match store.get(key.clone()).await {
            Ok(Some(value)) => {
                let chunk = Self::deserialize_chunk(value)?;
                Ok(Some(chunk))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StoreError::DatabaseError(format!(
                "Failed to get chunk: {:?}",
                e
            ))),
        }
    }

    async fn put_chunk(&self, id: ChunkId, chunk: &ChunkRecord) -> Result<(), StoreError> {
        let tx = self
            .db
            .transaction(&[DOCUMENTS_STORE], TransactionMode::ReadWrite)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(DOCUMENTS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(&Self::chunk_id_to_key(id));
        let value = Self::serialize_chunk(chunk)?;

        store
            .put(&value, Some(&key))
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to put chunk: {:?}", e)))?;

        tx.done()
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit: {:?}", e)))?;

        Ok(())
    }

    async fn delete_chunk(&self, id: ChunkId) -> Result<(), StoreError> {
        let tx = self
            .db
            .transaction(&[DOCUMENTS_STORE], TransactionMode::ReadWrite)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(DOCUMENTS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(&Self::chunk_id_to_key(id));

        store
            .delete(key)
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to delete chunk: {:?}", e)))?;

        tx.done()
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit: {:?}", e)))?;

        Ok(())
    }

    async fn get_chunks_batch(&self, ids: &[ChunkId]) -> Result<Vec<ChunkRecord>, StoreError> {
        let tx = self
            .db
            .transaction(&[DOCUMENTS_STORE], TransactionMode::ReadOnly)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(DOCUMENTS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let mut chunks = Vec::with_capacity(ids.len());
        for id in ids {
            let key = JsValue::from_str(&Self::chunk_id_to_key(*id));
            if let Ok(Some(value)) = store.get(key.clone()).await {
                if let Ok(chunk) = Self::deserialize_chunk(value) {
                    chunks.push(chunk);
                }
            }
        }

        Ok(chunks)
    }

    // =========================================================================
    // Embedding Operations
    // =========================================================================

    async fn get_embedding(&self, id: ChunkId) -> Result<Option<Vec<f32>>, StoreError> {
        let tx = self
            .db
            .transaction(&[EMBEDDINGS_STORE], TransactionMode::ReadOnly)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(EMBEDDINGS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(&Self::chunk_id_to_key(id));
        match store.get(key.clone()).await {
            Ok(Some(value)) => {
                let embedding = Self::deserialize_embedding(value)?;
                Ok(Some(embedding))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StoreError::DatabaseError(format!(
                "Failed to get embedding: {:?}",
                e
            ))),
        }
    }

    async fn put_embedding(&self, id: ChunkId, embedding: &[f32]) -> Result<(), StoreError> {
        let tx = self
            .db
            .transaction(&[EMBEDDINGS_STORE], TransactionMode::ReadWrite)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(EMBEDDINGS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(&Self::chunk_id_to_key(id));
        let value = Self::serialize_embedding(embedding);

        store
            .put(&value, Some(&key))
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to put embedding: {:?}", e)))?;

        tx.done()
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit: {:?}", e)))?;

        Ok(())
    }

    async fn delete_embedding(&self, id: ChunkId) -> Result<(), StoreError> {
        let tx = self
            .db
            .transaction(&[EMBEDDINGS_STORE], TransactionMode::ReadWrite)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(EMBEDDINGS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(&Self::chunk_id_to_key(id));

        store.delete(key).await.map_err(|e| {
            StoreError::DatabaseError(format!("Failed to delete embedding: {:?}", e))
        })?;

        tx.done()
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit: {:?}", e)))?;

        Ok(())
    }

    async fn iter_embeddings(&self) -> Result<Vec<(ChunkId, Vec<f32>)>, StoreError> {
        let tx = self
            .db
            .transaction(&[EMBEDDINGS_STORE], TransactionMode::ReadOnly)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(EMBEDDINGS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        // Use scan() to get (key, value) pairs
        let entries = store.scan(None, None, None, None).await.map_err(|e| {
            StoreError::DatabaseError(format!("Failed to scan embeddings: {:?}", e))
        })?;

        let mut embeddings = Vec::new();
        for (key, value) in entries {
            if let Some(key_str) = key.as_string() {
                if let Some(chunk_id) = Self::key_to_chunk_id(&key_str) {
                    if let Ok(embedding) = Self::deserialize_embedding(value) {
                        embeddings.push((chunk_id, embedding));
                    }
                }
            }
        }

        Ok(embeddings)
    }

    // =========================================================================
    // Source Operations
    // =========================================================================

    async fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>, StoreError> {
        let tx = self
            .db
            .transaction(&[SOURCES_STORE], TransactionMode::ReadOnly)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(SOURCES_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(source_id);
        match store.get(key.clone()).await {
            Ok(Some(value)) => {
                let record = Self::deserialize_source(value)?;
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StoreError::DatabaseError(format!(
                "Failed to get source: {:?}",
                e
            ))),
        }
    }

    async fn put_source(&self, source_id: &str, record: &SourceRecord) -> Result<(), StoreError> {
        let tx = self
            .db
            .transaction(&[SOURCES_STORE], TransactionMode::ReadWrite)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(SOURCES_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(source_id);
        let value = Self::serialize_source(record)?;

        store
            .put(&value, Some(&key))
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to put source: {:?}", e)))?;

        tx.done()
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit: {:?}", e)))?;

        Ok(())
    }

    async fn delete_source(&self, source_id: &str) -> Result<(), StoreError> {
        let tx = self
            .db
            .transaction(&[SOURCES_STORE], TransactionMode::ReadWrite)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(SOURCES_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(source_id);

        store
            .delete(key)
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to delete source: {:?}", e)))?;

        tx.done()
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit: {:?}", e)))?;

        Ok(())
    }

    async fn list_sources(&self) -> Result<Vec<String>, StoreError> {
        let tx = self
            .db
            .transaction(&[SOURCES_STORE], TransactionMode::ReadOnly)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(SOURCES_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        // Use get_all_keys() to get just the keys
        let keys = store
            .get_all_keys(None, None)
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to list sources: {:?}", e)))?;

        let sources: Vec<String> = keys.into_iter().filter_map(|key| key.as_string()).collect();

        Ok(sources)
    }

    // =========================================================================
    // Metadata Operations
    // =========================================================================

    async fn get_tombstones(&self) -> Result<HashSet<usize>, StoreError> {
        let tx = self
            .db
            .transaction(&[METADATA_STORE], TransactionMode::ReadOnly)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(METADATA_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(TOMBSTONES_KEY);
        match store.get(key.clone()).await {
            Ok(Some(value)) => Self::deserialize_tombstones(value),
            Ok(None) => Ok(HashSet::new()),
            Err(e) => Err(StoreError::DatabaseError(format!(
                "Failed to get tombstones: {:?}",
                e
            ))),
        }
    }

    async fn put_tombstones(&self, tombstones: &HashSet<usize>) -> Result<(), StoreError> {
        let tx = self
            .db
            .transaction(&[METADATA_STORE], TransactionMode::ReadWrite)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(METADATA_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let key = JsValue::from_str(TOMBSTONES_KEY);
        let value = Self::serialize_tombstones(tombstones)?;

        store
            .put(&value, Some(&key))
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to put tombstones: {:?}", e)))?;

        tx.done()
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to commit: {:?}", e)))?;

        Ok(())
    }

    // =========================================================================
    // Utility Operations
    // =========================================================================

    async fn chunk_count(&self) -> Result<usize, StoreError> {
        let tx = self
            .db
            .transaction(&[DOCUMENTS_STORE], TransactionMode::ReadOnly)
            .map_err(|e| {
                StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
            })?;

        let store = tx
            .store(DOCUMENTS_STORE)
            .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

        let count = store
            .count(None)
            .await
            .map_err(|e| StoreError::DatabaseError(format!("Failed to count chunks: {:?}", e)))?;

        Ok(count as usize)
    }

    async fn clear(&self) -> Result<(), StoreError> {
        // Clear all stores
        for store_name in &[
            DOCUMENTS_STORE,
            EMBEDDINGS_STORE,
            SOURCES_STORE,
            METADATA_STORE,
        ] {
            let tx = self
                .db
                .transaction(&[store_name], TransactionMode::ReadWrite)
                .map_err(|e| {
                    StoreError::DatabaseError(format!("Failed to start transaction: {:?}", e))
                })?;

            let store = tx
                .store(store_name)
                .map_err(|e| StoreError::DatabaseError(format!("Failed to get store: {:?}", e)))?;

            store.clear().await.map_err(|e| {
                StoreError::DatabaseError(format!("Failed to clear store: {:?}", e))
            })?;

            tx.done()
                .await
                .map_err(|e| StoreError::DatabaseError(format!("Failed to commit: {:?}", e)))?;
        }

        Ok(())
    }
}
