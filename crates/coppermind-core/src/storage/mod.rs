//! Cross-platform storage backend traits for persisting search indexes.
//!
//! This module provides platform-agnostic storage abstractions that allow
//! the search engine to persist and load data across different platforms.
//!
//! # Storage Abstractions
//!
//! ## [`StorageBackend`] (Legacy)
//! Simple key-value blob storage. Being replaced by DocumentStore for better
//! scalability.
//!
//! ## [`DocumentStore`] (New)
//! Efficient KV store with O(log n) lookups for documents, embeddings, and sources.
//! Designed to scale to millions of chunks without full-corpus serialization.
//!
//! # Implementations
//!
//! - [`InMemoryStorage`] / [`InMemoryDocumentStore`] - No-op storage for testing
//! - `NativeStorage` / `RedbDocumentStore` - Native filesystem (in app crate, desktop)
//! - `OpfsStorage` / `IndexedDbDocumentStore` - Browser storage (in app crate, WASM)

mod document_store;

#[cfg(feature = "redb-store")]
mod redb_store;

pub use document_store::{DocumentStore, InMemoryDocumentStore, StoreError};

#[cfg(feature = "redb-store")]
pub use redb_store::RedbDocumentStore;

use thiserror::Error;

/// Storage backend abstraction for cross-platform persistence.
///
/// This trait provides a generic key-value storage interface that can be
/// implemented using different backends (OPFS for web, native filesystem for desktop).
#[async_trait::async_trait(?Send)]
pub trait StorageBackend {
    /// Save binary data to storage with a key.
    #[must_use = "Storage save failures should be handled"]
    async fn save(&self, key: &str, data: &[u8]) -> Result<(), StorageError>;

    /// Load binary data from storage by key.
    #[must_use = "Storage load failures should be handled"]
    async fn load(&self, key: &str) -> Result<Vec<u8>, StorageError>;

    /// Check if a key exists in storage.
    #[must_use = "Storage check failures should be handled"]
    async fn exists(&self, key: &str) -> Result<bool, StorageError>;

    /// Delete data by key.
    #[must_use = "Storage delete failures should be handled"]
    async fn delete(&self, key: &str) -> Result<(), StorageError>;

    /// List all keys in storage (useful for debugging/management).
    #[must_use = "Storage listing failures should be handled"]
    async fn list_keys(&self) -> Result<Vec<String>, StorageError>;

    /// Clear all stored data.
    #[must_use = "Storage clear failures should be handled"]
    async fn clear(&self) -> Result<(), StorageError>;
}

/// Storage error types.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Key not found: {0}")]
    NotFound(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Browser API unavailable")]
    BrowserApiUnavailable,
    #[error("OPFS unavailable - please use a modern browser in standard mode")]
    OpfsUnavailable,
}

/// In-memory storage backend that doesn't persist data.
/// Useful for testing or when persistence is disabled.
#[derive(Default)]
pub struct InMemoryStorage;

impl InMemoryStorage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait(?Send)]
impl StorageBackend for InMemoryStorage {
    async fn save(&self, _key: &str, _data: &[u8]) -> Result<(), StorageError> {
        // No-op: don't persist anything
        Ok(())
    }

    async fn load(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        // Nothing is persisted, so always return NotFound
        Err(StorageError::NotFound(key.to_string()))
    }

    async fn exists(&self, _key: &str) -> Result<bool, StorageError> {
        // Nothing is persisted
        Ok(false)
    }

    async fn delete(&self, _key: &str) -> Result<(), StorageError> {
        // No-op: nothing to delete
        Ok(())
    }

    async fn list_keys(&self) -> Result<Vec<String>, StorageError> {
        // No keys stored
        Ok(Vec::new())
    }

    async fn clear(&self) -> Result<(), StorageError> {
        // No-op: nothing to clear
        Ok(())
    }
}
