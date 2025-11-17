//! Cross-platform storage backend for persisting search indexes.
//!
//! This module provides a platform-agnostic storage abstraction that allows
//! the search engine to persist and load data across different platforms:
//!
//! - **Web (WASM)**: Uses OPFS (Origin Private File System) for large binary data
//! - **Desktop**: Uses native filesystem (`tokio::fs`)
//!
//! # Storage Backend Trait
//!
//! The [`StorageBackend`] trait provides a simple key-value interface:
//! - `save(key, data)` - Store binary data
//! - `load(key)` - Retrieve binary data
//! - `exists(key)` - Check if key exists
//! - `delete(key)` - Remove data
//! - `list_keys()` - List all stored keys
//! - `clear()` - Delete all data
//!
//! # Platform-Specific Implementations
//!
//! ## OPFS (Web)
//!
//! Origin Private File System provides:
//! - **Large storage quota**: Gigabytes of storage for embeddings
//! - **Persistent**: Data survives page refreshes
//! - **Private**: Not accessible to other origins
//! - **Fast**: Direct filesystem access (not IndexedDB)
//!
//! ```ignore
//! #[cfg(target_arch = "wasm32")]
//! use coppermind::storage::OpfsStorage;
//!
//! let storage = OpfsStorage::new().await?;
//! storage.save("embeddings", &embedding_bytes).await?;
//! ```
//!
//! ## Native Filesystem (Desktop)
//!
//! Uses `tokio::fs` for async file I/O:
//! - **Path**: Configurable data directory
//! - **Atomic writes**: Temp file + rename for safety
//! - **Async**: Non-blocking I/O via tokio
//!
//! ```ignore
//! #[cfg(not(target_arch = "wasm32"))]
//! use coppermind::storage::NativeStorage;
//!
//! let storage = NativeStorage::new("./data").await?;
//! storage.save("embeddings", &embedding_bytes).await?;
//! ```
//!
//! # Usage Example
//!
//! ```ignore
//! use coppermind::storage::{StorageBackend, StorageError};
//!
//! async fn persist_index<S: StorageBackend>(
//!     storage: &S,
//!     index_data: Vec<u8>
//! ) -> Result<(), StorageError> {
//!     storage.save("search_index", &index_data).await?;
//!     Ok(())
//! }
//! ```
//!
//! # Error Handling
//!
//! All storage operations return `Result<T, StorageError>` with variants:
//! - `NotFound` - Key doesn't exist
//! - `IoError` - Filesystem/network error
//! - `SerializationError` - Data encoding/decoding failed
//! - `BrowserApiUnavailable` - Web API not supported
//! - `OpfsUnavailable` - OPFS not available in this browser

use thiserror::Error;

/// Storage backend abstraction for cross-platform persistence
///
/// This trait provides a generic key-value storage interface that can be
/// implemented using different backends (OPFS for web, native filesystem for desktop).
#[async_trait::async_trait(?Send)]
#[allow(dead_code)] // Public API trait
pub trait StorageBackend {
    /// Save binary data to storage with a key
    #[must_use = "Storage save failures should be handled"]
    async fn save(&self, key: &str, data: &[u8]) -> Result<(), StorageError>;

    /// Load binary data from storage by key
    #[must_use = "Storage load failures should be handled"]
    async fn load(&self, key: &str) -> Result<Vec<u8>, StorageError>;

    /// Check if a key exists in storage
    #[must_use = "Storage check failures should be handled"]
    async fn exists(&self, key: &str) -> Result<bool, StorageError>;

    /// Delete data by key
    #[must_use = "Storage delete failures should be handled"]
    async fn delete(&self, key: &str) -> Result<(), StorageError>;

    /// List all keys in storage (useful for debugging/management)
    #[must_use = "Storage listing failures should be handled"]
    async fn list_keys(&self) -> Result<Vec<String>, StorageError>;

    /// Clear all stored data
    #[must_use = "Storage clear failures should be handled"]
    async fn clear(&self) -> Result<(), StorageError>;
}

/// Storage error types
#[derive(Debug, Error)]
#[allow(dead_code)] // Public API enum
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

// Platform-specific implementations
#[cfg(target_arch = "wasm32")]
pub mod opfs;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

// Re-export the platform-specific storage (public API)
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
pub use opfs::OpfsStorage;

#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
pub use native::NativeStorage;

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
