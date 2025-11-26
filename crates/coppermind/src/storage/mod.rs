//! Cross-platform storage backend for persisting search indexes.
//!
//! This module re-exports the storage trait and types from `coppermind_core`
//! and provides platform-specific implementations:
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

// Re-export trait and types from core
pub use coppermind_core::storage::{
    DocumentStore, InMemoryDocumentStore, InMemoryStorage, StorageBackend, StorageError, StoreError,
};

// Platform-specific implementations (app-only, not in core)
#[cfg(target_arch = "wasm32")]
pub mod opfs;

#[cfg(target_arch = "wasm32")]
pub mod indexeddb_store;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

// Re-export the platform-specific storage (public API)
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
pub use opfs::OpfsStorage;

#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
pub use indexeddb_store::IndexedDbDocumentStore;

#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
pub use native::NativeStorage;

// Re-export RedbDocumentStore when available (desktop/mobile platforms)
// The redb-store feature is enabled automatically via desktop/mobile features
#[cfg(any(feature = "desktop", feature = "mobile"))]
pub use coppermind_core::storage::RedbDocumentStore;
