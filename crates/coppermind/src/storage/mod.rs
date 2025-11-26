//! Cross-platform storage backend for persisting search indexes.
//!
//! This module re-exports the storage trait and types from `coppermind_core`
//! and provides platform-specific implementations:
//!
//! - **Web (WASM)**: Uses IndexedDB for structured document storage
//! - **Desktop**: Uses redb (embedded B-tree database)
//!
//! # DocumentStore Trait
//!
//! The [`DocumentStore`] trait provides efficient key-value storage:
//! - O(log n) or O(1) lookups for documents, embeddings, and sources
//! - Incremental writes without full-corpus serialization
//! - Source tracking for update detection
//!
//! # Platform-Specific Implementations
//!
//! ## IndexedDB (Web)
//!
//! Browser-native storage with zero bundle cost:
//! - **O(1) key lookups**: Direct access by DocId
//! - **Persistent**: Survives page refreshes
//! - **Structured storage**: Separate object stores for documents, embeddings, sources
//!
//! ```ignore
//! #[cfg(target_arch = "wasm32")]
//! use coppermind::storage::IndexedDbDocumentStore;
//!
//! let store = IndexedDbDocumentStore::open().await?;
//! store.put_document(doc_id, record)?;
//! ```
//!
//! ## Redb (Desktop)
//!
//! Pure Rust embedded B-tree database:
//! - **O(log n) lookups**: Efficient B-tree access
//! - **ACID**: Full transaction support
//! - **Compact**: ~200KB binary size
//!
//! ```ignore
//! #[cfg(not(target_arch = "wasm32"))]
//! use coppermind::storage::RedbDocumentStore;
//!
//! let store = RedbDocumentStore::open("./data/index.redb")?;
//! store.put_document(doc_id, record)?;
//! ```
//!
//! # Error Handling
//!
//! All storage operations return `Result<T, StoreError>` with variants:
//! - `NotFound` - Key doesn't exist
//! - `Database` - Backend-specific errors
//! - `Serialization` - Data encoding/decoding failed

// Re-export trait and types from core
pub use coppermind_core::storage::{DocumentStore, InMemoryDocumentStore, StoreError};

// Platform-specific implementations (app-only, not in core)
#[cfg(target_arch = "wasm32")]
pub mod indexeddb_store;

// Re-export the platform-specific storage (public API)
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
pub use indexeddb_store::IndexedDbDocumentStore;

// Re-export RedbDocumentStore when available (desktop/mobile platforms)
// The redb-store feature is enabled automatically via desktop/mobile features
#[cfg(any(feature = "desktop", feature = "mobile"))]
pub use coppermind_core::storage::RedbDocumentStore;
