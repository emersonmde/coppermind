//! Cross-platform storage backend traits for persisting search indexes.
//!
//! This module provides platform-agnostic storage abstractions that allow
//! the search engine to persist and load data across different platforms.
//!
//! # Storage Abstraction
//!
//! ## [`DocumentStore`]
//! Efficient KV store with O(log n) lookups for documents, embeddings, and sources.
//! Designed to scale to millions of chunks without full-corpus serialization.
//!
//! # Implementations
//!
//! - [`InMemoryDocumentStore`] - No persistence, for testing
//! - `RedbDocumentStore` - Native filesystem (desktop/mobile, via feature flag)
//! - `IndexedDbDocumentStore` - Browser storage (WASM, in app crate)

mod document_store;

#[cfg(feature = "redb-store")]
mod redb_store;

pub use document_store::{DocumentStore, InMemoryDocumentStore, StoreError};

#[cfg(feature = "redb-store")]
pub use redb_store::RedbDocumentStore;
