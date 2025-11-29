//! Hybrid search engine combining vector and keyword search.
//!
//! This module implements a hybrid search system that combines:
//! - **Vector search** (semantic similarity via HNSW)
//! - **Keyword search** (exact matching via BM25)
//! - **Reciprocal Rank Fusion** (RRF) to merge rankings
//!
//! # Architecture
//!
//! - `types`: Core types (DocId, Document, SearchResult, FileSearchResult, SearchError)
//! - `engine`: HybridSearchEngine orchestrating vector + keyword search
//! - `vector`: HNSW-based semantic similarity search
//! - `keyword`: BM25 full-text keyword search
//! - `fusion`: Reciprocal Rank Fusion algorithm for merging results
//! - `aggregation`: File-level aggregation for chunk-based search results
//!
//! # Usage
//!
//! ```ignore
//! use coppermind_core::search::{HybridSearchEngine, Document, DocumentMetadata};
//! use coppermind_core::storage::InMemoryDocumentStore;
//!
//! // Create search engine with a document store
//! let store = InMemoryDocumentStore::new();
//! let mut engine = HybridSearchEngine::new(store, 512).await?;
//!
//! // Index documents with pre-computed embeddings
//! let doc = Document {
//!     text: "Rust is a systems programming language".to_string(),
//!     metadata: DocumentMetadata::default(),
//! };
//! let embedding = vec![0.1; 512]; // From embedding model
//! engine.add_document(doc, embedding).await?;
//!
//! // Search (combines vector + keyword + RRF fusion)
//! let query_embedding = vec![0.1; 512];
//! let results = engine.search("Rust programming", query_embedding, 10).await?;
//! ```
//!
//! # Algorithm Details
//!
//! **Vector Search (HNSW)**:
//! - Builds hierarchical graph for approximate nearest neighbor search
//! - Uses cosine similarity for semantic matching
//! - Fast O(log n) search with high recall
//!
//! **Keyword Search (BM25)**:
//! - Term frequency-inverse document frequency scoring
//! - Handles exact keyword matches and boolean queries
//! - Tuned parameters: k1=1.2, b=0.75 (standard)
//!
//! **Reciprocal Rank Fusion (RRF)**:
//! - Formula: `score = 1 / (k + rank)` where k=60
//! - Merges vector and keyword rankings without score normalization
//! - Robust to scale differences between algorithms
//!
//! # Performance Characteristics
//!
//! - **Indexing**: O(n log n) for HNSW construction, O(n) for BM25
//! - **Search**: O(log n) for vector, O(n) for keyword (small corpus)
//! - **Memory**: ~1KB per document + embedding storage
//!
//! # Implemented Features
//!
//! - ✅ Persistence via [`DocumentStore`](crate::storage::DocumentStore) trait
//! - ✅ Incremental index updates (HNSW supports online insertion)
//! - ✅ Tombstone-based deletion with automatic compaction
//! - ✅ Source tracking for re-upload detection
//!
//! # Future Extensions
//!
//! - Query expansion and reranking
//! - Multi-field search (title, body, metadata)

pub mod types;

// Internal modules - exposed for benchmarking but hidden from docs
mod aggregation;
mod engine;
#[doc(hidden)]
pub mod fusion;
#[doc(hidden)]
pub mod keyword;
#[doc(hidden)]
pub mod vector;

// Re-export main types (public API)
#[allow(unused_imports)]
pub use types::{
    validate_dimension, ChunkMetadata, CompactionStats, DocId, Document, DocumentMetadata,
    DocumentRecord, FileSearchResult, SearchError, SearchResult, SourceRecord,
};

// Re-export search engine and aggregation
pub use aggregation::aggregate_chunks_by_file;
pub use engine::HybridSearchEngine;
