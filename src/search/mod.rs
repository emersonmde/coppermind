//! Hybrid search engine combining vector and keyword search.
//!
//! This module implements a hybrid search system that combines:
//! - **Vector search** (semantic similarity via HNSW)
//! - **Keyword search** (exact matching via BM25)
//! - **Reciprocal Rank Fusion** (RRF) to merge rankings
//!
//! # Architecture
//!
//! - [`types`]: Core types (DocId, Document, SearchResult, SearchError)
//! - [`engine`]: HybridSearchEngine orchestrating vector + keyword search
//! - [`vector`]: HNSW-based semantic similarity search
//! - [`keyword`]: BM25 full-text keyword search
//! - [`fusion`]: Reciprocal Rank Fusion algorithm for merging results
//!
//! # Usage
//!
//! ```ignore
//! use coppermind::search::{HybridSearchEngine, Document, DocumentMetadata};
//! use coppermind::storage::NativeStorage;
//!
//! // Create search engine
//! let storage = NativeStorage::new("./data").await?;
//! let mut engine = HybridSearchEngine::new(storage, 512).await?;
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
//! # Future Extensions
//!
//! - Persistence/serialization of indexes
//! - Incremental index updates
//! - Query expansion and reranking
//! - Multi-field search (title, body, metadata)

pub mod types;

// Internal modules (will be implemented in phases)
mod engine;
mod fusion;
mod keyword;
mod vector;

// Re-export main types (public API)
#[allow(unused_imports)]
pub use types::{DocId, Document, DocumentMetadata, DocumentRecord, SearchError, SearchResult};

// Re-export search engine (will implement later)
pub use engine::HybridSearchEngine;
