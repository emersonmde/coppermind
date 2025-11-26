//! Coppermind - Local-first hybrid search engine.
//!
//! A cross-platform semantic search application that runs entirely on the user's device,
//! combining vector embeddings (semantic similarity) with keyword search (BM25) using
//! Reciprocal Rank Fusion (RRF).
//!
//! # Architecture
//!
//! - **Embedding**: JinaBERT model for generating 512-dimensional embeddings
//! - **Vector Search**: HNSW (Hierarchical Navigable Small World) for semantic similarity
//! - **Keyword Search**: BM25 (Best Matching 25) for exact keyword matching
//! - **Fusion**: Reciprocal Rank Fusion to combine rankings
//! - **Storage**: Platform-specific backends (IndexedDB for web, redb for desktop)
//!
//! See the [architecture documentation](https://github.com/example/coppermind/blob/main/docs/architecture-design.md)
//! for detailed design information.
//!
//! # Platform Support
//!
//! - **Web (WASM)**: Runs in browser with CPU inference
//! - **Desktop**: macOS/Windows/Linux with GPU acceleration (Metal/CUDA/Accelerate/MKL)
//!
//! # Examples
//!
//! ```ignore
//! use coppermind::embedding::compute_embedding;
//! use coppermind::search::engine::HybridSearchEngine;
//!
//! // Generate embedding
//! let result = compute_embedding("hello world").await?;
//!
//! // Search documents
//! let results = engine.search("semantic query", 10).await?;
//! ```

// Enforce memory safety: forbid all unsafe code
#![forbid(unsafe_code)]

pub mod components;
pub mod embedding;
pub mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
pub mod metrics;
pub mod platform;
pub mod processing;
pub mod search;
pub mod storage;
pub mod utils;
pub mod workers;

// Web crawler (desktop-only for now due to CORS restrictions on web)
#[cfg(not(target_arch = "wasm32"))]
pub mod crawler;
