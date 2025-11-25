//! File processing and indexing utilities.
//!
//! This module handles the CPU-intensive work of:
//! - Chunking text files
//! - Computing embeddings
//! - Indexing chunks in the search engine
//!
//! # Architecture
//!
//! - [`embedder`]: Platform-specific embedding abstraction
//! - [`processor`]: File and batch processing logic
//!
//! # Platform Strategy
//!
//! - **Web**: Uses `EmbeddingWorkerClient` for parallel embedding
//! - **Desktop**: Uses direct embedding with `tokio::spawn_blocking`

pub mod embedder;
pub mod processor;

// Re-export key types
pub use embedder::PlatformEmbedder;
pub use processor::{process_file_chunks, ChunkProcessingResult};
