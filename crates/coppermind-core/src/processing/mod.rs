//! Document processing and indexing pipeline.
//!
//! This module provides a platform-independent pipeline for processing documents:
//! - Chunk text using semantic chunking (text, markdown, code)
//! - Compute embeddings for each chunk
//! - Track progress for UI feedback
//!
//! # Architecture
//!
//! The `IndexingPipeline` coordinates:
//! 1. **Chunking**: Splits text into semantic units based on file type
//! 2. **Tokenization**: Converts chunks to token IDs
//! 3. **Embedding**: Computes vector embeddings for each chunk
//! 4. **Progress**: Reports progress via callbacks
//!
//! # Example
//!
//! ```ignore
//! use coppermind_core::processing::{IndexingPipeline, IndexingProgress};
//! use coppermind_core::embedding::{JinaBertEmbedder, TokenizerHandle};
//!
//! let pipeline = IndexingPipeline::new(
//!     Arc::new(embedder),
//!     Arc::new(tokenizer),
//! );
//!
//! let results = pipeline.process_text(
//!     &content,
//!     Some("README.md"),
//!     512,
//!     |progress| {
//!         println!("{}% complete", progress.percent_complete());
//!     }
//! )?;
//! ```

mod pipeline;
mod progress;

pub use pipeline::{ChunkResult, IndexingPipeline, ProcessingResult};
pub use progress::{BatchProgress, IndexingProgress};
