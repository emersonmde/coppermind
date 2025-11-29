//! Types for embedding results.
//!
//! This module defines result types used by embedding operations.

/// Result of computing an embedding for a single text.
#[derive(Clone, Debug)]
pub struct EmbeddingResult {
    /// Number of tokens processed
    pub token_count: usize,
    /// Embedding vector (dimension matches model config)
    pub embedding: Vec<f32>,
}

/// Result of embedding a text chunk.
///
/// Used when processing documents that are split into chunks.
#[derive(Clone, Debug)]
pub struct ChunkEmbeddingResult {
    /// Index of this chunk in the document (0-based)
    pub chunk_index: usize,
    /// Number of tokens in this chunk
    pub token_count: usize,
    /// Text content of this chunk
    pub text: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
}
