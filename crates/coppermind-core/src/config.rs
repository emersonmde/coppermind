//! Production configuration constants.
//!
//! This module contains constants that define the production configuration
//! for Coppermind. These values are used throughout the codebase and in
//! benchmarks to ensure consistency.
//!
//! # Usage
//!
//! ```
//! use coppermind_core::config::{EMBEDDING_DIM, MAX_CHUNK_TOKENS};
//!
//! // Create a vector with the correct dimension
//! let embedding = vec![0.0f32; EMBEDDING_DIM];
//!
//! // Configure chunking with production token limit
//! let chunker_config = MAX_CHUNK_TOKENS;
//! ```

// =============================================================================
// JinaBERT Model Configuration
// =============================================================================

/// Embedding vector dimension (JinaBERT hidden_size).
///
/// JinaBERT v2 small produces 512-dimensional embeddings.
/// This must match the model's `hidden_size` configuration.
///
/// Source: `crates/coppermind/src/embedding/config.rs` - `JinaBertConfig::default()`
pub const EMBEDDING_DIM: usize = 512;

/// Whether embeddings are L2-normalized.
///
/// JinaBERT produces normalized embeddings by default, meaning each
/// vector has unit length (||v|| = 1). This enables using dot product
/// as an efficient proxy for cosine similarity.
pub const EMBEDDINGS_NORMALIZED: bool = true;

// =============================================================================
// Text Chunking Configuration
// =============================================================================

/// Maximum tokens per chunk.
///
/// Chunks are sized to fit within this token limit while preserving
/// semantic boundaries (sentences, paragraphs). The actual token count
/// may be slightly lower due to boundary alignment.
///
/// Source: `crates/coppermind/src/processing/processor.rs` - `embed_text_chunks_auto(..., 512, ...)`
pub const MAX_CHUNK_TOKENS: usize = 512;

/// Approximate characters per token for English text.
///
/// English text averages ~4 characters per token with most tokenizers.
/// This is used for estimating chunk sizes in characters.
///
/// Note: This varies by language and content type:
/// - English prose: ~4 chars/token
/// - Code: ~3-4 chars/token (more symbols)
/// - CJK languages: ~1-2 chars/token
pub const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Target chunk size in characters.
///
/// Derived from `MAX_CHUNK_TOKENS * CHARS_PER_TOKEN_ESTIMATE`.
/// Useful for generating realistic test data.
pub const TARGET_CHUNK_CHARS: usize = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN_ESTIMATE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dim_matches_jina_small() {
        // JinaBERT v2 small uses 512-dimensional embeddings
        assert_eq!(EMBEDDING_DIM, 512);
    }

    #[test]
    fn test_target_chunk_chars_calculation() {
        // 512 tokens * 4 chars/token = 2048 chars
        assert_eq!(TARGET_CHUNK_CHARS, 2048);
    }

    #[test]
    fn test_max_chunk_tokens_reasonable() {
        // Should be within model's max sequence length (2048-8192 for JinaBERT)
        // Using explicit comparisons to avoid clippy::assertions_on_constants
        let max_tokens = MAX_CHUNK_TOKENS;
        assert!(max_tokens <= 2048, "MAX_CHUNK_TOKENS exceeds model limit");
        assert!(
            max_tokens >= 128,
            "MAX_CHUNK_TOKENS too small for useful chunks"
        );
    }
}
