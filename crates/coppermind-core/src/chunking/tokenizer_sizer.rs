//! TokenizerSizer for text-splitter integration.
//!
//! This module provides a ChunkSizer implementation that uses HuggingFace
//! tokenizers for accurate token counting during chunk sizing.

use text_splitter::ChunkSizer;
use tokenizers::Tokenizer;

/// ChunkSizer implementation for HuggingFace Tokenizer.
///
/// Wraps a tokenizer to implement text-splitter's ChunkSizer trait,
/// allowing token-based chunk sizing without the onig dependency.
///
/// # Token Count Consistency
///
/// **Important:** The same tokenizer instance used here for chunk sizing MUST be
/// the same tokenizer used for embedding. This ensures chunk sizes accurately
/// predict token counts during embedding, preventing truncation or wasted capacity.
pub struct TokenizerSizer<'a> {
    pub tokenizer: &'a Tokenizer,
}

impl ChunkSizer for TokenizerSizer<'_> {
    /// Returns the token count for the given text chunk.
    ///
    /// Uses the HuggingFace tokenizer to encode the text and count tokens.
    /// This is used by text-splitter to determine chunk boundaries.
    fn size(&self, chunk: &str) -> usize {
        self.tokenizer
            .encode(chunk, false)
            .map(|encoding| encoding.len())
            .unwrap_or(0)
    }
}
