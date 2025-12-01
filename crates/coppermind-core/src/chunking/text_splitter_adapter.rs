//! Text-splitter crate adapter for semantic chunking.
//!
//! This module wraps the `text-splitter` crate to provide semantic chunking
//! using Unicode-aware sentence boundary detection from ICU4X.

use super::{
    calculate_chunk_boundaries, tokenizer_sizer::TokenizerSizer, ChunkingStrategy, TextChunk,
};
use crate::error::ChunkingError;
use text_splitter::ChunkConfig;
use tokenizers::Tokenizer;

/// Text splitter adapter using the `text-splitter` crate.
///
/// Provides semantic chunking with Unicode-aware sentence boundary detection
/// and support for custom tokenizers (HuggingFace tokenizers).
///
/// # Algorithm
///
/// The text-splitter crate uses a hierarchical semantic approach:
/// 1. Identifies semantic boundaries at multiple levels (grapheme, word, sentence, etc.)
/// 2. Selects the highest semantic level where content fits within chunk size
/// 3. Merges neighboring sections while respecting higher-level boundaries
pub struct TextSplitterAdapter {
    max_tokens: usize,
    tokenizer: &'static Tokenizer,
}

impl TextSplitterAdapter {
    /// Creates a new text-splitter adapter.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens per chunk
    /// * `tokenizer` - Reference to HuggingFace tokenizer (must be static lifetime)
    pub fn new(max_tokens: usize, tokenizer: &'static Tokenizer) -> Self {
        Self {
            max_tokens,
            tokenizer,
        }
    }
}

impl ChunkingStrategy for TextSplitterAdapter {
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>, ChunkingError> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Create our custom ChunkSizer
        let sizer = TokenizerSizer {
            tokenizer: self.tokenizer,
        };

        // Create chunk config with our tokenizer sizer
        let chunk_config = ChunkConfig::new(self.max_tokens)
            .with_sizer(sizer)
            .with_trim(true);

        // Use text-splitter's semantic chunking
        let splitter = text_splitter::TextSplitter::new(chunk_config);

        // Use shared helper to calculate boundaries correctly for duplicate text
        let chunks = calculate_chunk_boundaries(text, splitter.chunks(text));

        Ok(chunks)
    }

    fn name(&self) -> &'static str {
        "text-splitter"
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::load_test_tokenizer;

    #[test]
    fn test_text_splitter_basic() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
        // With 512 token limit, this should fit in one chunk
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        let chunks = chunker.chunk("").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_whitespace_only() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        let chunks = chunker.chunk("   \n\t  ").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_small_chunk_size() {
        let tokenizer = load_test_tokenizer();
        // Very small chunk size to force multiple chunks
        let chunker = TextSplitterAdapter::new(5, tokenizer);

        let text = "This is a longer piece of text that should be split into multiple chunks.";
        let chunks = chunker.chunk(text).unwrap();

        // Should create multiple chunks
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_unicode_handling() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        let text = "Hello 世界. This is a test. Здравствуй мир!";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunking_strategy_trait() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        assert_eq!(chunker.name(), "text-splitter");
        assert_eq!(chunker.max_tokens(), 512);
    }
}
