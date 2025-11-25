//! Text-splitter crate adapter for semantic chunking.
//!
//! This module wraps the `text-splitter` crate to provide semantic chunking
//! using Unicode-aware sentence boundary detection from ICU4X.
//!
//! # Benefits over custom implementation
//!
//! - **Unicode-correct**: Uses ICU4X segmenter for proper sentence detection
//! - **Battle-tested**: Used in production LLM applications (428+ dependents)
//! - **Maintained**: Active development, bug fixes from community
//! - **Pure Rust**: ICU4X is pure Rust, WASM-compatible
//! - **Extensible**: Support for markdown/code chunking in future
//!
//! # ChunkSizer Implementation
//!
//! We implement our own `ChunkSizer` for HuggingFace tokenizers instead of using
//! text-splitter's built-in tokenizers feature, which would pull in onig (C library)
//! and break WASM builds.

use super::{ChunkingStrategy, TextChunk};
use crate::error::EmbeddingError;
use text_splitter::{ChunkConfig, ChunkSizer};
use tokenizers::Tokenizer;

/// ChunkSizer implementation for HuggingFace Tokenizer.
///
/// Wraps our tokenizer to implement text-splitter's ChunkSizer trait,
/// allowing token-based chunk sizing without the onig dependency.
struct TokenizerSizer {
    tokenizer: &'static Tokenizer,
}

impl ChunkSizer for TokenizerSizer {
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
///
/// # Examples
///
/// ```ignore
/// use crate::embedding::chunking::text_splitter_adapter::TextSplitterAdapter;
/// use crate::embedding::ensure_tokenizer;
///
/// let tokenizer_bytes = load_tokenizer_bytes();
/// let tokenizer = ensure_tokenizer(tokenizer_bytes, 2048)?;
/// let chunker = TextSplitterAdapter::new(512, tokenizer);
/// let chunks = chunker.chunk("First sentence. Second sentence.")?;
/// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let chunker = TextSplitterAdapter::new(512, tokenizer);
    /// ```
    pub fn new(max_tokens: usize, tokenizer: &'static Tokenizer) -> Self {
        Self {
            max_tokens,
            tokenizer,
        }
    }
}

impl ChunkingStrategy for TextSplitterAdapter {
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>, EmbeddingError> {
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

        let chunks: Vec<_> = splitter
            .chunks(text)
            .enumerate()
            .map(|(index, chunk)| {
                // Calculate character positions
                // Note: text-splitter gives us the chunk text, we need to find positions
                let start_char = if index == 0 {
                    0
                } else {
                    // This is approximate - text-splitter doesn't provide exact byte offsets
                    // For now, we'll calculate based on accumulated text length
                    // This is a limitation but acceptable for our use case
                    text.find(chunk).unwrap_or(0)
                };
                let end_char = start_char + chunk.len();

                TextChunk {
                    index,
                    text: chunk.to_string(),
                    start_char,
                    end_char,
                }
            })
            .collect();

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
    use crate::embedding::tokenizer::ensure_tokenizer;

    // Helper to load test tokenizer
    fn load_test_tokenizer() -> &'static Tokenizer {
        use std::fs;
        let tokenizer_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/models/jina-bert-tokenizer.json"
        );
        let tokenizer_bytes = fs::read(tokenizer_path).expect("Failed to read tokenizer file");
        ensure_tokenizer(tokenizer_bytes, 2048).expect("Failed to load test tokenizer")
    }

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
    fn test_single_sentence() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        let text = "This is a single sentence.";
        let chunks = chunker.chunk(text).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_multiple_sentences() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        let text = "First. Second. Third. Fourth. Fifth.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
        // All chunks should be indexed sequentially
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn test_small_chunk_size() {
        let tokenizer = load_test_tokenizer();
        // Very small chunk size to force multiple chunks
        let chunker = TextSplitterAdapter::new(5, tokenizer);

        let text = "This is a longer piece of text that should be split into multiple chunks because we have a very small chunk size limit.";
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
        // Should handle Unicode correctly
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_chunking_strategy_trait() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        assert_eq!(chunker.name(), "text-splitter");
        assert_eq!(chunker.max_tokens(), 512);
    }

    #[test]
    fn test_chunk_boundaries() {
        let tokenizer = load_test_tokenizer();
        let chunker = TextSplitterAdapter::new(512, tokenizer);

        let text = "First sentence. Second sentence.";
        let chunks = chunker.chunk(text).unwrap();

        // Verify chunks have valid boundaries
        for chunk in &chunks {
            assert!(chunk.end_char >= chunk.start_char);
            assert!(chunk.end_char <= text.len());
        }
    }
}
