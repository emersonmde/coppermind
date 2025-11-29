//! Markdown-specific chunking adapter.
//!
//! This module wraps the `text-splitter` crate's `MarkdownSplitter` to provide
//! semantic chunking for Markdown documents with awareness of Markdown structure
//! (headings, lists, code blocks, etc.).

use super::{
    calculate_chunk_boundaries, tokenizer_sizer::TokenizerSizer, ChunkingStrategy, TextChunk,
};
use crate::error::ChunkingError;
use text_splitter::{ChunkConfig, MarkdownSplitter};
use tokenizers::Tokenizer;

/// Markdown splitter adapter using the `text-splitter` crate.
///
/// Provides semantic chunking for Markdown documents with awareness of
/// Markdown structure (headings, lists, code blocks, emphasis, etc.).
///
/// # Algorithm
///
/// The MarkdownSplitter uses a hierarchical semantic approach:
/// 1. Parses Markdown using pulldown-cmark (CommonMark spec)
/// 2. Identifies semantic boundaries at multiple levels (headings, paragraphs, code blocks, etc.)
/// 3. Selects the highest semantic level where content fits within chunk size
/// 4. Merges neighboring sections while respecting higher-level boundaries
pub struct MarkdownSplitterAdapter {
    max_tokens: usize,
    tokenizer: &'static Tokenizer,
}

impl MarkdownSplitterAdapter {
    /// Creates a new markdown-splitter adapter.
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

impl ChunkingStrategy for MarkdownSplitterAdapter {
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

        // Use text-splitter's markdown chunking
        let splitter = MarkdownSplitter::new(chunk_config);

        // Use shared helper to calculate boundaries correctly for duplicate text
        let chunks = calculate_chunk_boundaries(text, splitter.chunks(text));

        Ok(chunks)
    }

    fn name(&self) -> &'static str {
        "markdown-splitter"
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to load test tokenizer
    fn load_test_tokenizer() -> &'static Tokenizer {
        use once_cell::sync::OnceCell;
        use std::fs;

        static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();

        TOKENIZER.get_or_init(|| {
            let tokenizer_path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../coppermind/assets/models/jina-bert-tokenizer.json"
            );
            let tokenizer_bytes = fs::read(tokenizer_path).expect("Failed to read tokenizer file");
            Tokenizer::from_bytes(tokenizer_bytes).expect("Failed to load tokenizer")
        })
    }

    #[test]
    fn test_markdown_splitter_basic() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let text = "# Heading\n\nParagraph text.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let chunks = chunker.chunk("").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_markdown_headings() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let text = "# Level 1\n\nContent.\n\n## Level 2\n\nMore content.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn test_markdown_code_blocks() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let text = "Text before.\n\n```rust\nfn main() {}\n```\n\nText after.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunking_strategy_trait() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        assert_eq!(chunker.name(), "markdown-splitter");
        assert_eq!(chunker.max_tokens(), 512);
    }
}
