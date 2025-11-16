//! Markdown-specific chunking adapter.
//!
//! This module wraps the `text-splitter` crate's `MarkdownSplitter` to provide
//! semantic chunking for Markdown documents with awareness of Markdown structure
//! (headings, lists, code blocks, etc.).
//!
//! # Benefits over generic text splitting
//!
//! - **Markdown-aware**: Respects heading boundaries, code blocks, lists
//! - **Semantic units**: Chunks preserve complete Markdown elements
//! - **Pure Rust**: Uses pulldown-cmark (WASM-compatible)
//! - **CommonMark compliant**: Follows CommonMark specification

use super::{ChunkingStrategy, TextChunk};
use crate::error::EmbeddingError;
use text_splitter::{ChunkConfig, ChunkSizer, MarkdownSplitter};
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
///
/// # Examples
///
/// ```ignore
/// use crate::embedding::chunking::markdown_splitter_adapter::MarkdownSplitterAdapter;
/// use crate::embedding::ensure_tokenizer;
///
/// let tokenizer_bytes = load_tokenizer_bytes();
/// let tokenizer = ensure_tokenizer(tokenizer_bytes, 2048)?;
/// let chunker = MarkdownSplitterAdapter::new(512, tokenizer);
/// let chunks = chunker.chunk("# Heading\n\nParagraph text.")?;
/// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let chunker = MarkdownSplitterAdapter::new(512, tokenizer);
    /// ```
    pub fn new(max_tokens: usize, tokenizer: &'static Tokenizer) -> Self {
        Self {
            max_tokens,
            tokenizer,
        }
    }
}

impl ChunkingStrategy for MarkdownSplitterAdapter {
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

        // Use text-splitter's markdown chunking
        let splitter = MarkdownSplitter::new(chunk_config);

        let chunks: Vec<_> = splitter
            .chunks(text)
            .enumerate()
            .map(|(index, chunk)| {
                // Calculate character positions
                let start_char = if index == 0 {
                    0
                } else {
                    // Find position in original text
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
        "markdown-splitter"
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
    fn test_markdown_splitter_basic() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let text = "# Heading\n\nParagraph text.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
        // With 512 token limit, this should fit in one chunk
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
    fn test_whitespace_only() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let chunks = chunker.chunk("   \n\t  ").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_markdown_headings() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let text = "# Level 1\n\nContent.\n\n## Level 2\n\nMore content.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
        // All chunks should be indexed sequentially
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
    fn test_markdown_lists() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let text = "# List\n\n- Item 1\n- Item 2\n- Item 3";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_small_chunk_size() {
        let tokenizer = load_test_tokenizer();
        // Very small chunk size to force multiple chunks
        let chunker = MarkdownSplitterAdapter::new(5, tokenizer);

        let text = "# Heading\n\nThis is a longer piece of markdown text that should be split into multiple chunks because we have a very small chunk size limit.\n\n## Subheading\n\nMore text here.";
        let chunks = chunker.chunk(text).unwrap();

        // Should create multiple chunks
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_chunking_strategy_trait() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        assert_eq!(chunker.name(), "markdown-splitter");
        assert_eq!(chunker.max_tokens(), 512);
    }

    #[test]
    fn test_chunk_boundaries() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let text = "# Heading\n\nParagraph.";
        let chunks = chunker.chunk(text).unwrap();

        // Verify chunks have valid boundaries
        for chunk in &chunks {
            assert!(chunk.end_char >= chunk.start_char);
            assert!(chunk.end_char <= text.len());
        }
    }

    #[test]
    fn test_markdown_links_and_emphasis() {
        let tokenizer = load_test_tokenizer();
        let chunker = MarkdownSplitterAdapter::new(512, tokenizer);

        let text = "Here is a [link](https://example.com) and **bold** text and *italic* text.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
    }
}
