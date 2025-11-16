//! Text chunking strategies for document processing.
//!
//! This module provides pluggable chunking strategies that split documents
//! into semantically coherent pieces before tokenization and embedding.
//!
//! # WASM Compatibility
//!
//! All chunkers are pure Rust with minimal dependencies, ensuring they work
//! identically on web (WASM) and desktop platforms. We avoid heavy dependencies
//! like ICU for Unicode segmentation in favor of regex-based sentence detection.
//!
//! # Strategy Selection by File Type
//!
//! Different content types benefit from different chunking approaches:
//! - **Text documents**: Sentence-based (preserves complete thoughts)
//! - **HTML**: Hierarchical by semantic tags (future)
//! - **Code**: Function/class boundaries (future)
//! - **Markdown**: Header-based hierarchy (future)
//!
//! # Why Chunk Before Tokenizing?
//!
//! 1. **Coherence**: Preserves natural boundaries (sentences, paragraphs) rather than
//!    breaking mid-thought
//! 2. **Quality**: Models produce better embeddings for complete semantic units
//! 3. **Overlap**: Can overlap at semantic boundaries for context preservation
//! 4. **User experience**: Search results show readable passages, not fragments

pub mod fixed;
pub mod sentence;
pub mod text_splitter_adapter;

use crate::error::EmbeddingError;

/// A chunk of text with metadata about its position in the source document.
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// Index of this chunk in the document (0-based)
    pub index: usize,
    /// The text content of this chunk
    pub text: String,
    /// Character offset where this chunk starts in the original document
    pub start_char: usize,
    /// Character offset where this chunk ends in the original document
    pub end_char: usize,
}

/// Trait for text chunking strategies.
///
/// Implementations define how to split text into coherent chunks suitable
/// for embedding. Each strategy makes different trade-offs between chunk size,
/// overlap, and semantic coherence.
///
/// # Examples
///
/// ```ignore
/// use coppermind::embedding::chunking::{ChunkingStrategy, sentence::SentenceChunker};
///
/// let chunker = SentenceChunker::new(512, 2); // 512 tokens, 2 sentence overlap
/// let chunks = chunker.chunk("First sentence. Second sentence. Third sentence.")?;
/// ```
pub trait ChunkingStrategy: Send + Sync {
    /// Splits text into chunks according to this strategy.
    ///
    /// # Arguments
    ///
    /// * `text` - The source text to chunk
    ///
    /// # Returns
    ///
    /// Vector of text chunks with metadata. Chunks should be ordered by their
    /// position in the source document (ascending `start_char`).
    ///
    /// # Errors
    ///
    /// Returns error if chunking fails (e.g., text too short, invalid input).
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>, EmbeddingError>;

    /// Returns a human-readable name for this strategy.
    ///
    /// Used for logging and debugging.
    fn name(&self) -> &'static str;

    /// Returns the maximum target tokens per chunk.
    ///
    /// Note: Actual chunks may exceed this slightly if a single semantic unit
    /// (sentence, paragraph) is longer than the limit.
    fn max_tokens(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_chunk_ordering() {
        let chunk1 = TextChunk {
            index: 0,
            text: "First".to_string(),
            start_char: 0,
            end_char: 5,
        };
        let chunk2 = TextChunk {
            index: 1,
            text: "Second".to_string(),
            start_char: 6,
            end_char: 12,
        };

        assert!(chunk1.start_char < chunk2.start_char);
    }
}
