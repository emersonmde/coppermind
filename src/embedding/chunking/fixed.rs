//! Fixed-size text chunking strategy.
//!
//! This strategy splits text into fixed-size chunks based on estimated token count,
//! with configurable overlap. Useful as a fallback when semantic chunking isn't
//! appropriate or as a baseline for comparison.

use super::{ChunkingStrategy, TextChunk};
use crate::error::EmbeddingError;

/// Fixed-size chunking with overlap.
///
/// Splits text into chunks of approximately `chunk_tokens` tokens, with
/// `overlap_tokens` tokens overlapping between consecutive chunks.
///
/// # Algorithm
///
/// 1. Estimate token count using character-based heuristic
/// 2. Calculate chunk size in characters
/// 3. Slide a window across the text with specified overlap
///
/// # Trade-offs
///
/// **Pros:**
/// - Simple and predictable
/// - Guaranteed chunk sizes (no large outliers)
/// - Fast (no sentence parsing)
///
/// **Cons:**
/// - May break mid-sentence or mid-word
/// - Lower embedding quality due to fragments
/// - Poor user experience (search results show fragments)
///
/// # When to Use
///
/// - Fallback when sentence detection fails
/// - Baseline for benchmarking other strategies
/// - Content without clear sentence boundaries (e.g., logs, data dumps)
///
/// # Examples
///
/// ```ignore
/// let chunker = FixedSizeChunker::new(512, 128);
/// let chunks = chunker.chunk("Long text...")?;
/// ```
pub struct FixedSizeChunker {
    chunk_tokens: usize,
    overlap_tokens: usize,
}

impl FixedSizeChunker {
    /// Creates a new fixed-size chunker.
    ///
    /// # Arguments
    ///
    /// * `chunk_tokens` - Target tokens per chunk
    /// * `overlap_tokens` - Tokens to overlap between chunks
    ///
    /// # Panics
    ///
    /// Panics if `overlap_tokens >= chunk_tokens` (would create infinite loop).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let chunker = FixedSizeChunker::new(512, 128); // 25% overlap
    /// ```
    pub fn new(chunk_tokens: usize, overlap_tokens: usize) -> Self {
        assert!(
            overlap_tokens < chunk_tokens,
            "Overlap must be less than chunk size"
        );

        Self {
            chunk_tokens,
            overlap_tokens,
        }
    }

    /// Estimates character count for a given token count.
    ///
    /// Inverse of token estimation: ~4 characters per token.
    fn tokens_to_chars(&self, tokens: usize) -> usize {
        tokens * 4
    }
}

impl ChunkingStrategy for FixedSizeChunker {
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>, EmbeddingError> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(vec![]);
        }

        let chunk_chars = self.tokens_to_chars(self.chunk_tokens);
        let overlap_chars = self.tokens_to_chars(self.overlap_tokens);
        let step_size = chunk_chars - overlap_chars;

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < text.len() {
            let end = (start + chunk_chars).min(text.len());
            let chunk_text = &text[start..end];

            chunks.push(TextChunk {
                index,
                text: chunk_text.to_string(),
                start_char: start,
                end_char: end,
            });

            // If we've reached the end, break
            if end >= text.len() {
                break;
            }

            start += step_size;
            index += 1;
        }

        Ok(chunks)
    }

    fn name(&self) -> &'static str {
        "fixed"
    }

    fn max_tokens(&self) -> usize {
        self.chunk_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_size_no_overlap() {
        let chunker = FixedSizeChunker::new(10, 0); // ~40 chars per chunk
        let text = "a".repeat(100); // 100 chars
        let chunks = chunker.chunk(&text).unwrap();

        // Should create 3 chunks: 40, 40, 20
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text.len(), 40);
        assert_eq!(chunks[1].text.len(), 40);
        assert_eq!(chunks[2].text.len(), 20);
    }

    #[test]
    fn test_fixed_size_with_overlap() {
        let chunker = FixedSizeChunker::new(10, 2); // ~40 chars, ~8 char overlap
        let text = "a".repeat(100);
        let chunks = chunker.chunk(&text).unwrap();

        // Chunks should overlap
        assert!(chunks.len() > 1);
        for i in 1..chunks.len() {
            // Each chunk should start before previous chunk ends
            assert!(chunks[i].start_char < chunks[i - 1].end_char);
        }
    }

    #[test]
    fn test_empty_text() {
        let chunker = FixedSizeChunker::new(512, 128);
        let chunks = chunker.chunk("").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_single_small_chunk() {
        let chunker = FixedSizeChunker::new(512, 128);
        let text = "Short text.";
        let chunks = chunker.chunk(text).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
    }

    #[test]
    #[should_panic(expected = "Overlap must be less than chunk size")]
    fn test_invalid_overlap() {
        FixedSizeChunker::new(10, 10); // overlap == chunk size
    }

    #[test]
    fn test_chunking_strategy_trait() {
        let chunker = FixedSizeChunker::new(512, 128);
        assert_eq!(chunker.name(), "fixed");
        assert_eq!(chunker.max_tokens(), 512);
    }

    #[test]
    fn test_chunk_boundaries() {
        let chunker = FixedSizeChunker::new(10, 2);
        let text = "The quick brown fox jumps over the lazy dog";
        let chunks = chunker.chunk(text).unwrap();

        // Verify chunk boundaries are correct
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
            assert_eq!(&text[chunk.start_char..chunk.end_char], chunk.text);
        }
    }
}
