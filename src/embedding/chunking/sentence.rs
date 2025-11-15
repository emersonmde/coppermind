//! Sentence-based text chunking strategy.
//!
//! This strategy splits text at sentence boundaries, preserving complete thoughts
//! and producing coherent chunks suitable for embedding.

use super::{ChunkingStrategy, TextChunk};
use crate::error::EmbeddingError;
use once_cell::sync::Lazy;
use regex::Regex;

// Global regex pattern for sentence detection (compiled once)
static SENTENCE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    // Matches: `. ! ?` followed by whitespace or end of string
    // Note: Doesn't handle all abbreviations perfectly (e.g., "Dr. Smith")
    // but good enough for chunking purposes (slight over-splitting is acceptable)
    Regex::new(r"[.!?]+(?:\s+|$)").expect("Invalid sentence regex pattern")
});

/// Sentence-based chunking with configurable overlap.
///
/// Splits text into sentences, then groups sentences into chunks that respect
/// a maximum token limit. Includes overlap between chunks to preserve context
/// at boundaries.
///
/// # Algorithm
///
/// 1. Split text into sentences using regex (handles `. ! ?` with common edge cases)
/// 2. Estimate tokens per sentence (rough heuristic: ~1 token per 4 characters)
/// 3. Group sentences into chunks respecting `max_tokens`
/// 4. Add `overlap_sentences` from previous chunk to preserve context
///
/// # Edge Cases
///
/// - **Long sentence exceeds limit**: Include anyway (chunk may exceed max_tokens)
/// - **Very short text**: Returns single chunk
/// - **Empty text**: Returns empty vector
///
/// # Examples
///
/// ```ignore
/// let chunker = SentenceChunker::new(512, 2);
/// let chunks = chunker.chunk("First. Second. Third. Fourth.")?;
/// // Chunk 0: [First, Second, Third] (3 sentences)
/// // Chunk 1: [Second, Third, Fourth] (2 overlap + 1 new)
/// ```
pub struct SentenceChunker {
    max_tokens: usize,
    overlap_sentences: usize,
}

impl SentenceChunker {
    /// Creates a new sentence-based chunker.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Target maximum tokens per chunk
    /// * `overlap_sentences` - Number of sentences to overlap between chunks
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 512 tokens per chunk, 2 sentences overlap
    /// let chunker = SentenceChunker::new(512, 2);
    /// ```
    pub fn new(max_tokens: usize, overlap_sentences: usize) -> Self {
        Self {
            max_tokens,
            overlap_sentences,
        }
    }

    /// Splits text into sentences.
    ///
    /// Uses regex to detect sentence boundaries while handling common edge cases
    /// like abbreviations.
    fn split_sentences(&self, text: &str) -> Vec<Sentence> {
        let text = text.trim();
        if text.is_empty() {
            return vec![];
        }

        let mut sentences = Vec::new();
        let mut last_end = 0;

        for mat in SENTENCE_PATTERN.find_iter(text) {
            let end = mat.end();
            let sentence_text = text[last_end..end].trim();

            if !sentence_text.is_empty() {
                sentences.push(Sentence {
                    text: sentence_text.to_string(),
                    start_char: last_end,
                    end_char: end,
                    estimated_tokens: estimate_token_count(sentence_text),
                });
            }

            last_end = end;
        }

        // Handle final sentence if it doesn't end with punctuation
        if last_end < text.len() {
            let sentence_text = text[last_end..].trim();
            if !sentence_text.is_empty() {
                sentences.push(Sentence {
                    text: sentence_text.to_string(),
                    start_char: last_end,
                    end_char: text.len(),
                    estimated_tokens: estimate_token_count(sentence_text),
                });
            }
        }

        // Fallback: if no sentences detected, treat entire text as one sentence
        if sentences.is_empty() && !text.is_empty() {
            sentences.push(Sentence {
                text: text.to_string(),
                start_char: 0,
                end_char: text.len(),
                estimated_tokens: estimate_token_count(text),
            });
        }

        sentences
    }

    /// Groups sentences into chunks respecting token limits and overlap.
    fn group_into_chunks(&self, sentences: Vec<Sentence>) -> Vec<TextChunk> {
        if sentences.is_empty() {
            return vec![];
        }

        let mut chunks = Vec::new();
        let mut current_sentences: Vec<Sentence> = Vec::new();
        let mut current_tokens = 0;
        let mut chunk_index = 0;

        for sentence in sentences {
            let sentence_tokens = sentence.estimated_tokens;

            // Check if adding this sentence would exceed limit
            if current_tokens + sentence_tokens > self.max_tokens && !current_sentences.is_empty() {
                // Flush current chunk
                chunks.push(self.build_chunk(&current_sentences, chunk_index));
                chunk_index += 1;

                // Start new chunk with overlap from previous chunk
                let overlap_start = current_sentences
                    .len()
                    .saturating_sub(self.overlap_sentences);
                current_sentences = current_sentences[overlap_start..].to_vec();
                current_tokens = current_sentences.iter().map(|s| s.estimated_tokens).sum();
            }

            current_sentences.push(sentence);
            current_tokens += sentence_tokens;
        }

        // Flush final chunk
        if !current_sentences.is_empty() {
            chunks.push(self.build_chunk(&current_sentences, chunk_index));
        }

        chunks
    }

    /// Builds a TextChunk from a group of sentences.
    fn build_chunk(&self, sentences: &[Sentence], index: usize) -> TextChunk {
        let text = sentences
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let start_char = sentences.first().map(|s| s.start_char).unwrap_or(0);
        let end_char = sentences.last().map(|s| s.end_char).unwrap_or(0);

        TextChunk {
            index,
            text,
            start_char,
            end_char,
        }
    }
}

impl ChunkingStrategy for SentenceChunker {
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>, EmbeddingError> {
        let sentences = self.split_sentences(text);
        Ok(self.group_into_chunks(sentences))
    }

    fn name(&self) -> &'static str {
        "sentence"
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

/// Internal representation of a sentence with metadata.
#[derive(Debug, Clone)]
struct Sentence {
    text: String,
    start_char: usize,
    end_char: usize,
    estimated_tokens: usize,
}

/// Estimates token count for text using a simple heuristic.
///
/// Rule of thumb: ~1 token per 4 characters for English text.
/// This is a rough estimate - actual tokenization may differ by ±20%.
///
/// More accurate estimation would require running the tokenizer, but that
/// defeats the purpose of chunking before tokenizing.
fn estimate_token_count(text: &str) -> usize {
    // Simple heuristic: average ~4 characters per token
    // Add word count for better accuracy (whitespace-separated words)
    let char_estimate = text.len() / 4;
    let word_count = text.split_whitespace().count();

    // Average the two estimates (tends to be more accurate)
    (char_estimate + word_count) / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_splitting() {
        let chunker = SentenceChunker::new(512, 0);
        let sentences = chunker.split_sentences("First sentence. Second sentence! Third sentence?");

        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].text.contains("First"));
        assert!(sentences[1].text.contains("Second"));
        assert!(sentences[2].text.contains("Third"));
    }

    #[test]
    fn test_abbreviation_handling() {
        let chunker = SentenceChunker::new(512, 0);
        let sentences = chunker.split_sentences("Dr. Smith went to the store. He bought milk.");

        // Note: Simplified regex splits on "Dr." (3 sentences instead of ideal 2)
        // This is acceptable for chunking purposes - slight over-splitting is fine
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_empty_text() {
        let chunker = SentenceChunker::new(512, 0);
        let chunks = chunker.chunk("").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_single_sentence() {
        let chunker = SentenceChunker::new(512, 0);
        let chunks = chunker.chunk("Single sentence.").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_overlap() {
        // Very small chunks (3 tokens max) to force multiple chunks
        // Each single-word sentence is ~1-2 tokens
        let chunker = SentenceChunker::new(3, 1);

        let text = "First. Second. Third. Fourth.";
        let chunks = chunker.chunk(text).unwrap();

        // With 3 token limit, should create multiple chunks
        // Chunk 0: First, Second (overlap=1 means Second appears in both)
        // Chunk 1: Second, Third
        // Chunk 2: Third, Fourth
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks, got {}",
            chunks.len()
        );

        if chunks.len() >= 2 {
            // Second chunk should contain overlapping sentence
            assert!(chunks[1].text.contains("Second") || chunks[1].text.contains("Third"));
        }
    }

    #[test]
    fn test_token_estimation() {
        // "Hello world" ~ 2-3 tokens
        let estimate = estimate_token_count("Hello world");
        assert!((2..=3).contains(&estimate));

        // Longer text
        let estimate = estimate_token_count("The quick brown fox jumps over the lazy dog");
        assert!((8..=12).contains(&estimate));
    }

    #[test]
    fn test_chunking_strategy_trait() {
        let chunker = SentenceChunker::new(512, 2);
        assert_eq!(chunker.name(), "sentence");
        assert_eq!(chunker.max_tokens(), 512);
    }
}
