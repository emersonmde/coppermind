//! Text chunking strategies for document processing.
//!
//! This module provides semantic chunking that splits documents into coherent
//! pieces before tokenization and embedding.
//!
//! # WASM Compatibility
//!
//! Uses the `text-splitter` crate with ICU4X for Unicode-aware sentence detection.
//! All dependencies are pure Rust and WASM-compatible.
//!
//! # Strategy Selection by File Type
//!
//! Current implementation supports:
//! - **Text documents**: Semantic sentence-based chunking (via text-splitter)
//!
//! Future support:
//! - **HTML**: Hierarchical by semantic tags
//! - **Code**: Function/class boundaries (text-splitter supports this)
//! - **Markdown**: Header-based hierarchy (text-splitter supports this)
//!
//! # Why Chunk Before Tokenizing?
//!
//! 1. **Coherence**: Preserves natural boundaries (sentences, paragraphs) rather than
//!    breaking mid-thought
//! 2. **Quality**: Models produce better embeddings for complete semantic units
//! 3. **Overlap**: Can overlap at semantic boundaries for context preservation
//! 4. **User experience**: Search results show readable passages, not fragments

pub mod markdown_splitter_adapter;
pub mod text_splitter_adapter;

use crate::error::EmbeddingError;
use std::path::Path;

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
/// use coppermind::embedding::chunking::{ChunkingStrategy, text_splitter_adapter::TextSplitterAdapter};
/// use coppermind::embedding::ensure_tokenizer;
///
/// let tokenizer = ensure_tokenizer(2048).await?;
/// let chunker = TextSplitterAdapter::new(512, tokenizer);
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

/// File type for chunking strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Markdown files (*.md, *.markdown)
    Markdown,
    /// Plain text files (everything else)
    Text,
}

/// Detects file type from filename or path.
///
/// Uses file extension to determine the appropriate chunking strategy:
/// - `.md`, `.markdown` → FileType::Markdown
/// - Everything else → FileType::Text
///
/// # Arguments
///
/// * `filename` - Filename or path to analyze
///
/// # Returns
///
/// FileType indicating which chunking strategy should be used.
///
/// # Examples
///
/// ```ignore
/// use crate::embedding::chunking::detect_file_type;
///
/// assert_eq!(detect_file_type("README.md"), FileType::Markdown);
/// assert_eq!(detect_file_type("document.txt"), FileType::Text);
/// assert_eq!(detect_file_type("script.py"), FileType::Text);
/// ```
pub fn detect_file_type<P: AsRef<Path>>(filename: P) -> FileType {
    let path = filename.as_ref();

    // Check extension
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        match ext_str.as_str() {
            "md" | "markdown" => return FileType::Markdown,
            _ => {}
        }
    }

    // Default to text for everything else
    FileType::Text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_markdown_files() {
        assert_eq!(detect_file_type("README.md"), FileType::Markdown);
        assert_eq!(detect_file_type("doc.markdown"), FileType::Markdown);
        assert_eq!(detect_file_type("path/to/file.md"), FileType::Markdown);
        assert_eq!(detect_file_type("NOTES.MD"), FileType::Markdown); // case insensitive
    }

    #[test]
    fn test_detect_text_files() {
        assert_eq!(detect_file_type("document.txt"), FileType::Text);
        assert_eq!(detect_file_type("file.log"), FileType::Text);
        assert_eq!(detect_file_type("data.json"), FileType::Text);
        assert_eq!(detect_file_type("no_extension"), FileType::Text);
    }

    #[test]
    fn test_detect_code_files_as_text() {
        // Code files should use text chunking for now
        // (code chunking requires tree-sitter which has WASM issues)
        assert_eq!(detect_file_type("script.py"), FileType::Text);
        assert_eq!(detect_file_type("main.rs"), FileType::Text);
        assert_eq!(detect_file_type("program.java"), FileType::Text);
        assert_eq!(detect_file_type("code.cpp"), FileType::Text);
        assert_eq!(detect_file_type("app.js"), FileType::Text);
    }

    #[test]
    fn test_detect_with_full_paths() {
        assert_eq!(
            detect_file_type("/Users/name/Documents/README.md"),
            FileType::Markdown
        );
        assert_eq!(
            detect_file_type("/var/log/app.log"),
            FileType::Text
        );
    }

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
