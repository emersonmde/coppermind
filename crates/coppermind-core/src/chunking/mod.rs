//! Text chunking strategies for document processing.
//!
//! This module provides semantic chunking that splits documents into coherent
//! pieces before tokenization and embedding.
//!
//! # WASM Compatibility
//!
//! Uses the `text-splitter` crate with ICU4X for Unicode-aware sentence detection.
//! All dependencies are pure Rust and WASM-compatible (except tree-sitter code chunking).
//!
//! # Strategy Selection by File Type
//!
//! Supports:
//! - **Text documents**: Semantic sentence-based chunking (via text-splitter)
//! - **Markdown**: Header-based hierarchy (text-splitter + pulldown-cmark)
//! - **Code**: Function/class boundaries (text-splitter + tree-sitter, native only)
//!
//! # Why Chunk Before Tokenizing?
//!
//! 1. **Coherence**: Preserves natural boundaries (sentences, paragraphs)
//! 2. **Quality**: Models produce better embeddings for complete semantic units
//! 3. **User experience**: Search results show readable passages

pub mod markdown_splitter_adapter;
pub mod text_splitter_adapter;

// Code chunking is only available on native platforms (tree-sitter uses C code)
#[cfg(not(target_arch = "wasm32"))]
pub mod code_splitter_adapter;

mod tokenizer_sizer;
mod types;

use crate::error::ChunkingError;
use std::path::Path;
use tokenizers::Tokenizer;

pub use tokenizer_sizer::TokenizerSizer;
pub use types::TextChunk;

#[cfg(not(target_arch = "wasm32"))]
pub use code_splitter_adapter::CodeLanguage;

/// Trait for text chunking strategies.
///
/// Implementations define how to split text into coherent chunks suitable
/// for embedding. Each strategy makes different trade-offs between chunk size,
/// overlap, and semantic coherence.
pub trait ChunkingStrategy: Send + Sync {
    /// Splits text into chunks according to this strategy.
    ///
    /// # Arguments
    ///
    /// * `text` - The source text to chunk
    ///
    /// # Returns
    ///
    /// Vector of text chunks with metadata. Chunks are ordered by their
    /// position in the source document (ascending `start_char`).
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>, ChunkingError>;

    /// Returns a human-readable name for this strategy.
    fn name(&self) -> &'static str;

    /// Returns the maximum target tokens per chunk.
    fn max_tokens(&self) -> usize;
}

/// File type for chunking strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Markdown files (*.md, *.markdown)
    Markdown,
    /// Code files (*.rs, *.py, *.js, *.java, etc.) - native platforms only
    #[cfg(not(target_arch = "wasm32"))]
    Code(code_splitter_adapter::CodeLanguage),
    /// Plain text files (everything else, or code files on WASM)
    Text,
}

/// Creates a chunker appropriate for the given file type.
///
/// # Arguments
///
/// * `file_type` - Detected file type (use `detect_file_type` to get this)
/// * `max_tokens` - Maximum tokens per chunk
/// * `tokenizer` - Reference to HuggingFace tokenizer
///
/// # Returns
///
/// A boxed `ChunkingStrategy` implementation appropriate for the file type.
pub fn create_chunker(
    file_type: FileType,
    max_tokens: usize,
    tokenizer: &'static Tokenizer,
) -> Box<dyn ChunkingStrategy> {
    match file_type {
        FileType::Markdown => Box::new(markdown_splitter_adapter::MarkdownSplitterAdapter::new(
            max_tokens, tokenizer,
        )),
        #[cfg(not(target_arch = "wasm32"))]
        FileType::Code(language) => Box::new(code_splitter_adapter::CodeSplitterAdapter::new(
            max_tokens, language, tokenizer,
        )),
        FileType::Text => Box::new(text_splitter_adapter::TextSplitterAdapter::new(
            max_tokens, tokenizer,
        )),
    }
}

/// Detects file type from filename or path.
///
/// Uses file extension to determine the appropriate chunking strategy:
/// - `.md`, `.markdown` -> FileType::Markdown
/// - Code extensions (`.rs`, `.py`, etc.) -> FileType::Code (native only)
/// - Everything else -> FileType::Text
///
/// On WASM, code files fall back to FileType::Text since tree-sitter doesn't
/// compile to WASM.
pub fn detect_file_type<P: AsRef<Path>>(filename: P) -> FileType {
    let path = filename.as_ref();

    // Check extension
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();

        // Check for markdown first
        if matches!(ext_str.as_str(), "md" | "markdown") {
            return FileType::Markdown;
        }

        // Check for code files (native only - tree-sitter uses C code)
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(language) = code_splitter_adapter::CodeLanguage::from_extension(&ext_str) {
                return FileType::Code(language);
            }
        }
    }

    // Default to text for everything else
    FileType::Text
}

/// Helper to calculate chunk boundaries by tracking cumulative position.
///
/// Unlike `text.find(chunk)` which fails with duplicate text, this tracks
/// position cumulatively as we iterate through chunks.
#[allow(dead_code)] // Used by chunking adapters in feature-gated code
pub(crate) fn calculate_chunk_boundaries<'a, I>(text: &str, chunks_iter: I) -> Vec<TextChunk>
where
    I: Iterator<Item = &'a str>,
{
    let mut result = Vec::new();
    let mut search_start = 0;

    for (index, chunk) in chunks_iter.enumerate() {
        // Search for chunk starting from where we left off
        let start_char = if let Some(pos) = text[search_start..].find(chunk) {
            search_start + pos
        } else {
            // Fallback: search from beginning (shouldn't happen with well-behaved splitters)
            text.find(chunk).unwrap_or(0)
        };
        let end_char = start_char + chunk.len();

        result.push(TextChunk {
            index,
            text: chunk.to_string(),
            start_char,
            end_char,
        });

        // Move search position past this chunk for next iteration
        search_start = end_char;
    }

    result
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

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_code_files() {
        use code_splitter_adapter::CodeLanguage;
        assert_eq!(
            detect_file_type("script.py"),
            FileType::Code(CodeLanguage::Python)
        );
        assert_eq!(
            detect_file_type("main.rs"),
            FileType::Code(CodeLanguage::Rust)
        );
        assert_eq!(
            detect_file_type("program.java"),
            FileType::Code(CodeLanguage::Java)
        );
    }

    #[test]
    fn test_detect_with_full_paths() {
        assert_eq!(
            detect_file_type("/Users/name/Documents/README.md"),
            FileType::Markdown
        );
        assert_eq!(detect_file_type("/var/log/app.log"), FileType::Text);
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
