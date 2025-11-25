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

// Code chunking is only available on native platforms (tree-sitter uses C code)
#[cfg(not(target_arch = "wasm32"))]
pub mod code_splitter_adapter;

use crate::error::EmbeddingError;
use std::path::Path;
use text_splitter::ChunkSizer;
use tokenizers::Tokenizer;

/// ChunkSizer implementation for HuggingFace Tokenizer.
///
/// Wraps our tokenizer to implement text-splitter's ChunkSizer trait,
/// allowing token-based chunk sizing without the onig dependency.
///
/// This is shared across all chunking adapters (text, markdown, code).
pub(crate) struct TokenizerSizer {
    pub tokenizer: &'static Tokenizer,
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

/// Helper to calculate chunk boundaries by tracking cumulative position.
///
/// Unlike `text.find(chunk)` which fails with duplicate text, this tracks
/// position cumulatively as we iterate through chunks.
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
    /// Code files (*.rs, *.py, *.js, *.java, etc.) - native platforms only
    #[cfg(not(target_arch = "wasm32"))]
    Code(code_splitter_adapter::CodeLanguage),
    /// Plain text files (everything else, or code files on WASM)
    Text,
}

/// Creates a chunker appropriate for the given file type.
///
/// This factory function encapsulates the platform-specific logic for selecting
/// the right chunking strategy based on file type.
///
/// # Arguments
///
/// * `file_type` - Detected file type (use `detect_file_type` to get this)
/// * `max_tokens` - Maximum tokens per chunk
/// * `tokenizer` - Reference to HuggingFace tokenizer (must be static lifetime)
///
/// # Returns
///
/// A boxed `ChunkingStrategy` implementation appropriate for the file type.
///
/// # Examples
///
/// ```ignore
/// use coppermind::embedding::chunking::{create_chunker, detect_file_type, FileType};
///
/// let file_type = detect_file_type("README.md");
/// let chunker = create_chunker(file_type, 512, tokenizer);
/// let chunks = chunker.chunk("# Heading\n\nContent...")?;
/// ```
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
/// - `.md`, `.markdown` → FileType::Markdown
/// - Code extensions (`.rs`, `.py`, etc.) → FileType::Code (native only)
/// - Everything else → FileType::Text
///
/// On WASM, code files fall back to FileType::Text since tree-sitter doesn't
/// compile to WASM.
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
/// // On native: FileType::Code(CodeLanguage::Python)
/// // On WASM: FileType::Text
/// let _ = detect_file_type("script.py");
/// ```
pub fn detect_file_type<P: AsRef<Path>>(filename: P) -> FileType {
    let path = filename.as_ref();

    // Check extension
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();

        // Check for markdown first
        if matches!(ext_str.as_str(), "md" | "markdown") {
            return FileType::Markdown;
        }

        // Check for code files (native only)
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(language) = code_splitter_adapter::CodeLanguage::from_extension(&ext_str) {
                return FileType::Code(language);
            }
        }
    }

    // Default to text for everything else (including code files on WASM)
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
    fn test_detect_code_files() {
        // On native platforms: Code files use syntax-aware chunking
        // On WASM: Code files fall back to text chunking (tree-sitter doesn't work)
        #[cfg(not(target_arch = "wasm32"))]
        {
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
            assert_eq!(
                detect_file_type("code.cpp"),
                FileType::Code(CodeLanguage::Cpp)
            );
            assert_eq!(
                detect_file_type("app.js"),
                FileType::Code(CodeLanguage::JavaScript)
            );
        }

        #[cfg(target_arch = "wasm32")]
        {
            assert_eq!(detect_file_type("script.py"), FileType::Text);
            assert_eq!(detect_file_type("main.rs"), FileType::Text);
            assert_eq!(detect_file_type("program.java"), FileType::Text);
            assert_eq!(detect_file_type("code.cpp"), FileType::Text);
            assert_eq!(detect_file_type("app.js"), FileType::Text);
        }
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
