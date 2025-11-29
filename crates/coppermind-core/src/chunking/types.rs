//! Types for text chunking.

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
