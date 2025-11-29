//! Tokenization utilities for text processing.
//!
//! This module provides the `TokenizerHandle` type for managing HuggingFace
//! tokenizers with proper truncation configuration.

use crate::error::EmbeddingError;
use tokenizers::tokenizer::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// Handle for a configured tokenizer.
///
/// Wraps a HuggingFace tokenizer with truncation settings. Unlike the singleton
/// pattern in the app crate, this is an owned type that the caller manages.
///
/// # Examples
///
/// ```ignore
/// let tokenizer_bytes = std::fs::read("tokenizer.json")?;
/// let handle = TokenizerHandle::from_bytes(tokenizer_bytes, 2048)?;
///
/// let tokens = handle.tokenize("Hello, world!")?;
/// println!("Token IDs: {:?}", tokens);
/// ```
pub struct TokenizerHandle {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl TokenizerHandle {
    /// Creates a tokenizer from JSON bytes with truncation configured.
    ///
    /// # Arguments
    ///
    /// * `tokenizer_bytes` - Serialized tokenizer JSON bytes
    /// * `max_length` - Maximum sequence length for truncation
    ///
    /// # Returns
    ///
    /// Configured tokenizer handle.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::TokenizerUnavailable` if initialization fails.
    pub fn from_bytes(tokenizer_bytes: Vec<u8>, max_length: usize) -> Result<Self, EmbeddingError> {
        let mut tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| {
            EmbeddingError::TokenizerUnavailable(format!("Failed to deserialize tokenizer: {}", e))
        })?;

        configure_truncation(&mut tokenizer, max_length)?;

        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    /// Returns the configured maximum length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Returns a reference to the underlying tokenizer.
    ///
    /// Useful for integrating with text-splitter's `ChunkSizer` trait.
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Tokenizes text into token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    ///
    /// Vector of token IDs, including special tokens (CLS, SEP).
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::TokenizationFailed` if encoding fails.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, EmbeddingError> {
        tokenize_text(&self.tokenizer, text)
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

impl Clone for TokenizerHandle {
    fn clone(&self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone(),
            max_length: self.max_length,
        }
    }
}

/// Configures tokenizer with truncation settings.
fn configure_truncation(
    tokenizer: &mut Tokenizer,
    max_length: usize,
) -> Result<(), EmbeddingError> {
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            stride: 0,
            strategy: TruncationStrategy::OnlyFirst,
            direction: TruncationDirection::Right,
        }))
        .map_err(|e| {
            EmbeddingError::InvalidConfig(format!(
                "Failed to configure tokenizer truncation: {}",
                e
            ))
        })?;

    Ok(())
}

/// Tokenizes text into token IDs.
///
/// # Arguments
///
/// * `tokenizer` - Reference to configured tokenizer
/// * `text` - Input text to tokenize
///
/// # Returns
///
/// Vector of token IDs, including special tokens (CLS, SEP).
pub fn tokenize_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, EmbeddingError> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| EmbeddingError::TokenizationFailed(format!("Encoding failed: {}", e)))?;

    let ids = encoding.get_ids();
    if ids.is_empty() {
        return Err(EmbeddingError::TokenizationFailed(
            "Tokenizer returned no tokens".to_string(),
        ));
    }

    Ok(ids.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load test tokenizer from assets
    fn load_test_tokenizer(max_length: usize) -> TokenizerHandle {
        // During tests, look for tokenizer in the coppermind crate's assets
        let tokenizer_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../coppermind/assets/models/jina-bert-tokenizer.json"
        );
        let tokenizer_bytes =
            std::fs::read(tokenizer_path).expect("Failed to read tokenizer file for tests");

        TokenizerHandle::from_bytes(tokenizer_bytes, max_length)
            .expect("Failed to create TokenizerHandle")
    }

    #[test]
    fn test_tokenize_basic() {
        let handle = load_test_tokenizer(512);
        let result = handle.tokenize("hello world");

        assert!(result.is_ok());
        let token_ids = result.unwrap();

        // Should have at least 2 tokens plus special tokens [CLS] and [SEP]
        assert!(
            token_ids.len() >= 4,
            "Expected at least 4 tokens, got {}",
            token_ids.len()
        );

        // First token should be [CLS] (token ID 101 for BERT-like models)
        assert_eq!(token_ids[0], 101, "First token should be [CLS]");

        // Last token should be [SEP] (token ID 102 for BERT-like models)
        assert_eq!(
            token_ids[token_ids.len() - 1],
            102,
            "Last token should be [SEP]"
        );
    }

    #[test]
    fn test_tokenize_empty_string() {
        let handle = load_test_tokenizer(512);
        let result = handle.tokenize("");

        // Empty string should still return special tokens [CLS] [SEP]
        assert!(result.is_ok());
        let token_ids = result.unwrap();
        assert_eq!(token_ids.len(), 2, "Empty string should return [CLS] [SEP]");
        assert_eq!(token_ids[0], 101); // [CLS]
        assert_eq!(token_ids[1], 102); // [SEP]
    }

    #[test]
    fn test_truncation() {
        let max_length = 10;
        let handle = load_test_tokenizer(max_length);

        // Create a long text that will exceed max_length
        let long_text = "word ".repeat(100);
        let result = handle.tokenize(&long_text);

        assert!(result.is_ok());
        let token_ids = result.unwrap();

        assert!(
            token_ids.len() <= max_length,
            "Expected <= {} tokens, got {}",
            max_length,
            token_ids.len()
        );
    }

    #[test]
    fn test_max_length() {
        let handle = load_test_tokenizer(256);
        assert_eq!(handle.max_length(), 256);
    }

    #[test]
    fn test_vocab_size() {
        let handle = load_test_tokenizer(512);
        let vocab_size = handle.vocab_size();
        // JinaBERT tokenizer should have ~30k tokens
        assert!(
            vocab_size > 20000,
            "Expected vocab size > 20000, got {}",
            vocab_size
        );
    }

    #[test]
    fn test_clone() {
        let handle = load_test_tokenizer(512);
        let cloned = handle.clone();

        assert_eq!(handle.max_length(), cloned.max_length());
        assert_eq!(handle.vocab_size(), cloned.vocab_size());

        // Both should tokenize the same
        let tokens1 = handle.tokenize("test").unwrap();
        let tokens2 = cloned.tokenize("test").unwrap();
        assert_eq!(tokens1, tokens2);
    }
}
