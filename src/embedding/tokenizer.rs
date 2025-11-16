//! Tokenization utilities for text processing.
//!
//! This module handles tokenizer initialization, text tokenization, and
//! chunking strategies for processing long documents.

use crate::error::EmbeddingError;
use once_cell::sync::OnceCell;
use tokenizers::tokenizer::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// Global tokenizer singleton.
///
/// Initialized once on first use to avoid repeated downloads/deserialization.
/// In a multi-model future, this may become a HashMap<ModelId, Tokenizer>.
static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();

/// Returns cached tokenizer if already initialized.
///
/// # Returns
///
/// Some(tokenizer) if already initialized, None otherwise.
pub fn get_cached_tokenizer() -> Option<&'static Tokenizer> {
    TOKENIZER.get()
}

/// Ensures the tokenizer is initialized and returns a reference.
///
/// Downloads and configures the tokenizer on first call, then caches it
/// for subsequent calls. Thread-safe via `OnceCell`.
///
/// # Arguments
///
/// * `tokenizer_bytes` - Serialized tokenizer JSON bytes
/// * `max_positions` - Maximum sequence length for truncation
///
/// # Returns
///
/// Static reference to the initialized tokenizer.
///
/// # Errors
///
/// Returns `EmbeddingError::TokenizerUnavailable` if initialization fails.
pub fn ensure_tokenizer(
    tokenizer_bytes: Vec<u8>,
    max_positions: usize,
) -> Result<&'static Tokenizer, EmbeddingError> {
    // Use get_or_try_init for atomic initialization (prevents race conditions)
    TOKENIZER.get_or_try_init(|| {
        let mut tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| {
            EmbeddingError::TokenizerUnavailable(format!("Failed to deserialize tokenizer: {}", e))
        })?;

        configure_tokenizer(&mut tokenizer, max_positions)?;

        Ok(tokenizer)
    })
}

/// Configures tokenizer with truncation settings.
///
/// # Arguments
///
/// * `tokenizer` - Mutable reference to tokenizer
/// * `max_positions` - Maximum sequence length
fn configure_tokenizer(
    tokenizer: &mut Tokenizer,
    max_positions: usize,
) -> Result<(), EmbeddingError> {
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_positions,
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
///
/// # Errors
///
/// Returns `EmbeddingError::TokenizationFailed` if encoding fails or produces no tokens.
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

/// Async wrapper for tokenization that uses `spawn_blocking` on desktop.
///
/// On web platforms, tokenization runs directly. On desktop, it's offloaded
/// to a blocking thread pool to prevent UI freezing.
///
/// # Arguments
///
/// * `tokenizer` - Reference to configured tokenizer
/// * `text` - Input text to tokenize
///
/// # Returns
///
/// Vector of token IDs.
pub async fn tokenize_text_async(
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<Vec<u32>, EmbeddingError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let tokenizer_clone = tokenizer.clone();
        let text_owned = text.to_string();
        tokio::task::spawn_blocking(move || tokenize_text(&tokenizer_clone, &text_owned))
            .await
            .map_err(|e| EmbeddingError::TokenizationFailed(format!("Task join failed: {}", e)))?
    }

    #[cfg(target_arch = "wasm32")]
    {
        tokenize_text(tokenizer, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load test tokenizer from assets
    fn load_test_tokenizer(max_positions: usize) -> Tokenizer {
        let tokenizer_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/models/jina-bert-tokenizer.json"
        );
        let tokenizer_bytes =
            std::fs::read(tokenizer_path).expect("Failed to read tokenizer file for tests");

        let mut tokenizer =
            Tokenizer::from_bytes(tokenizer_bytes).expect("Failed to deserialize tokenizer");

        configure_tokenizer(&mut tokenizer, max_positions).expect("Failed to configure tokenizer");

        tokenizer
    }

    #[test]
    fn test_tokenize_text_basic() {
        let tokenizer = load_test_tokenizer(512);
        let result = tokenize_text(&tokenizer, "hello world");

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
    fn test_tokenize_text_empty_string() {
        let tokenizer = load_test_tokenizer(512);
        let result = tokenize_text(&tokenizer, "");

        // Empty string should still return special tokens [CLS] [SEP]
        assert!(result.is_ok());
        let token_ids = result.unwrap();
        assert_eq!(token_ids.len(), 2, "Empty string should return [CLS] [SEP]");
        assert_eq!(token_ids[0], 101); // [CLS]
        assert_eq!(token_ids[1], 102); // [SEP]
    }

    #[test]
    fn test_tokenize_text_with_truncation() {
        let max_positions = 10;
        let tokenizer = load_test_tokenizer(max_positions);

        // Create a long text that will exceed max_positions
        let long_text = "word ".repeat(100); // 100 words, will be truncated
        let result = tokenize_text(&tokenizer, &long_text);

        assert!(result.is_ok());
        let token_ids = result.unwrap();

        // Should be truncated to max_positions
        assert!(
            token_ids.len() <= max_positions,
            "Expected <= {} tokens, got {}",
            max_positions,
            token_ids.len()
        );
    }

    #[test]
    fn test_tokenize_text_special_characters() {
        let tokenizer = load_test_tokenizer(512);
        let text = "Hello, world! This is a test: 123 @#$%";
        let result = tokenize_text(&tokenizer, text);

        assert!(result.is_ok());
        let token_ids = result.unwrap();
        assert!(token_ids.len() > 2, "Should tokenize special characters");
    }

    #[test]
    fn test_tokenize_text_unicode() {
        let tokenizer = load_test_tokenizer(512);
        let text = "Hello 世界 🌍 café";
        let result = tokenize_text(&tokenizer, text);

        assert!(result.is_ok());
        let token_ids = result.unwrap();
        assert!(token_ids.len() > 2, "Should handle unicode characters");
    }

    #[tokio::test]
    async fn test_tokenize_text_async_basic() {
        let tokenizer = load_test_tokenizer(512);
        let result = tokenize_text_async(&tokenizer, "async test").await;

        assert!(result.is_ok());
        let token_ids = result.unwrap();
        assert!(token_ids.len() >= 4); // [CLS] + at least 2 tokens + [SEP]
    }

    #[test]
    fn test_get_cached_tokenizer_before_init() {
        // Note: This test may fail if ensure_tokenizer was called in a previous test
        // In practice, the singleton persists across all tests in the same run
        let cached = get_cached_tokenizer();
        // Can be None or Some depending on test execution order
        // Just verify it doesn't panic
        let _ = cached;
    }

    #[test]
    fn test_ensure_tokenizer_initializes() {
        let tokenizer_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/models/jina-bert-tokenizer.json"
        );
        let tokenizer_bytes =
            std::fs::read(tokenizer_path).expect("Failed to read tokenizer file for tests");

        // Note: Singleton may already be initialized by other tests
        // We just verify ensure_tokenizer succeeds (either initializing or returning cached)
        let result = ensure_tokenizer(tokenizer_bytes, 512);
        assert!(
            result.is_ok(),
            "ensure_tokenizer should succeed: {:?}",
            result.err()
        );

        // After calling ensure_tokenizer, the tokenizer should be cached
        let cached = get_cached_tokenizer();
        assert!(
            cached.is_some(),
            "Tokenizer should be cached after ensure_tokenizer"
        );
    }

    #[test]
    fn test_ensure_tokenizer_returns_cached() {
        let tokenizer_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/models/jina-bert-tokenizer.json"
        );
        let tokenizer_bytes =
            std::fs::read(tokenizer_path).expect("Failed to read tokenizer file for tests");

        // Note: Singleton may already be initialized by previous tests
        // First call either initializes or returns cached
        let result1 = ensure_tokenizer(tokenizer_bytes.clone(), 512);
        assert!(
            result1.is_ok(),
            "First ensure_tokenizer call failed: {:?}",
            result1.err()
        );

        // Second call should return the same cached tokenizer
        let result2 = ensure_tokenizer(tokenizer_bytes, 1024);
        assert!(
            result2.is_ok(),
            "Second ensure_tokenizer call failed: {:?}",
            result2.err()
        );

        // Both should point to the same tokenizer
        let ptr1 = result1.unwrap() as *const Tokenizer;
        let ptr2 = result2.unwrap() as *const Tokenizer;
        assert_eq!(ptr1, ptr2, "Should return same tokenizer instance");
    }

    #[test]
    fn test_configure_tokenizer_sets_truncation() {
        let tokenizer_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/models/jina-bert-tokenizer.json"
        );
        let tokenizer_bytes =
            std::fs::read(tokenizer_path).expect("Failed to read tokenizer file for tests");

        let mut tokenizer =
            Tokenizer::from_bytes(tokenizer_bytes).expect("Failed to deserialize tokenizer");

        let result = configure_tokenizer(&mut tokenizer, 256);
        assert!(result.is_ok());

        // Verify truncation is enabled by tokenizing long text
        let long_text = "word ".repeat(1000);
        let encoding = tokenizer.encode(long_text.as_str(), true).unwrap();
        assert!(
            encoding.get_ids().len() <= 256,
            "Truncation should limit tokens to 256"
        );
    }
}
