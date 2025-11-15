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
    if let Some(tokenizer) = TOKENIZER.get() {
        return Ok(tokenizer);
    }

    let mut tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| {
        EmbeddingError::TokenizerUnavailable(format!("Failed to deserialize tokenizer: {}", e))
    })?;

    configure_tokenizer(&mut tokenizer, max_positions)?;

    TOKENIZER.set(tokenizer).map_err(|_| {
        EmbeddingError::TokenizerUnavailable("Tokenizer already initialized".to_string())
    })?;

    TOKENIZER.get().ok_or_else(|| {
        EmbeddingError::TokenizerUnavailable(
            "Tokenizer unavailable after initialization".to_string(),
        )
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
