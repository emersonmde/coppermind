//! Test utilities for coppermind-core.
//!
//! This module provides shared helpers for unit tests, including tokenizer loading.
//! Only compiled when running tests.

use crate::embedding::tokenizer::TokenizerHandle;
use once_cell::sync::OnceCell;
use tokenizers::Tokenizer;

/// Path to the tokenizer file relative to CARGO_MANIFEST_DIR.
const TOKENIZER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../coppermind/assets/models/jina-bert-tokenizer.json"
);

/// Loads the raw test tokenizer (singleton, thread-safe).
///
/// This function caches the tokenizer on first call to avoid repeated file I/O
/// across test functions. The tokenizer is loaded from the coppermind app's
/// assets directory.
///
/// # Panics
///
/// Panics if the tokenizer file cannot be read or parsed. This is intentional
/// for tests - a missing tokenizer should fail loudly.
pub fn load_test_tokenizer() -> &'static Tokenizer {
    static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();

    TOKENIZER.get_or_init(|| {
        let tokenizer_bytes = std::fs::read(TOKENIZER_PATH).expect("Failed to read tokenizer file");
        Tokenizer::from_bytes(tokenizer_bytes).expect("Failed to load tokenizer")
    })
}

/// Creates a TokenizerHandle configured with the specified max_length.
///
/// Unlike `load_test_tokenizer()`, this creates a new handle each time since
/// the max_length may differ between tests.
///
/// # Panics
///
/// Panics if the tokenizer file cannot be read or parsed.
pub fn create_test_tokenizer_handle(max_length: usize) -> TokenizerHandle {
    let tokenizer_bytes = std::fs::read(TOKENIZER_PATH).expect("Failed to read tokenizer file");
    TokenizerHandle::from_bytes(tokenizer_bytes, max_length)
        .expect("Failed to create TokenizerHandle")
}
