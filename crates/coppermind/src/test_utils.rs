//! Test utilities for coppermind app crate.
//!
//! This module provides shared helpers for unit tests, including tokenizer loading.
//! Only compiled when running tests.

use crate::embedding::tokenizer::{configure_tokenizer, ensure_tokenizer};
use tokenizers::Tokenizer;

/// Path to the tokenizer file relative to CARGO_MANIFEST_DIR.
const TOKENIZER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/models/jina-bert-tokenizer.json"
);

/// Loads a configured tokenizer for testing with the specified max_positions.
///
/// Creates a new tokenizer instance each time since max_positions may differ.
///
/// # Panics
///
/// Panics if the tokenizer file cannot be read or configured.
pub fn create_configured_tokenizer(max_positions: usize) -> Tokenizer {
    let tokenizer_bytes = std::fs::read(TOKENIZER_PATH).expect("Failed to read tokenizer file");
    let mut tokenizer =
        Tokenizer::from_bytes(tokenizer_bytes).expect("Failed to deserialize tokenizer");
    configure_tokenizer(&mut tokenizer, max_positions).expect("Failed to configure tokenizer");
    tokenizer
}

/// Loads the test tokenizer via ensure_tokenizer (singleton, thread-safe).
///
/// This function uses the global tokenizer singleton and is appropriate for
/// chunking tests that need a static reference.
///
/// # Panics
///
/// Panics if the tokenizer file cannot be read or initialized.
pub fn load_test_tokenizer() -> &'static Tokenizer {
    // ensure_tokenizer already manages its own static singleton, so just call it directly
    let tokenizer_bytes = std::fs::read(TOKENIZER_PATH).expect("Failed to read tokenizer file");
    ensure_tokenizer(tokenizer_bytes, 2048).expect("Failed to load test tokenizer")
}
