//! Error types for the Coppermind application.

use thiserror::Error;

/// Errors that can occur during embedding operations.
#[derive(Debug, Clone, Error)]
pub enum EmbeddingError {
    /// Failed to load model from bytes
    #[error("Failed to load model: {0}")]
    ModelLoad(String),
    /// Failed to create tensor during inference
    #[error("Failed to create tensor: {0}")]
    TensorCreation(String),
    /// Forward pass through the model failed
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    /// Failed to tokenize text
    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),
    /// Failed to fetch asset from network or filesystem
    #[error("Asset fetch failed: {0}")]
    AssetFetch(String),
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    /// Tokenizer not available or initialization failed
    #[error("Tokenizer unavailable: {0}")]
    TokenizerUnavailable(String),
    /// Model not available or initialization failed
    #[error("Model unavailable: {0}")]
    ModelUnavailable(String),
    /// Text chunking failed
    #[error("Chunking failed: {0}")]
    ChunkingFailed(String),
}

/// Errors that can occur during file processing operations.
#[derive(Debug, Clone, Error)]
pub enum FileProcessingError {
    /// Failed to read file
    #[error("Failed to read file: {0}")]
    FileRead(String),
    /// File appears to be binary (not text)
    #[error("Binary file detected: {0}")]
    BinaryFile(String),
    /// Failed to embed file contents
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(String),
    /// Failed to index chunks in search engine
    #[error("Indexing failed: {0}")]
    IndexingFailed(String),
    /// I/O error during directory traversal
    #[error("Directory traversal error: {0}")]
    DirectoryTraversal(String),
}

/// Convert from String to EmbeddingError for platform::run_blocking compatibility
impl From<String> for EmbeddingError {
    fn from(s: String) -> Self {
        EmbeddingError::InferenceFailed(s)
    }
}

/// Convert from EmbeddingError to String for backward compatibility
impl From<EmbeddingError> for String {
    fn from(err: EmbeddingError) -> String {
        err.to_string()
    }
}

/// Convert from String to FileProcessingError for platform::run_blocking compatibility
impl From<String> for FileProcessingError {
    fn from(s: String) -> Self {
        FileProcessingError::EmbeddingFailed(s)
    }
}

/// Convert from FileProcessingError to String for backward compatibility
impl From<FileProcessingError> for String {
    fn from(err: FileProcessingError) -> String {
        err.to_string()
    }
}

#[cfg(target_arch = "wasm32")]
impl From<EmbeddingError> for wasm_bindgen::JsValue {
    fn from(err: EmbeddingError) -> wasm_bindgen::JsValue {
        wasm_bindgen::JsValue::from_str(&err.to_string())
    }
}
