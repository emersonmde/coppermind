//! Error types for the Coppermind application.

use std::fmt;

/// Errors that can occur during embedding operations.
#[derive(Debug, Clone)]
pub enum EmbeddingError {
    /// Failed to load model from bytes
    ModelLoad(String),
    /// Failed to create tensor during inference
    TensorCreation(String),
    /// Forward pass through the model failed
    InferenceFailed(String),
    /// Failed to tokenize text
    TokenizationFailed(String),
    /// Failed to fetch asset from network or filesystem
    AssetFetch(String),
    /// Invalid configuration
    InvalidConfig(String),
    /// Tokenizer not available or initialization failed
    TokenizerUnavailable(String),
    /// Model not available or initialization failed
    ModelUnavailable(String),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelLoad(msg) => write!(f, "Failed to load model: {}", msg),
            Self::TensorCreation(msg) => write!(f, "Failed to create tensor: {}", msg),
            Self::InferenceFailed(msg) => write!(f, "Inference failed: {}", msg),
            Self::TokenizationFailed(msg) => write!(f, "Tokenization failed: {}", msg),
            Self::AssetFetch(msg) => write!(f, "Asset fetch failed: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::TokenizerUnavailable(msg) => write!(f, "Tokenizer unavailable: {}", msg),
            Self::ModelUnavailable(msg) => write!(f, "Model unavailable: {}", msg),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// Errors that can occur during file processing operations.
#[derive(Debug, Clone)]
pub enum FileProcessingError {
    /// Failed to read file
    FileRead(String),
    /// File appears to be binary (not text)
    BinaryFile(String),
    /// Failed to embed file contents
    EmbeddingFailed(String),
    /// Failed to index chunks in search engine
    IndexingFailed(String),
    /// I/O error during directory traversal
    DirectoryTraversal(String),
}

impl fmt::Display for FileProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileRead(msg) => write!(f, "Failed to read file: {}", msg),
            Self::BinaryFile(msg) => write!(f, "Binary file detected: {}", msg),
            Self::EmbeddingFailed(msg) => write!(f, "Embedding failed: {}", msg),
            Self::IndexingFailed(msg) => write!(f, "Indexing failed: {}", msg),
            Self::DirectoryTraversal(msg) => write!(f, "Directory traversal error: {}", msg),
        }
    }
}

impl std::error::Error for FileProcessingError {}

/// Convert from EmbeddingError to String for backward compatibility
impl From<EmbeddingError> for String {
    fn from(err: EmbeddingError) -> String {
        err.to_string()
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
