//! Error types for coppermind-core.
//!
//! This module defines error types that are used across the core library,
//! including embedding, chunking, asset loading, and GPU scheduling errors.

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

/// Errors that can occur during asset loading.
#[derive(Debug, Clone, Error)]
pub enum AssetError {
    /// Failed to load asset from source
    #[error("Failed to load asset: {0}")]
    LoadFailed(String),
    /// Asset not found at expected location
    #[error("Asset not found: {0}")]
    NotFound(String),
    /// Asset data is invalid or corrupted
    #[error("Invalid asset data: {0}")]
    InvalidData(String),
}

/// Errors that can occur during GPU scheduling.
#[derive(Debug, Clone, Error)]
pub enum GpuError {
    /// Scheduler not initialized
    #[error("GPU scheduler not initialized")]
    NotInitialized,
    /// Failed to send request to scheduler
    #[error("Failed to send request: {0}")]
    SendFailed(String),
    /// Failed to receive response from scheduler
    #[error("Failed to receive response: {0}")]
    ReceiveFailed(String),
    /// Embedding operation failed
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(String),
    /// Model creation failed on GPU thread
    #[error("Model creation failed: {0}")]
    ModelCreationFailed(String),
    /// Worker thread panicked
    #[error("Worker thread panicked")]
    WorkerPanicked,
}

/// Errors that can occur during text chunking.
#[derive(Debug, Clone, Error)]
pub enum ChunkingError {
    /// Failed to chunk text
    #[error("Failed to chunk text: {0}")]
    ChunkFailed(String),
    /// Invalid chunking configuration
    #[error("Invalid chunking config: {0}")]
    InvalidConfig(String),
    /// Tokenizer error during chunk sizing
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
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

// Conversion implementations for error chaining

impl From<EmbeddingError> for String {
    fn from(err: EmbeddingError) -> String {
        err.to_string()
    }
}

impl From<String> for EmbeddingError {
    fn from(s: String) -> Self {
        EmbeddingError::InferenceFailed(s)
    }
}

impl From<AssetError> for EmbeddingError {
    fn from(err: AssetError) -> Self {
        EmbeddingError::ModelLoad(err.to_string())
    }
}

impl From<ChunkingError> for EmbeddingError {
    fn from(err: ChunkingError) -> Self {
        EmbeddingError::ChunkingFailed(err.to_string())
    }
}

impl From<GpuError> for EmbeddingError {
    fn from(err: GpuError) -> Self {
        match err {
            GpuError::NotInitialized => {
                EmbeddingError::ModelUnavailable("GPU scheduler not initialized".to_string())
            }
            _ => EmbeddingError::InferenceFailed(err.to_string()),
        }
    }
}

impl From<FileProcessingError> for String {
    fn from(err: FileProcessingError) -> String {
        err.to_string()
    }
}

impl From<String> for FileProcessingError {
    fn from(s: String) -> Self {
        FileProcessingError::EmbeddingFailed(s)
    }
}
