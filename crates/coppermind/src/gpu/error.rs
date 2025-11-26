//! Error types for the GPU scheduler.

use super::types::ModelId;
use thiserror::Error;

/// Errors that can occur in the GPU scheduler.
#[derive(Debug, Clone, Error)]
pub enum GpuError {
    /// Scheduler initialization failed
    #[error("Scheduler initialization failed: {0}")]
    SchedulerInit(String),

    /// Failed to spawn worker thread
    #[error("Failed to spawn GPU worker thread: {0}")]
    ThreadSpawnFailed(String),

    /// Request channel disconnected (scheduler shutting down)
    #[error("GPU scheduler channel disconnected")]
    ChannelDisconnected,

    /// Response channel failed
    #[error("Response channel failed: {0}")]
    ResponseFailed(String),

    /// Model not loaded in registry
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(ModelId),

    /// Model already loaded
    #[error("Model already loaded: {0}")]
    ModelAlreadyLoaded(ModelId),

    /// Model loading failed
    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),

    /// Embedding inference failed
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(String),

    /// LLM generation failed (future)
    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    /// Invalid request (e.g., empty batch)
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Scheduler not initialized
    #[error("GPU scheduler not initialized")]
    NotInitialized,

    /// Scheduler already initialized
    #[error("GPU scheduler already initialized")]
    AlreadyInitialized,
}

impl From<GpuError> for String {
    fn from(err: GpuError) -> String {
        err.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GpuError::ModelNotLoaded(ModelId::JinaBert);
        assert_eq!(err.to_string(), "Model not loaded: jina-bert");

        let err = GpuError::EmbeddingFailed("inference error".to_string());
        assert_eq!(err.to_string(), "Embedding failed: inference error");
    }
}
