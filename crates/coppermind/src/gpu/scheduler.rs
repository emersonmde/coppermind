//! GPU scheduler trait definition.
//!
//! This module defines the `GpuScheduler` trait that abstracts GPU execution
//! strategy. The current implementation uses a serial scheduler (single thread
//! owns GPU) to work around Candle's Metal threading bug. When Candle is fixed,
//! we can swap in a parallel scheduler without changing calling code.

use super::error::GpuError;
use super::types::{
    BatchEmbedRequest, EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, ModelId,
    ModelLoadConfig, SchedulerStats,
};
use async_trait::async_trait;

/// GPU scheduler trait - abstracts execution strategy.
///
/// This trait defines the interface for submitting GPU work. It supports:
/// - Single embedding requests (for search, interactive)
/// - Batch embedding requests (for background processing)
/// - LLM generation (future)
/// - Model loading/unloading
///
/// # Implementations
///
/// - `SerialScheduler`: Single thread owns GPU (current - Candle Metal workaround)
/// - `ParallelScheduler`: Thread pool (future - when Candle fixed)
///
/// # Priority Handling
///
/// The scheduler processes requests in priority order:
/// - P0 (Immediate): Search queries, must complete quickly
/// - P1 (Interactive): User-triggered actions
/// - P2 (Background): Bulk processing, can be delayed
///
/// Within the same priority level, requests are processed FIFO.
///
/// # Example
///
/// ```ignore
/// use coppermind::gpu::{GpuScheduler, EmbedRequest, Priority};
///
/// async fn search_query(scheduler: &impl GpuScheduler, tokens: Vec<u32>) -> Result<Vec<f32>, GpuError> {
///     let response = scheduler.embed(EmbedRequest::immediate(tokens)).await?;
///     Ok(response.embedding)
/// }
/// ```
#[async_trait]
pub trait GpuScheduler: Send + Sync {
    /// Submit a single embedding request.
    ///
    /// # Arguments
    ///
    /// * `request` - Embedding request with tokens, model ID, and priority
    ///
    /// # Returns
    ///
    /// Embedding response with the generated embedding vector.
    ///
    /// # Errors
    ///
    /// Returns `GpuError` if:
    /// - Model is not loaded
    /// - Inference fails
    /// - Scheduler is shutting down
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, GpuError>;

    /// Submit a batch embedding request.
    ///
    /// More efficient than individual requests for background work. The GPU
    /// processes the entire batch in a single forward pass, reducing overhead.
    ///
    /// # Arguments
    ///
    /// * `request` - Batch embedding request with multiple token sequences
    ///
    /// # Returns
    ///
    /// Vector of embedding responses, one per input sequence, in order.
    ///
    /// # Errors
    ///
    /// Returns `GpuError` if:
    /// - Model is not loaded
    /// - Batch is empty
    /// - Inference fails
    /// - Scheduler is shutting down
    async fn embed_batch(&self, request: BatchEmbedRequest)
        -> Result<Vec<EmbedResponse>, GpuError>;

    /// Generate text with an LLM (future).
    ///
    /// Placeholder for future LLM integration.
    ///
    /// # Arguments
    ///
    /// * `request` - Generation request with prompt, max tokens, and priority
    ///
    /// # Returns
    ///
    /// Generation response with generated tokens and text.
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, GpuError>;

    /// Load a model into the registry.
    ///
    /// Downloads/loads model weights and initializes for inference.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Identifier for the model
    /// * `config` - Model loading configuration (weights, tokenizer, vocab size)
    ///
    /// # Errors
    ///
    /// Returns `GpuError` if:
    /// - Model already loaded
    /// - Weight loading fails
    /// - Model initialization fails
    async fn load_model(&self, model_id: ModelId, config: ModelLoadConfig) -> Result<(), GpuError>;

    /// Unload a model from the registry.
    ///
    /// Frees GPU memory used by the model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Identifier of model to unload
    ///
    /// # Errors
    ///
    /// Returns `GpuError` if model is not loaded.
    async fn unload_model(&self, model_id: &ModelId) -> Result<(), GpuError>;

    /// Check if a specific model is loaded.
    fn is_model_loaded(&self, model_id: &ModelId) -> bool;

    /// Check if scheduler is ready (at least one model loaded).
    fn is_ready(&self) -> bool;

    /// Get scheduler statistics.
    ///
    /// Returns current queue depth, requests completed, and models loaded.
    fn stats(&self) -> SchedulerStats;
}
