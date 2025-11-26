//! GPU scheduler for Metal thread-safe model inference.
//!
//! This module provides a GPU scheduler that serializes access to GPU resources
//! to work around Candle's Metal threading bug. The scheduler supports:
//!
//! - **Priority scheduling**: Search queries (P0) processed before background work (P2)
//! - **Multi-model support**: Model registry for multiple embedding models + future LLM
//! - **Batch processing**: Efficient batched inference for background work
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                          Application Code                           │
//! │  compute_embedding(), search(), embed_chunks()                      │
//! └────────────────────────────────┬────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                         GpuScheduler Trait                          │
//! │  embed(), embed_batch(), load_model(), is_ready()                   │
//! └────────────────────────────────┬────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        SerialScheduler                              │
//! │  Single thread owns Metal device, priority queue for requests       │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Background
//!
//! Candle's Metal backend has a threading bug where concurrent access from
//! multiple threads causes crashes:
//!
//! - **0.8.x/0.9.1**: "A command encoder is already encoding to this command buffer"
//! - **0.9.2-alpha.1**: SIGSEGV in `allocate_zeros`
//!
//! The root cause is that Metal's `MTLCommandBuffer` is not thread-safe.
//! This scheduler works around the issue by having a single dedicated thread
//! own the Metal device and process all requests serially.
//!
//! See `docs/adrs/006-gpu-scheduler.md` for full design documentation.
//!
//! # Platform Support
//!
//! - **Desktop (macOS)**: Uses `SerialScheduler` with Metal acceleration
//! - **Desktop (Linux/Windows)**: Uses `SerialScheduler` with CUDA/CPU
//! - **Web (WASM)**: Direct execution (no threading issues on CPU)
//!
//! # Example
//!
//! ```ignore
//! use coppermind::gpu::{get_scheduler, EmbedRequest, Priority};
//!
//! // Get the global scheduler
//! let scheduler = get_scheduler()?;
//!
//! // Submit a search query (highest priority)
//! let response = scheduler.embed(EmbedRequest::immediate(tokens)).await?;
//! println!("Embedding: {:?}", response.embedding);
//!
//! // Submit background batch
//! let responses = scheduler.embed_batch(BatchEmbedRequest::new(batches)).await?;
//! ```

pub mod error;
pub mod scheduler;
#[cfg(not(target_arch = "wasm32"))]
pub mod serial_scheduler;
pub mod types;

// Re-export key types
pub use error::GpuError;
pub use scheduler::GpuScheduler;
#[cfg(not(target_arch = "wasm32"))]
pub use serial_scheduler::SerialScheduler;
pub use types::{
    BatchEmbedRequest, EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, ModelId,
    ModelLoadConfig, Priority, SchedulerStats,
};

use once_cell::sync::OnceCell;
use std::sync::Arc;

/// Global GPU scheduler instance (desktop only).
#[cfg(not(target_arch = "wasm32"))]
static SCHEDULER: OnceCell<Arc<dyn GpuScheduler>> = OnceCell::new();

/// Initialize the global GPU scheduler.
///
/// Must be called once at application startup. Creates a `SerialScheduler`
/// with a dedicated worker thread that owns the GPU device.
///
/// # Errors
///
/// Returns `GpuError::AlreadyInitialized` if called more than once.
/// Returns `GpuError::SchedulerInit` if scheduler creation fails.
///
/// # Example
///
/// ```ignore
/// // In main.rs or app initialization
/// gpu::init_scheduler()?;
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub fn init_scheduler() -> Result<(), GpuError> {
    let scheduler = SerialScheduler::new()?;
    SCHEDULER
        .set(Arc::new(scheduler) as Arc<dyn GpuScheduler>)
        .map_err(|_| GpuError::AlreadyInitialized)
}

/// Get the global GPU scheduler.
///
/// # Errors
///
/// Returns `GpuError::NotInitialized` if `init_scheduler()` hasn't been called.
///
/// # Example
///
/// ```ignore
/// let scheduler = gpu::get_scheduler()?;
/// let response = scheduler.embed(request).await?;
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub fn get_scheduler() -> Result<Arc<dyn GpuScheduler>, GpuError> {
    SCHEDULER.get().cloned().ok_or(GpuError::NotInitialized)
}

/// Check if the GPU scheduler is initialized.
#[cfg(not(target_arch = "wasm32"))]
pub fn is_scheduler_initialized() -> bool {
    SCHEDULER.get().is_some()
}
