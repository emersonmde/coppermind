//! Platform-specific embedding abstraction.
//!
//! This module provides a unified interface for computing embeddings across
//! different platforms (web with workers, desktop with direct execution).

use crate::embedding::EmbeddingComputation;
use crate::error::EmbeddingError;

/// Platform-agnostic embedder trait.
///
/// Implementations handle platform-specific embedding strategies:
/// - **Web**: Delegates to web worker (`EmbeddingWorkerClient`)
/// - **Desktop**: Direct embedding via `compute_embedding()`
#[async_trait::async_trait(?Send)]
pub trait PlatformEmbedder {
    /// Compute embedding for a text chunk.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to embed
    ///
    /// # Returns
    ///
    /// Embedding computation result with token count and embedding vector.
    async fn embed(&self, text: &str) -> Result<EmbeddingComputation, EmbeddingError>;

    /// Check if embedder is ready to process requests.
    ///
    /// Returns `true` if embedder is initialized and ready.
    fn is_ready(&self) -> bool;

    /// Get embedder status message (e.g., "Ready", "Initializing", "Failed: ...").
    fn status_message(&self) -> String;
}

// ============================================================================
// Web Platform: Worker-based Embedder
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub mod web {
    use super::*;
    use crate::components::worker::{use_worker_state, WorkerStatus};
    use crate::workers::EmbeddingWorkerClient;
    use dioxus::prelude::*;

    /// Web embedder using worker client.
    ///
    /// Delegates embedding to a web worker to prevent UI freezing.
    pub struct WebEmbedder {
        worker: Option<EmbeddingWorkerClient>,
        status: String,
    }

    impl WebEmbedder {
        /// Create embedder from worker state signal.
        ///
        /// Reads the current worker status and creates an embedder instance.
        pub fn from_worker_state(worker_state: Signal<WorkerStatus>) -> Self {
            let state = worker_state.read().clone();
            match state {
                WorkerStatus::Ready(client) => Self {
                    worker: Some(client),
                    status: "Ready".to_string(),
                },
                WorkerStatus::Pending => Self {
                    worker: None,
                    status: "Initializing worker...".to_string(),
                },
                WorkerStatus::Failed(err) => Self {
                    worker: None,
                    status: format!("Worker failed: {}", err),
                },
            }
        }

        /// Get embedder from current Dioxus context.
        ///
        /// Reads worker state from context and creates embedder.
        pub fn from_context() -> Self {
            let worker_state = use_worker_state();
            Self::from_worker_state(worker_state)
        }
    }

    #[async_trait::async_trait(?Send)]
    impl PlatformEmbedder for WebEmbedder {
        async fn embed(&self, text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
            match &self.worker {
                Some(client) => client
                    .embed(text.to_string())
                    .await
                    .map_err(EmbeddingError::InferenceFailed),
                None => Err(EmbeddingError::ModelUnavailable(self.status.clone())),
            }
        }

        fn is_ready(&self) -> bool {
            self.worker.is_some()
        }

        fn status_message(&self) -> String {
            self.status.clone()
        }
    }
}

// ============================================================================
// Desktop Platform: Direct Embedder
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub mod desktop {
    use super::*;

    /// Desktop embedder using direct computation.
    ///
    /// Calls `compute_embedding` directly, which internally uses
    /// `tokio::spawn_blocking` via `platform::run_blocking`.
    pub struct DesktopEmbedder;

    impl DesktopEmbedder {
        pub fn new() -> Self {
            Self
        }
    }

    impl Default for DesktopEmbedder {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait::async_trait(?Send)]
    impl PlatformEmbedder for DesktopEmbedder {
        async fn embed(&self, text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
            crate::embedding::compute_embedding(text).await
        }

        fn is_ready(&self) -> bool {
            true // Desktop embedder is always ready (model loads on first use)
        }

        fn status_message(&self) -> String {
            "Ready".to_string()
        }
    }
}

// ============================================================================
// Platform-specific re-exports
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub use web::WebEmbedder;

#[cfg(not(target_arch = "wasm32"))]
pub use desktop::DesktopEmbedder;

// ============================================================================
// Convenience Hook
// ============================================================================

/// Returns a platform-appropriate embedder.
///
/// # Platform Behavior
///
/// - **Web**: Returns `WebEmbedder` that delegates to worker
/// - **Desktop**: Returns `DesktopEmbedder` that calls `compute_embedding()` directly
///
/// # Note
///
/// On WASM, this uses `use_worker_state()` hook and must be called from a
/// component context (not inside a coroutine).
///
/// # Example
///
/// ```ignore
/// let embedder = get_platform_embedder();
/// if embedder.is_ready() {
///     let result = embedder.embed("Hello, world!").await?;
///     println!("Embedding dimension: {}", result.embedding.len());
/// }
/// ```
#[cfg(target_arch = "wasm32")]
pub fn get_platform_embedder() -> WebEmbedder {
    WebEmbedder::from_context()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn get_platform_embedder() -> DesktopEmbedder {
    DesktopEmbedder::new()
}

// ============================================================================
// Async Embedding Helper
// ============================================================================

/// Computes an embedding for the given text using the platform-appropriate method.
///
/// # Platform Behavior
///
/// - **Web**: Uses the worker via the provided `worker_state` signal
/// - **Desktop**: Uses direct `compute_embedding()` call
///
/// # Arguments
///
/// * `text` - The text to embed
/// * `worker_state` (WASM only) - Signal containing the worker status
///
/// # Returns
///
/// The embedding computation result, or an error message.
///
/// # Example
///
/// ```ignore
/// // In a coroutine where you have access to worker_state signal:
/// #[cfg(target_arch = "wasm32")]
/// let result = embed_text(&query, worker_state).await;
///
/// #[cfg(not(target_arch = "wasm32"))]
/// let result = embed_text(&query).await;
/// ```
#[cfg(target_arch = "wasm32")]
pub async fn embed_text(
    text: &str,
    worker_state: dioxus::prelude::Signal<crate::components::worker::WorkerStatus>,
) -> Result<crate::embedding::EmbeddingComputation, String> {
    let embedder = WebEmbedder::from_worker_state(worker_state);
    if !embedder.is_ready() {
        return Err(embedder.status_message());
    }
    embedder
        .embed(text)
        .await
        .map_err(|e| format!("Embedding failed: {}", e))
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn embed_text(text: &str) -> Result<crate::embedding::EmbeddingComputation, String> {
    DesktopEmbedder::new()
        .embed(text)
        .await
        .map_err(|e| format!("Embedding failed: {}", e))
}
