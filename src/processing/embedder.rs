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
