//! Web Worker state management for embedding operations (web platform only)

#[cfg(target_arch = "wasm32")]
use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::workers::{set_global_worker, EmbeddingWorkerClient};

#[cfg(target_arch = "wasm32")]
use dioxus::logger::tracing::{error, info};

/// Status of the embedding web worker (web platform only)
#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
pub enum WorkerStatus {
    /// Worker is being initialized
    Pending,
    /// Worker is ready and can process embeddings
    Ready(EmbeddingWorkerClient),
    /// Worker failed to initialize
    Failed(String),
}

/// Access the worker state from context (web platform only)
#[cfg(target_arch = "wasm32")]
pub fn use_worker_state() -> Signal<WorkerStatus> {
    use_context::<Signal<WorkerStatus>>()
}

/// Initialize and provide worker state to the component tree (web platform only)
#[cfg(target_arch = "wasm32")]
pub fn provide_worker_state() -> Signal<WorkerStatus> {
    let state = use_signal(|| WorkerStatus::Pending);
    use_context_provider(|| state);

    // Initialize worker in effect
    let worker_signal = state;
    use_effect(move || {
        let mut worker_state = worker_signal;
        if matches!(*worker_state.read(), WorkerStatus::Pending) {
            info!("🔧 Initializing embedding worker…");
            match EmbeddingWorkerClient::new() {
                Ok(client) => {
                    // Also set global worker for use outside component context
                    set_global_worker(client.clone());
                    worker_state.set(WorkerStatus::Ready(client));
                }
                Err(err) => {
                    error!("❌ Embedding worker failed to start: {}", err);
                    worker_state.set(WorkerStatus::Failed(err));
                }
            }
        }
    });

    state
}
