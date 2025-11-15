use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::workers::EmbeddingWorkerClient;

#[cfg(target_arch = "wasm32")]
use dioxus::logger::tracing::{error, info};

use super::{use_search_engine_status, SearchEngineStatus};

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
pub enum WorkerStatus {
    Pending,
    Ready(EmbeddingWorkerClient),
    Failed(String),
}

// Export worker state for use by other components
#[cfg(target_arch = "wasm32")]
pub fn use_worker_state() -> Signal<WorkerStatus> {
    use_context::<Signal<WorkerStatus>>()
}

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
                Ok(client) => worker_state.set(WorkerStatus::Ready(client)),
                Err(err) => {
                    error!("❌ Embedding worker failed to start: {}", err);
                    worker_state.set(WorkerStatus::Failed(err));
                }
            }
        }
    });

    state
}

#[component]
pub fn Hero() -> Element {
    let search_engine_status = use_search_engine_status();

    #[cfg(target_arch = "wasm32")]
    let worker_status_view: Element = {
        let worker_state = use_worker_state();
        let status_label = match worker_state.read().clone() {
            WorkerStatus::Pending => "Web Worker: starting…".to_string(),
            WorkerStatus::Ready(_) => "Web Worker: ready ✅".to_string(),
            WorkerStatus::Failed(err) => format!("Web Worker error: {}", err),
        };

        rsx! { p { class: "worker-status", "{status_label}" } }
    };

    #[cfg(not(target_arch = "wasm32"))]
    let worker_status_view: Element = rsx! { Fragment {} };

    let search_status_view: Element = {
        let status_label = match search_engine_status.read().clone() {
            SearchEngineStatus::Pending => "Search Index: initializing…".to_string(),
            SearchEngineStatus::Ready { doc_count } => {
                if doc_count == 0 {
                    "Search Index: ready (empty) ✅".to_string()
                } else {
                    format!("Search Index: ready ({} documents) ✅", doc_count)
                }
            }
            SearchEngineStatus::Failed(err) => format!("Search Index error: {}", err),
        };

        rsx! { p { class: "worker-status", "{status_label}" } }
    };

    rsx! {
        div { class: "hero-section",
            h1 { class: "hero-title", "Coppermind" }
            p { class: "hero-subtitle",
                "Local-first semantic search engine powered by Rust + WASM"
            }
            {worker_status_view}
            {search_status_view}
        }
    }
}
