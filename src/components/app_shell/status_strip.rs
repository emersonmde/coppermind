use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::components::hero::{use_worker_state, WorkerStatus};

use crate::components::{use_search_engine_status, SearchEngineStatus};

/// Status strip showing worker and index status pills
#[component]
pub fn StatusStrip() -> Element {
    let engine_status = use_search_engine_status();

    #[cfg(target_arch = "wasm32")]
    let worker_state = use_worker_state();

    // Worker status pill (web only)
    #[cfg(target_arch = "wasm32")]
    let worker_pill = {
        let status = worker_state.read().clone();
        match status {
            WorkerStatus::Pending => rsx! {
                span { class: "cm-status-pill cm-status-pill--warn",
                    span { class: "cm-status-dot cm-status-dot--warn" }
                    "Web worker starting…"
                }
            },
            WorkerStatus::Ready(_) => rsx! {
                span { class: "cm-status-pill cm-status-pill--ok",
                    span { class: "cm-status-dot cm-status-dot--ok" }
                    "Web worker ready"
                }
            },
            WorkerStatus::Failed(err) => rsx! {
                span { class: "cm-status-pill cm-status-pill--warn",
                    span { class: "cm-status-dot cm-status-dot--warn" }
                    "Web worker error: {err}"
                }
            },
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    let worker_pill = rsx! { Fragment {} };

    // Index status pill
    let index_pill = {
        let status = engine_status.read().clone();
        match status {
            SearchEngineStatus::Pending => rsx! {
                span { class: "cm-status-pill cm-status-pill--warn",
                    span { class: "cm-status-dot cm-status-dot--warn" }
                    "Index initializing…"
                }
            },
            SearchEngineStatus::Ready { doc_count } => {
                if doc_count == 0 {
                    rsx! {
                        span { class: "cm-status-pill cm-status-pill--ok",
                            span { class: "cm-status-dot cm-status-dot--ok" }
                            "Index: ready (empty)"
                        }
                    }
                } else {
                    // Calculate approximate token count (assuming ~400 tokens per doc as placeholder)
                    let approx_tokens = doc_count * 400;
                    rsx! {
                        span { class: "cm-status-pill cm-status-pill--ok",
                            span { class: "cm-status-dot cm-status-dot--ok" }
                            "Index: {doc_count} documents • ~{approx_tokens} tokens"
                        }
                    }
                }
            }
            SearchEngineStatus::Failed(err) => rsx! {
                span { class: "cm-status-pill cm-status-pill--warn",
                    span { class: "cm-status-dot cm-status-dot--warn" }
                    "Index error: {err}"
                }
            },
        }
    };

    rsx! {
        section { class: "cm-status-strip",
            {worker_pill}
            {index_pill}
            // TODO: Add indexing progress pill when files are being processed
            // Example: "Indexing 4 files… View details"
        }
    }
}
