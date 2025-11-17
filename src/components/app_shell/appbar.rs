use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::components::worker::{use_worker_state, WorkerStatus};

use crate::components::{
    use_model_status, use_search_engine_status, ModelStatus, SearchEngineStatus,
};

/// View selection enum for navigation
#[derive(Clone, Copy, PartialEq)]
pub enum View {
    Search,
    Index,
}

/// Global app bar with logo, navigation, status pills, and metrics toggle
#[component]
pub fn AppBar(
    current_view: ReadSignal<View>,
    on_view_change: EventHandler<View>,
    on_metrics_toggle: EventHandler<()>,
    metrics_collapsed: ReadSignal<bool>,
) -> Element {
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

    // Model status pill (shows embedding model state)
    let model_pill = {
        let model_status_signal = use_model_status();
        let status = model_status_signal.read().clone();
        match status {
            ModelStatus::Cold => rsx! {
                span { class: "cm-status-pill cm-status-pill--muted",
                    span { class: "cm-status-dot cm-status-dot--muted" }
                    "Model: cold"
                }
            },
            ModelStatus::Loading => rsx! {
                span { class: "cm-status-pill cm-status-pill--warn",
                    span { class: "cm-status-dot cm-status-dot--warn" }
                    "Model: loading…"
                }
            },
            ModelStatus::Ready => rsx! {
                span { class: "cm-status-pill cm-status-pill--ok",
                    span { class: "cm-status-dot cm-status-dot--ok" }
                    "Model: ready"
                }
            },
            ModelStatus::Failed(err) => rsx! {
                span { class: "cm-status-pill cm-status-pill--error",
                    span { class: "cm-status-dot cm-status-dot--error" }
                    "Model: {err}"
                }
            },
        }
    };

    // Index status pill (clickable to toggle metrics)
    let index_pill = {
        let status = engine_status.read().clone();
        let chevron = if metrics_collapsed() { "▼" } else { "▲" };

        match status {
            SearchEngineStatus::Pending => rsx! {
                button {
                    class: "cm-status-pill cm-status-pill--warn cm-status-pill--clickable",
                    onclick: move |_| on_metrics_toggle.call(()),
                    "aria-label": "Toggle metrics panel",
                    span { class: "cm-status-dot cm-status-dot--warn" }
                    "Index initializing…"
                    span { class: "cm-status-chevron", "{chevron}" }
                }
            },
            SearchEngineStatus::Ready { doc_count } => {
                if doc_count == 0 {
                    rsx! {
                        button {
                            class: "cm-status-pill cm-status-pill--ok cm-status-pill--clickable",
                            onclick: move |_| on_metrics_toggle.call(()),
                            "aria-label": "Toggle metrics panel",
                            span { class: "cm-status-dot cm-status-dot--ok" }
                            "Index: ready (empty)"
                            span { class: "cm-status-chevron", "{chevron}" }
                        }
                    }
                } else {
                    rsx! {
                        button {
                            class: "cm-status-pill cm-status-pill--ok cm-status-pill--clickable",
                            onclick: move |_| on_metrics_toggle.call(()),
                            "aria-label": "Toggle metrics panel",
                            span { class: "cm-status-dot cm-status-dot--ok" }
                            "Index: {doc_count} docs"
                            span { class: "cm-status-chevron", "{chevron}" }
                        }
                    }
                }
            }
            SearchEngineStatus::Failed(err) => rsx! {
                button {
                    class: "cm-status-pill cm-status-pill--warn cm-status-pill--clickable",
                    onclick: move |_| on_metrics_toggle.call(()),
                    "aria-label": "Toggle metrics panel",
                    span { class: "cm-status-dot cm-status-dot--warn" }
                    "Index error: {err}"
                    span { class: "cm-status-chevron", "{chevron}" }
                }
            },
        }
    };

    rsx! {
        header { class: "cm-appbar",
            div { class: "cm-appbar-left",
                div { class: "cm-logo",
                    span { class: "cm-logo-word", "Copper" }
                    span { class: "cm-logo-word cm-logo-word--accent", "mind" }
                }
            }
            nav { class: "cm-appbar-center",
                button {
                    class: if current_view() == View::Search {
                        "cm-nav-link cm-nav-link--active"
                    } else {
                        "cm-nav-link"
                    },
                    onclick: move |_| on_view_change.call(View::Search),
                    "Search"
                }
                button {
                    class: if current_view() == View::Index {
                        "cm-nav-link cm-nav-link--active"
                    } else {
                        "cm-nav-link"
                    },
                    onclick: move |_| on_view_change.call(View::Index),
                    "Index"
                }
            }
            div { class: "cm-appbar-right",
                {worker_pill}
                {model_pill}
                {index_pill}
            }
        }
    }
}
