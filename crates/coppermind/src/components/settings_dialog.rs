//! Settings dialog component for managing application preferences and storage.

use crate::components::{
    use_batches, use_search_engine, use_search_engine_status, SearchEngineStatus,
};
use coppermind_core::metrics::global_metrics;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;

/// Settings dialog with clear storage functionality.
///
/// Displayed as a modal overlay when the user clicks the settings button.
#[component]
pub fn SettingsDialog(on_close: EventHandler<()>) -> Element {
    let search_engine = use_search_engine();
    let mut engine_status = use_search_engine_status();
    let mut batches = use_batches();
    let mut confirm_clear = use_signal(|| false);
    let mut clearing = use_signal(|| false);

    // Handle clear storage
    let handle_clear = move |_| {
        if !confirm_clear() {
            // First click: show confirmation
            confirm_clear.set(true);
        } else {
            // Second click: perform clear
            clearing.set(true);

            // Clone the engine lock before entering async block to avoid holding read across await
            let engine_opt = search_engine.read().clone();
            spawn(async move {
                if let Some(engine_lock) = engine_opt {
                    let mut engine = engine_lock.lock().await;
                    match engine.clear_all().await {
                        Ok(()) => {
                            info!("Storage cleared successfully");

                            // Reset engine status
                            engine_status.set(SearchEngineStatus::Ready {
                                doc_count: 0,
                                total_tokens: 0,
                            });

                            // Clear batches list
                            batches.set(Vec::new());

                            // Clear performance metrics
                            global_metrics().clear();

                            info!("All metrics and batches cleared");
                        }
                        Err(e) => {
                            error!("Failed to clear storage: {:?}", e);
                        }
                    }
                }
                clearing.set(false);
                confirm_clear.set(false);
            });
        }
    };

    // Cancel confirmation
    let handle_cancel_confirm = move |_| {
        confirm_clear.set(false);
    };

    // Get current doc count for display
    let doc_count = match engine_status.read().clone() {
        SearchEngineStatus::Ready { doc_count, .. } => doc_count,
        _ => 0,
    };

    rsx! {
        // Modal backdrop
        div {
            class: "cm-modal-backdrop",
            onclick: move |_| on_close.call(()),

            // Modal content (stop propagation to prevent close on content click)
            div {
                class: "cm-modal",
                onclick: move |e| e.stop_propagation(),

                // Header
                div { class: "cm-modal-header",
                    h2 { class: "cm-modal-title", "Settings" }
                    button {
                        class: "cm-modal-close",
                        onclick: move |_| on_close.call(()),
                        "aria-label": "Close settings",
                        "\u{2715}" // Unicode X
                    }
                }

                // Content
                div { class: "cm-modal-content",
                    // Storage section
                    section { class: "cm-settings-section",
                        h3 { class: "cm-settings-section-title", "Storage" }

                        // Storage info
                        p { class: "cm-settings-info",
                            "Index contains {doc_count} documents."
                        }

                        // Clear storage button
                        div { class: "cm-settings-action",
                            if confirm_clear() {
                                // Confirmation state
                                p { class: "cm-settings-warning",
                                    "This will permanently delete all indexed documents. Are you sure?"
                                }
                                div { class: "cm-settings-buttons",
                                    button {
                                        class: "cm-btn cm-btn--secondary",
                                        onclick: handle_cancel_confirm,
                                        disabled: clearing(),
                                        "Cancel"
                                    }
                                    button {
                                        class: "cm-btn cm-btn--danger",
                                        onclick: handle_clear,
                                        disabled: clearing(),
                                        if clearing() {
                                            "Clearing..."
                                        } else {
                                            "Yes, Clear All"
                                        }
                                    }
                                }
                            } else {
                                // Normal state
                                button {
                                    class: "cm-btn cm-btn--danger",
                                    onclick: handle_clear,
                                    disabled: doc_count == 0,
                                    "Clear Storage"
                                }
                                if doc_count == 0 {
                                    p { class: "cm-settings-hint",
                                        "No documents to clear."
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
