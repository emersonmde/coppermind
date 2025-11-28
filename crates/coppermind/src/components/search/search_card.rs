use dioxus::prelude::*;

use crate::components::{
    use_model_status, use_search_engine_status, ModelStatus, SearchEngineStatus,
};

/// Search card with input, hints, and search button
#[component]
pub fn SearchCard(
    search_query: Signal<String>,
    on_search: EventHandler<String>,
    searching: ReadSignal<bool>,
) -> Element {
    let engine_status = use_search_engine_status();
    let model_status = use_model_status();

    // Get doc count and token count from engine status
    let (doc_count, token_count) = match engine_status.read().clone() {
        SearchEngineStatus::Ready {
            doc_count,
            total_tokens,
        } => (doc_count, total_tokens),
        _ => (0, 0),
    };

    // Check if search is ready (both model and index must be ready)
    let engine_status_clone = engine_status.read().clone();
    let model_status_clone = model_status.read().clone();
    let index_loading = matches!(
        engine_status_clone,
        SearchEngineStatus::Pending | SearchEngineStatus::Loading
    );
    let model_loading = matches!(model_status_clone, ModelStatus::Cold | ModelStatus::Loading);
    let search_ready = !index_loading && !model_loading;

    // Determine placeholder text based on loading state
    let placeholder = if index_loading && model_loading {
        "Initializing index and model…"
    } else if index_loading {
        "Loading index…"
    } else if model_loading {
        "Loading model…"
    } else {
        "Search your knowledge base…"
    };

    let handle_keypress = move |evt: KeyboardEvent| {
        if evt.key() == Key::Enter {
            let query = search_query.read().clone();
            if !query.trim().is_empty() {
                on_search.call(query);
            }
        }
    };

    rsx! {
        section { class: "cm-search-card",
            div { class: "cm-search-card-top",
                div { class: "cm-search-index-selector cm-search-index-selector--disabled",
                    "All indexes"
                    span { class: "cm-caret", "▾" }
                }
            }
            div { class: "cm-search-input-row",
                input {
                    class: if !search_ready { "cm-search-input cm-search-input--loading" } else { "cm-search-input" },
                    r#type: "text",
                    placeholder: "{placeholder}",
                    value: "{search_query}",
                    disabled: searching() || !search_ready,
                    oninput: move |evt| search_query.set(evt.value()),
                    onkeypress: handle_keypress,
                }
                button {
                    class: "cm-btn cm-btn--primary",
                    disabled: searching() || !search_ready,
                    onclick: move |_| {
                        let query = search_query.read().clone();
                        if !query.trim().is_empty() {
                            on_search.call(query);
                        }
                    },
                    if searching() {
                        "Searching…"
                    } else if !search_ready {
                        "Loading…"
                    } else {
                        "Search"
                    }
                }
            }
            div { class: "cm-search-hints",
                span { "Local-first • Hybrid semantic + keyword search" }
                span { "{doc_count} documents indexed • {token_count} tokens" }
            }
        }
    }
}
