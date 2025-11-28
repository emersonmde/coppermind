use dioxus::prelude::*;

use crate::components::{use_search_engine_status, SearchEngineStatus};

/// Search card with input, hints, and search button
#[component]
pub fn SearchCard(
    search_query: Signal<String>,
    on_search: EventHandler<String>,
    searching: ReadSignal<bool>,
) -> Element {
    let engine_status = use_search_engine_status();

    // Get doc count and token count from engine status
    let (doc_count, token_count) = match engine_status.read().clone() {
        SearchEngineStatus::Ready {
            doc_count,
            total_tokens,
        } => (doc_count, total_tokens),
        _ => (0, 0),
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
                    class: "cm-search-input",
                    r#type: "text",
                    placeholder: "Search your knowledge base…",
                    value: "{search_query}",
                    disabled: searching(),
                    oninput: move |evt| search_query.set(evt.value()),
                    onkeypress: handle_keypress,
                }
                button {
                    class: "cm-btn cm-btn--primary",
                    disabled: searching(),
                    onclick: move |_| {
                        let query = search_query.read().clone();
                        if !query.trim().is_empty() {
                            on_search.call(query);
                        }
                    },
                    if searching() {
                        "Searching…"
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
