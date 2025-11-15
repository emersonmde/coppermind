use crate::search::types::SearchResult;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;

#[cfg(target_arch = "wasm32")]
use super::hero::{use_worker_state, WorkerStatus};

// Import the search engine context hook (will be defined in mod.rs)
use super::SearchEngineStatus;
use super::{use_search_engine, use_search_engine_status};

// Messages for search coroutine
enum SearchMessage {
    RunSearch(String), // query text
}

#[component]
pub fn Search() -> Element {
    let mut search_query = use_signal(String::new);
    let mut search_results = use_signal(Vec::<SearchResult>::new);
    let mut searching = use_signal(|| false);
    let search_status = use_signal(String::new);

    let search_engine = use_search_engine();
    let engine_status = use_search_engine_status();

    #[cfg(target_arch = "wasm32")]
    let worker_state = use_worker_state();

    // Search coroutine - runs in background
    let search_task = use_coroutine({
        let mut results = search_results;
        let mut status = search_status;
        let mut is_searching = searching;
        let engine = search_engine;

        move |mut rx: UnboundedReceiver<SearchMessage>| async move {
            while let Some(msg) = rx.next().await {
                match msg {
                    SearchMessage::RunSearch(query) => {
                        if query.trim().is_empty() {
                            status.set("Please enter a search query.".into());
                            is_searching.set(false);
                            continue;
                        }

                        info!("🔍 Searching for: '{}'", query);
                        status.set(format!("Embedding query '{}'...", query));

                        // Embed the query
                        let query_embedding = {
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                // Desktop: Use direct embedding
                                match crate::embedding::compute_embedding(&query).await {
                                    Ok(computation) => Some(computation.embedding),
                                    Err(e) => {
                                        error!("❌ Failed to embed query: {}", e);
                                        status.set(format!("Failed to embed query: {}", e));
                                        is_searching.set(false);
                                        None
                                    }
                                }
                            }

                            #[cfg(target_arch = "wasm32")]
                            {
                                // Web: Use embedding worker
                                let worker_snapshot = worker_state.read().clone();
                                match worker_snapshot {
                                    WorkerStatus::Pending => {
                                        status.set(
                                            "Embedding worker is starting… please retry.".into(),
                                        );
                                        is_searching.set(false);
                                        None
                                    }
                                    WorkerStatus::Failed(err) => {
                                        status
                                            .set(format!("Embedding worker unavailable: {}", err));
                                        is_searching.set(false);
                                        None
                                    }
                                    WorkerStatus::Ready(client) => {
                                        match client.embed(query.clone()).await {
                                            Ok(computation) => Some(computation.embedding),
                                            Err(e) => {
                                                error!("❌ Failed to embed query: {}", e);
                                                status.set(format!("Failed to embed query: {}", e));
                                                is_searching.set(false);
                                                None
                                            }
                                        }
                                    }
                                }
                            }
                        };

                        if let Some(embedding) = query_embedding {
                            status.set("Running hybrid search...".into());

                            // Clone Arc from signal (cheap operation)
                            let engine_arc = engine.read().clone();
                            if let Some(engine) = engine_arc {
                                // Acquire lock and run search
                                let search_engine = engine.lock().await;
                                match search_engine.search(&embedding, &query, 10).await {
                                    Ok(search_results) => {
                                        info!(
                                            "✅ Search completed: {} results",
                                            search_results.len()
                                        );
                                        if search_results.is_empty() {
                                            status.set("No results found.".into());
                                        } else {
                                            status.set(format!(
                                                "Found {} results",
                                                search_results.len()
                                            ));
                                        }
                                        results.set(search_results);
                                    }
                                    Err(e) => {
                                        error!("❌ Search failed: {:?}", e);
                                        status.set(format!("Search failed: {}", e));
                                    }
                                }
                            } else {
                                status.set(
                                    "Search engine not initialized yet. Please wait...".into(),
                                );
                            }
                        }

                        is_searching.set(false);
                    }
                }
            }
        }
    });

    rsx! {
        div { class: "main-section",
            div { class: "section-header",
                div { class: "section-header-content",
                    h2 { "Search Your Knowledge Base" }
                    if let SearchEngineStatus::Ready { doc_count } = engine_status.read().clone() {
                        span { class: "doc-count-badge",
                            {format!("{} document{} indexed", doc_count, if doc_count != 1 { "s" } else { "" })}
                        }
                    }
                }
                p { class: "section-description",
                    "Use semantic and keyword search to find relevant documents. Results are ranked using hybrid search fusion."
                }
            }

            // Search input area
            div { class: "upload-zone search-zone",
                div { class: "upload-content",
                    // Search icon
                    svg {
                        class: "upload-icon",
                        xmlns: "http://www.w3.org/2000/svg",
                        width: "48",
                        height: "48",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "1.5",
                        circle { cx: "11", cy: "11", r: "8" }
                        path { d: "m21 21-4.35-4.35" }
                    }

                    if searching() {
                        div { class: "upload-processing",
                            div { class: "spinner" }
                            p { class: "upload-text-primary", "Searching..." }
                            p { class: "upload-text-secondary", "{search_status.read()}" }
                        }
                    } else {
                        form {
                            class: "search-form",
                            onsubmit: move |evt| {
                                evt.prevent_default();
                                if !searching() && !search_query.read().trim().is_empty() {
                                    searching.set(true);
                                    search_results.set(Vec::new());
                                    let query = search_query.read().clone();
                                    search_task.send(SearchMessage::RunSearch(query));
                                }
                            },

                            div { class: "search-input-wrapper",
                                input {
                                    class: "search-input",
                                    r#type: "text",
                                    placeholder: "What would you like to find?",
                                    value: "{search_query.read()}",
                                    disabled: searching(),
                                    oninput: move |evt| {
                                        search_query.set(evt.value().clone());
                                    }
                                }

                                button {
                                    class: "btn-primary search-submit-button",
                                    r#type: "submit",
                                    disabled: searching() || search_query.read().trim().is_empty(),
                                    svg {
                                        xmlns: "http://www.w3.org/2000/svg",
                                        width: "20",
                                        height: "20",
                                        view_box: "0 0 24 24",
                                        fill: "none",
                                        stroke: "currentColor",
                                        stroke_width: "2",
                                        circle { cx: "11", cy: "11", r: "8" }
                                        path { d: "m21 21-4.35-4.35" }
                                    }
                                    span { "Search" }
                                }
                            }

                            // Status message
                            if !search_status.read().is_empty() && !searching() {
                                p { class: "upload-text-secondary", "{search_status.read()}" }
                            }
                        }
                    }
                }
            }

            // Search results
            if !search_results.read().is_empty() {
                div { class: "search-results-container",
                    div { class: "search-results-header",
                        h3 { "Search Results" }
                        p { class: "results-count",
                            "{search_results.read().len()} result(s) for \"{search_query.read()}\""
                        }
                    }

                    div { class: "search-results-list",
                        for (idx, result) in search_results.read().iter().enumerate() {
                            div {
                                key: "{result.doc_id.as_u64()}",
                                class: "search-result-item",

                                div { class: "search-result-header",
                                    span { class: "search-result-rank", "#{idx + 1}" }
                                    span { class: "search-result-score-main",
                                        "RRF Score: {result.score:.4}"
                                    }
                                }

                                div { class: "search-result-text", "{result.text}" }

                                // Individual scores
                                div { class: "search-result-scores",
                                    if let Some(vector_score) = result.vector_score {
                                        div { class: "score-badge vector-score",
                                            span { class: "score-label", "Vector (Semantic)" }
                                            span { class: "score-value", "{vector_score:.4}" }
                                        }
                                    }
                                    if let Some(keyword_score) = result.keyword_score {
                                        div { class: "score-badge keyword-score",
                                            span { class: "score-label", "Keyword (BM25)" }
                                            span { class: "score-value", "{keyword_score:.4}" }
                                        }
                                    }
                                }

                                // Metadata
                                if let Some(filename) = &result.metadata.filename {
                                    div { class: "search-result-source", "Source: {filename}" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
