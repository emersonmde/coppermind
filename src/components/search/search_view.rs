use crate::search::types::{DocId, DocumentMetadata, SearchResult};
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;

#[cfg(target_arch = "wasm32")]
use crate::components::hero::{use_worker_state, WorkerStatus};

use crate::components::{use_search_engine, use_search_engine_status, SearchEngineStatus, View};

use super::{EmptyState, ResultCard, SearchCard};

// Messages for search coroutine
enum SearchMessage {
    RunSearch(String), // query text
}

/// Easter egg: hardcoded result that appears when searching for "coppermind"
/// This is NOT indexed and does NOT affect document counts - it's purely a UI-level feature.
fn create_easter_egg_result() -> SearchResult {
    SearchResult {
        doc_id: DocId::from_u64(u64::MAX), // Special ID to avoid conflicts
        score: 1.0,
        vector_score: Some(1.0),
        keyword_score: Some(1.0),
        text: "Coppermind is a local-first semantic search engine powered by Rust and WebAssembly. \
               It uses hybrid search combining vector similarity (JinaBERT embeddings) with keyword \
               matching (BM25) and fuses results using Reciprocal Rank Fusion. All processing happens \
               on your device with no cloud dependencies. The search engine uses HNSW for fast \
               approximate nearest neighbor search and provides detailed fusion scores for transparency."
            .to_string(),
        metadata: DocumentMetadata {
            filename: Some("welcome.md".to_string()),
            source: Some("Built-in example document".to_string()),
            ..Default::default()
        },
    }
}

/// Main search view with search card and results
#[component]
pub fn SearchView(on_navigate: EventHandler<View>) -> Element {
    let search_query = use_signal(String::new);
    let search_results = use_signal(Vec::<SearchResult>::new);
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
                            // Read lock is dropped immediately after clone
                            let engine_arc = { engine.read().clone() };
                            if let Some(engine) = engine_arc {
                                // Acquire lock and run search
                                let search_engine = engine.lock().await;
                                match search_engine.search(&embedding, &query, 10).await {
                                    Ok(mut search_results) => {
                                        // Easter egg: prepend hardcoded result when searching for "coppermind"
                                        if query.trim().to_lowercase() == "coppermind" {
                                            let easter_egg = create_easter_egg_result();
                                            search_results.insert(0, easter_egg);
                                            info!("🥚 Easter egg activated!");
                                        }

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

    let handle_search = move |query: String| {
        searching.set(true);
        search_task.send(SearchMessage::RunSearch(query));
    };

    let handle_show_source = move |result: SearchResult| {
        // TODO Phase 5: Implement source preview overlay
        info!("Show source for: {:?}", result.metadata.filename);
    };

    // Determine what to show
    let engine_ready = matches!(engine_status.read().clone(), SearchEngineStatus::Ready { doc_count } if doc_count > 0);
    let has_results = !search_results.read().is_empty();
    let show_empty_state = !engine_ready && !searching();

    rsx! {
        section {
            class: "cm-view cm-view--search cm-view--active",
            "data-view": "search",

            SearchCard {
                search_query,
                on_search: handle_search,
                searching,
            }

            // Show empty state if no documents indexed
            if show_empty_state {
                EmptyState {
                    on_navigate_to_index: on_navigate
                }
            }

            // Show results section if we have results or are searching
            if has_results || searching() {
                section { class: "cm-results-section",
                    header { class: "cm-results-header",
                        span { class: "cm-results-count",
                            if searching() {
                                "Searching…"
                            } else {
                                "{search_results.read().len()} result(s) for \"{search_query}\""
                            }
                        }
                    }

                    // Result cards
                    for (idx, result) in search_results.read().iter().enumerate() {
                        ResultCard {
                            key: "{result.doc_id.as_u64()}",
                            rank: idx + 1,
                            result: result.clone(),
                            on_show_source: handle_show_source,
                        }
                    }

                    // Load more button (placeholder for future pagination)
                    if has_results && search_results.read().len() >= 10 {
                        div { class: "cm-results-load-more",
                            button {
                                class: "cm-btn cm-btn--secondary",
                                "Load more results"
                            }
                        }
                    }
                }
            }

            // Status message (for debugging)
            if !search_status.read().is_empty() && cfg!(debug_assertions) {
                div { class: "cm-page-subtitle",
                    style: "margin-top: 1rem; opacity: 0.7;",
                    "{search_status}"
                }
            }
        }
    }
}
