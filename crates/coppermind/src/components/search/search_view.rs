use crate::processing::embed_text;
use crate::search::types::{ChunkId, ChunkSourceMetadata, SearchResult};
use crate::search::{aggregate_chunks_by_file, types::FileSearchResult};
use coppermind_core::metrics::global_metrics;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;
use instant::Instant;

#[cfg(target_arch = "wasm32")]
use crate::components::worker::use_worker_state;

use crate::components::{
    trigger_metrics_refresh, use_search_engine, use_search_engine_status, SearchEngineStatus, View,
};

use super::{EmptyState, ResultCard, SearchCard, SourcePreviewOverlay};

// Messages for search coroutine
enum SearchMessage {
    RunSearch(String), // query text
}

/// Easter egg: hardcoded file result that appears when searching for "coppermind"
/// This is NOT indexed and does NOT affect document counts - it's purely a UI-level feature.
fn create_easter_egg_result() -> FileSearchResult {
    // First git commit timestamp: 2025-10-31 01:16:40 -0400
    const FIRST_COMMIT_TIMESTAMP: u64 = 1761887800;

    let chunk = SearchResult {
        chunk_id: ChunkId::from_u64(u64::MAX), // Special ID to avoid conflicts
        score: 1.0,
        vector_score: Some(1.0),
        keyword_score: Some(1.0),
        text: "Coppermind is a local-first semantic search engine powered by Rust and WebAssembly. \
               It uses hybrid search combining vector similarity (JinaBERT embeddings) with keyword \
               matching (BM25) and fuses results using Reciprocal Rank Fusion. All processing happens \
               on your device with no cloud dependencies. The search engine uses HNSW for fast \
               approximate nearest neighbor search and provides detailed fusion scores for transparency."
            .to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("welcome.md (chunk 1)".to_string()),
            source: Some("welcome.md".to_string()),
            created_at: FIRST_COMMIT_TIMESTAMP,
        },
    };

    FileSearchResult {
        file_path: "welcome.md".to_string(),
        file_name: "welcome.md".to_string(),
        score: 1.0,
        vector_score: Some(1.0),
        keyword_score: Some(1.0),
        chunks: vec![chunk],
        created_at: FIRST_COMMIT_TIMESTAMP,
    }
}

/// Main search view with search card and results
#[component]
pub fn SearchView(on_navigate: EventHandler<View>) -> Element {
    let search_query = use_signal(String::new);
    let search_results = use_signal(Vec::<FileSearchResult>::new);
    let mut searching = use_signal(|| false);
    let search_status = use_signal(String::new);
    let mut preview_result = use_signal(|| None::<FileSearchResult>);

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

                        // Time query embedding
                        let embed_start = Instant::now();

                        // Embed the query using platform-appropriate method
                        #[cfg(target_arch = "wasm32")]
                        let embed_result = embed_text(&query, worker_state).await;

                        #[cfg(not(target_arch = "wasm32"))]
                        let embed_result = embed_text(&query).await;

                        let embed_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

                        let query_embedding = match embed_result {
                            Ok(computation) => Some(computation.embedding),
                            Err(e) => {
                                error!("❌ Failed to embed query: {}", e);
                                status.set(format!("Failed to embed query: {}", e));
                                is_searching.set(false);
                                None
                            }
                        };

                        if let Some(embedding) = query_embedding {
                            status.set("Running hybrid search...".into());

                            // Clone Arc from signal (cheap operation)
                            // Read lock is dropped immediately after clone
                            let engine_arc = { engine.read().clone() };
                            if let Some(engine) = engine_arc {
                                // Acquire lock and run search with timing
                                let mut search_engine = engine.lock().await;
                                match search_engine
                                    .search_with_timings(&embedding, &query, 20)
                                    .await
                                {
                                    Ok((chunk_results, timings)) => {
                                        // Calculate score statistics for metrics
                                        let top_score = chunk_results.first().map(|r| r.score);
                                        let median_score = if chunk_results.is_empty() {
                                            None
                                        } else {
                                            chunk_results
                                                .get(chunk_results.len() / 2)
                                                .map(|r| r.score)
                                        };

                                        // Record search metrics
                                        global_metrics().record_search(
                                            embed_ms,
                                            timings.vector_ms,
                                            timings.keyword_ms,
                                            timings.fusion_ms,
                                            chunk_results.len(),
                                            timings.vector_count,
                                            timings.keyword_count,
                                            top_score,
                                            median_score,
                                        );

                                        info!(
                                            "✅ Search completed: {} results in {:.1}ms (embed: {:.1}ms, vector: {:.1}ms, keyword: {:.1}ms, fusion: {:.1}ms)",
                                            chunk_results.len(),
                                            timings.total_ms + embed_ms,
                                            embed_ms,
                                            timings.vector_ms,
                                            timings.keyword_ms,
                                            timings.fusion_ms
                                        );

                                        // Aggregate chunks into file-level results
                                        let mut file_results =
                                            aggregate_chunks_by_file(chunk_results);

                                        // Easter egg: prepend hardcoded result when searching for "coppermind"
                                        if query.trim().to_lowercase() == "coppermind" {
                                            let easter_egg = create_easter_egg_result();
                                            file_results.insert(0, easter_egg);
                                            info!("🥚 Easter egg activated!");
                                        }

                                        info!(
                                            "📁 Aggregated to {} file results",
                                            file_results.len()
                                        );

                                        // Calculate total query time (embedding + search)
                                        let total_query_ms = timings.total_ms + embed_ms;

                                        if file_results.is_empty() {
                                            status.set(format!(
                                                "No results found ({:.0} ms)",
                                                total_query_ms
                                            ));
                                        } else {
                                            let file_count = file_results.len();
                                            let chunk_count: usize =
                                                file_results.iter().map(|f| f.chunks.len()).sum();

                                            let file_word =
                                                if file_count == 1 { "file" } else { "files" };
                                            let chunk_word =
                                                if chunk_count == 1 { "chunk" } else { "chunks" };

                                            // Google/Bing style: "About X results (0.42 seconds)"
                                            status.set(format!(
                                                "{} {} ({} {}) in {:.0} ms",
                                                file_count,
                                                file_word,
                                                chunk_count,
                                                chunk_word,
                                                total_query_ms
                                            ));
                                        }
                                        results.set(file_results);

                                        // Trigger metrics pane refresh to show updated search stats
                                        trigger_metrics_refresh();
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

    let mut preview_signal = preview_result;
    let handle_show_source = move |file_result: FileSearchResult| {
        info!("Show source for: {}", file_result.file_name);
        preview_signal.set(Some(file_result));
    };

    let handle_close_preview = move |_| {
        preview_result.set(None);
    };

    // Determine what to show
    let engine_status_val = engine_status.read().clone();
    let engine_loading = matches!(
        engine_status_val,
        SearchEngineStatus::Pending | SearchEngineStatus::Loading
    );
    let engine_ready_with_docs = matches!(
        engine_status_val,
        SearchEngineStatus::Ready { doc_count, .. } if doc_count > 0
    );
    let has_results = !search_results.read().is_empty();
    // Show empty state only when index is ready but has no documents
    // Don't show empty state while loading (search card already indicates loading)
    let show_empty_state = !engine_loading && !engine_ready_with_docs && !searching();

    // Compute result count text with proper pluralization
    let result_count_text = if !searching() && has_results {
        let count = search_results.read().len();
        let result_word = if count == 1 { "result" } else { "results" };
        format!("{} {} for \"{}\"", count, result_word, search_query.read())
    } else if searching() {
        "Searching…".to_string()
    } else {
        String::new()
    };

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
                            "{result_count_text}"
                        }
                    }

                    // File result cards
                    for (idx, file_result) in search_results.read().iter().enumerate() {
                        ResultCard {
                            key: "{file_result.file_path}",
                            rank: idx + 1,
                            file_result: file_result.clone(),
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

        // Source preview overlay (shown when user clicks "Show Source")
        SourcePreviewOverlay {
            file_result: preview_result,
            on_close: handle_close_preview,
        }
    }
}
