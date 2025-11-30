use crate::processing::embed_text;
use crate::search::types::{
    ChunkId, ChunkSourceMetadata, DocumentId, DocumentMetainfo, DocumentSearchResult, SearchResult,
};
use coppermind_core::metrics::global_metrics;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;
use instant::Instant;

#[cfg(target_arch = "wasm32")]
use crate::components::worker::use_worker_state;

use crate::components::{
    trigger_metrics_refresh, use_metrics_refresh, use_search_engine, use_search_engine_status,
    SearchEngineStatus, View,
};

use super::{
    EmptyState, ResultCard, SearchCard, SearchMode, SkeletonResultCard, SourcePreviewOverlay,
};

/// Default number of documents to return
const DEFAULT_RESULT_COUNT: usize = 20;

// Messages for search coroutine
enum SearchMessage {
    /// Run a fresh search, replacing all results
    RunSearch { query: String, result_count: usize },
    /// Load more results, appending to existing results
    LoadMore {
        query: String,
        current_count: usize,
        load_count: usize,
    },
}

/// Easter egg: hardcoded document result that appears when searching for "coppermind"
/// This is NOT indexed and does NOT affect document counts - it's purely a UI-level feature.
fn create_easter_egg_result() -> DocumentSearchResult {
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

    DocumentSearchResult {
        doc_id: DocumentId::from_u64(u64::MAX), // Special ID to avoid conflicts
        score: 1.0,
        doc_keyword_score: Some(1.0),
        best_chunk_score: Some(1.0),
        metadata: DocumentMetainfo {
            source_id: "welcome.md".to_string(),
            title: "welcome.md".to_string(),
            mime_type: Some("text/markdown".to_string()),
            content_hash: "easter_egg".to_string(),
            created_at: FIRST_COMMIT_TIMESTAMP,
            updated_at: FIRST_COMMIT_TIMESTAMP,
            char_count: chunk.text.len(),
            chunk_count: 1,
        },
        chunks: vec![chunk],
    }
}

/// Main search view with search card and results
#[component]
pub fn SearchView(on_navigate: EventHandler<View>) -> Element {
    let search_query = use_signal(String::new);
    let search_results = use_signal(Vec::<DocumentSearchResult>::new);
    let mut searching = use_signal(|| false);
    let mut loading_more = use_signal(|| false); // True when appending results (no skeleton)
    let search_status = use_signal(String::new);
    let mut preview_result = use_signal(|| None::<DocumentSearchResult>);

    // Track the last executed query for "load more" functionality
    let last_query = use_signal(String::new);

    // Track if more results can be loaded (None = unknown, true = more available, false = exhausted)
    let can_load_more = use_signal(|| None::<bool>);

    // Search controls
    let result_count = use_signal(|| DEFAULT_RESULT_COUNT);
    let search_mode = use_signal(SearchMode::default);

    let search_engine = use_search_engine();
    let engine_status = use_search_engine_status();

    #[cfg(target_arch = "wasm32")]
    let worker_state = use_worker_state();

    // Search coroutine - runs in background
    let search_task = use_coroutine({
        let mut results = search_results;
        let mut status = search_status;
        let mut is_searching = searching;
        let mut is_loading_more = loading_more;
        let engine = search_engine;
        let mut stored_query = last_query;
        let mut loadable = can_load_more;

        move |mut rx: UnboundedReceiver<SearchMessage>| async move {
            while let Some(msg) = rx.next().await {
                match msg {
                    SearchMessage::RunSearch {
                        query,
                        result_count: k,
                    } => {
                        if query.trim().is_empty() {
                            status.set("Please enter a search query.".into());
                            is_searching.set(false);
                            continue;
                        }

                        // Store the query for "load more" functionality
                        stored_query.set(query.clone());

                        info!("🔍 Searching for: '{}' (k={} documents)", query, k);
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
                                // Acquire lock and run document-level search with timing
                                let mut search_engine = engine.lock().await;
                                match search_engine
                                    .search_documents_with_timings(&embedding, &query, k)
                                    .await
                                {
                                    Ok((mut doc_results, timings)) => {
                                        // Calculate score statistics for metrics
                                        let top_score = doc_results.first().map(|r| r.score);
                                        let median_score = if doc_results.is_empty() {
                                            None
                                        } else {
                                            doc_results.get(doc_results.len() / 2).map(|r| r.score)
                                        };

                                        // Count total chunks across all documents
                                        let total_chunks: usize =
                                            doc_results.iter().map(|d| d.chunks.len()).sum();

                                        // Record search metrics
                                        global_metrics().record_search(
                                            embed_ms,
                                            timings.vector_ms,
                                            timings.keyword_ms,
                                            timings.fusion_ms,
                                            total_chunks,
                                            timings.vector_count,
                                            timings.keyword_count,
                                            top_score,
                                            median_score,
                                        );

                                        info!(
                                            "✅ Search completed: {} documents in {:.1}ms (embed: {:.1}ms, vector: {:.1}ms, keyword: {:.1}ms, fusion: {:.1}ms)",
                                            doc_results.len(),
                                            timings.total_ms + embed_ms,
                                            embed_ms,
                                            timings.vector_ms,
                                            timings.keyword_ms,
                                            timings.fusion_ms
                                        );

                                        // Easter egg: prepend hardcoded result when searching for "coppermind"
                                        if query.trim().to_lowercase() == "coppermind" {
                                            let easter_egg = create_easter_egg_result();
                                            doc_results.insert(0, easter_egg);
                                            info!("🥚 Easter egg activated!");
                                        }

                                        // Calculate total query time (embedding + search)
                                        let total_query_ms = timings.total_ms + embed_ms;

                                        if doc_results.is_empty() {
                                            status.set(format!(
                                                "No results found ({:.0} ms)",
                                                total_query_ms
                                            ));
                                        } else {
                                            let doc_count = doc_results.len();
                                            let chunk_count: usize =
                                                doc_results.iter().map(|d| d.chunks.len()).sum();

                                            let doc_word = if doc_count == 1 {
                                                "document"
                                            } else {
                                                "documents"
                                            };
                                            let chunk_word =
                                                if chunk_count == 1 { "chunk" } else { "chunks" };

                                            // Google/Bing style: "About X results (0.42 seconds)"
                                            status.set(format!(
                                                "{} {} ({} {}) in {:.0} ms",
                                                doc_count,
                                                doc_word,
                                                chunk_count,
                                                chunk_word,
                                                total_query_ms
                                            ));
                                        }
                                        results.set(doc_results);

                                        // Reset load more state for fresh search
                                        loadable.set(None);

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

                    SearchMessage::LoadMore {
                        query,
                        current_count,
                        load_count,
                    } => {
                        // Fetch current_count + load_count results, then append only the new ones
                        let total_k = current_count + load_count;

                        info!(
                            "📄 Loading more: fetching {} total (currently have {})",
                            total_k, current_count
                        );

                        // Time query embedding
                        let embed_start = Instant::now();

                        #[cfg(target_arch = "wasm32")]
                        let embed_result = embed_text(&query, worker_state).await;

                        #[cfg(not(target_arch = "wasm32"))]
                        let embed_result = embed_text(&query).await;

                        let embed_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

                        let query_embedding = match embed_result {
                            Ok(computation) => Some(computation.embedding),
                            Err(e) => {
                                error!("❌ Failed to embed query: {}", e);
                                is_searching.set(false);
                                is_loading_more.set(false);
                                None
                            }
                        };

                        if let Some(embedding) = query_embedding {
                            let engine_arc = { engine.read().clone() };
                            if let Some(engine) = engine_arc {
                                let mut search_engine = engine.lock().await;
                                match search_engine
                                    .search_documents_with_timings(&embedding, &query, total_k)
                                    .await
                                {
                                    Ok((doc_results, timings)) => {
                                        // Only append results beyond current_count
                                        if doc_results.len() > current_count {
                                            let new_results: Vec<_> = doc_results
                                                .into_iter()
                                                .skip(current_count)
                                                .collect();
                                            let new_count = new_results.len();

                                            info!(
                                                "✅ Loaded {} more documents in {:.1}ms",
                                                new_count,
                                                timings.total_ms + embed_ms
                                            );

                                            // Append to existing results
                                            results.write().extend(new_results);

                                            // Update status with new total
                                            let total_docs = results.read().len();
                                            let total_chunks: usize =
                                                results.read().iter().map(|d| d.chunks.len()).sum();
                                            let doc_word = if total_docs == 1 {
                                                "document"
                                            } else {
                                                "documents"
                                            };
                                            let chunk_word =
                                                if total_chunks == 1 { "chunk" } else { "chunks" };
                                            status.set(format!(
                                                "{} {} ({} {})",
                                                total_docs, doc_word, total_chunks, chunk_word
                                            ));

                                            // If we got fewer results than requested, we've exhausted
                                            if new_count < load_count {
                                                loadable.set(Some(false));
                                            }
                                        } else {
                                            info!("No more results to load");
                                            loadable.set(Some(false));
                                        }
                                    }
                                    Err(e) => {
                                        error!("❌ Load more failed: {:?}", e);
                                    }
                                }
                            }
                        }

                        is_searching.set(false);
                        is_loading_more.set(false);
                    }
                }
            }
        }
    });

    let handle_search = move |query: String| {
        searching.set(true);
        let k = result_count();
        search_task.send(SearchMessage::RunSearch {
            query,
            result_count: k,
        });
    };

    // Load more results by appending to existing results
    let handle_load_more = move |_| {
        let query = last_query.read().clone();
        if !query.is_empty() {
            let current_count = search_results.read().len();
            let load_count = result_count(); // Load as many as the "Max" setting
            searching.set(true);
            loading_more.set(true);
            search_task.send(SearchMessage::LoadMore {
                query,
                current_count,
                load_count,
            });
        }
    };

    let mut preview_signal = preview_result;
    let handle_show_source = move |doc_result: DocumentSearchResult| {
        info!("Show source for: {}", doc_result.metadata.title);
        preview_signal.set(Some(doc_result));
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

    // Get metrics refresh signal - used for result card keys to reset expansion state
    let metrics_refresh = use_metrics_refresh();

    // Google-style results header: show search_status only after search completes
    // search_status already contains "X files (Y chunks) in Z ms" format
    let result_count_text = if searching() {
        "Searching…".to_string()
    } else if has_results {
        search_status.read().clone()
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
                result_count,
                search_mode,
            }

            // Show empty state if no documents indexed
            if show_empty_state {
                EmptyState {
                    on_navigate_to_index: on_navigate
                }
            }

            // Show results section if we have results or are searching (but not just loading more)
            if has_results || (searching() && !loading_more()) {
                section { class: "cm-results-section",
                    header { class: "cm-results-header",
                        span { class: "cm-results-count",
                            "{result_count_text}"
                        }
                    }

                    // Show skeleton cards only for fresh searches (not load more)
                    if searching() && !loading_more() && !has_results {
                        // Show skeleton placeholders (show 3 cards as preview)
                        for rank in 1..=3 {
                            SkeletonResultCard { key: "skeleton-{rank}", rank }
                        }
                    }

                    // Document result cards - always show existing results
                    // Key includes metrics_refresh counter to force remount on new search,
                    // which resets expansion state (chunks/details collapsed)
                    for (idx, doc_result) in search_results.read().iter().enumerate() {
                        ResultCard {
                            key: "{metrics_refresh()}-{idx}-{doc_result.metadata.source_id}",
                            rank: idx + 1,
                            doc_result: doc_result.clone(),
                            on_show_source: handle_show_source,
                        }
                    }

                    // Load more section - show button or "no more results" message
                    if has_results {
                        div { class: "cm-results-load-more",
                            if can_load_more() == Some(false) {
                                // All results loaded - show message
                                span {
                                    class: "cm-results-exhausted",
                                    "No more results"
                                }
                            } else {
                                // More results may be available - show button
                                button {
                                    class: "cm-btn cm-btn--secondary",
                                    disabled: searching(),
                                    onclick: handle_load_more,
                                    if loading_more() {
                                        "Loading..."
                                    } else {
                                        "Load more results"
                                    }
                                }
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
