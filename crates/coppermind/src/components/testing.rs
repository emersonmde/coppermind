use crate::search::types::{Document, DocumentMetadata};
use crate::search::HybridSearchEngine;
use crate::storage::StorageError;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;

#[cfg(not(target_arch = "wasm32"))]
use crate::embedding::run_embedding;

#[cfg(target_arch = "wasm32")]
use crate::embedding::format_embedding_summary;

#[cfg(target_arch = "wasm32")]
use super::worker::{use_worker_state, WorkerStatus};

use super::{use_search_engine, use_search_engine_status, SearchEngineStatus};

// Mock storage for testing
struct MockStorage;

#[async_trait::async_trait(?Send)]
impl crate::storage::StorageBackend for MockStorage {
    async fn save(&self, _key: &str, _data: &[u8]) -> Result<(), StorageError> {
        Ok(())
    }
    async fn load(&self, _key: &str) -> Result<Vec<u8>, StorageError> {
        Err(StorageError::NotFound("test".to_string()))
    }
    async fn exists(&self, _key: &str) -> Result<bool, StorageError> {
        Ok(false)
    }
    async fn delete(&self, _key: &str) -> Result<(), StorageError> {
        Ok(())
    }
    async fn list_keys(&self) -> Result<Vec<String>, StorageError> {
        Ok(vec![])
    }
    async fn clear(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

// Messages for embedding test coroutine
enum EmbeddingMessage {
    RunTest(String),
}

#[component]
pub fn DeveloperTesting() -> Element {
    let embedding_result = use_signal(String::new);
    let search_results = use_signal(Vec::<crate::search::types::SearchResult>::new);
    let search_query = use_signal(String::new);
    let clear_status = use_signal(String::new);
    let debug_dump = use_signal(String::new);

    #[cfg(target_arch = "wasm32")]
    let worker_state = use_worker_state();

    let search_engine = use_search_engine();
    let search_engine_status = use_search_engine_status();

    // Embedding test coroutine
    let embedding_task = use_coroutine({
        let mut result = embedding_result;

        move |mut rx: UnboundedReceiver<EmbeddingMessage>| async move {
            while let Some(msg) = rx.next().await {
                match msg {
                    EmbeddingMessage::RunTest(text) => {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            // Desktop: Direct async call
                            match run_embedding(&text).await {
                                Ok(res) => result.set(res),
                                Err(e) => result.set(format!("Error: {}", e)),
                            }
                        }

                        #[cfg(target_arch = "wasm32")]
                        {
                            let worker_snapshot = worker_state.read().clone();
                            match worker_snapshot {
                                WorkerStatus::Pending => {
                                    result
                                        .set("Embedding worker is starting… please retry.".into());
                                }
                                WorkerStatus::Failed(err) => {
                                    result.set(format!("Embedding worker unavailable: {}", err));
                                }
                                WorkerStatus::Ready(client) => {
                                    match client.embed(text.clone()).await {
                                        Ok(computation) => {
                                            let formatted = format_embedding_summary(
                                                &text,
                                                computation.token_count,
                                                &computation.embedding,
                                            );
                                            result.set(formatted);
                                        }
                                        Err(e) => {
                                            result.set(format!("Worker embedding failed: {}", e));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    rsx! {
        div { class: "testing-section",
            div { class: "section-header",
                h2 { class: "testing-header", "Developer Testing" }
                p { class: "section-description",
                    "Test individual components of the search system"
                }
            }

            div { class: "test-grid",
                // Text Embedding Test
                div { class: "test-card",
                    h3 { class: "test-card-title", "Text Embedding Test" }
                    p { class: "test-card-description",
                        "Test JinaBERT text embeddings using Candle ML framework"
                    }

                    button {
                        class: "btn-primary",
                        onclick: move |_| {
                            let test_text = "This is a test sentence for text embedding.";
                            embedding_task.send(EmbeddingMessage::RunTest(test_text.to_string()));
                        },
                        "Run Embedding Test"
                    }

                    if !embedding_result.read().is_empty() {
                        div { class: "test-results", "{embedding_result.read()}" }
                    }
                }

                // Hybrid Search Test
                div { class: "test-card",
                    h3 { class: "test-card-title", "Hybrid Search Test" }
                    p { class: "test-card-description",
                        "Test vector (semantic) + keyword (BM25) search with RRF fusion"
                    }

                    button {
                        class: "btn-primary",
                        onclick: move |_| {
                            let mut results_signal = search_results;
                            let mut query_signal = search_query;
                            spawn(async move {
                                info!("🔍 Testing Hybrid Search...");

                                // Clear previous results
                                results_signal.set(Vec::new());

                                let storage = MockStorage;
                                let mut engine = match HybridSearchEngine::new(storage, 512).await {
                                    Ok(e) => {
                                        info!("✓ HybridSearchEngine created (embedding_dim=512)");
                                        e
                                    }
                                    Err(e) => {
                                        error!("❌ Failed to create engine: {:?}", e);
                                        return;
                                    }
                                };

                                // Add test documents
                                let docs = vec![
                                    (
                                        "Machine learning is a subset of artificial intelligence",
                                        vec![0.9, 0.1, 0.0, 0.3, 0.5],
                                    ),
                                    (
                                        "Deep neural networks are powerful for pattern recognition",
                                        vec![0.8, 0.2, 0.1, 0.4, 0.6],
                                    ),
                                    (
                                        "Natural language processing enables computers to understand text",
                                        vec![0.7, 0.3, 0.2, 0.5, 0.4],
                                    ),
                                ];

                                for (i, (text, mut embedding)) in docs.into_iter().enumerate() {
                                    embedding.resize(512, 0.0);

                                    let doc = Document {
                                        text: text.to_string(),
                                        metadata: DocumentMetadata {
                                            filename: Some(format!("doc{}.txt", i + 1)),
                                            source: None,
                                            created_at: i as u64,
                                        },
                                    };

                                    match engine.add_document(doc, embedding).await {
                                        Ok(doc_id) => {
                                            info!("✓ Added document {}: {:?}", doc_id.as_u64(), text);
                                        }
                                        Err(e) => {
                                            error!("❌ Failed to add document: {:?}", e);
                                        }
                                    }
                                }

                                // Perform search
                                let query = "machine learning neural networks";
                                info!("🔎 Searching for '{}'...", query);
                                query_signal.set(query.to_string());

                                let query_embedding = {
                                    let mut emb = vec![0.85, 0.15, 0.05, 0.35, 0.55];
                                    emb.resize(512, 0.0);
                                    emb
                                };

                                match engine.search(&query_embedding, query, 3).await {
                                    Ok(results) => {
                                        info!("📊 Final RRF fused results (top {}):", results.len());
                                        for (i, result) in results.iter().enumerate() {
                                            info!(
                                                "  {}. [RRF: {:.4}] {}",
                                                i + 1,
                                                result.score,
                                                result.text
                                            );
                                        }
                                        info!("🎉 Hybrid search test completed successfully!");
                                        results_signal.set(results);
                                    }
                                    Err(e) => {
                                        error!("❌ Search failed: {:?}", e);
                                    }
                                }
                            });
                        },
                        "Run Search Test"
                    }

                    if !search_results.read().is_empty() {
                        div { class: "search-results-container",
                            div { class: "search-query-header",
                                span { class: "search-query-label", "Query:" }
                                span { class: "search-query-text", "\"{search_query.read()}\"" }
                            }

                            div { class: "search-results-list",
                                for (idx, result) in search_results.read().iter().enumerate() {
                                    div {
                                        key: "{result.doc_id.as_u64()}",
                                        class: "search-result-item",
                                        div { class: "search-result-header",
                                            span { class: "search-result-rank", "#{idx + 1}" }
                                            span { class: "search-result-score",
                                                "RRF: {result.score:.4}"
                                            }
                                        }
                                        div { class: "search-result-scores",
                                            if let Some(vector_score) = result.vector_score {
                                                span { class: "score-badge vector-score",
                                                    "Vector: {vector_score:.4}"
                                                }
                                            }
                                            if let Some(keyword_score) = result.keyword_score {
                                                span { class: "score-badge keyword-score",
                                                    "Keyword: {keyword_score:.4}"
                                                }
                                            }
                                        }
                                        div { class: "search-result-text", "{result.text}" }
                                        if let Some(filename) = &result.metadata.filename {
                                            div { class: "search-result-source", "Source: {filename}" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Clear Index Test
                div { class: "test-card",
                    h3 { class: "test-card-title", "Clear Search Index" }
                    p { class: "test-card-description",
                        "Clear all indexed documents from the search engine (useful for testing)"
                    }

                    button {
                        class: "btn-primary",
                        onclick: move |_| {
                            let mut status = clear_status;
                            let engine = search_engine;
                            let mut engine_status = search_engine_status;
                            spawn(async move {
                                info!("🗑️ Clearing search index...");
                                let engine_arc = engine.read().clone();
                                if let Some(engine_lock) = engine_arc {
                                    let mut search_engine = engine_lock.lock().await;
                                    search_engine.clear();
                                    info!("✅ Search index cleared");
                                    status.set("✓ Search index cleared successfully".into());
                                    engine_status.set(SearchEngineStatus::Ready { doc_count: 0 });
                                } else {
                                    error!("❌ Search engine not initialized");
                                    status.set("Search engine not initialized".into());
                                }
                            });
                        },
                        "Clear Search Index"
                    }

                    if !clear_status.read().is_empty() {
                        div { class: "test-results", "{clear_status.read()}" }
                    }
                }

                // Debug Dump Test
                div { class: "test-card",
                    h3 { class: "test-card-title", "Debug Index Dump" }
                    p { class: "test-card-description",
                        "Dump the current search index state for debugging (shows all indexed documents)"
                    }

                    button {
                        class: "btn-primary",
                        onclick: move |_| {
                            let mut dump = debug_dump;
                            let engine = search_engine;
                            spawn(async move {
                                info!("🔍 Dumping search index...");
                                let engine_arc = engine.read().clone();
                                if let Some(engine_lock) = engine_arc {
                                    let search_engine = engine_lock.lock().await;
                                    let dump_text = search_engine.debug_dump();
                                    info!("{}", dump_text);
                                    dump.set(dump_text);
                                } else {
                                    error!("❌ Search engine not initialized");
                                    dump.set("Search engine not initialized".into());
                                }
                            });
                        },
                        "Dump Index State"
                    }

                    if !debug_dump.read().is_empty() {
                        div { class: "test-results",
                            style: "white-space: pre-wrap; font-family: monospace; font-size: 0.9em;",
                            "{debug_dump.read()}"
                        }
                    }
                }
            }
        }
    }
}
