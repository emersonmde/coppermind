use crate::cpu::spawn_worker;
use crate::embedding::{embed_text_chunks, run_embedding, ChunkEmbeddingResult};
use crate::search::types::{Document, DocumentMetadata};
use crate::search::HybridSearchEngine;
use crate::storage::StorageError;
use crate::wgpu::test_webgpu;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;

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

#[component]
pub fn TestControls() -> Element {
    let mut cpu_running = use_signal(|| false);
    let mut cpu_results = use_signal(Vec::<String>::new);
    let gpu_result = use_signal(String::new);
    let embedding_result = use_signal(String::new);
    let file_processing = use_signal(|| false);
    let mut file_status = use_signal(String::new);
    let mut file_chunks = use_signal(Vec::<ChunkEmbeddingResult>::new);
    let mut file_name = use_signal(String::new);

    rsx! {
        div { class: "test-controls",
            div { class: "test-section",
                h2 { "CPU Workers Test foo" }
                p { class: "description", "Spawns 16 Web Workers for parallel CPU computation" }

                div { class: "button-group",
                    button {
                        class: "btn-primary",
                        disabled: cpu_running(),
                        onclick: move |_| {
                            cpu_running.set(true);
                            cpu_results.set(Vec::new());

                            let num_workers = 16;
                            for i in 0..num_workers {
                                spawn({
                                    let mut results = cpu_results;
                                    let mut running = cpu_running;
                                    async move {
                                        match spawn_worker(i, "test", 10000).await {
                                            Ok(result) => {
                                                results.write().push(result);
                                                if results.read().len() == num_workers {
                                                    running.set(false);
                                                }
                                            }
                                            Err(e) => {
                                                results.write().push(format!("Error: {}", e));
                                            }
                                        }
                                    }
                                });
                            }
                        },
                        "Start CPU Test"
                    }

                    if cpu_running() {
                        button {
                            class: "btn-secondary",
                            onclick: move |_| {
                                cpu_running.set(false);
                                cpu_results.set(Vec::new());
                            },
                            "Stop"
                        }
                    }
                }

                if cpu_running() {
                    div { class: "status",
                        "Running... ({cpu_results.read().len()}/16 workers completed)"
                    }
                }

                if !cpu_results.read().is_empty() && !cpu_running() {
                    div { class: "results",
                        "✓ All workers completed successfully"
                    }
                }
            }

            div { class: "test-section",
                p { class: "description", "WebGPU compute shader with 1M+ parallel operations" }
                h2 { "GPU Compute Test" }

                button {
                    class: "btn-primary",
                    onclick: move |_| {
                        let mut result = gpu_result;
                        spawn(async move {
                            match test_webgpu().await {
                                Ok(msg) => result.set(msg),
                                Err(e) => result.set(format!("Error: {}", e)),
                            }
                        });
                    },
                    "Test GPU"
                }

                if !gpu_result.read().is_empty() {
                    div { class: "results",
                        "{gpu_result.read()}"
                    }
                }
            }

            div { class: "test-section",
                h2 { "Text Embedding Test" }
                p { class: "description", "JinaBERT text embeddings using Candle ML framework" }

                button {
                    class: "btn-primary",
                    onclick: move |_| {
                        let mut result = embedding_result;
                        spawn(async move {
                            let test_text = "This is a test sentence for text embedding.";
                            match run_embedding(test_text).await {
                                Ok(msg) => result.set(msg),
                                Err(e) => result.set(format!("Error: {}", e)),
                            }
                        });
                    },
                    "Test Embedding"
                }

                if !embedding_result.read().is_empty() {
                    div { class: "results",
                        "{embedding_result.read()}"
                    }
                }
            }

            div { class: "test-section",
                h2 { "File Embedding Demo" }
                p { class: "description", "Upload a .txt file, chunk it in Rust, and embed each chunk entirely in WASM" }

                input {
                    r#type: "file",
                    accept: ".txt",
                    multiple: false,
                    onchange: move |evt: dioxus::events::FormEvent| {
                        if file_processing() {
                            file_status.set("Already processing a file. Please wait for it to finish.".into());
                            return;
                        }

                        if let Some(engine) = evt.files() {
                            let files = engine.files();
                            if let Some(first_name) = files.first() {
                                file_name.set(first_name.clone());
                                let file_label = first_name.clone();
                                let engine = engine.clone();
                                let mut status = file_status;
                                let mut chunks = file_chunks;
                                let mut processing = file_processing;
                                let mut selected_name = file_name;
                                spawn(async move {
                                    processing.set(true);
                                    chunks.set(Vec::new());
                                    info!("📂 Selected file: {file_label}");
                                    status.set(format!("Reading {file_label}..."));
                                    match engine.read_file_to_string(&file_label).await {
                                        Some(contents) => {
                                            let byte_len = contents.len();
                                            info!(
                                                "🧮 File size: {} bytes (~{:.2} KB)",
                                                byte_len,
                                                byte_len as f64 / 1024.0
                                            );
                                            status.set(format!("Embedding {file_label} ({} bytes)...", byte_len));
                                            match embed_text_chunks(&contents, 512).await {
                                                Ok(results) => {
                                                    let chunk_count = results.len();
                                                    if chunk_count == 0 {
                                                        status.set(format!("File {file_label} did not produce any tokens."));
                                                    } else {
                                                        for chunk in results.iter() {
                                                            info!(
                                                                "📦 Chunk {} embedded ({} tokens)",
                                                                chunk.chunk_index,
                                                                chunk.token_count
                                                            );
                                                        }
                                                        status.set(format!("✓ Embedded {chunk_count} chunks from {file_label}"));
                                                    }
                                                    chunks.set(results);
                                                }
                                                Err(e) => {
                                                    error!("❌ Embedding failed: {e}");
                                                    status.set(format!("Embedding failed: {e}"));
                                                    selected_name.set(String::new());
                                                }
                                            }
                                        }
                                        None => {
                                            error!("❌ Failed to read {file_label}");
                                            status.set(format!("Failed to read {file_label}"));
                                            selected_name.set(String::new());
                                        }
                                    }
                                    processing.set(false);
                                });
                            } else {
                                file_status.set("No file selected.".into());
                                file_chunks.set(Vec::new());
                                file_name.set(String::new());
                            }
                        } else {
                            file_status.set("Browser did not provide a file list.".into());
                            file_chunks.set(Vec::new());
                            file_name.set(String::new());
                        }
                    }
                }

                if file_processing() {
                    div { class: "status",
                        "Processing {file_name.read()}..."
                    }
                }

                if !file_status.read().is_empty() {
                    div { class: "status",
                        "{file_status.read()}"
                    }
                }

                if !file_chunks.read().is_empty() {
                    div { class: "results",
                        h3 { "Chunk Results" }
                        ul {
                            for chunk in file_chunks.read().iter() {
                                li {
                                    key: "chunk-{chunk.chunk_index}",
                                    {format!(
                                        "Chunk #{}: {} tokens - preview [{}]",
                                        chunk.chunk_index,
                                        chunk.token_count,
                                        chunk
                                            .embedding
                                            .iter()
                                            .take(6)
                                            .map(|v| format!("{:.4}", v))
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    )}
                                }
                            }
                        }
                    }
                }
            }

            div { class: "test-section",
                h2 { "Hybrid Search Test" }
                p { class: "description", "Test vector (semantic) + keyword (BM25) search with RRF fusion" }

                button {
                    class: "btn-primary",
                    onclick: move |_| {
                        spawn(async move {
                            info!("🔍 Testing Hybrid Search...");

                            // Create mock storage and hybrid search engine
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
                                    vec![0.9, 0.1, 0.0, 0.3, 0.5], // Dummy 5D embedding for demo
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

                            // Pad embeddings to 512 dimensions
                            for (i, (text, mut embedding)) in docs.into_iter().enumerate() {
                                // Pad to 512 dimensions
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
                                        info!(
                                            "✓ Added document {}: {:?}",
                                            doc_id.as_u64(),
                                            text
                                        );
                                    }
                                    Err(e) => {
                                        error!("❌ Failed to add document: {:?}", e);
                                    }
                                }
                            }

                            // Perform hybrid search
                            info!("🔎 Searching for 'machine learning neural networks'...");

                            let query_embedding = {
                                let mut emb = vec![0.85, 0.15, 0.05, 0.35, 0.55];
                                emb.resize(512, 0.0);
                                emb
                            };

                            match engine.search(&query_embedding, "machine learning neural networks", 3).await {
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
                                }
                                Err(e) => {
                                    error!("❌ Search failed: {:?}", e);
                                }
                            }
                        });
                    },
                    "Test Hybrid Search"
                }

                div { class: "results",
                    "Check browser console for detailed search results"
                }
            }

        }
    }
}
