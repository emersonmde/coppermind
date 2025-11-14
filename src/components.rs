use crate::embedding::{embed_text_chunks, ChunkEmbeddingResult};

#[cfg(not(target_arch = "wasm32"))]
use crate::embedding::run_embedding;

#[cfg(target_arch = "wasm32")]
use crate::embedding::format_embedding_summary;
use crate::search::types::{Document, DocumentMetadata};
use crate::search::HybridSearchEngine;
use crate::storage::StorageError;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;

#[cfg(target_arch = "wasm32")]
use crate::workers::EmbeddingWorkerClient;

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
enum WorkerStatus {
    Pending,
    Ready(EmbeddingWorkerClient),
    Failed(String),
}

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

// Messages for the embedding coroutine
enum EmbeddingMessage {
    RunTest(String),
}

// Messages for the file processing coroutine
enum FileMessage {
    ProcessFile(String, String), // filename, contents
}

#[component]
pub fn TestControls() -> Element {
    let embedding_result = use_signal(String::new);
    let mut file_processing = use_signal(|| false);
    let mut file_status = use_signal(String::new);
    let mut file_chunks = use_signal(Vec::<ChunkEmbeddingResult>::new);
    let mut file_name = use_signal(String::new);

    #[cfg(target_arch = "wasm32")]
    let worker_state = use_signal(|| WorkerStatus::Pending);

    #[cfg(target_arch = "wasm32")]
    {
        let worker_signal = worker_state.clone();
        use_effect(move || {
            let mut worker_state = worker_signal.clone();
            if matches!(*worker_state.read(), WorkerStatus::Pending) {
                info!("🔧 Initializing embedding worker…");
                match EmbeddingWorkerClient::new() {
                    Ok(client) => worker_state.set(WorkerStatus::Ready(client)),
                    Err(err) => {
                        error!("❌ Embedding worker failed to start: {}", err);
                        worker_state.set(WorkerStatus::Failed(err));
                    }
                }
            };
        });
    }

    #[cfg(target_arch = "wasm32")]
    let worker_status_view: Element = {
        let status_label = match worker_state.read().clone() {
            WorkerStatus::Pending => "Web Worker: starting…".to_string(),
            WorkerStatus::Ready(_) => "Web Worker: ready ✅".to_string(),
            WorkerStatus::Failed(err) => format!("Web Worker error: {}", err),
        };

        rsx! { p { class: "status", "{status_label}" } }
    };

    #[cfg(not(target_arch = "wasm32"))]
    let worker_status_view: Element = {
        rsx! { Fragment {} }
    };

    // Embedding test coroutine - runs in background
    let embedding_task = use_coroutine({
        let mut result = embedding_result;
        #[cfg(target_arch = "wasm32")]
        let worker_state = worker_state.clone();
        move |mut rx: UnboundedReceiver<EmbeddingMessage>| async move {
            while let Some(msg) = rx.next().await {
                match msg {
                    EmbeddingMessage::RunTest(text) => {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            // Desktop: Direct async call (Dioxus already has async runtime)
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

    // File processing coroutine - runs in background
    let file_task = use_coroutine({
        let mut status = file_status;
        let mut chunks = file_chunks;
        let mut name = file_name;
        let mut processing = file_processing;
        #[cfg(target_arch = "wasm32")]
        let worker_state = worker_state.clone();

        move |mut rx: UnboundedReceiver<FileMessage>| async move {
            while let Some(msg) = rx.next().await {
                match msg {
                    FileMessage::ProcessFile(file_label, contents) => {
                        let byte_len = contents.len();
                        info!(
                            "🧮 File size: {} bytes (~{:.2} KB)",
                            byte_len,
                            byte_len as f64 / 1024.0
                        );
                        status.set(format!("Embedding {file_label} ({} bytes)...", byte_len));

                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            // Desktop: Direct async call (Dioxus already has async runtime)
                            match embed_text_chunks(&contents, 512).await {
                                Ok(results) => {
                                    let chunk_count = results.len();
                                    if chunk_count == 0 {
                                        status.set(format!(
                                            "File {file_label} did not produce any tokens."
                                        ));
                                    } else {
                                        for chunk in results.iter() {
                                            info!(
                                                "📦 Chunk {} embedded ({} tokens)",
                                                chunk.chunk_index, chunk.token_count
                                            );
                                        }
                                        status.set(format!(
                                            "✓ Embedded {chunk_count} chunks from {file_label}"
                                        ));
                                    }
                                    chunks.set(results);
                                }
                                Err(e) => {
                                    error!("❌ Embedding failed: {e}");
                                    status.set(format!("Embedding failed: {e}"));
                                    name.set(String::new());
                                }
                            }
                        }

                        #[cfg(target_arch = "wasm32")]
                        {
                            // Web: Use embedding worker for non-blocking processing
                            let worker_snapshot = worker_state.read().clone();
                            match worker_snapshot {
                                WorkerStatus::Pending => {
                                    status.set("Embedding worker is starting… please retry.".into());
                                    processing.set(false);
                                }
                                WorkerStatus::Failed(err) => {
                                    status.set(format!("Embedding worker unavailable: {}", err));
                                    name.set(String::new());
                                    processing.set(false);
                                }
                                WorkerStatus::Ready(client) => {
                                    // Split text into chunks (roughly 2000 chars each, will be re-chunked by worker)
                                    let chunk_size = 2000;
                                    let text_chunks: Vec<String> = contents
                                        .chars()
                                        .collect::<Vec<_>>()
                                        .chunks(chunk_size)
                                        .map(|chunk| chunk.iter().collect())
                                        .collect();

                                    let total_chunks = text_chunks.len();
                                    info!("📄 Split file into {} text chunks", total_chunks);

                                    let mut results = Vec::new();

                                    for (idx, chunk_text) in text_chunks.into_iter().enumerate() {
                                        status.set(format!(
                                            "Embedding chunk {}/{} from {}...",
                                            idx + 1,
                                            total_chunks,
                                            file_label
                                        ));

                                        match client.embed(chunk_text).await {
                                            Ok(computation) => {
                                                info!(
                                                    "📦 Chunk {} embedded ({} tokens)",
                                                    idx,
                                                    computation.token_count
                                                );
                                                results.push(ChunkEmbeddingResult {
                                                    chunk_index: idx,
                                                    token_count: computation.token_count,
                                                    embedding: computation.embedding,
                                                });
                                            }
                                            Err(e) => {
                                                error!("❌ Failed to embed chunk {}: {}", idx, e);
                                                status.set(format!(
                                                    "Failed to embed chunk {}/{}: {}",
                                                    idx + 1,
                                                    total_chunks,
                                                    e
                                                ));
                                                name.set(String::new());
                                                processing.set(false);
                                                return;
                                            }
                                        }
                                    }

                                    let chunk_count = results.len();
                                    if chunk_count == 0 {
                                        status.set(format!(
                                            "File {file_label} did not produce any tokens."
                                        ));
                                    } else {
                                        status.set(format!(
                                            "✓ Embedded {chunk_count} chunks from {file_label}"
                                        ));
                                    }
                                    chunks.set(results);
                                }
                            }
                        }
                        processing.set(false);
                    }
                }
            }
        }
    });

    rsx! {
        div { class: "test-controls",
            div { class: "test-section",
                h2 { "Text Embedding Test" }
                p { class: "description", "JinaBERT text embeddings using Candle ML framework (background processing)" }

                {worker_status_view}

                button {
                    class: "btn-primary",
                    onclick: move |_| {
                        let test_text = "This is a test sentence for text embedding.";
                        embedding_task.send(EmbeddingMessage::RunTest(test_text.to_string()));
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
                p { class: "description", "Upload a .txt file, chunk it in Rust, and embed each chunk (background processing)" }

                input {
                    r#type: "file",
                    accept: ".txt",
                    multiple: false,
                    onchange: move |evt: dioxus::events::FormEvent| {
                        if file_processing() {
                            file_status.set("Already processing a file. Please wait for it to finish.".into());
                            return;
                        }

                        let files = evt.files();
                        if let Some(file) = files.first() {
                            file_name.set(file.name().clone());
                            let file_label = file.name().clone();
                            let file_data = file.clone();
                            let task = file_task;

                            spawn(async move {
                                file_processing.set(true);
                                file_chunks.set(Vec::new());
                                info!("📂 Selected file: {file_label}");
                                file_status.set(format!("Reading {file_label}..."));

                                match file_data.read_string().await {
                                    Ok(contents) => {
                                        // Send to background coroutine for processing
                                        task.send(FileMessage::ProcessFile(file_label, contents));
                                    }
                                    Err(e) => {
                                        error!("❌ Failed to read {file_label}: {e}");
                                        file_status.set(format!("Failed to read {file_label}: {e}"));
                                        file_name.set(String::new());
                                        file_processing.set(false);
                                    }
                                }
                            });
                        } else {
                            file_status.set("No file selected.".into());
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
