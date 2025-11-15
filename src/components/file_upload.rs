use crate::embedding::ChunkEmbeddingResult;
use crate::search::types::{Document, DocumentMetadata};
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;
use instant::Instant;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(target_arch = "wasm32")]
use web_sys;

#[cfg(not(target_arch = "wasm32"))]
use crate::embedding::embed_text_chunks;

#[cfg(target_arch = "wasm32")]
use super::hero::{use_worker_state, WorkerStatus};

use super::{use_search_engine, use_search_engine_status, SearchEngineStatus};

// Messages for file processing coroutine
enum FileMessage {
    ProcessFiles(Vec<(String, String)>), // Vec of (filename, contents)
}

// Helper function to detect if content is binary (not text)
// Standard approach: check for null bytes which appear in binary but never in text
fn is_likely_binary(content: &str) -> bool {
    // Since we successfully read as String, it's valid UTF-8
    // Binary files contain null bytes which don't appear in text files
    content.contains('\0')
}

// Processing metrics for display
#[derive(Clone, Default)]
#[allow(dead_code)] // Fields are accessed via signal reads in rsx! macro
struct ProcessingMetrics {
    total_tokens: usize,
    total_chunks: usize,
    elapsed_secs: f64,
    tokens_per_sec: f64,
    chunks_per_sec: f64,
    avg_time_per_chunk_ms: f64,
}

// Progress tracking for visual feedback
#[derive(Clone, Default)]
struct ProcessingProgress {
    current: usize,
    total: usize,
    percentage: f64,
}

#[component]
pub fn FileUpload() -> Element {
    let mut file_processing = use_signal(|| false);
    let mut file_status = use_signal(String::new);
    let mut file_chunks = use_signal(Vec::<ChunkEmbeddingResult>::new);
    let mut file_name = use_signal(String::new);
    let metrics = use_signal(ProcessingMetrics::default);
    let progress = use_signal(ProcessingProgress::default);
    let mut warnings = use_signal(Vec::<String>::new);

    #[cfg(target_arch = "wasm32")]
    let worker_state = use_worker_state();

    let search_engine = use_search_engine();
    let mut search_engine_status = use_search_engine_status();

    // Set webkitdirectory attribute on the directory input (web only)
    #[cfg(target_arch = "wasm32")]
    use_effect(move || {
        use wasm_bindgen::JsCast;
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                if let Some(input) = document.get_element_by_id("file-input-directory") {
                    if let Some(input_element) = input.dyn_ref::<web_sys::HtmlInputElement>() {
                        let _ = input_element.set_attribute("webkitdirectory", "");
                        let _ = input_element.set_attribute("directory", "");
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
        let mut metrics_signal = metrics;
        let mut progress_signal = progress;
        let mut warnings_signal = warnings;
        let engine = search_engine;
        let mut engine_status = search_engine_status;
        #[cfg(target_arch = "wasm32")]
        let worker_state = worker_state;

        move |mut rx: UnboundedReceiver<FileMessage>| async move {
            while let Some(msg) = rx.next().await {
                match msg {
                    FileMessage::ProcessFiles(files) => {
                        warnings_signal.set(Vec::new()); // Clear previous warnings
                        let mut processed_files = 0;
                        let total_files = files.len();

                        for (file_label, contents) in files {
                            // Check if file is binary
                            if is_likely_binary(&contents) {
                                let warning = format!("⚠️ Skipped '{}': appears to be a binary file", file_label);
                                info!("{}", warning);
                                warnings_signal.write().push(warning);
                                continue;
                            }

                            processed_files += 1;
                            name.set(format!("{} ({}/{})", file_label, processed_files, total_files));
                        let start_time = Instant::now();
                        let byte_len = contents.len();
                        info!(
                            "🧮 File size: {} bytes (~{:.2} KB)",
                            byte_len,
                            byte_len as f64 / 1024.0
                        );
                        status.set(format!("Embedding {file_label} ({} bytes)...", byte_len));

                        // Reset metrics and progress
                        metrics_signal.set(ProcessingMetrics::default());
                        progress_signal.set(ProcessingProgress::default());

                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            // Desktop: Direct async call (Dioxus already has async runtime)
                            match embed_text_chunks(&contents, 512).await {
                                Ok(results) => {
                                    let elapsed = start_time.elapsed();
                                    let chunk_count = results.len();
                                    let total_tokens: usize =
                                        results.iter().map(|c| c.token_count).sum();

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

                                        // Index chunks in search engine
                                        status.set("Indexing chunks in search engine...".into());
                                        let mut indexed_count = 0;
                                        let engine_arc = engine.read().clone();
                                        if let Some(engine_lock) = engine_arc {
                                            // Split original text into chunks for indexing
                                            let chunk_size = 2000;
                                            let chunks_text: Vec<String> = contents
                                                .chars()
                                                .collect::<Vec<_>>()
                                                .chunks(chunk_size)
                                                .map(|chunk| chunk.iter().collect())
                                                .collect();

                                            // Add all documents in batch (deferred index rebuild)
                                            {
                                                let mut search_engine = engine_lock.lock().await;
                                                for (idx, chunk_result) in results.iter().enumerate() {
                                                    if let Some(chunk_text) = chunks_text.get(idx) {
                                                        let doc = Document {
                                                            text: chunk_text.clone(),
                                                            metadata: DocumentMetadata {
                                                                filename: Some(format!(
                                                                    "{} (chunk {})",
                                                                    file_label,
                                                                    idx + 1
                                                                )),
                                                                source: Some(file_label.clone()),
                                                                created_at: std::time::SystemTime::now(
                                                                )
                                                                .duration_since(std::time::UNIX_EPOCH)
                                                                .unwrap()
                                                                .as_secs(),
                                                            },
                                                        };

                                                        match search_engine
                                                            .add_document_deferred(
                                                                doc,
                                                                chunk_result.embedding.clone(),
                                                            )
                                                            .await
                                                        {
                                                            Ok(_) => indexed_count += 1,
                                                            Err(e) => {
                                                                error!(
                                                                    "❌ Failed to index chunk {}: {:?}",
                                                                    idx, e
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                            } // Release lock before rebuilding index

                                            // Rebuild vector index (CPU-intensive, runs in thread pool on desktop)
                                            status.set("Building search index...".into());
                                            {
                                                let mut search_engine = engine_lock.lock().await;
                                                search_engine.rebuild_vector_index().await;
                                            }

                                            info!(
                                                "✅ Indexed {} chunks in search engine",
                                                indexed_count
                                            );

                                            // Update search engine status with new document count
                                            let doc_count = {
                                                let search_engine = engine_lock.lock().await;
                                                search_engine.len()
                                            };
                                            engine_status.set(SearchEngineStatus::Ready { doc_count });
                                        }

                                        status.set(format!(
                                            "✓ Indexed {chunk_count} chunks from {file_label}"
                                        ));

                                        // Calculate metrics
                                        let elapsed_secs = elapsed.as_secs_f64();
                                        metrics_signal.set(ProcessingMetrics {
                                            total_tokens,
                                            total_chunks: chunk_count,
                                            elapsed_secs,
                                            tokens_per_sec: total_tokens as f64 / elapsed_secs,
                                            chunks_per_sec: chunk_count as f64 / elapsed_secs,
                                            avg_time_per_chunk_ms: (elapsed_secs * 1000.0)
                                                / chunk_count as f64,
                                        });

                                        // Set progress to 100%
                                        progress_signal.set(ProcessingProgress {
                                            current: chunk_count,
                                            total: chunk_count,
                                            percentage: 100.0,
                                        });
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
                                    status
                                        .set("Embedding worker is starting… please retry.".into());
                                    processing.set(false);
                                }
                                WorkerStatus::Failed(err) => {
                                    status.set(format!("Embedding worker unavailable: {}", err));
                                    name.set(String::new());
                                    processing.set(false);
                                }
                                WorkerStatus::Ready(client) => {
                                    // Split text into chunks (roughly 2000 chars each)
                                    let chunk_size = 2000;
                                    let text_chunks: Vec<String> = contents
                                        .chars()
                                        .collect::<Vec<_>>()
                                        .chunks(chunk_size)
                                        .map(|chunk| chunk.iter().collect())
                                        .collect();

                                    let total_chunks = text_chunks.len();
                                    info!("📄 Split file into {} text chunks", total_chunks);

                                    // Initialize progress
                                    progress_signal.set(ProcessingProgress {
                                        current: 0,
                                        total: total_chunks,
                                        percentage: 0.0,
                                    });

                                    let mut results = Vec::new();

                                    for (idx, chunk_text) in text_chunks.iter().enumerate() {
                                        status.set(format!(
                                            "Embedding chunk {}/{} from {}...",
                                            idx + 1,
                                            total_chunks,
                                            file_label
                                        ));

                                        match client.embed(chunk_text.clone()).await {
                                            Ok(computation) => {
                                                info!(
                                                    "📦 Chunk {} embedded ({} tokens)",
                                                    idx, computation.token_count
                                                );
                                                results.push(ChunkEmbeddingResult {
                                                    chunk_index: idx,
                                                    token_count: computation.token_count,
                                                    embedding: computation.embedding,
                                                });

                                                // Update metrics and progress in real-time
                                                let elapsed = start_time.elapsed();
                                                let elapsed_secs = elapsed.as_secs_f64();
                                                let total_tokens: usize =
                                                    results.iter().map(|c| c.token_count).sum();
                                                let chunk_count = results.len();

                                                metrics_signal.set(ProcessingMetrics {
                                                    total_tokens,
                                                    total_chunks: chunk_count,
                                                    elapsed_secs,
                                                    tokens_per_sec: total_tokens as f64
                                                        / elapsed_secs,
                                                    chunks_per_sec: chunk_count as f64
                                                        / elapsed_secs,
                                                    avg_time_per_chunk_ms: (elapsed_secs * 1000.0)
                                                        / chunk_count as f64,
                                                });

                                                // Update progress
                                                let percentage = (chunk_count as f64
                                                    / total_chunks as f64)
                                                    * 100.0;
                                                progress_signal.set(ProcessingProgress {
                                                    current: chunk_count,
                                                    total: total_chunks,
                                                    percentage,
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
                                        // Index chunks in search engine
                                        status.set("Indexing chunks in search engine...".into());
                                        let mut indexed_count = 0;
                                        let engine_arc = engine.read().clone();
                                        if let Some(engine_lock) = engine_arc {
                                            // Add all documents in batch (deferred index rebuild)
                                            {
                                                let mut search_engine = engine_lock.lock().await;
                                                for (idx, chunk_result) in results.iter().enumerate() {
                                                    // Get the corresponding text chunk
                                                    if let Some(chunk_text) = text_chunks.get(idx) {
                                                        let doc = Document {
                                                            text: chunk_text.clone(),
                                                            metadata: DocumentMetadata {
                                                                filename: Some(format!(
                                                                    "{} (chunk {})",
                                                                    file_label,
                                                                    idx + 1
                                                                )),
                                                                source: Some(file_label.clone()),
                                                                created_at: instant::SystemTime::now()
                                                                    .duration_since(
                                                                        instant::SystemTime::UNIX_EPOCH,
                                                                    )
                                                                    .unwrap()
                                                                    .as_secs(),
                                                            },
                                                        };

                                                        match search_engine
                                                            .add_document_deferred(
                                                                doc,
                                                                chunk_result.embedding.clone(),
                                                            )
                                                            .await
                                                        {
                                                            Ok(_) => indexed_count += 1,
                                                            Err(e) => {
                                                                error!(
                                                                    "❌ Failed to index chunk {}: {:?}",
                                                                    idx, e
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                            } // Release lock before rebuilding index

                                            // Rebuild vector index
                                            status.set("Building search index...".into());
                                            {
                                                let mut search_engine = engine_lock.lock().await;
                                                search_engine.rebuild_vector_index().await;
                                            }

                                            info!(
                                                "✅ Indexed {} chunks in search engine",
                                                indexed_count
                                            );

                                            // Update search engine status with new document count
                                            let doc_count = {
                                                let search_engine = engine_lock.lock().await;
                                                search_engine.len()
                                            };
                                            engine_status.set(SearchEngineStatus::Ready { doc_count });
                                        }

                                        status.set(format!(
                                            "✓ Indexed {chunk_count} chunks from {file_label}"
                                        ));
                                    }
                                    chunks.set(results);
                                }
                            }
                        }
                        } // End for loop over files

                        processing.set(false);
                    }
                }
            }
        }
    });

    rsx! {
        div { class: "main-section",
            div { class: "section-header",
                h2 { "Index Your Documents" }
                p { class: "section-description",
                    "Upload text files to build your local searchable knowledge base. All processing happens on your device."
                }
            }

            // File upload area
            div { class: "upload-zone",
                div { class: "upload-content",
                    svg {
                        class: "upload-icon",
                        xmlns: "http://www.w3.org/2000/svg",
                        width: "64",
                        height: "64",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "1.5",
                        path { d: "M9 13h6m-3-3v6m-9 1V7a2 2 0 0 1 2-2h6l2 2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" }
                    }

                    if file_processing() {
                        div { class: "upload-processing",
                            div { class: "spinner" }
                            p { class: "upload-text-primary", "Processing {file_name.read()}..." }
                            p { class: "upload-text-secondary", "{file_status.read()}" }

                            // Progress bar
                            if progress.read().total > 0 {
                                div { class: "progress-container",
                                    div { class: "progress-info",
                                        span { class: "progress-label",
                                            "Chunk {progress.read().current}/{progress.read().total}"
                                        }
                                        span { class: "progress-percentage",
                                            "{progress.read().percentage:.1}%"
                                        }
                                    }
                                    div { class: "progress-bar-bg",
                                        div {
                                            class: "progress-bar-fill",
                                            style: "width: {progress.read().percentage}%"
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        p { class: "upload-text-primary", "Drop files or folder here" }
                        p { class: "upload-text-secondary",
                            "Supports multiple files • Any text format • Processed locally"
                        }
                        div { class: "upload-buttons",
                            label {
                                class: "upload-mode-button",
                                r#for: "file-input-files",
                                "📄 Select Files"
                            }
                            label {
                                class: "upload-mode-button",
                                r#for: "file-input-directory",
                                "📁 Select Folder"
                            }
                        }
                    }

                    // Hidden file input for selecting multiple files
                    input {
                        id: "file-input-files",
                        class: "file-input-hidden",
                        r#type: "file",
                        accept: "*/*",
                        multiple: true,
                        disabled: file_processing(),
                        onchange: move |evt: dioxus::events::FormEvent| {
                            if file_processing() {
                                file_status
                                    .set(
                                        "Already processing files. Please wait for it to finish.".into(),
                                    );
                                return;
                            }

                            let files = evt.files();
                            if files.is_empty() {
                                file_status.set("No files selected.".into());
                                file_chunks.set(Vec::new());
                                file_name.set(String::new());
                                warnings.set(Vec::new());
                                return;
                            }

                            let file_count = files.len();
                            let task = file_task;

                            spawn(async move {
                                file_processing.set(true);
                                file_chunks.set(Vec::new());
                                warnings.set(Vec::new());
                                info!("📂 Selected {} file(s)", file_count);
                                file_status.set(format!("Reading {} file(s)...", file_count));

                                let mut file_contents = Vec::new();

                                // Read all files
                                for file in files.into_iter() {
                                    let file_label = file.name().clone();
                                    match file.read_string().await {
                                        Ok(contents) => {
                                            file_contents.push((file_label, contents));
                                        }
                                        Err(e) => {
                                            error!("❌ Failed to read {}: {}", file_label, e);
                                            let warning = format!("⚠️ Failed to read '{}': {}", file_label, e);
                                            warnings.write().push(warning);
                                        }
                                    }
                                }

                                if file_contents.is_empty() {
                                    file_status.set("All files failed to read.".into());
                                    file_processing.set(false);
                                } else {
                                    // Send to background coroutine for processing
                                    task.send(FileMessage::ProcessFiles(file_contents));
                                }
                            });
                        }
                    }

                    // Hidden directory input for selecting folders (recursive)
                    // Note: We set webkitdirectory via script since Dioxus doesn't support it directly
                    input {
                        id: "file-input-directory",
                        class: "file-input-hidden",
                        r#type: "file",
                        multiple: true,
                        disabled: file_processing(),
                        onchange: move |evt: dioxus::events::FormEvent| {
                            if file_processing() {
                                file_status
                                    .set(
                                        "Already processing files. Please wait for it to finish.".into(),
                                    );
                                return;
                            }

                            let files = evt.files();
                            if files.is_empty() {
                                file_status.set("No files selected.".into());
                                file_chunks.set(Vec::new());
                                file_name.set(String::new());
                                warnings.set(Vec::new());
                                return;
                            }

                            let file_count = files.len();
                            let task = file_task;

                            spawn(async move {
                                file_processing.set(true);
                                file_chunks.set(Vec::new());
                                warnings.set(Vec::new());
                                info!("📂 Selected folder with {} file(s)", file_count);
                                file_status.set(format!("Reading {} file(s) from folder...", file_count));

                                let mut file_contents = Vec::new();

                                // Read all files from directory
                                for file in files.into_iter() {
                                    let file_label = file.name().clone();
                                    match file.read_string().await {
                                        Ok(contents) => {
                                            file_contents.push((file_label, contents));
                                        }
                                        Err(e) => {
                                            error!("❌ Failed to read {}: {}", file_label, e);
                                            let warning = format!("⚠️ Failed to read '{}': {}", file_label, e);
                                            warnings.write().push(warning);
                                        }
                                    }
                                }

                                if file_contents.is_empty() {
                                    file_status.set("All files failed to read.".into());
                                    file_processing.set(false);
                                } else {
                                    // Send to background coroutine for processing
                                    task.send(FileMessage::ProcessFiles(file_contents));
                                }
                            });
                        }
                    }
                }
            }

            // Processing metrics (show during or after processing)
            if file_processing() || (metrics.read().total_chunks > 0 && !file_chunks.read().is_empty()) {
                div { class: "metrics-card",
                    h3 { class: "metrics-title",
                        if file_processing() {
                            "Processing Metrics (Live)"
                        } else {
                            "Processing Metrics"
                        }
                    }

                    div { class: "metrics-grid",
                        div { class: "metric-item",
                            div { class: "metric-icon", "⚡" }
                            div { class: "metric-content",
                                div { class: "metric-label", "Tokens/Second" }
                                div { class: "metric-value",
                                    "{metrics.read().tokens_per_sec:.1}"
                                }
                            }
                        }

                        div { class: "metric-item",
                            div { class: "metric-icon", "⏱️" }
                            div { class: "metric-content",
                                div { class: "metric-label", "Total Time" }
                                div { class: "metric-value",
                                    "{metrics.read().elapsed_secs:.2}s"
                                }
                            }
                        }

                        div { class: "metric-item",
                            div { class: "metric-icon", "📦" }
                            div { class: "metric-content",
                                div { class: "metric-label", "Chunks/Second" }
                                div { class: "metric-value",
                                    "{metrics.read().chunks_per_sec:.2}"
                                }
                            }
                        }

                        div { class: "metric-item",
                            div { class: "metric-icon", "⏲️" }
                            div { class: "metric-content",
                                div { class: "metric-label", "Avg Time/Chunk" }
                                div { class: "metric-value",
                                    "{metrics.read().avg_time_per_chunk_ms:.0}ms"
                                }
                            }
                        }
                    }
                }
            }

            // Processing results
            if !file_chunks.read().is_empty() {
                div { class: "success-card",
                    div { class: "success-header",
                        svg {
                            class: "success-icon",
                            xmlns: "http://www.w3.org/2000/svg",
                            width: "24",
                            height: "24",
                            view_box: "0 0 24 24",
                            fill: "none",
                            stroke: "currentColor",
                            stroke_width: "2",
                            path { d: "M9 12l2 2 4-4m6 2a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" }
                        }
                        div {
                            h3 { class: "success-title", "File Indexed Successfully" }
                            p { class: "success-subtitle", "{file_name.read()}" }
                        }
                    }
                    div { class: "chunk-stats",
                        div { class: "stat-item",
                            span { class: "stat-label", "Chunks" }
                            span { class: "stat-value", "{file_chunks.read().len()}" }
                        }
                        div { class: "stat-item",
                            span { class: "stat-label", "Total Tokens" }
                            span { class: "stat-value",
                                "{file_chunks.read().iter().map(|c| c.token_count).sum::<usize>()}"
                            }
                        }
                    }
                }
            }

            // Warnings display (binary files skipped, read errors, etc.)
            if !warnings.read().is_empty() {
                div { class: "warning-card",
                    div { class: "warning-header",
                        svg {
                            class: "warning-icon",
                            xmlns: "http://www.w3.org/2000/svg",
                            width: "24",
                            height: "24",
                            view_box: "0 0 24 24",
                            fill: "none",
                            stroke: "currentColor",
                            stroke_width: "2",
                            path { d: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" }
                        }
                        h3 { class: "warning-title", "Warnings" }
                    }
                    div { class: "warning-list",
                        for warning in warnings.read().iter() {
                            div { class: "warning-item", "{warning}" }
                        }
                    }
                }
            }
        }
    }
}
