use crate::cpu::spawn_worker;
use crate::embedding::{embed_text_chunks_streaming, run_embedding, ChunkEmbeddingResult};
use crate::wgpu::test_webgpu;
use dioxus::prelude::*;
use futures_util::StreamExt;

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
                p { class: "description", "JinaBert text embeddings using Candle ML framework" }

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
                                    web_sys::console::log_1(&format!("📂 Selected file: {file_label}").into());
                                    status.set(format!("Reading {file_label}..."));
                                    match engine.read_file_to_string(&file_label).await {
                                        Some(contents) => {
                                            let byte_len = contents.len();
                                            web_sys::console::log_1(&format!(
                                                "🧮 File size: {} bytes (~{:.2} KB)",
                                                byte_len,
                                                byte_len as f64 / 1024.0
                                            ).into());
                                            status.set(format!("Embedding {file_label} ({} bytes)...", byte_len));

                                            // Use streaming API to process chunks incrementally
                                            match embed_text_chunks_streaming(&contents, 512).await {
                                                Ok(mut receiver) => {
                                                    let mut processed_chunks = 0;

                                                    // Process chunks as they arrive
                                                    while let Some(chunk_result) = receiver.next().await {
                                                        match chunk_result {
                                                            Ok(chunk) => {
                                                                processed_chunks += 1;

                                                                // Update status with progress
                                                                status.set(format!(
                                                                    "Embedding {file_label}: chunk {} ({} tokens)",
                                                                    processed_chunks,
                                                                    chunk.token_count
                                                                ));

                                                                // Add chunk directly to signal's vector
                                                                chunks.write().push(chunk);
                                                            }
                                                            Err(e) => {
                                                                web_sys::console::error_1(&format!("❌ Chunk embedding failed: {e}").into());
                                                                status.set(format!("Embedding failed: {e}"));
                                                                selected_name.set(String::new());
                                                                break;
                                                            }
                                                        }
                                                    }

                                                    // Final status update
                                                    if processed_chunks == 0 {
                                                        status.set(format!("File {file_label} did not produce any tokens."));
                                                    } else {
                                                        status.set(format!("✓ Embedded {processed_chunks} chunks from {file_label}"));
                                                    }
                                                }
                                                Err(e) => {
                                                    web_sys::console::error_1(&format!("❌ Embedding failed: {e}").into());
                                                    status.set(format!("Embedding failed: {e}"));
                                                    selected_name.set(String::new());
                                                }
                                            }
                                        }
                                        None => {
                                            web_sys::console::error_1(&format!("❌ Failed to read {file_label}").into());
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

        }
    }
}
