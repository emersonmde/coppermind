use dioxus::prelude::*;
use crate::cpu::spawn_worker;
use crate::wgpu::test_webgpu;
use crate::embedding::run_embedding;

#[component]
pub fn TestControls() -> Element {
    let mut cpu_running = use_signal(|| false);
    let mut cpu_results = use_signal(Vec::<String>::new);
    let gpu_result = use_signal(|| String::new());
    let embedding_result = use_signal(|| String::new());

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
                                    let mut results = cpu_results.clone();
                                    let mut running = cpu_running.clone();
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
                        let mut result = gpu_result.clone();
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
                        let mut result = embedding_result.clone();
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

        }
    }
}
