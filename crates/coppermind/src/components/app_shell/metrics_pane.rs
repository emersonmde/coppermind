use dioxus::prelude::*;

use crate::components::{
    calculate_engine_metrics, calculate_live_metrics, use_batches, use_search_engine,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::{get_scheduler, is_scheduler_initialized};
use crate::metrics::global_metrics;

/// Collapsible metrics panel showing engine statistics
#[component]
pub fn MetricsPane(collapsed: ReadSignal<bool>) -> Element {
    let panel_class = if collapsed() {
        "cm-metrics-panel cm-metrics-panel--collapsed"
    } else {
        "cm-metrics-panel"
    };

    // Get batches from context and calculate metrics
    let batches = use_batches();
    let batches_list = batches.read();

    // Get search engine for index-specific metrics
    let search_engine_signal = use_search_engine();
    let search_engine_arc = search_engine_signal.read().clone();

    // Get index-specific metrics from search engine
    let (vector_chunks, vector_tokens, vector_avg_tokens) =
        if let Some(engine_lock) = &search_engine_arc {
            // Try to get lock without blocking (UI thread)
            if let Some(engine) = engine_lock.try_lock() {
                // Use sync version for UI - can't await in component render
                engine.get_index_metrics_sync()
            } else {
                // Lock held, show zeros temporarily
                (0, 0, 0.0)
            }
        } else {
            (0, 0, 0.0)
        };

    // GPU Scheduler stats (desktop only)
    #[cfg(not(target_arch = "wasm32"))]
    let (gpu_queue_depth, gpu_requests_completed, gpu_models_loaded) = {
        if is_scheduler_initialized() {
            if let Ok(scheduler) = get_scheduler() {
                let stats = scheduler.stats();
                (
                    stats.queue_depth,
                    stats.requests_completed,
                    stats.models_loaded,
                )
            } else {
                (0, 0, 0)
            }
        } else {
            (0, 0, 0)
        }
    };
    #[cfg(target_arch = "wasm32")]
    let (gpu_queue_depth, gpu_requests_completed, gpu_models_loaded) = (0usize, 0u64, 0usize);

    // Memory estimate: chunks × embedding_dim × 4 bytes (f32)
    // Plus ~20% overhead for HNSW graph structure
    let embedding_dim = 512usize;
    let vector_memory_bytes = vector_chunks * embedding_dim * 4;
    let estimated_memory_bytes = (vector_memory_bytes as f64 * 1.2) as usize;
    let memory_display = if estimated_memory_bytes > 1_000_000 {
        format!("{:.1} MB", estimated_memory_bytes as f64 / 1_000_000.0)
    } else if estimated_memory_bytes > 1_000 {
        format!("{:.1} KB", estimated_memory_bytes as f64 / 1_000.0)
    } else {
        format!("{} B", estimated_memory_bytes)
    };

    // Format values for display
    let formatted_avg_tokens = format!("{:.0}", vector_avg_tokens);

    // Calculate aggregate engine metrics from all completed batches (for historical tracking)
    let engine_metrics = calculate_engine_metrics(&batches_list);
    let batch_total_docs = engine_metrics.total_docs;

    // Calculate live/historical indexing metrics
    let live_metrics = calculate_live_metrics(&batches_list);

    // Extract metrics and state
    let (tokens_per_sec, chunks_per_sec, avg_chunk_time_ms, state_mode) =
        if let Some(metrics) = &live_metrics {
            if metrics.is_live {
                (
                    metrics.tokens_per_sec,
                    metrics.chunks_per_sec,
                    metrics.avg_chunk_time_ms,
                    "live",
                )
            } else {
                (
                    metrics.tokens_per_sec,
                    metrics.chunks_per_sec,
                    metrics.avg_chunk_time_ms,
                    "historical",
                )
            }
        } else {
            // No batches indexed yet - show zeros with idle state
            (0.0, 0.0, 0.0, "idle")
        };

    // Format values for display
    let formatted_tokens_per_sec = format!("{:.1}", tokens_per_sec);
    let formatted_chunks_per_sec = format!("{:.1}", chunks_per_sec);
    let formatted_chunk_time = format!("{:.0}", avg_chunk_time_ms);

    // State indicator and caption based on mode
    let (state_class, state_label, state_caption) = match state_mode {
        "live" => (
            "cm-metrics-state cm-metrics-state--live",
            "Live",
            "Vector engines processing…",
        ),
        "historical" => (
            "cm-metrics-state cm-metrics-state--historical",
            "Last Batch",
            "Most recent indexing performance",
        ),
        _ => (
            "cm-metrics-state cm-metrics-state--idle",
            "Idle",
            "Waiting for first batch…",
        ),
    };

    // Get rolling average metrics (60-second window)
    let perf_snapshot = global_metrics().snapshot();

    // Format rolling averages for display
    let chunking_avg = perf_snapshot
        .chunking_avg_ms
        .map(|v| format!("{:.1}", v))
        .unwrap_or_else(|| "-".to_string());
    let tokenization_avg = perf_snapshot
        .tokenization_avg_ms
        .map(|v| format!("{:.1}", v))
        .unwrap_or_else(|| "-".to_string());
    let embedding_avg = perf_snapshot
        .embedding_avg_ms
        .map(|v| format!("{:.1}", v))
        .unwrap_or_else(|| "-".to_string());
    let hnsw_avg = perf_snapshot
        .hnsw_avg_ms
        .map(|v| format!("{:.2}", v))
        .unwrap_or_else(|| "-".to_string());
    let bm25_avg = perf_snapshot
        .bm25_avg_ms
        .map(|v| format!("{:.2}", v))
        .unwrap_or_else(|| "-".to_string());

    // Throughput metrics
    let embedding_throughput = format!("{:.1}", perf_snapshot.embedding_throughput);

    // Check if we have any rolling metrics data
    let has_rolling_data = perf_snapshot.chunking_count > 0
        || perf_snapshot.tokenization_count > 0
        || perf_snapshot.embedding_count > 0;

    rsx! {
        section {
            class: panel_class,
            "data-role": "metrics-panel",
            header { class: "cm-metrics-header",
                h2 { class: "cm-metrics-title", "Engine Metrics" }
            }

            // Two-column layout for Aggregate Stats and GPU Scheduler
            div { class: "cm-metrics-columns",
                // Left column: Aggregate Statistics
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "Aggregate" }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Files" }
                            div { class: "cm-metric-value", "{batch_total_docs}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Chunks" }
                            div { class: "cm-metric-value", "{vector_chunks}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Tokens" }
                            div { class: "cm-metric-value", "{vector_tokens}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Avg Tok/Chunk" }
                            div { class: "cm-metric-value", "{formatted_avg_tokens}" }
                        }
                    }
                }

                // Right column: GPU Scheduler
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "GPU Scheduler" }
                        if gpu_queue_depth > 0 {
                            span { class: "cm-metrics-state cm-metrics-state--live",
                                span { class: "cm-metrics-state-indicator" }
                                "Active"
                            }
                        }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Queue Depth" }
                            div { class: "cm-metric-value", "{gpu_queue_depth}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Completed" }
                            div { class: "cm-metric-value", "{gpu_requests_completed}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Models" }
                            div { class: "cm-metric-value", "{gpu_models_loaded}" }
                        }
                    }
                }
            }

            // Per-index metrics: HNSW and BM25
            div { class: "cm-metrics-separator cm-metrics-separator--tight" }
            div { class: "cm-metrics-columns",
                // HNSW Vector Index
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "HNSW Vector Index" }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Chunks" }
                            div { class: "cm-metric-value", "{vector_chunks}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Tokens" }
                            div { class: "cm-metric-value", "{vector_tokens}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Memory" }
                            div { class: "cm-metric-value", "{memory_display}" }
                        }
                    }
                }

                // BM25 Keyword Index
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "BM25 Keyword Index" }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Chunks" }
                            div { class: "cm-metric-value", "{vector_chunks}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Tokens" }
                            div { class: "cm-metric-value", "{vector_tokens}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Avg Tok/Chunk" }
                            div { class: "cm-metric-value", "{formatted_avg_tokens}" }
                        }
                    }
                }
            }

            // Performance metrics (live/historical indexing stats)
            div { class: "cm-metrics-separator cm-metrics-separator--tight" }
            div { class: "cm-metrics-section-header",
                h3 { class: "cm-metrics-section-title", "Indexing Performance" }
                span { class: state_class,
                    span { class: "cm-metrics-state-indicator" }
                    "{state_label}"
                }
            }
            div { class: "cm-metrics-grid cm-metrics-grid--compact",
                div { class: "cm-metric-card cm-metric-card--compact cm-metric-card--rate",
                    div { class: "cm-metric-label", "Tokens / sec" }
                    div { class: "cm-metric-value", "{formatted_tokens_per_sec}" }
                }
                div { class: "cm-metric-card cm-metric-card--compact cm-metric-card--rate",
                    div { class: "cm-metric-label", "Chunks / sec" }
                    div { class: "cm-metric-value", "{formatted_chunks_per_sec}" }
                }
                div { class: "cm-metric-card cm-metric-card--compact cm-metric-card--rate",
                    div { class: "cm-metric-label", "Avg Time / Chunk" }
                    div { class: "cm-metric-value", "{formatted_chunk_time}", span { class: "cm-metric-unit", "ms" } }
                }
            }
            p { class: "cm-metrics-caption",
                "{state_caption}"
            }

            // Rolling averages (60-second window)
            if has_rolling_data {
                div { class: "cm-metrics-separator cm-metrics-separator--tight" }
                div { class: "cm-metrics-section-header",
                    h3 { class: "cm-metrics-section-title", "Operation Timings" }
                    span { class: "cm-metrics-state cm-metrics-state--historical",
                        "60s avg"
                    }
                }
                div { class: "cm-metrics-columns",
                    // Processing Pipeline
                    div { class: "cm-metrics-column",
                        div { class: "cm-metrics-grid cm-metrics-grid--compact",
                            div { class: "cm-metric-card cm-metric-card--compact",
                                div { class: "cm-metric-label", "Chunking" }
                                div { class: "cm-metric-value", "{chunking_avg}", span { class: "cm-metric-unit", "ms" } }
                            }
                            div { class: "cm-metric-card cm-metric-card--compact",
                                div { class: "cm-metric-label", "Tokenize" }
                                div { class: "cm-metric-value", "{tokenization_avg}", span { class: "cm-metric-unit", "ms" } }
                            }
                            div { class: "cm-metric-card cm-metric-card--compact",
                                div { class: "cm-metric-label", "Embed" }
                                div { class: "cm-metric-value", "{embedding_avg}", span { class: "cm-metric-unit", "ms" } }
                            }
                        }
                    }
                    // Indexing
                    div { class: "cm-metrics-column",
                        div { class: "cm-metrics-grid cm-metrics-grid--compact",
                            div { class: "cm-metric-card cm-metric-card--compact",
                                div { class: "cm-metric-label", "HNSW Insert" }
                                div { class: "cm-metric-value", "{hnsw_avg}", span { class: "cm-metric-unit", "ms" } }
                            }
                            div { class: "cm-metric-card cm-metric-card--compact",
                                div { class: "cm-metric-label", "BM25 Insert" }
                                div { class: "cm-metric-value", "{bm25_avg}", span { class: "cm-metric-unit", "ms" } }
                            }
                            div { class: "cm-metric-card cm-metric-card--compact",
                                div { class: "cm-metric-label", "Embed/sec" }
                                div { class: "cm-metric-value", "{embedding_throughput}" }
                            }
                        }
                    }
                }
            }
        }
    }
}
