use dioxus::prelude::*;

use crate::components::{use_batches, use_metrics_refresh, use_search_engine, BatchStatus};
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::{get_scheduler, is_scheduler_initialized, DeviceType};
use coppermind_core::metrics::global_metrics;

/// Embedding dimension for JinaBERT model (used for memory calculations)
const EMBEDDING_DIM: usize = 512;

/// Cached index metrics to avoid flickering when lock is held during indexing
#[derive(Clone, Default)]
struct CachedIndexMetrics {
    /// Number of documents (files, URLs) - user-facing
    docs: usize,
    /// Number of chunks in index
    chunks: usize,
    /// HNSW live entries (excludes tombstones)
    hnsw_live: usize,
    /// Number of tombstoned entries in HNSW
    tombstone_count: usize,
    /// Ratio of tombstones to total (0.0 - 1.0)
    tombstone_ratio: f32,
    /// Whether compaction is recommended (>30% tombstones)
    needs_compaction: bool,
    /// Chunk-level BM25 index size
    bm25_chunk_count: usize,
    /// Document-level BM25 index size (for proper IDF)
    bm25_doc_count: usize,
}

/// Format bytes as human-readable string
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Estimate HNSW memory usage
/// Formula: entries × embedding_dim × 4 bytes (f32) × 1.2 (graph overhead)
fn estimate_hnsw_memory(entries: usize) -> usize {
    let vector_bytes = entries * EMBEDDING_DIM * 4;
    (vector_bytes as f64 * 1.2) as usize
}

/// Estimate BM25 memory usage (rough approximation)
/// BM25 stores term frequencies and document lengths
/// Estimate: ~100 bytes per entry for term data + metadata
fn estimate_bm25_memory(chunk_entries: usize, doc_entries: usize) -> usize {
    (chunk_entries + doc_entries) * 100
}

/// Format an optional f64 metric value with specified decimal precision.
/// Returns "-" if the value is None.
fn format_metric(value: Option<f64>, decimals: usize) -> String {
    match value {
        Some(v) => match decimals {
            0 => format!("{:.0}", v),
            1 => format!("{:.1}", v),
            2 => format!("{:.2}", v),
            _ => format!("{:.1}", v), // Default to 1 decimal
        },
        None => "-".to_string(),
    }
}

/// Format an optional usize metric value.
/// Returns "-" if the value is None.
fn format_count(value: Option<usize>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

/// Collapsible metrics panel showing engine statistics
#[component]
pub fn MetricsPane(collapsed: ReadSignal<bool>) -> Element {
    let panel_class = if collapsed() {
        "cm-metrics-panel cm-metrics-panel--collapsed"
    } else {
        "cm-metrics-panel"
    };

    // Subscribe to metrics refresh signal - triggers re-render when search completes
    let metrics_refresh = use_metrics_refresh();
    let _refresh_tick = metrics_refresh(); // Read to subscribe

    // Get batches from context
    let batches = use_batches();
    let batches_list = batches.read();

    // Check if any batch is currently processing
    let is_indexing = batches_list
        .iter()
        .any(|b| matches!(b.status, BatchStatus::Processing));

    // Get search engine for index-specific metrics
    let search_engine_signal = use_search_engine();
    let search_engine_arc = search_engine_signal.read().clone();

    // Cache for index metrics - only update when lock is available and not mid-indexing
    let mut cached_metrics = use_signal(CachedIndexMetrics::default);

    // Try to update cached metrics from engine (only when not actively indexing)
    if let Some(engine_lock) = &search_engine_arc {
        if let Some(engine) = engine_lock.try_lock() {
            if !is_indexing {
                let (docs, chunks, _tokens, _avg) = engine.get_index_metrics_sync();
                let (tombstones, _total, ratio) = engine.compaction_stats();
                let needs_compact = engine.needs_compaction();

                let hnsw_size = engine.vector_index_len();
                let bm25_chunk_size = engine.keyword_index_len();
                let bm25_doc_size = engine.document_keyword_index_len();

                cached_metrics.set(CachedIndexMetrics {
                    docs,
                    chunks,
                    hnsw_live: hnsw_size,
                    tombstone_count: tombstones,
                    tombstone_ratio: ratio,
                    needs_compaction: needs_compact,
                    bm25_chunk_count: bm25_chunk_size,
                    bm25_doc_count: bm25_doc_size,
                });
            }
        }
    }

    let index_metrics = cached_metrics.read();

    // GPU Scheduler stats (desktop only)
    #[cfg(not(target_arch = "wasm32"))]
    let (scheduler_name, device_type, gpu_queue_depth, gpu_requests_completed, _gpu_models_loaded) = {
        if is_scheduler_initialized() {
            if let Ok(scheduler) = get_scheduler() {
                let stats = scheduler.stats();
                (
                    stats.scheduler_name,
                    stats.device_type,
                    stats.queue_depth,
                    stats.requests_completed,
                    stats.models_loaded,
                )
            } else {
                ("N/A", DeviceType::Cpu, 0, 0, 0)
            }
        } else {
            ("Not initialized", DeviceType::Cpu, 0, 0, 0)
        }
    };
    #[cfg(target_arch = "wasm32")]
    let (scheduler_name, device_type, gpu_queue_depth, gpu_requests_completed, _gpu_models_loaded) =
        ("Web Worker", "CPU (WASM)", 0usize, 0u64, 0usize);

    // Memory calculations
    let hnsw_memory = estimate_hnsw_memory(index_metrics.hnsw_live);
    let bm25_memory =
        estimate_bm25_memory(index_metrics.bm25_chunk_count, index_metrics.bm25_doc_count);
    let total_memory = hnsw_memory + bm25_memory;

    // Tombstone percentage
    let tombstone_pct = index_metrics.tombstone_ratio * 100.0;

    // Get rolling average metrics (60-second window)
    let perf_snapshot = global_metrics().snapshot();

    // Determine state based on activity
    let has_recent_activity = perf_snapshot.embedding_count > 0;
    let state_mode = if is_indexing {
        "live"
    } else if has_recent_activity {
        "historical"
    } else {
        "idle"
    };

    let chunks_per_sec = perf_snapshot.embedding_throughput;
    let avg_chunk_time_ms = perf_snapshot.embedding_avg_ms.unwrap_or(0.0);

    let formatted_chunks_per_sec = format!("{:.1}", chunks_per_sec);
    let formatted_chunk_time = format!("{:.0}", avg_chunk_time_ms);

    let (state_class, state_label, state_caption) = match state_mode {
        "live" => (
            "cm-metrics-state cm-metrics-state--live",
            "Live",
            "Embedding in progress (60s avg)",
        ),
        "historical" => (
            "cm-metrics-state cm-metrics-state--historical",
            "60s Avg",
            "Rolling average from last 60 seconds",
        ),
        _ => (
            "cm-metrics-state cm-metrics-state--idle",
            "Idle",
            "Waiting for first batch...",
        ),
    };

    // Format rolling averages using helper
    let chunking_avg = format_metric(perf_snapshot.chunking_avg_ms, 1);
    let tokenization_avg = format_metric(perf_snapshot.tokenization_avg_ms, 1);
    let embedding_avg = format_metric(perf_snapshot.embedding_avg_ms, 1);
    let hnsw_avg = format_metric(perf_snapshot.hnsw_avg_ms, 2);
    let bm25_avg = format_metric(perf_snapshot.bm25_avg_ms, 2);
    let embedding_throughput = format!("{:.1}", perf_snapshot.embedding_throughput);

    let has_rolling_data = perf_snapshot.chunking_count > 0
        || perf_snapshot.tokenization_count > 0
        || perf_snapshot.embedding_count > 0;

    // Search metrics using helper
    let has_search_data = perf_snapshot.search.query_count > 0;
    let search_total_avg = format_metric(perf_snapshot.search.total_latency_avg_ms, 1);
    let search_vector_avg = format_metric(perf_snapshot.search.vector_search_avg_ms, 1);
    let search_keyword_avg = format_metric(perf_snapshot.search.keyword_search_avg_ms, 1);
    let search_fusion_avg = format_metric(perf_snapshot.search.fusion_avg_ms, 2);
    let last_result_count = format_count(perf_snapshot.search.last_result_count);
    let last_top_score = format_metric(perf_snapshot.search.last_top_score.map(|v| v as f64), 2);

    // Scheduler timing metrics using helper
    let has_scheduler_data = perf_snapshot.scheduler.requests_completed > 0;
    let scheduler_queue_wait_avg = format_metric(perf_snapshot.scheduler.queue_wait_avg_ms, 1);
    let scheduler_inference_avg = format_metric(perf_snapshot.scheduler.inference_avg_ms, 1);
    let scheduler_queue_depth_metric = perf_snapshot.scheduler.queue_depth;

    rsx! {
        section {
            class: panel_class,
            "data-role": "metrics-panel",
            header { class: "cm-metrics-header",
                h2 { class: "cm-metrics-title", "Engine Metrics" }
            }

            // Row 1: Search Performance (always visible, top priority)
            div { class: "cm-metrics-section-header",
                h3 { class: "cm-metrics-section-title", "Search Performance" }
                if has_search_data {
                    span { class: "cm-metrics-state cm-metrics-state--historical",
                        "5 min avg"
                    }
                } else {
                    span { class: "cm-metrics-state cm-metrics-state--idle",
                        "No searches yet"
                    }
                }
            }
            div { class: "cm-metrics-columns",
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Total Latency" }
                            div { class: "cm-metric-value", "{search_total_avg}", span { class: "cm-metric-unit", "ms" } }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Vector Search" }
                            div { class: "cm-metric-value", "{search_vector_avg}", span { class: "cm-metric-unit", "ms" } }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Keyword Search" }
                            div { class: "cm-metric-value", "{search_keyword_avg}", span { class: "cm-metric-unit", "ms" } }
                        }
                    }
                }
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "RRF Fusion" }
                            div { class: "cm-metric-value", "{search_fusion_avg}", span { class: "cm-metric-unit", "ms" } }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Results" }
                            div { class: "cm-metric-value", "{last_result_count}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Top Score" }
                            div { class: "cm-metric-value", "{last_top_score}" }
                        }
                    }
                }
            }

            // Row 2: Index Summary and Compute
            div { class: "cm-metrics-separator cm-metrics-separator--tight" }
            div { class: "cm-metrics-columns",
                // Index Summary
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "Index Summary" }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Documents" }
                            div { class: "cm-metric-value", "{index_metrics.docs}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Chunks" }
                            div { class: "cm-metric-value", "{index_metrics.chunks}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Total Memory" }
                            div { class: "cm-metric-value", "{format_bytes(total_memory)}" }
                        }
                    }
                }

                // Compute Backend
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "Compute" }
                        if gpu_queue_depth > 0 {
                            span { class: "cm-metrics-state cm-metrics-state--live",
                                span { class: "cm-metrics-state-indicator" }
                                "Active"
                            }
                        }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Scheduler" }
                            div { class: "cm-metric-value", "{scheduler_name}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Device" }
                            div { class: "cm-metric-value", "{device_type}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Completed" }
                            div { class: "cm-metric-value", "{gpu_requests_completed}" }
                        }
                    }
                }
            }

            // Row 3: HNSW and BM25 Index Details
            div { class: "cm-metrics-separator cm-metrics-separator--tight" }
            div { class: "cm-metrics-columns",
                // HNSW Vector Index
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "HNSW Vector Index" }
                        if index_metrics.needs_compaction {
                            span { class: "cm-metrics-state cm-metrics-state--warning",
                                "Compact Now"
                            }
                        }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Entries" }
                            div { class: "cm-metric-value", "{index_metrics.hnsw_live}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Tombstones" }
                            div { class: "cm-metric-value",
                                "{index_metrics.tombstone_count}",
                                span { class: "cm-metric-unit", " ({tombstone_pct:.0}%)" }
                            }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Memory" }
                            div { class: "cm-metric-value", "{format_bytes(hnsw_memory)}" }
                        }
                    }
                }

                // BM25 Keyword Indexes
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "BM25 Keyword Index" }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Chunk Index" }
                            div { class: "cm-metric-value", "{index_metrics.bm25_chunk_count}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Doc Index" }
                            div { class: "cm-metric-value", "{index_metrics.bm25_doc_count}" }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Memory" }
                            div { class: "cm-metric-value", "{format_bytes(bm25_memory)}" }
                        }
                    }
                }
            }

            // Row 4: Embedding Performance
            div { class: "cm-metrics-separator cm-metrics-separator--tight" }
            div { class: "cm-metrics-section-header",
                h3 { class: "cm-metrics-section-title", "Embedding Performance" }
                span { class: state_class,
                    span { class: "cm-metrics-state-indicator" }
                    "{state_label}"
                }
            }
            div { class: "cm-metrics-grid cm-metrics-grid--compact",
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

            // Row 4: Operation Timings (only when data available)
            if has_rolling_data {
                div { class: "cm-metrics-separator cm-metrics-separator--tight" }
                div { class: "cm-metrics-section-header",
                    h3 { class: "cm-metrics-section-title", "Operation Timings" }
                    span { class: "cm-metrics-state cm-metrics-state--historical",
                        "60s avg"
                    }
                }
                div { class: "cm-metrics-columns",
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

            // Row 6: GPU Scheduler Timing (only when scheduler data available, desktop only)
            if has_scheduler_data {
                div { class: "cm-metrics-separator cm-metrics-separator--tight" }
                div { class: "cm-metrics-section-header",
                    h3 { class: "cm-metrics-section-title", "Scheduler Timing" }
                    if scheduler_queue_depth_metric > 0 {
                        span { class: "cm-metrics-state cm-metrics-state--live",
                            span { class: "cm-metrics-state-indicator" }
                            "{scheduler_queue_depth_metric} pending"
                        }
                    } else {
                        span { class: "cm-metrics-state cm-metrics-state--historical",
                            "60s avg"
                        }
                    }
                }
                div { class: "cm-metrics-grid cm-metrics-grid--compact",
                    div { class: "cm-metric-card cm-metric-card--compact",
                        div { class: "cm-metric-label", "Queue Wait" }
                        div { class: "cm-metric-value", "{scheduler_queue_wait_avg}", span { class: "cm-metric-unit", "ms" } }
                    }
                    div { class: "cm-metric-card cm-metric-card--compact",
                        div { class: "cm-metric-label", "Inference" }
                        div { class: "cm-metric-value", "{scheduler_inference_avg}", span { class: "cm-metric-unit", "ms" } }
                    }
                    div { class: "cm-metric-card cm-metric-card--compact",
                        div { class: "cm-metric-label", "Completed" }
                        div { class: "cm-metric-value", "{perf_snapshot.scheduler.requests_completed}" }
                    }
                }
            }
        }
    }
}
