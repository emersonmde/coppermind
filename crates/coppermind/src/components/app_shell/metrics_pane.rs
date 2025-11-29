use dioxus::prelude::*;

use crate::components::{use_batches, use_search_engine, BatchStatus};
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::{get_scheduler, is_scheduler_initialized};
use crate::metrics::global_metrics;

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
            // Only update cache when not actively indexing to avoid flickering
            if !is_indexing {
                let (docs, chunks, _tokens, _avg) = engine.get_index_metrics_sync();
                let (tombstones, _total, ratio) = engine.compaction_stats();
                let needs_compact = engine.needs_compaction();

                // Get individual index sizes
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

    // Read cached values
    let index_metrics = cached_metrics.read();

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

    // Memory estimate for HNSW: live_entries × embedding_dim × 4 bytes (f32)
    // Plus ~20% overhead for HNSW graph structure (M=16 bidirectional links per node)
    let embedding_dim = 512usize;
    let vector_memory_bytes = index_metrics.hnsw_live * embedding_dim * 4;
    let estimated_memory_bytes = (vector_memory_bytes as f64 * 1.2) as usize;
    let memory_display = if estimated_memory_bytes > 1_000_000 {
        format!("{:.1} MB", estimated_memory_bytes as f64 / 1_000_000.0)
    } else if estimated_memory_bytes > 1_000 {
        format!("{:.1} KB", estimated_memory_bytes as f64 / 1_000.0)
    } else {
        format!("{} B", estimated_memory_bytes)
    };

    // Tombstone ratio as percentage and progress toward 30% compaction threshold
    let tombstone_pct = index_metrics.tombstone_ratio * 100.0;
    // Show how close we are to the 30% threshold (capped at 100%)
    let compaction_progress = ((index_metrics.tombstone_ratio / 0.30) * 100.0).min(100.0);

    // Note: batches_list is used only to check is_indexing status above
    // All displayed metrics come from persisted index state via cached_metrics

    // Get rolling average metrics (60-second window) for stable performance display
    let perf_snapshot = global_metrics().snapshot();

    // Determine state based on whether we have recent activity
    let has_recent_activity = perf_snapshot.embedding_count > 0;
    let state_mode = if is_indexing {
        "live"
    } else if has_recent_activity {
        "historical"
    } else {
        "idle"
    };

    // Use rolling averages for throughput metrics (stable, not per-chunk)
    let chunks_per_sec = perf_snapshot.embedding_throughput;
    let avg_chunk_time_ms = perf_snapshot.embedding_avg_ms.unwrap_or(0.0);

    // Format values for display
    let formatted_chunks_per_sec = format!("{:.1}", chunks_per_sec);
    let formatted_chunk_time = format!("{:.0}", avg_chunk_time_ms);

    // State indicator and caption based on mode
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
            "Waiting for first batch…",
        ),
    };

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

            // Two-column layout for Index Summary and GPU Scheduler
            div { class: "cm-metrics-columns",
                // Left column: Index Summary (persisted metrics)
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
                            div { class: "cm-metric-label", "Memory" }
                            div { class: "cm-metric-value", "{memory_display}" }
                        }
                    }
                }

                // Right column: GPU Scheduler (desktop only, session metrics)
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
                // HNSW Vector Index (semantic search)
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
                            div { class: "cm-metric-label", "Live Entries" }
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
                            div { class: "cm-metric-label", "Compaction" }
                            div { class: "cm-metric-value",
                                "{compaction_progress:.0}",
                                span { class: "cm-metric-unit", "% of 30%" }
                            }
                        }
                    }
                }

                // BM25 Keyword Indexes (chunk-level and document-level)
                div { class: "cm-metrics-column",
                    div { class: "cm-metrics-section-header",
                        h3 { class: "cm-metrics-section-title", "BM25 Keyword Index" }
                    }
                    div { class: "cm-metrics-grid cm-metrics-grid--compact",
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Chunk Index" }
                            div { class: "cm-metric-value",
                                "{index_metrics.bm25_chunk_count}",
                                span { class: "cm-metric-unit", " entries" }
                            }
                        }
                        div { class: "cm-metric-card cm-metric-card--compact",
                            div { class: "cm-metric-label", "Doc Index" }
                            div { class: "cm-metric-value",
                                "{index_metrics.bm25_doc_count}",
                                span { class: "cm-metric-unit", " entries" }
                            }
                        }
                    }
                }
            }

            // Performance metrics (rolling 60-second averages)
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

            // Rolling averages (60-second window) - only shown during/after indexing
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
