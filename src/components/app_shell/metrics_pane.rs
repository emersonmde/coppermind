use dioxus::prelude::*;

use crate::components::{
    calculate_engine_metrics, calculate_live_metrics, use_batches, use_search_engine,
};

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
                let (chunks, tokens, avg) = engine.get_index_metrics();
                (chunks, tokens, avg)
            } else {
                // Lock held, show zeros temporarily
                (0, 0, 0.0)
            }
        } else {
            (0, 0, 0.0)
        };

    // BM25 and HNSW share the same documents, so metrics are identical
    let (keyword_chunks, keyword_tokens, keyword_avg_tokens) =
        (vector_chunks, vector_tokens, vector_avg_tokens);

    // Format values for display
    let formatted_vector_avg = format!("{:.0}", vector_avg_tokens);
    let formatted_keyword_avg = format!("{:.0}", keyword_avg_tokens);

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

    rsx! {
        section {
            class: panel_class,
            "data-role": "metrics-panel",
            header { class: "cm-metrics-header",
                h2 { class: "cm-metrics-title", "Engine Metrics" }
            }

            div { class: "cm-metrics-section-header",
                h3 { class: "cm-metrics-section-title", "Aggregate (All Indexes)" }
            }
            div { class: "cm-metrics-grid",
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Files Indexed" }
                    div { class: "cm-metric-value", "{batch_total_docs}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Total Chunks" }
                    div { class: "cm-metric-value", "{vector_chunks}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Total Tokens" }
                    div { class: "cm-metric-value", "{vector_tokens}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Avg Tokens / Chunk" }
                    div { class: "cm-metric-value", "{formatted_vector_avg}" }
                }
            }

            // Per-index metrics
            div { class: "cm-metrics-separator" }
            div { class: "cm-metrics-section-header",
                h3 { class: "cm-metrics-section-title", "HNSW Vector Index" }
            }
            div { class: "cm-metrics-grid",
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Chunks" }
                    div { class: "cm-metric-value", "{vector_chunks}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Tokens" }
                    div { class: "cm-metric-value", "{vector_tokens}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Avg Tokens / Chunk" }
                    div { class: "cm-metric-value", "{formatted_vector_avg}" }
                }
            }

            div { class: "cm-metrics-separator" }
            div { class: "cm-metrics-section-header",
                h3 { class: "cm-metrics-section-title", "BM25 Keyword Index" }
            }
            div { class: "cm-metrics-grid",
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Chunks" }
                    div { class: "cm-metric-value", "{keyword_chunks}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Tokens" }
                    div { class: "cm-metric-value", "{keyword_tokens}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Avg Tokens / Chunk" }
                    div { class: "cm-metric-value", "{formatted_keyword_avg}" }
                }
            }

            // Performance metrics (live/historical indexing stats)
            div { class: "cm-metrics-separator" }
            div { class: "cm-metrics-section-header",
                h3 { class: "cm-metrics-section-title", "Indexing Performance" }
                span { class: state_class,
                    span { class: "cm-metrics-state-indicator" }
                    "{state_label}"
                }
            }
            div { class: "cm-metrics-grid cm-metrics-grid--performance",
                div { class: "cm-metric-card cm-metric-card--rate",
                    div { class: "cm-metric-label", "Tokens / sec" }
                    div { class: "cm-metric-value", "{formatted_tokens_per_sec}" }
                }
                div { class: "cm-metric-card cm-metric-card--rate",
                    div { class: "cm-metric-label", "Chunks / sec" }
                    div { class: "cm-metric-value", "{formatted_chunks_per_sec}" }
                }
                div { class: "cm-metric-card cm-metric-card--rate",
                    div { class: "cm-metric-label", "Avg Time / Chunk" }
                    div { class: "cm-metric-value", "{formatted_chunk_time}", span { class: "cm-metric-unit", "ms" } }
                }
            }
            p { class: "cm-metrics-caption",
                "{state_caption}"
            }
        }
    }
}
