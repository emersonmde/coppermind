use dioxus::prelude::*;

/// Collapsible metrics panel showing engine statistics
#[component]
pub fn MetricsPane(collapsed: ReadSignal<bool>) -> Element {
    let panel_class = if collapsed() {
        "cm-metrics-panel cm-metrics-panel--collapsed"
    } else {
        "cm-metrics-panel"
    };

    // TODO Phase 5: Wire to real engine metrics
    // For now, showing placeholder data
    let total_docs = 0;
    let total_chunks = 0;
    let total_tokens = 0;
    let avg_tokens_per_chunk = 0;

    // Live indexing metrics (shown only when indexing is active)
    let is_indexing = false;
    let tokens_per_sec = 0.0;
    let chunks_per_sec = 0.0;
    let avg_chunk_time_ms = 0.0;

    // Format values for display
    let formatted_tokens_per_sec = format!("{:.1}", tokens_per_sec);
    let formatted_chunks_per_sec = format!("{:.1}", chunks_per_sec);
    let formatted_chunk_time = format!("{:.0}", avg_chunk_time_ms);

    rsx! {
        section {
            class: panel_class,
            "data-role": "metrics-panel",
            header { class: "cm-metrics-header",
                h2 { class: "cm-metrics-title", "Engine Metrics" }
                span { class: "cm-tag cm-tag--live", "Live" }
            }
            div { class: "cm-metrics-grid",
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Total Documents" }
                    div { class: "cm-metric-value", "{total_docs}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Total Chunks" }
                    div { class: "cm-metric-value", "{total_chunks}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Total Tokens" }
                    div { class: "cm-metric-value", "{total_tokens}" }
                }
                div { class: "cm-metric-card",
                    div { class: "cm-metric-label", "Avg Tokens / Chunk" }
                    div { class: "cm-metric-value", "{avg_tokens_per_chunk}" }
                }
            }

            if is_indexing {
                div { class: "cm-metrics-sub",
                    div { class: "cm-metrics-row",
                        div {
                            div { class: "cm-metric-label", "Tokens / sec" }
                            div { class: "cm-metric-value cm-metric-value--sub", "{formatted_tokens_per_sec}" }
                        }
                        div {
                            div { class: "cm-metric-label", "Chunks / sec" }
                            div { class: "cm-metric-value cm-metric-value--sub", "{formatted_chunks_per_sec}" }
                        }
                        div {
                            div { class: "cm-metric-label", "Avg Chunk Time" }
                            div { class: "cm-metric-value cm-metric-value--sub", "{formatted_chunk_time}ms" }
                        }
                    }
                    p { class: "cm-metrics-caption",
                        "Vector engines spinning up… ⚙️"
                    }
                }
            }
        }
    }
}
