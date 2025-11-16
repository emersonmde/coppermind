use dioxus::prelude::*;

use crate::search::types::SearchResult;

/// Individual search result card with expandable details
#[component]
pub fn ResultCard(
    rank: usize,
    result: SearchResult,
    on_show_source: EventHandler<SearchResult>,
) -> Element {
    let mut details_expanded = use_signal(|| false);

    // Format scores for display
    let fusion_score_fmt = format!("{:.4}", result.score);
    let vector_score_fmt = result
        .vector_score
        .map(|s| format!("{:.4}", s))
        .unwrap_or_else(|| "N/A".to_string());
    let bm25_score_fmt = result
        .keyword_score
        .map(|s| format!("{:.4}", s))
        .unwrap_or_else(|| "N/A".to_string());

    // Calculate percentage bars for detail view
    let vector_pct = result.vector_score.map(|s| (s * 100.0) as u32).unwrap_or(0);
    let bm25_pct = result
        .keyword_score
        .map(|s| (s * 100.0) as u32)
        .unwrap_or(0);

    // Extract metadata
    let file_name = result
        .metadata
        .filename
        .clone()
        .unwrap_or_else(|| "Unknown".to_string());
    let source = result
        .metadata
        .source
        .clone()
        .unwrap_or_else(|| "Unknown".to_string());

    // Format timestamp (simple format for now)
    let indexed_at = format!("{}", result.metadata.created_at);

    // Clone result for closures
    let result_for_title_click = result.clone();
    let result_for_show_source = result.clone();

    let details_class = if details_expanded() {
        "cm-result-details"
    } else {
        "cm-result-details cm-result-details--collapsed"
    };

    rsx! {
        article { class: "cm-result-card",
            header { class: "cm-result-header",
                div { class: "cm-result-rank", "#{rank}" }
                div { class: "cm-result-main",
                    a {
                        class: "cm-result-title",
                        href: "#",
                        onclick: move |evt| {
                            evt.prevent_default();
                            on_show_source.call(result_for_title_click.clone());
                        },
                        "{file_name}"
                    }
                    div { class: "cm-result-meta",
                        span { class: "cm-result-source", "{source}" }
                        span { class: "cm-meta-dot", "•" }
                        span { class: "cm-result-index", "Index: Local Docs" }
                    }
                }
                div { class: "cm-result-score",
                    span { class: "cm-score-label", "RRF" }
                    span { class: "cm-score-value", "{fusion_score_fmt}" }
                }
            }

            p { class: "cm-result-snippet",
                "{result.text}"
            }

            footer { class: "cm-result-footer",
                button {
                    class: "cm-btn cm-btn--ghost",
                    onclick: move |evt| {
                        evt.prevent_default();
                        on_show_source.call(result_for_show_source.clone());
                    },
                    "Show source"
                }
                button {
                    class: "cm-btn cm-btn--ghost cm-btn--subtle",
                    onclick: move |_| {
                        details_expanded.set(!details_expanded());
                    },
                    "Details"
                }
            }

            // Expandable details section
            div { class: details_class,
                div { class: "cm-result-detail-row",
                    div { class: "cm-detail-label", "Fusion breakdown" }
                    div { class: "cm-detail-bars",
                        if let Some(_vec_score) = result.vector_score {
                            div { class: "cm-detail-bar",
                                div { class: "cm-detail-bar-label",
                                    span { class: "cm-detail-bar-name", "HNSW" }
                                    span { class: "cm-detail-bar-score", "{vector_score_fmt}" }
                                }
                                div { class: "cm-progress-bar cm-progress-bar--metric",
                                    span { style: "width: {vector_pct}%;" }
                                }
                            }
                        }
                        if let Some(_kw_score) = result.keyword_score {
                            div { class: "cm-detail-bar",
                                div { class: "cm-detail-bar-label",
                                    span { class: "cm-detail-bar-name", "BM25" }
                                    span { class: "cm-detail-bar-score", "{bm25_score_fmt}" }
                                }
                                div { class: "cm-progress-bar cm-progress-bar--metric",
                                    span { style: "width: {bm25_pct}%;" }
                                }
                            }
                        }
                    }
                }
                div { class: "cm-result-detail-row cm-result-detail-row--meta",
                    div { "Doc ID: ", "{result.doc_id.as_u64()}" }
                    div { "Indexed: {indexed_at}" }
                }
            }
        }
    }
}
