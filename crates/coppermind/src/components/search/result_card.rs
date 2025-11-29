use dioxus::prelude::*;

use crate::search::types::{DocumentSearchResult, SearchResult};
use crate::utils::formatting::format_timestamp;

/// Skeleton result card shown while search is in progress.
/// Mimics the structure of ResultCard but with animated placeholder blocks.
#[component]
pub fn SkeletonResultCard(rank: usize) -> Element {
    rsx! {
        article { class: "cm-result-card cm-result-card--skeleton",
            header { class: "cm-result-header",
                div { class: "cm-result-rank cm-skeleton-text cm-skeleton-text--sm" }
                div { class: "cm-result-main",
                    div { class: "cm-skeleton-text cm-skeleton-text--title" }
                    div { class: "cm-result-meta",
                        span { class: "cm-skeleton-text cm-skeleton-text--md" }
                    }
                }
                div { class: "cm-result-score",
                    span { class: "cm-skeleton-text cm-skeleton-text--sm" }
                }
            }

            // Skeleton snippet - multiple lines
            div { class: "cm-result-snippet cm-skeleton-snippet",
                div { class: "cm-skeleton-text cm-skeleton-text--line" }
                div { class: "cm-skeleton-text cm-skeleton-text--line cm-skeleton-text--short" }
            }

            footer { class: "cm-result-footer",
                span { class: "cm-skeleton-text cm-skeleton-text--btn" }
                span { class: "cm-skeleton-text cm-skeleton-text--btn" }
            }
        }
    }
}

/// Document-level search result card with expandable chunk list and fusion details.
///
/// Displays a document as the primary search result (like Google shows pages),
/// with the best-matching chunk as the preview snippet. Users can expand
/// to see all chunks and fusion score diagnostics.
#[component]
pub fn ResultCard(
    rank: usize,
    doc_result: DocumentSearchResult,
    on_show_source: EventHandler<DocumentSearchResult>,
) -> Element {
    let mut details_expanded = use_signal(|| false);
    let mut chunks_expanded = use_signal(|| false);

    // Best chunk determines document's displayed scores
    let best_chunk = &doc_result.chunks[0];

    // Format scores for display
    let fusion_score_fmt = format!("{:.4}", doc_result.score);
    let vector_score_fmt = doc_result
        .best_chunk_score
        .map(|s| format!("{:.4}", s))
        .unwrap_or_else(|| "N/A".to_string());
    let bm25_score_fmt = doc_result
        .doc_keyword_score
        .map(|s| format!("{:.4}", s))
        .unwrap_or_else(|| "N/A".to_string());

    // Calculate percentage bars for detail view
    let vector_pct = doc_result
        .best_chunk_score
        .map(|s| (s * 100.0) as u32)
        .unwrap_or(0);
    let bm25_pct = doc_result
        .doc_keyword_score
        .map(|s| (s * 100.0) as u32)
        .unwrap_or(0);

    // Extract document info
    let doc_name = doc_result.metadata.title.clone();
    let doc_path = doc_result.metadata.source_id.clone();
    let chunk_count = doc_result.chunks.len();

    // Format timestamp (human-readable)
    let indexed_at = format_timestamp(doc_result.metadata.created_at);

    // Clone doc_result for closures
    let doc_for_title_click = doc_result.clone();
    let doc_for_show_source = doc_result.clone();

    let details_class = if details_expanded() {
        "cm-result-details"
    } else {
        "cm-result-details cm-result-details--collapsed"
    };

    let chunks_class = if chunks_expanded() {
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
                            on_show_source.call(doc_for_title_click.clone());
                        },
                        "{doc_name}"
                    }
                    div { class: "cm-result-meta",
                        span { class: "cm-result-source", "{doc_path}" }
                        span { class: "cm-meta-dot", "•" }
                        span { class: "cm-result-index",
                            if chunk_count == 1 {
                                "1 chunk"
                            } else {
                                "{chunk_count} chunks"
                            }
                        }
                        span { class: "cm-meta-dot", "•" }
                        span { class: "cm-result-timestamp", "{indexed_at}" }
                    }
                }
                div { class: "cm-result-score",
                    span { class: "cm-score-label", "RRF" }
                    span { class: "cm-score-value", "{fusion_score_fmt}" }
                }
            }

            // Snippet: Best chunk's text
            p { class: "cm-result-snippet",
                "{best_chunk.text}"
            }

            footer { class: "cm-result-footer",
                button {
                    class: "cm-btn cm-btn--ghost",
                    onclick: move |evt| {
                        evt.prevent_default();
                        on_show_source.call(doc_for_show_source.clone());
                    },
                    "Show source"
                }
                button {
                    class: "cm-btn cm-btn--ghost",
                    onclick: move |_| {
                        chunks_expanded.set(!chunks_expanded());
                    },
                    if chunks_expanded() {
                        "Hide chunks"
                    } else {
                        "Show chunks ({chunk_count})"
                    }
                }
                button {
                    class: "cm-btn cm-btn--ghost cm-btn--subtle",
                    onclick: move |_| {
                        details_expanded.set(!details_expanded());
                    },
                    "Details"
                }
            }

            // Expandable chunks section
            div { class: chunks_class,
                div { class: "cm-result-detail-row",
                    div { class: "cm-detail-label",
                        if chunk_count == 1 {
                            "Chunk (1 total)"
                        } else {
                            "All chunks ({chunk_count} total, sorted by relevance)"
                        }
                    }
                    div { class: "cm-detail-bars",
                        for (idx, chunk) in doc_result.chunks.iter().enumerate() {
                            ChunkPreview {
                                key: "{chunk.chunk_id.as_u64()}",
                                chunk_number: idx + 1,
                                chunk: chunk.clone()
                            }
                        }
                    }
                }
            }

            // Expandable details section (fusion diagnostics)
            div { class: details_class,
                div { class: "cm-result-detail-row",
                    div { class: "cm-detail-label", "Fusion breakdown" }
                    div { class: "cm-detail-bars",
                        if doc_result.best_chunk_score.is_some() {
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
                        if doc_result.doc_keyword_score.is_some() {
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
                    div { "Source: {doc_path}" }
                    div { "Indexed: {indexed_at}" }
                }
            }
        }
    }
}

/// Compact chunk preview within document result.
///
/// Shows chunk number, all scores (RRF, Vector, BM25), and text snippet.
/// Follows the same visual design as the Details section.
#[component]
fn ChunkPreview(chunk_number: usize, chunk: SearchResult) -> Element {
    // Format all scores
    let rrf_score_fmt = format!("{:.4}", chunk.score);
    let vector_score_fmt = chunk
        .vector_score
        .map(|s| format!("{:.4}", s))
        .unwrap_or_else(|| "N/A".to_string());
    let bm25_score_fmt = chunk
        .keyword_score
        .map(|s| format!("{:.4}", s))
        .unwrap_or_else(|| "N/A".to_string());

    // Extract chunk index from filename if available (e.g., "file.md (chunk 3)")
    let chunk_label = chunk
        .metadata
        .filename
        .as_ref()
        .and_then(|f| {
            f.strip_prefix(&format!(
                "{} (chunk ",
                chunk.metadata.source.as_deref().unwrap_or("")
            ))
            .and_then(|s| s.strip_suffix(")"))
        })
        .map(|num| format!("Chunk {}", num))
        .unwrap_or_else(|| format!("Chunk {}", chunk_number));

    // Truncate text for preview (show first 200 chars)
    // Use char-based truncation to avoid panicking on multi-byte UTF-8 characters
    let preview_text = if chunk.text.chars().count() > 200 {
        let truncated: String = chunk.text.chars().take(200).collect();
        format!("{}...", truncated)
    } else {
        chunk.text.clone()
    };

    rsx! {
        div {
            class: "cm-chunk-preview",

            // Header: Chunk label + primary RRF score
            div { class: "cm-chunk-preview-header",
                span { class: "cm-chunk-label", "{chunk_label}" }
                div { class: "cm-chunk-score",
                    span { class: "cm-chunk-score-label-inline", "RRF" }
                    span { class: "cm-chunk-score-value-inline", "{rrf_score_fmt}" }
                }
            }

            // Text snippet
            p { class: "cm-chunk-text",
                "{preview_text}"
            }

            // Score breakdown (same pattern as Details section)
            div { class: "cm-chunk-scores",
                // RRF score
                div { class: "cm-chunk-score-item",
                    span { class: "cm-chunk-score-label", "RRF" }
                    span { class: "cm-chunk-score-value", "{rrf_score_fmt}" }
                }
                // Vector score
                if chunk.vector_score.is_some() {
                    div { class: "cm-chunk-score-item",
                        span { class: "cm-chunk-score-label", "HNSW" }
                        span { class: "cm-chunk-score-value", "{vector_score_fmt}" }
                    }
                }
                // BM25 score
                if chunk.keyword_score.is_some() {
                    div { class: "cm-chunk-score-item",
                        span { class: "cm-chunk-score-label", "BM25" }
                        span { class: "cm-chunk-score-value", "{bm25_score_fmt}" }
                    }
                }
            }
        }
    }
}
