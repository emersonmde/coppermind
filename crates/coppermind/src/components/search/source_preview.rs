use dioxus::prelude::*;

use crate::search::types::DocumentSearchResult;
use crate::utils::formatting::format_timestamp;

/// Source preview overlay showing reconstructed document content from all chunks.
///
/// Takes a DocumentSearchResult and reconstructs the full document by concatenating
/// all chunks in order (sorted by chunk index). This provides the "Show Source"
/// functionality matching the UX spec requirements.
#[component]
pub fn SourcePreviewOverlay(
    file_result: ReadSignal<Option<DocumentSearchResult>>,
    on_close: EventHandler<()>,
) -> Element {
    let result_data = file_result.read();

    if result_data.is_none() {
        return rsx! { div {} };
    }

    let doc = result_data.as_ref().unwrap();

    // Reconstruct full document text from chunks
    // Chunks are stored with filenames like "file.md (chunk 3)"
    // Sort by chunk number, then concatenate
    let full_text = reconstruct_file_from_chunks(&doc.chunks);

    // Calculate total token count (approximate)
    let token_count: usize = doc
        .chunks
        .iter()
        .map(|c| c.text.split_whitespace().count())
        .sum();

    // Extract metadata
    let doc_name = &doc.metadata.title;
    let source = &doc.metadata.source_id;

    // Format timestamp
    let date_str = format_timestamp(doc.metadata.created_at);

    rsx! {
        // Overlay backdrop
        div {
            class: "cm-overlay",
            onclick: move |_| on_close.call(()),

            // Preview panel
            div {
                class: "cm-source-preview",
                onclick: move |e| e.stop_propagation(), // Prevent closing when clicking inside

                // Header with metadata
                header { class: "cm-source-preview-header",
                    div { class: "cm-source-preview-title",
                        h2 { "{doc_name}" }
                        button {
                            class: "cm-icon-button",
                            onclick: move |_| on_close.call(()),
                            "aria-label": "Close preview",
                            "✕"
                        }
                    }
                    div { class: "cm-source-preview-meta",
                        div { class: "cm-meta-item",
                            span { class: "cm-meta-label", "Source: " }
                            span { class: "cm-meta-value", "{source}" }
                        }
                        div { class: "cm-meta-item",
                            span { class: "cm-meta-label", "Indexed: " }
                            span { class: "cm-meta-value", "{date_str}" }
                        }
                        div { class: "cm-meta-item",
                            span { class: "cm-meta-label", "Chunks: " }
                            span { class: "cm-meta-value", "{doc.chunks.len()}" }
                        }
                        div { class: "cm-meta-item",
                            span { class: "cm-meta-label", "Tokens: " }
                            span { class: "cm-meta-value", "~{token_count}" }
                        }
                    }
                }

                // Reconstructed document content (from all chunks)
                div { class: "cm-source-preview-content",
                    pre { class: "cm-source-text",
                        code { "{full_text}" }
                    }
                }
            }
        }
    }
}

/// Reconstructs full file text from chunks.
///
/// Chunks have filenames like "file.md (chunk 3)". This function:
/// 1. Extracts chunk indices from filenames
/// 2. Sorts chunks by index
/// 3. Concatenates chunk text with newlines
///
/// If chunk indices can't be extracted, chunks are concatenated in order.
fn reconstruct_file_from_chunks(chunks: &[crate::search::types::SearchResult]) -> String {
    use crate::search::types::SearchResult;

    // Try to extract chunk index from filename
    let mut indexed_chunks: Vec<(usize, &SearchResult)> = chunks
        .iter()
        .enumerate()
        .map(|(idx, chunk)| {
            // Try to parse chunk index from filename like "file.md (chunk 3)"
            let chunk_idx = chunk
                .metadata
                .filename
                .as_ref()
                .and_then(|f| {
                    f.rsplit(" (chunk ")
                        .next()
                        .and_then(|s| s.strip_suffix(")"))
                        .and_then(|num| num.parse::<usize>().ok())
                })
                .unwrap_or(idx + 1); // Fallback: use position in array

            (chunk_idx, chunk)
        })
        .collect();

    // Sort by chunk index
    indexed_chunks.sort_by_key(|(idx, _)| *idx);

    // Concatenate chunks with double newlines for readability
    indexed_chunks
        .iter()
        .map(|(_, chunk)| chunk.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}
