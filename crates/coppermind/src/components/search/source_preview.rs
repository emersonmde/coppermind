use dioxus::prelude::*;

use crate::search::types::FileSearchResult;

/// Source preview overlay showing reconstructed file content from all chunks.
///
/// Takes a FileSearchResult and reconstructs the full file by concatenating
/// all chunks in order (sorted by chunk index). This provides the "Show Source"
/// functionality matching the UX spec requirements.
#[component]
pub fn SourcePreviewOverlay(
    file_result: ReadSignal<Option<FileSearchResult>>,
    on_close: EventHandler<()>,
) -> Element {
    let result_data = file_result.read();

    if result_data.is_none() {
        return rsx! { div {} };
    }

    let file = result_data.as_ref().unwrap();

    // Reconstruct full file text from chunks
    // Chunks are stored with filenames like "file.md (chunk 3)"
    // Sort by chunk number, then concatenate
    let full_text = reconstruct_file_from_chunks(&file.chunks);

    // Calculate total token count (approximate)
    let token_count: usize = file
        .chunks
        .iter()
        .map(|c| c.text.split_whitespace().count())
        .sum();

    // Extract metadata
    let filename = &file.file_name;
    let source = &file.file_path;

    // Format timestamp
    let date_str = format_timestamp(file.created_at);

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
                        h2 { "{filename}" }
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
                            span { class: "cm-meta-value", "{file.chunks.len()}" }
                        }
                        div { class: "cm-meta-item",
                            span { class: "cm-meta-label", "Tokens: " }
                            span { class: "cm-meta-value", "~{token_count}" }
                        }
                    }
                }

                // Reconstructed file content (from all chunks)
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

/// Format Unix timestamp to human-readable date
fn format_timestamp(timestamp: u64) -> String {
    // Simple formatting - could be improved with a date library
    // For now, just show the timestamp
    // In a real app, you'd use chrono or similar
    if timestamp == 0 {
        "Unknown".to_string()
    } else {
        // Convert to hours/days ago for simplicity
        #[cfg(target_arch = "wasm32")]
        {
            let now = instant::SystemTime::now()
                .duration_since(instant::SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let elapsed = now.saturating_sub(timestamp);
            format_duration(elapsed)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let elapsed = now.saturating_sub(timestamp);
            format_duration(elapsed)
        }
    }
}

/// Format duration in seconds to human-readable string
fn format_duration(seconds: u64) -> String {
    match seconds {
        0..=59 => "Just now".to_string(),
        60..=3599 => {
            let mins = seconds / 60;
            if mins == 1 {
                "1 minute ago".to_string()
            } else {
                format!("{} minutes ago", mins)
            }
        }
        3600..=86399 => {
            let hours = seconds / 3600;
            if hours == 1 {
                "1 hour ago".to_string()
            } else {
                format!("{} hours ago", hours)
            }
        }
        86400..=2591999 => {
            let days = seconds / 86400;
            if days == 1 {
                "1 day ago".to_string()
            } else {
                format!("{} days ago", days)
            }
        }
        _ => {
            let days = seconds / 86400;
            format!("{} days ago", days)
        }
    }
}
