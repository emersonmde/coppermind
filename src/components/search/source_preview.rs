use dioxus::prelude::*;

use crate::search::types::SearchResult;

/// Source preview overlay showing document content
#[component]
pub fn SourcePreviewOverlay(
    result: ReadSignal<Option<SearchResult>>,
    on_close: EventHandler<()>,
) -> Element {
    let result_data = result.read();

    if result_data.is_none() {
        return rsx! { div {} };
    }

    let search_result = result_data.as_ref().unwrap();

    // Extract metadata
    let filename = search_result
        .metadata
        .filename
        .as_deref()
        .unwrap_or("Untitled");
    let source = search_result
        .metadata
        .source
        .as_deref()
        .unwrap_or("Unknown source");

    // Format timestamp
    let created_at = search_result.metadata.created_at;
    let date_str = format_timestamp(created_at);

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
                            span { class: "cm-meta-label", "Doc ID: " }
                            span { class: "cm-meta-value", "{search_result.doc_id.as_u64()}" }
                        }
                    }
                }

                // Document text content
                div { class: "cm-source-preview-content",
                    pre { class: "cm-source-text",
                        code { "{search_result.text}" }
                    }
                }
            }
        }
    }
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
