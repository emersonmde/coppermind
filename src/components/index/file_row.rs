use dioxus::prelude::*;

/// File processing status for display
#[derive(Clone, PartialEq, Debug)]
pub enum FileStatus {
    Queued,
    Processing { current: usize, total: usize },
    Completed,
    Failed(String),
}

/// File metrics for advanced view
#[derive(Clone, Default, PartialEq, Debug)]
pub struct FileMetrics {
    pub tokens_processed: usize,
    pub chunks_embedded: usize,
    pub chunks_total: usize,
    pub elapsed_ms: u64,
}

/// Individual file row with progress indicator
#[component]
pub fn FileRow(
    file_name: String,
    status: FileStatus,
    progress_pct: f64,
    metrics: Option<FileMetrics>,
) -> Element {
    let mut details_expanded = use_signal(|| false);

    // Status text
    let status_text = match &status {
        FileStatus::Queued => "Queued".to_string(),
        FileStatus::Processing { current, total } => {
            format!("Embedding chunks… {} / {}", current, total)
        }
        FileStatus::Completed => "Completed".to_string(),
        FileStatus::Failed(err) => format!("Failed: {}", err),
    };

    // Progress bar class
    let progress_class = match status {
        FileStatus::Queued => "cm-progress-bar cm-progress-bar--idle",
        FileStatus::Failed(_) => "cm-progress-bar",
        _ => "cm-progress-bar",
    };

    let details_class = if details_expanded() {
        "cm-file-advanced"
    } else {
        "cm-file-advanced cm-file-advanced--collapsed"
    };

    rsx! {
        div { class: "cm-file-row",
            div { class: "cm-file-main",
                div { class: "cm-file-name", "{file_name}" }
                div { class: "cm-file-sub", "{status_text}" }
            }
            div { class: "cm-file-progress",
                div { class: progress_class,
                    span { style: "width: {progress_pct}%;" }
                }
                div { class: "cm-file-percent", "{progress_pct:.0}%" }
            }
            button {
                class: "cm-icon-button cm-file-details-toggle",
                onclick: move |_| details_expanded.set(!details_expanded()),
                "▾"
            }
        }

        // Advanced details (collapsible)
        if let Some(m) = metrics {
            div { class: details_class,
                div { class: "cm-file-advanced-grid",
                    div {
                        div { class: "cm-metric-label", "Tokens processed" }
                        div { class: "cm-metric-value cm-metric-value--sub", "{m.tokens_processed}" }
                    }
                    div {
                        div { class: "cm-metric-label", "Chunks embedded" }
                        div { class: "cm-metric-value cm-metric-value--sub", "{m.chunks_embedded} / {m.chunks_total}" }
                    }
                    div {
                        div { class: "cm-metric-label", "Elapsed" }
                        div { class: "cm-metric-value cm-metric-value--sub", "{m.elapsed_ms}ms" }
                    }
                }
            }
        }
    }
}
