use dioxus::prelude::*;

use super::batch::{Batch, BatchStatus};
use super::file_row::FileRow;

/// Unified batch list showing pending, processing, and completed batches
#[component]
pub fn BatchList(batches: ReadSignal<Vec<Batch>>) -> Element {
    let batch_list = batches.read();

    // Only show if there are batches
    if batch_list.is_empty() {
        return rsx! { div {} };
    }

    // Reverse order: newest (highest batch number) at the top
    let mut reversed_batches: Vec<_> = batch_list.iter().collect();
    reversed_batches.reverse();

    // Find the index of the first non-completed batch (for auto-expand)
    let first_active_idx = reversed_batches
        .iter()
        .position(|b| !matches!(b.status, BatchStatus::Completed));

    rsx! {
        section { class: "cm-indexing-section",
            header { class: "cm-section-header",
                h2 { class: "cm-section-title", "Indexing batches" }
            }
            div { class: "cm-batch-list",
                for (idx, batch) in reversed_batches.iter().enumerate() {
                    BatchCard {
                        key: "{batch.batch_number}",
                        batch: (*batch).clone(),
                        auto_expand: first_active_idx == Some(idx)
                    }
                }
            }
        }
    }
}

/// Individual batch card with expandable file list
#[component]
fn BatchCard(batch: Batch, auto_expand: bool) -> Element {
    let mut details_expanded = use_signal(|| auto_expand);
    let has_files = !batch.files.is_empty();

    // Determine batch card styling based on status
    let (status_class, status_label) = match batch.status {
        BatchStatus::Pending => ("cm-batch--pending", "Queued"),
        BatchStatus::Processing => ("cm-batch--processing", "Processing"),
        BatchStatus::Completed => ("cm-batch--completed", "Completed"),
    };

    rsx! {
        article { class: "cm-batch-card {status_class}",
            // Batch header with expand button and status
            header {
                class: "cm-batch-card-header",
                onclick: move |_| {
                    if has_files {
                        details_expanded.set(!details_expanded());
                    }
                },

                div { class: "cm-batch-card-title-row",
                    h3 { class: "cm-batch-card-title", "Batch #{batch.batch_number}" }

                    div { class: "cm-batch-card-actions",
                        span { class: "cm-tag cm-tag--{status_class}", "{status_label}" }
                        if has_files {
                            button {
                                class: if details_expanded() { "cm-expand-btn cm-expand-btn--expanded" } else { "cm-expand-btn" },
                                r#type: "button",
                                "aria-label": if details_expanded() { "Collapse files" } else { "Expand files" },
                                "▼"
                            }
                        }
                    }
                }

                // Progress bar for pending/processing batches
                if !matches!(batch.status, BatchStatus::Completed) {
                    div { class: "cm-batch-progress",
                        div {
                            class: "cm-batch-progress-bar",
                            style: "width: {batch.progress_pct()}%"
                        }
                    }
                }

                // Subtitle with batch summary
                div { class: "cm-batch-card-subtitle",
                    "{batch.subtitle()}"
                }
            }

            // Expandable file list
            if details_expanded() && has_files {
                div { class: "cm-batch-files",
                    for file in batch.files.iter() {
                        FileRow {
                            key: "{file.name}",
                            file_name: file.name.clone(),
                            status: file.status.clone(),
                            progress_pct: file.progress_pct,
                            metrics: file.metrics.clone(),
                        }
                    }
                }
            }
        }
    }
}
