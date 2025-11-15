use dioxus::prelude::*;

/// Summary of a completed batch for history display
#[derive(Clone, PartialEq)]
pub struct BatchSummary {
    pub batch_number: usize,
    pub file_count: usize,
    pub chunk_count: usize,
    pub token_count: usize,
    pub duration_ms: u64,
}

/// Individual batch card showing summary stats
#[component]
pub fn BatchCard(batch: BatchSummary) -> Element {
    let duration_secs = batch.duration_ms as f64 / 1000.0;

    rsx! {
        article { class: "cm-batch-card",
            header { class: "cm-batch-header",
                h3 { class: "cm-batch-title", "Batch #{batch.batch_number}" }
                span { class: "cm-tag cm-tag--success", "Completed" }
            }
            div { class: "cm-batch-body",
                div { class: "cm-batch-row",
                    span { "Files" }
                    span { "{batch.file_count}" }
                }
                div { class: "cm-batch-row",
                    span { "Chunks" }
                    span { "{batch.chunk_count}" }
                }
                div { class: "cm-batch-row",
                    span { "Tokens" }
                    span { "{batch.token_count}" }
                }
                div { class: "cm-batch-row",
                    span { "Duration" }
                    span { "{duration_secs:.1}s" }
                }
            }
        }
    }
}

/// Previous batches section showing history
#[component]
pub fn PreviousBatches(batches: ReadSignal<Vec<BatchSummary>>) -> Element {
    let batch_list = batches.read();

    // Only show if there are batches
    if batch_list.is_empty() {
        return rsx! { div {} };
    }

    rsx! {
        section { class: "cm-batches-section",
            header { class: "cm-section-header",
                h2 { class: "cm-section-title", "Recent batches" }
            }
            div { class: "cm-batch-grid",
                for batch in batch_list.iter() {
                    BatchCard {
                        key: "{batch.batch_number}",
                        batch: batch.clone()
                    }
                }
            }
        }
    }
}
