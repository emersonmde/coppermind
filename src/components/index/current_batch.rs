use dioxus::prelude::*;

use super::file_row::{FileMetrics, FileRow, FileStatus};

/// File being processed in current batch
#[derive(Clone, PartialEq)]
pub struct FileInBatch {
    pub name: String,
    pub status: FileStatus,
    pub progress_pct: f64,
    pub metrics: Option<FileMetrics>,
}

/// Current indexing batch section showing files being processed
#[component]
pub fn CurrentBatch(files: ReadSignal<Vec<FileInBatch>>, subtitle: ReadSignal<String>) -> Element {
    let files_list = files.read();

    // Only show section if there are files
    if files_list.is_empty() {
        return rsx! { div {} };
    }

    rsx! {
        section { class: "cm-indexing-section",
            header { class: "cm-section-header",
                h2 { class: "cm-section-title", "Current indexing batch" }
                span { class: "cm-section-subtitle", "{subtitle}" }
            }

            div { class: "cm-file-list",
                for file in files_list.iter() {
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
