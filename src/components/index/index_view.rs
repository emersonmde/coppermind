use dioxus::logger::tracing::info;
use dioxus::prelude::*;

use crate::components::{use_batches, use_processing_sender, ProcessingMessage};

use super::batch_list::BatchList;
use super::upload_card::UploadCard;

/// Main index view with upload and batch management
#[component]
pub fn IndexView() -> Element {
    // Get batches state from context (persists across view switches)
    let batches = use_batches();

    // Get processing coroutine from context
    let processing_task = use_processing_sender();

    let handle_files_selected = move |file_contents: Vec<(String, String)>| {
        info!("📂 Selected {} file(s)", file_contents.len());
        processing_task.send(ProcessingMessage::ProcessFiles(file_contents));
    };

    rsx! {
        section {
            class: "cm-view cm-view--index cm-view--active",
            "data-view": "index",

            UploadCard {
                on_files_selected: handle_files_selected
            }

            BatchList {
                batches
            }
        }
    }
}
