use dioxus::logger::tracing::info;
use dioxus::prelude::*;

use crate::components::{use_batches, use_processing_sender, ProcessingMessage};

use super::batch_list::BatchList;
use super::upload_card::UploadCard;
use crate::components::WebCrawlerCard;

/// Main index view with upload and batch management
#[component]
pub fn IndexView() -> Element {
    // Get batches state from context (persists across view switches)
    let batches = use_batches();

    // Get processing coroutine from context
    let processing_task = use_processing_sender();

    // Tab state: "crawler" or "files"
    let mut active_tab = use_signal(|| {
        if cfg!(not(target_arch = "wasm32")) {
            "crawler" // Desktop: default to crawler
        } else {
            "files" // Web: default to files (crawler doesn't work)
        }
    });

    let handle_files_selected = move |file_contents: Vec<(String, String)>| {
        info!("📂 Selected {} file(s)", file_contents.len());
        processing_task.send(ProcessingMessage::ProcessFiles(file_contents));
    };

    rsx! {
        section {
            class: "cm-view cm-view--index cm-view--active",
            "data-view": "index",

            // Tab system
            div { class: "cm-upload-tabs-container",
                div { class: "cm-upload-tabs",
                    button {
                        class: if active_tab() == "crawler" {
                            "cm-upload-tab cm-upload-tab--active"
                        } else {
                            "cm-upload-tab"
                        },
                        onclick: move |_| active_tab.set("crawler"),
                        "Web Crawler"
                    }
                    button {
                        class: if active_tab() == "files" {
                            "cm-upload-tab cm-upload-tab--active"
                        } else {
                            "cm-upload-tab"
                        },
                        onclick: move |_| active_tab.set("files"),
                        "Files / Folders"
                    }
                }
            }

            // Tab content
            if active_tab() == "crawler" {
                WebCrawlerCard {}
            } else {
                UploadCard {
                    on_files_selected: handle_files_selected
                }
            }

            BatchList {
                batches
            }
        }
    }
}
