//! Web crawler UI component (desktop-only).
//!
//! This component provides a simple interface for crawling web pages on desktop.
//! On web/WASM, this component renders nothing due to CORS restrictions.

// Desktop implementation (full crawler functionality)
#[cfg(not(target_arch = "wasm32"))]
mod desktop {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use dioxus::logger::tracing::{error, info};
    use dioxus::prelude::*;

    use crate::components::{use_processing_sender, ProcessingMessage};
    use crate::crawler::{CrawlConfig, CrawlEngine, CrawlProgress, CrawlResult};

    /// Web crawler card for entering URLs and crawling pages
    #[component]
    pub fn WebCrawlerCard() -> Element {
        let mut url = use_signal(String::new);
        let mut max_depth = use_signal(|| 999); // Default to All (unlimited)
        let mut status = use_signal(|| CrawlStatus::Idle);
        let mut progress = use_signal(|| None::<CrawlProgress>);
        let mut cancel_flag = use_signal(|| Arc::new(AtomicBool::new(false)));
        let processing_task = use_processing_sender();

        let mut handle_crawl = move || {
            let url_value = url();
            if url_value.is_empty() {
                return;
            }

            let depth = max_depth();

            // Reset status and create new cancel flag
            status.set(CrawlStatus::Crawling);
            let flag = Arc::new(AtomicBool::new(false));
            cancel_flag.set(flag.clone());
            progress.set(None);

            // Spawn crawl task
            spawn(async move {
                info!("Starting crawl from: {} (depth: {})", url_value, depth);

                let config = CrawlConfig {
                    start_url: url_value.clone(),
                    max_depth: depth,
                    same_origin_only: true,
                    max_pages: 100,
                    delay_ms: 500,
                };

                let mut engine = CrawlEngine::new(config);

                // Progress callback to update UI
                let progress_callback = move |p: CrawlProgress| {
                    progress.set(Some(p));
                };

                // Collect all pages (no callback, batch at the end)
                match engine
                    .crawl_with_progress(
                        Some(progress_callback),
                        None::<fn(&CrawlResult)>,
                        Some(flag),
                    )
                    .await
                {
                    Ok(results) => {
                        info!("Crawl complete: {} pages", results.len());

                        // Collect all successful pages into one batch
                        let file_contents: Vec<(String, String)> = results
                            .iter()
                            .filter(|r| r.success && !r.text.is_empty())
                            .map(|r| (r.url.clone(), r.text.clone()))
                            .collect();

                        if !file_contents.is_empty() {
                            let count = file_contents.len();
                            // Send all pages as ONE batch - they'll be processed one by one
                            processing_task.send(ProcessingMessage::ProcessFiles(file_contents));

                            status.set(CrawlStatus::Success { pages: count });
                        } else {
                            status.set(CrawlStatus::Error {
                                message: "No content extracted from pages".to_string(),
                            });
                        }
                        progress.set(None);
                    }
                    Err(e) => {
                        error!("Crawl failed: {}", e);
                        status.set(CrawlStatus::Error {
                            message: e.to_string(),
                        });
                        progress.set(None);
                    }
                }
            });
        };

        let handle_stop = move || {
            info!("Stopping crawl");
            cancel_flag().store(true, Ordering::Relaxed);
        };

        let is_crawling = matches!(status(), CrawlStatus::Crawling);

        rsx! {
            section { class: "cm-upload-card",
                div { class: "cm-upload-body",
                    p { class: "cm-crawler-subtitle",
                        "Enter a URL to crawl and index web pages for semantic search"
                    }

                    // Controls row: URL input + max depth + buttons
                    div { class: "cm-crawler-controls",
                        div { class: "cm-crawler-input-group",
                            input {
                                r#type: "url",
                                class: "cm-crawler-url-input",
                                placeholder: "https://example.com/docs",
                                value: "{url}",
                                disabled: is_crawling,
                                oninput: move |evt| url.set(evt.value()),
                                onkeydown: move |evt: Event<KeyboardData>| {
                                    if evt.key() == Key::Enter && !is_crawling {
                                        handle_crawl();
                                    }
                                }
                            }

                            select {
                                class: "cm-crawler-depth-select",
                                disabled: is_crawling,
                                value: "{max_depth}",
                                onchange: move |evt| {
                                    if let Ok(depth) = evt.value().parse::<usize>() {
                                        max_depth.set(depth);
                                    }
                                },
                                option { value: "0", "Current page only" }
                                option { value: "1", selected: max_depth() == 1, "Depth 1" }
                                option { value: "2", "Depth 2" }
                                option { value: "3", "Depth 3" }
                                option { value: "5", "Depth 5" }
                                option { value: "999", "All (unlimited)" }
                            }
                        }

                        div { class: "cm-crawler-button-group",
                            button {
                                class: "cm-crawler-button",
                                disabled: url().is_empty() || is_crawling,
                                onclick: move |_| handle_crawl(),
                                "Crawl"
                            }

                            if is_crawling {
                                button {
                                    class: "cm-crawler-button cm-crawler-button--stop",
                                    onclick: move |_| handle_stop(),
                                    "Stop"
                                }
                            }
                        }
                    }

                    // Progress cards (show when crawling)
                    if let Some(p) = progress() {
                        div { class: "cm-crawler-progress",
                            // Current target card
                            if let Some(current) = &p.current_url {
                                div { class: "cm-crawler-progress-card",
                                    div { class: "cm-crawler-progress-label", "Currently crawling:" }
                                    div { class: "cm-crawler-progress-url", "{current}" }
                                    div { class: "cm-crawler-progress-stats",
                                        "{p.completed_count} completed · {p.visited_count} visited"
                                    }
                                }
                            }

                            // Queue card
                            if !p.queue.is_empty() {
                                div { class: "cm-crawler-progress-card cm-crawler-progress-card--queue",
                                    div { class: "cm-crawler-progress-label",
                                        "Queue ({p.queue.len()} pending):"
                                    }
                                    div { class: "cm-crawler-queue-list",
                                        for (idx, queued_url) in p.queue.iter().take(5).enumerate() {
                                            div { class: "cm-crawler-queue-item",
                                                key: "{idx}",
                                                "{queued_url}"
                                            }
                                        }
                                        if p.queue.len() > 5 {
                                            div { class: "cm-crawler-queue-more",
                                                "...and {p.queue.len() - 5} more"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Status display
                    match status() {
                        CrawlStatus::Idle => rsx! {
                            div { class: "cm-crawler-status cm-crawler-status--idle",
                                "Ready to crawl"
                            }
                        },
                        CrawlStatus::Crawling => rsx! {
                            div { class: "cm-crawler-status cm-crawler-status--crawling",
                                if let Some(p) = progress() {
                                    "Crawling... {p.completed_count} pages indexed"
                                } else {
                                    "Initializing crawler..."
                                }
                            }
                        },
                        CrawlStatus::Success { pages } => rsx! {
                            div { class: "cm-crawler-status cm-crawler-status--success",
                                "Successfully crawled {pages} pages"
                            }
                        },
                        CrawlStatus::Error { message } => rsx! {
                            div { class: "cm-crawler-status cm-crawler-status--error",
                                "Error: {message}"
                            }
                        },
                    }
                }
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    enum CrawlStatus {
        Idle,
        Crawling,
        Success { pages: usize },
        Error { message: String },
    }
}

// WASM stub (no-op, CORS restrictions prevent web crawling)
#[cfg(target_arch = "wasm32")]
mod wasm {
    use dioxus::prelude::*;

    /// Stub component for WASM (renders nothing due to CORS restrictions)
    #[component]
    pub fn WebCrawlerCard() -> Element {
        rsx! {}
    }
}

// Re-export the appropriate implementation
#[cfg(not(target_arch = "wasm32"))]
pub use desktop::WebCrawlerCard;

#[cfg(target_arch = "wasm32")]
pub use wasm::WebCrawlerCard;
