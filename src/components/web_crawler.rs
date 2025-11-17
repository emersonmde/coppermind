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
        let mut parallel_requests = use_signal(|| 2); // Default to 2 parallel requests
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
            let workers = parallel_requests();

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
                    parallel_requests: workers,
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

        // Compute crawling status text with transparency about failures
        // Note: "downloaded" = successfully fetched, not yet indexed (indexing happens after crawl completes)
        let crawling_status_text = if matches!(status(), CrawlStatus::Crawling) {
            if let Some(p) = progress() {
                let failed = p.visited_count.saturating_sub(p.completed_count);
                if failed > 0 {
                    format!(
                        "Crawling... {} downloaded, {} failed ({} total)",
                        p.completed_count, failed, p.visited_count
                    )
                } else {
                    format!(
                        "Crawling... {} downloaded ({} crawled)",
                        p.completed_count, p.visited_count
                    )
                }
            } else {
                "Initializing crawler...".to_string()
            }
        } else {
            String::new()
        };

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

                            select {
                                class: "cm-crawler-depth-select",
                                disabled: is_crawling,
                                value: "{parallel_requests}",
                                onchange: move |evt| {
                                    if let Ok(workers) = evt.value().parse::<usize>() {
                                        parallel_requests.set(workers);
                                    }
                                },
                                option { value: "1", "1 request" }
                                option { value: "2", selected: parallel_requests() == 2, "2 parallel" }
                                option { value: "4", "4 parallel" }
                                option { value: "8", "8 parallel" }
                                option { value: "16", "16 parallel" }
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
                            // Currently fetching cards (show all parallel requests)
                            if !p.current_urls.is_empty() {
                                div { class: "cm-crawler-progress-card",
                                    div { class: "cm-crawler-progress-label",
                                        if p.current_urls.len() == 1 {
                                            "Currently fetching:"
                                        } else {
                                            "Currently fetching ({p.current_urls.len()} parallel):"
                                        }
                                    }
                                    div { class: "cm-crawler-queue-list",
                                        for (idx, current_url) in p.current_urls.iter().enumerate() {
                                            div { class: "cm-crawler-queue-item",
                                                key: "{idx}",
                                                "{current_url}"
                                            }
                                        }
                                    }
                                    div { class: "cm-crawler-progress-stats",
                                        "{p.completed_count} downloaded · {p.visited_count} crawled"
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
                                "{crawling_status_text}"
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

    /// Stub component for WASM showing desktop download prompt
    #[component]
    pub fn WebCrawlerCard() -> Element {
        rsx! {
            section { class: "cm-upload-card",
                div { class: "cm-upload-body",
                    p { class: "cm-crawler-subtitle",
                        "Web crawling is not available in the browser due to CORS restrictions."
                    }

                    div { class: "cm-crawler-desktop-prompt",
                        p { class: "cm-crawler-desktop-text",
                            "Download the Desktop version to use the web crawler"
                        }

                        a {
                            class: "cm-btn cm-btn--primary",
                            href: "https://github.com/emersonmde/coppermind",
                            target: "_blank",
                            rel: "noopener noreferrer",
                            "Download Desktop App"
                        }
                    }
                }
            }
        }
    }
}

// Re-export the appropriate implementation
#[cfg(not(target_arch = "wasm32"))]
pub use desktop::WebCrawlerCard;

#[cfg(target_arch = "wasm32")]
pub use wasm::WebCrawlerCard;
