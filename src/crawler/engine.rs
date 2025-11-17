//! Crawl engine with recursive page fetching and cycle detection.
//!
//! This module implements the main crawling logic:
//! - BFS traversal of linked pages
//! - Cycle detection (don't visit same URL twice)
//! - Depth limiting
//! - Same-origin filtering
//! - Politeness delays

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dioxus::logger::tracing::{error, info};
use tokio::time::sleep;

use super::fetcher::fetch_html;
use super::parser::{extract_links, extract_text};
use super::{CrawlConfig, CrawlError, CrawlProgress, CrawlResult};

/// Normalizes a URL for deduplication purposes.
///
/// This removes trailing slashes from paths to treat:
/// - `https://example.com/page/` and `https://example.com/page` as identical
/// - `https://example.com/` becomes `https://example.com`
///
/// This prevents crawling the same page multiple times with different URL formats.
fn normalize_url(url: &str) -> String {
    // Remove trailing slash from path (but not from root)
    if url.ends_with('/') && url.matches('/').count() > 2 {
        // Has path component beyond scheme:// - safe to remove trailing slash
        url.trim_end_matches('/').to_string()
    } else {
        url.to_string()
    }
}

/// Crawl engine for fetching and processing web pages.
pub struct CrawlEngine {
    config: CrawlConfig,
    visited: HashSet<String>,
    queue: VecDeque<(String, usize)>, // (url, depth)
}

impl CrawlEngine {
    /// Creates a new crawl engine with the given configuration.
    pub fn new(config: CrawlConfig) -> Self {
        Self {
            config,
            visited: HashSet::new(),
            queue: VecDeque::new(),
        }
    }

    /// Crawls pages starting from the configured start URL.
    ///
    /// Takes optional progress callback, page callback, and cancellation flag.
    ///
    /// Returns a vector of successfully crawled pages.
    ///
    /// **Example:**
    /// ```no_run
    /// use coppermind::crawler::{CrawlConfig, CrawlEngine};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let config = CrawlConfig {
    ///         start_url: "https://example.com".to_string(),
    ///         max_depth: 1,
    ///         same_origin_only: true,
    ///         max_pages: 10,
    ///         delay_ms: 1000,
    ///     };
    ///
    ///     let mut engine = CrawlEngine::new(config);
    ///     // Use the simple crawl() method for backward compatibility
    ///     let results = engine.crawl().await?;
    ///     println!("Crawled {} pages", results.len());
    ///     Ok(())
    /// }
    /// ```
    pub async fn crawl_with_progress<F, G>(
        &mut self,
        mut progress_callback: Option<F>,
        mut page_callback: Option<G>,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> Result<Vec<CrawlResult>, CrawlError>
    where
        F: FnMut(CrawlProgress),
        G: FnMut(&CrawlResult),
    {
        // Initialize queue with start URL at depth 0
        self.queue.push_back((self.config.start_url.clone(), 0));
        let mut results = Vec::new();

        // Parse start URL to get origin for same-origin checking
        let start_url = url::Url::parse(&self.config.start_url)
            .map_err(|e| CrawlError::InvalidUrl(format!("Invalid start URL: {}", e)))?;

        info!(
            "Starting crawl from {} (max_depth: {}, max_pages: {})",
            self.config.start_url, self.config.max_depth, self.config.max_pages
        );

        // BFS traversal
        while let Some((url, depth)) = self.queue.pop_front() {
            // Check cancellation
            if let Some(ref flag) = cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    info!("Crawl cancelled by user");
                    break;
                }
            }

            // Check limits
            if results.len() >= self.config.max_pages {
                info!(
                    "Reached max_pages limit ({}), stopping crawl",
                    self.config.max_pages
                );
                break;
            }

            if depth > self.config.max_depth {
                continue;
            }

            // Normalize URL to handle trailing slashes
            let normalized_url = normalize_url(&url);

            // Skip if already visited (using normalized URL)
            if self.visited.contains(&normalized_url) {
                continue;
            }

            self.visited.insert(normalized_url);

            // Send progress update
            if let Some(ref mut callback) = progress_callback {
                let progress = CrawlProgress {
                    current_url: Some(url.clone()),
                    queue: self.queue.iter().map(|(u, _)| u.clone()).collect(),
                    visited_count: self.visited.len(),
                    completed_count: results.len(),
                };
                callback(progress);
            }

            // Politeness delay (except for first page)
            if !results.is_empty() && self.config.delay_ms > 0 {
                sleep(Duration::from_millis(self.config.delay_ms)).await;
            }

            // Fetch and parse page
            info!("Crawling: {} (depth: {})", url, depth);

            match self.crawl_page(&url, depth, &start_url).await {
                Ok(result) => {
                    // Add discovered links to queue if we haven't reached max depth
                    if depth < self.config.max_depth {
                        for link in &result.links {
                            let normalized_link = normalize_url(link);
                            if !self.visited.contains(&normalized_link) {
                                self.queue.push_back((link.clone(), depth + 1));
                            }
                        }
                    }

                    // Call page callback immediately (for streaming indexing)
                    if let Some(ref mut callback) = page_callback {
                        callback(&result);
                    }

                    results.push(result);
                }
                Err(e) => {
                    // Use info level for expected skips (binary/non-HTML content)
                    // Use error level for actual failures (network errors, etc.)
                    let error_msg = e.to_string();
                    if error_msg.contains("Skipping") {
                        info!("Skipped {}: {}", url, error_msg);
                    } else {
                        error!("Failed to crawl {}: {}", url, e);
                    }
                    // Continue crawling other pages even if one fails
                    continue;
                }
            }
        }

        // Send final progress update
        if let Some(ref mut callback) = progress_callback {
            let progress = CrawlProgress {
                current_url: None,
                queue: Vec::new(),
                visited_count: self.visited.len(),
                completed_count: results.len(),
            };
            callback(progress);
        }

        info!("Crawl complete: {} pages crawled", results.len());
        Ok(results)
    }

    /// Legacy crawl method without progress tracking (for backward compatibility).
    pub async fn crawl(&mut self) -> Result<Vec<CrawlResult>, CrawlError> {
        self.crawl_with_progress::<fn(CrawlProgress), fn(&CrawlResult)>(None, None, None)
            .await
    }

    /// Crawls a single page and returns the result.
    async fn crawl_page(
        &self,
        url: &str,
        _depth: usize,
        start_url: &url::Url,
    ) -> Result<CrawlResult, CrawlError> {
        // Fetch HTML
        let (html, status_code) = fetch_html(url).await?;

        // Extract text content
        let text = extract_text(&html)?;

        // Extract links
        let mut links = extract_links(&html, url)?;

        // Filter same-origin links if configured
        if self.config.same_origin_only {
            links = self.filter_same_origin(links, start_url)?;
        }

        Ok(CrawlResult {
            url: url.to_string(),
            text,
            links,
            status_code,
            success: true,
        })
    }

    /// Filters links to only include same-origin URLs under the start path.
    ///
    /// For example, if start_url is `https://example.com/docs/guide`, only links
    /// under `https://example.com/docs/` will be included.
    fn filter_same_origin(
        &self,
        links: Vec<String>,
        start_url: &url::Url,
    ) -> Result<Vec<String>, CrawlError> {
        let mut filtered = Vec::new();

        // Extract base path from start URL (directory containing the start page)
        // e.g., "https://example.com/docs/guide" -> "/docs/"
        // e.g., "https://example.com/docs/" -> "/docs/"
        let start_path = start_url.path();
        let base_path = if start_path.ends_with('/') {
            start_path.to_string()
        } else {
            // Get parent directory by removing everything after the last '/'
            match start_path.rfind('/') {
                Some(idx) => start_path[..=idx].to_string(),
                None => "/".to_string(), // Shouldn't happen for valid URLs
            }
        };

        for link in links {
            let parsed = url::Url::parse(&link)
                .map_err(|e| CrawlError::InvalidUrl(format!("Invalid link {}: {}", link, e)))?;

            // Check if origin matches (scheme + host + port) and path is under base_path
            if parsed.origin() == start_url.origin() && parsed.path().starts_with(&base_path) {
                filtered.push(link);
            }
        }

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_same_origin_with_path() {
        let engine = CrawlEngine::new(CrawlConfig::default());
        let start_url = url::Url::parse("https://example.com/docs/intro").unwrap();

        let links = vec![
            "https://example.com/docs/guide".to_string(), // ✓ Under /docs/
            "https://example.com/docs/api/rest".to_string(), // ✓ Under /docs/
            "https://example.com/api".to_string(),        // ✗ Not under /docs/
            "https://other.com/docs/page".to_string(),    // ✗ Different origin
            "http://example.com/docs/insecure".to_string(), // ✗ Different scheme
        ];

        let filtered = engine.filter_same_origin(links, &start_url).unwrap();

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&"https://example.com/docs/guide".to_string()));
        assert!(filtered.contains(&"https://example.com/docs/api/rest".to_string()));
    }

    #[test]
    fn test_filter_same_origin_directory_url() {
        let engine = CrawlEngine::new(CrawlConfig::default());
        // Start URL is a directory (ends with /)
        let start_url = url::Url::parse("https://example.com/docs/").unwrap();

        let links = vec![
            "https://example.com/docs/guide".to_string(), // ✓ Under /docs/
            "https://example.com/docs/api/rest".to_string(), // ✓ Under /docs/
            "https://example.com/api".to_string(),        // ✗ Not under /docs/
        ];

        let filtered = engine.filter_same_origin(links, &start_url).unwrap();

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&"https://example.com/docs/guide".to_string()));
        assert!(filtered.contains(&"https://example.com/docs/api/rest".to_string()));
    }

    #[test]
    fn test_filter_same_origin_root_url() {
        let engine = CrawlEngine::new(CrawlConfig::default());
        // Start URL is the root - should allow all same-origin links
        let start_url = url::Url::parse("https://example.com/").unwrap();

        let links = vec![
            "https://example.com/docs".to_string(), // ✓ Under /
            "https://example.com/api".to_string(),  // ✓ Under /
            "https://other.com/page".to_string(),   // ✗ Different origin
        ];

        let filtered = engine.filter_same_origin(links, &start_url).unwrap();

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&"https://example.com/docs".to_string()));
        assert!(filtered.contains(&"https://example.com/api".to_string()));
    }

    #[test]
    fn test_normalize_url_removes_trailing_slash() {
        assert_eq!(
            normalize_url("https://example.com/page/"),
            "https://example.com/page"
        );
        assert_eq!(
            normalize_url("https://example.com/docs/guide/"),
            "https://example.com/docs/guide"
        );
    }

    #[test]
    fn test_normalize_url_handles_root() {
        // Root URLs with trailing slash get normalized
        assert_eq!(normalize_url("https://example.com/"), "https://example.com");
        // Already without trailing slash
        assert_eq!(normalize_url("https://example.com"), "https://example.com");
    }

    #[test]
    fn test_normalize_url_no_trailing_slash() {
        // URLs without trailing slash should remain unchanged
        assert_eq!(
            normalize_url("https://example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn test_normalize_url_deduplication() {
        // These should normalize to the same value
        let url1 = "https://example.com/page/";
        let url2 = "https://example.com/page";
        assert_eq!(normalize_url(url1), normalize_url(url2));
    }
}
