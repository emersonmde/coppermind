//! Web crawler module for fetching and indexing web pages.
//!
//! This module provides functionality to crawl web pages, extract text content,
//! and feed it into the existing embedding pipeline for semantic search.
//!
//! **Platform Support:**
//! - Desktop: Full support (no CORS restrictions)
//! - Web (WASM): Architecture ready, but disabled due to CORS restrictions
//!   - Future: Could enable via browser extension or CORS proxy
//!
//! **Module Organization:**
//! - `mod.rs`: Public API, configuration types, and error handling
//! - `fetcher.rs`: HTTP fetching with reqwest
//! - `parser.rs`: HTML parsing and text extraction with scraper
//! - `engine.rs`: Recursive crawl logic with cycle detection

use serde::{Deserialize, Serialize};
use thiserror::Error;

// Re-export submodules conditionally (desktop only for now)
#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
#[cfg(not(target_arch = "wasm32"))]
pub mod fetcher;
#[cfg(not(target_arch = "wasm32"))]
pub mod parser;

// Re-export main functionality
#[cfg(not(target_arch = "wasm32"))]
pub use engine::CrawlEngine;

/// Configuration for web crawling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlConfig {
    /// Starting URL to begin crawling from.
    pub start_url: String,

    /// Maximum depth to crawl (0 = single page, 1 = start page + linked pages, etc.).
    pub max_depth: usize,

    /// Only follow links with the same origin as start_url.
    ///
    /// **Example:**
    /// - Start: `https://example.com/docs`
    /// - Same origin: `https://example.com/docs/guide` ✅
    /// - Different origin: `https://other.com` ❌
    pub same_origin_only: bool,

    /// Maximum number of pages to crawl (prevents runaway crawling).
    pub max_pages: usize,

    /// Politeness delay between requests (milliseconds).
    pub delay_ms: u64,

    /// Number of pages to fetch in parallel.
    ///
    /// Higher values = faster crawling, but more load on target server.
    /// Recommended values: 1 (sequential), 2, 4, 8, 16.
    pub parallel_requests: usize,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            start_url: String::new(),
            max_depth: 1,
            same_origin_only: true,
            max_pages: 100,
            delay_ms: 1000,       // 1 second between requests
            parallel_requests: 2, // Default to 2 parallel requests
        }
    }
}

/// Result of crawling a single page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlResult {
    /// URL of the crawled page.
    pub url: String,

    /// Extracted visible text content (HTML tags stripped).
    pub text: String,

    /// All links found on the page (absolute URLs).
    pub links: Vec<String>,

    /// HTTP status code (200, 404, etc.).
    pub status_code: u16,

    /// Whether the fetch was successful.
    pub success: bool,
}

/// Progress update during crawling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlProgress {
    /// URLs currently being fetched in parallel.
    pub current_urls: Vec<String>,

    /// URLs in queue waiting to be crawled.
    pub queue: Vec<String>,

    /// URLs already visited.
    pub visited_count: usize,

    /// Pages successfully crawled so far.
    pub completed_count: usize,
}

/// Errors that can occur during web crawling.
#[derive(Debug, Error)]
pub enum CrawlError {
    /// HTTP request failed (network error, timeout, etc.).
    #[error("HTTP request failed: {0}")]
    RequestFailed(String),

    /// Invalid URL format.
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    /// HTML parsing failed.
    #[error("HTML parsing failed: {0}")]
    ParseError(String),

    /// Crawl limit exceeded (max_pages or max_depth).
    #[error("Crawl limit exceeded: {0}")]
    LimitExceeded(String),

    /// Feature not available on current platform (web/WASM).
    #[error("Web crawler not available on WASM platform (CORS restrictions)")]
    PlatformNotSupported,
}
