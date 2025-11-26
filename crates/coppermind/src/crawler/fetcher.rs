//! HTTP fetching utilities for web crawling.
//!
//! This module wraps reqwest to provide a simple interface for fetching HTML pages.
//! reqwest works on both native and WASM platforms:
//! - Native: Uses hyper with rustls-tls for HTTPS
//! - WASM: Uses browser fetch() API internally
//!
//! The HTTP client is pooled for connection reuse, improving performance significantly
//! when crawling multiple pages from the same domain.

use super::CrawlError;
use once_cell::sync::Lazy;
#[cfg(feature = "profile")]
use tracing::instrument;

/// Global HTTP client for connection pooling.
///
/// reqwest::Client handles connection pooling internally, so reusing a single
/// client across requests is much more efficient than creating one per request.
/// This is especially important for crawling where we make many requests to
/// the same domain.
///
/// Configured with:
/// - 30 second timeout per request
/// - Custom user agent identifying Coppermind
/// - Up to 10 idle connections per host for connection reuse
static HTTP_CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .user_agent("Coppermind/0.1.0 (local-first semantic search engine)")
        .timeout(std::time::Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .build()
        .expect("Failed to build HTTP client")
});

/// Fetches HTML content from a URL.
///
/// **Example:**
/// ```no_run
/// use coppermind::crawler::fetcher::fetch_html;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let (html, status_code) = fetch_html("https://example.com").await?;
///     println!("Fetched {} bytes (status: {})", html.len(), status_code);
///     Ok(())
/// }
/// ```
#[cfg_attr(feature = "profile", instrument(skip_all, fields(url)))]
pub async fn fetch_html(url: &str) -> Result<(String, u16), CrawlError> {
    // Validate URL format
    let parsed_url =
        url::Url::parse(url).map_err(|e| CrawlError::InvalidUrl(format!("{}: {}", url, e)))?;

    // Ensure HTTP or HTTPS scheme
    if parsed_url.scheme() != "http" && parsed_url.scheme() != "https" {
        return Err(CrawlError::InvalidUrl(format!(
            "Unsupported scheme: {} (only http/https allowed)",
            parsed_url.scheme()
        )));
    }

    // Get pooled HTTP client (reuses connections)
    let client = &*HTTP_CLIENT;

    // Fetch the page
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| CrawlError::RequestFailed(format!("Failed to fetch {}: {}", url, e)))?;

    let status_code = response.status().as_u16();

    // Check Content-Type header to skip non-HTML content
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    // Only accept text/html or text/plain (or unspecified)
    // Skip images, PDFs, videos, etc.
    if !content_type.is_empty()
        && !content_type.contains("text/html")
        && !content_type.contains("text/plain")
        && !content_type.contains("application/xhtml")
    {
        return Err(CrawlError::RequestFailed(format!(
            "Skipping non-HTML content: {}",
            content_type
        )));
    }

    // Get response body as bytes first to check if binary
    let bytes = response
        .bytes()
        .await
        .map_err(|e| CrawlError::RequestFailed(format!("Failed to read response body: {}", e)))?;

    // Check if content is binary (contains NULL bytes or non-UTF8)
    if content_inspector::inspect(&bytes).is_binary() {
        return Err(CrawlError::RequestFailed(
            "Skipping binary content".to_string(),
        ));
    }

    // Decode as UTF-8 text
    let html = String::from_utf8(bytes.to_vec())
        .map_err(|e| CrawlError::RequestFailed(format!("Content is not valid UTF-8: {}", e)))?;

    Ok((html, status_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_invalid_url() {
        let result = fetch_html("not a url").await;
        assert!(matches!(result, Err(CrawlError::InvalidUrl(_))));
    }

    #[tokio::test]
    async fn test_invalid_scheme() {
        let result = fetch_html("ftp://example.com").await;
        assert!(matches!(result, Err(CrawlError::InvalidUrl(_))));
    }
}
