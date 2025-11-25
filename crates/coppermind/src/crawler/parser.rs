//! HTML parsing and text extraction utilities.
//!
//! This module uses the scraper crate (pure Rust, WASM-compatible) to parse HTML
//! and extract visible text content and links.

use scraper::{Html, Selector};

use super::CrawlError;

/// Extracts visible text from HTML by stripping all tags.
///
/// This removes:
/// - All HTML tags (`<div>`, `<p>`, etc.)
/// - Script and style content
/// - HTML comments
///
/// **Example:**
/// ```
/// use coppermind::crawler::parser::extract_text;
///
/// let html = r#"<html><body><h1>Hello</h1><p>World</p></body></html>"#;
/// let text = extract_text(html).unwrap();
/// assert!(text.contains("Hello"));
/// assert!(text.contains("World"));
/// ```
pub fn extract_text(html: &str) -> Result<String, CrawlError> {
    let document = Html::parse_document(html);

    // Selectors for elements to skip (currently checked inline in extract_text_recursive)
    let _skip_selector = Selector::parse("script, style, noscript, iframe, svg, path")
        .map_err(|e| CrawlError::ParseError(format!("Invalid CSS selector: {:?}", e)))?;

    let body_selector = Selector::parse("body")
        .map_err(|e| CrawlError::ParseError(format!("Invalid CSS selector: {:?}", e)))?;

    // Extract visible text from body only (skip <head> content)
    let mut text_parts = Vec::new();

    if let Some(body) = document.select(&body_selector).next() {
        // Recursively extract text, skipping unwanted elements
        extract_text_recursive(body, &mut text_parts);
    } else {
        // Fallback: no <body> tag, extract from entire document but skip head
        let html_selector = Selector::parse("html").ok();
        if let Some(selector) = html_selector {
            if let Some(html_element) = document.select(&selector).next() {
                extract_text_recursive(html_element, &mut text_parts);
            }
        }
    }

    // Join with newlines and clean up excessive whitespace
    let text = text_parts
        .into_iter()
        .filter(|s| !s.is_empty() && !is_script_content(s))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(text)
}

/// Recursively extract text nodes while skipping unwanted elements
fn extract_text_recursive(element: scraper::element_ref::ElementRef, text_parts: &mut Vec<String>) {
    use scraper::node::Node;

    // Check if this element should be skipped
    if element.value().name() == "script"
        || element.value().name() == "style"
        || element.value().name() == "noscript"
        || element.value().name() == "iframe"
        || element.value().name() == "svg"
        || element.value().name() == "head"
    {
        return;
    }

    // Extract direct text nodes
    for child in element.children() {
        match child.value() {
            Node::Text(text) => {
                let trimmed = text.trim();
                if !trimmed.is_empty() && !is_script_content(trimmed) {
                    text_parts.push(trimmed.to_string());
                }
            }
            Node::Element(_) => {
                // Recursively process child elements
                if let Some(child_element) = scraper::ElementRef::wrap(child) {
                    extract_text_recursive(child_element, text_parts);
                }
            }
            _ => {}
        }
    }
}

/// Heuristic to detect JavaScript/CSS content that leaked through
fn is_script_content(text: &str) -> bool {
    // Check for common JS/CSS patterns
    let patterns = [
        "function(",
        "const ",
        "let ",
        "var ",
        "=>",
        "document.",
        "window.",
        "typeof",
        "undefined",
        "null===",
        "!==",
        "querySelector",
        "addEventListener",
        "{display:",
        "position:",
        "opacity:",
        "srcSet=",
        "data-",
    ];

    // If text contains multiple JS/CSS indicators, it's likely script content
    let indicator_count = patterns.iter().filter(|&p| text.contains(p)).count();
    indicator_count >= 2
}

/// Extracts all links from HTML and converts them to absolute URLs.
///
/// **Example:**
/// ```
/// use coppermind::crawler::parser::extract_links;
///
/// let html = r#"<html><body><a href="/docs">Docs</a></body></html>"#;
/// let base_url = "https://example.com/page";
/// let links = extract_links(html, base_url).unwrap();
/// assert_eq!(links, vec!["https://example.com/docs"]);
/// ```
pub fn extract_links(html: &str, base_url: &str) -> Result<Vec<String>, CrawlError> {
    let document = Html::parse_document(html);

    // Parse base URL for resolving relative links
    let base = url::Url::parse(base_url)
        .map_err(|e| CrawlError::InvalidUrl(format!("Invalid base URL {}: {}", base_url, e)))?;

    // Select all <a> tags with href attribute
    let link_selector = Selector::parse("a[href]")
        .map_err(|e| CrawlError::ParseError(format!("Invalid CSS selector: {:?}", e)))?;

    let mut links = Vec::new();

    for element in document.select(&link_selector) {
        if let Some(href) = element.value().attr("href") {
            // Skip empty hrefs, anchors, and javascript: links
            if href.is_empty() || href.starts_with('#') || href.starts_with("javascript:") {
                continue;
            }

            // Convert relative URLs to absolute
            match base.join(href) {
                Ok(absolute_url) => {
                    // Only include http/https URLs
                    if absolute_url.scheme() == "http" || absolute_url.scheme() == "https" {
                        links.push(absolute_url.to_string());
                    }
                }
                Err(_) => {
                    // Ignore malformed URLs
                    continue;
                }
            }
        }
    }

    // Deduplicate links (preserve order)
    let mut seen = std::collections::HashSet::new();
    links.retain(|link| seen.insert(link.clone()));

    Ok(links)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_basic() {
        let html = r#"<html><body><h1>Hello</h1><p>World</p></body></html>"#;
        let text = extract_text(html).unwrap();
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_extract_text_strips_scripts() {
        let html = r#"
            <html>
                <head><script>alert('bad');</script></head>
                <body><p>Good content</p></body>
            </html>
        "#;
        let text = extract_text(html).unwrap();
        assert!(text.contains("Good content"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn test_extract_links_absolute() {
        let html = r#"<a href="https://example.com/page">Link</a>"#;
        let links = extract_links(html, "https://base.com").unwrap();
        assert_eq!(links, vec!["https://example.com/page"]);
    }

    #[test]
    fn test_extract_links_relative() {
        let html = r#"<a href="/docs">Docs</a>"#;
        let links = extract_links(html, "https://example.com/page").unwrap();
        assert_eq!(links, vec!["https://example.com/docs"]);
    }

    #[test]
    fn test_extract_links_skips_anchors() {
        let html = r##"<a href="#section">Section</a>"##;
        let links = extract_links(html, "https://example.com").unwrap();
        assert!(links.is_empty());
    }

    #[test]
    fn test_extract_links_deduplicates() {
        let html = r#"
            <a href="/page">Link 1</a>
            <a href="/page">Link 2</a>
        "#;
        let links = extract_links(html, "https://example.com").unwrap();
        assert_eq!(links, vec!["https://example.com/page"]);
    }
}
