//! File processing utilities for document indexing.
//!
//! This module provides utilities for processing uploaded files, including:
//! - Binary file detection
//! - Content hashing for update detection
//! - Source ID generation (platform-specific)
//! - Search engine indexing with source tracking
//! - Directory traversal (desktop only)

use crate::embedding::ChunkEmbeddingResult;
use crate::error::FileProcessingError;
use crate::metrics::global_metrics;
#[cfg(not(target_arch = "wasm32"))]
use crate::platform::run_blocking;
use crate::search::types::{get_current_timestamp, Document, DocumentMetadata};
use crate::search::HybridSearchEngine;
use crate::storage::DocumentStore;
#[cfg(feature = "desktop")]
use dioxus::logger::tracing::error;
use dioxus::logger::tracing::info;
use futures::lock::Mutex;
use instant::Instant;
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[cfg(feature = "profile")]
use tracing::instrument;

#[cfg(feature = "desktop")]
use std::{future::Future, path::PathBuf, pin::Pin};

/// Detects if file contents appear to be binary (not text).
///
/// Uses the `content_inspector` crate which provides industry-standard heuristics:
/// - NULL bytes (`\0`) indicating binary content
/// - Byte order marks (BOMs) for text encoding detection
/// - UTF-8 validity checks
///
/// # Arguments
///
/// * `content` - File contents as a string (already validated as UTF-8)
///
/// # Returns
///
/// `true` if the content appears to be binary, `false` if it's likely text.
///
/// # Examples
///
/// ```ignore
/// // Internal module - not part of public API
/// use coppermind::components::file_processing::is_likely_binary;
///
/// assert!(!is_likely_binary("Hello, world!"));
/// assert!(is_likely_binary("Binary\0data"));
/// ```
pub fn is_likely_binary(content: &str) -> bool {
    content_inspector::inspect(content.as_bytes()).is_binary()
}

/// Computes a SHA-256 hash of content for change detection.
///
/// Returns the hash as a lowercase hexadecimal string.
pub fn compute_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// Generates a platform-specific source ID for a file or URL.
///
/// # Platform Behavior
///
/// - **Desktop**: Uses the input as-is. This works for:
///   - File paths (e.g., `/Users/matt/docs/README.md`)
///   - URLs from the web crawler (e.g., `https://example.com/docs/intro`)
/// - **Web**: Uses `web:{filename}` format (e.g., `web:README.md`)
///
/// This function handles the Dioxus web limitation where only filename is available
/// (no `webkitRelativePath` support per [issue #3136](https://github.com/DioxusLabs/dioxus/issues/3136)).
///
/// # Arguments
///
/// * `file_path` - Full path on desktop, URL for crawled pages, filename on web
///
/// # Crawler Integration
///
/// When the web crawler fetches pages, it passes the full URL as the "filename".
/// This means re-crawling the same URL will:
/// 1. Detect the existing source via `source_needs_update()`
/// 2. Compare content hashes to determine if the page changed
/// 3. Update (delete old + add new) if changed, or skip if unchanged
///
/// # Web Under-Indexing Strategy
///
/// On web, multiple files with the same name (from different directories) will
/// have the same source_id. The chosen strategy is "under-indexing": replace
/// the existing file rather than accumulate duplicates. This provides cleaner
/// search results and matches user expectations (re-upload = update).
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_source_id(file_path: &str) -> String {
    // Desktop: Use full file path or URL as stable anchor
    file_path.to_string()
}

#[cfg(target_arch = "wasm32")]
pub fn generate_source_id(file_path: &str) -> String {
    // Web: Only have filename, use web: prefix for clarity
    // Extract filename from path in case user provides relative path
    let filename = file_path.rsplit('/').next().unwrap_or(file_path);
    format!("web:{}", filename)
}

/// Result of checking if a source needs to be updated.
#[derive(Debug, Clone)]
pub enum SourceUpdateAction {
    /// New source, add all chunks normally
    Add,
    /// Existing source unchanged, skip processing
    Skip,
    /// Existing source changed, delete old chunks then add new
    Update { old_chunk_count: usize },
}

/// Checks if a source needs to be indexed, updated, or skipped.
///
/// # Arguments
///
/// * `engine` - Search engine to check against
/// * `source_id` - Stable identifier for the source
/// * `content_hash` - SHA-256 hash of current content
///
/// # Returns
///
/// - `Add`: New source, process normally
/// - `Skip`: Source exists with same hash, skip processing
/// - `Update`: Source exists with different hash, need to replace
pub async fn check_source_update<S: DocumentStore>(
    engine: &HybridSearchEngine<S>,
    source_id: &str,
    content_hash: &str,
) -> Result<SourceUpdateAction, FileProcessingError> {
    // Check if source exists and if it needs updating
    match engine.source_needs_update(source_id, content_hash).await {
        Ok(needs_update) => {
            if needs_update {
                // Source exists with different hash - need to update
                match engine.get_source(source_id).await {
                    Ok(Some(record)) => Ok(SourceUpdateAction::Update {
                        old_chunk_count: record.doc_ids.len(),
                    }),
                    Ok(None) => {
                        // Race condition: source was deleted between checks
                        Ok(SourceUpdateAction::Add)
                    }
                    Err(e) => Err(FileProcessingError::IndexingFailed(format!(
                        "Failed to get source info: {:?}",
                        e
                    ))),
                }
            } else {
                // Source exists with same hash - skip
                Ok(SourceUpdateAction::Skip)
            }
        }
        Err(crate::search::SearchError::NotFound) => {
            // Source doesn't exist - add new
            Ok(SourceUpdateAction::Add)
        }
        Err(e) => Err(FileProcessingError::IndexingFailed(format!(
            "Failed to check source update: {:?}",
            e
        ))),
    }
}

/// Indexes embedding chunks in the search engine with source tracking.
///
/// Takes embedding results (which include the decoded text for each chunk)
/// and indexes them in the search engine, registering them with a source
/// for future update detection.
///
/// # Type Parameters
///
/// * `S` - Storage backend type implementing `DocumentStore`
///
/// # Arguments
///
/// * `engine` - Arc-wrapped Mutex to the search engine
/// * `embedding_results` - Vector of embedding results with chunk indices, text, and embeddings
/// * `file_label` - Label for the file (used in metadata)
/// * `source_id` - Stable identifier for the source (path on desktop, web:{filename} on web)
/// * `content_hash` - SHA-256 hash of the source content
///
/// # Returns
///
/// Number of successfully indexed chunks, or an error if indexing fails.
///
/// # Platform-Specific Behavior
///
/// - **Web**: Uses `instant::SystemTime` for timestamps
/// - **Desktop**: Uses `std::time::SystemTime`
///
/// # Examples
///
/// ```ignore
/// let indexed = index_chunks(
///     engine.clone(),
///     embedding_results,
///     "example.txt",
///     "/path/to/example.txt",
///     "abc123...",
/// ).await?;
/// println!("Indexed {} chunks", indexed);
/// ```
// Desktop: Uses blocking thread pool for CPU-intensive work, requires Send + Sync
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(feature = "profile", instrument(skip_all, fields(chunks = embedding_results.len())))]
pub async fn index_chunks<S: DocumentStore + Send + Sync + 'static>(
    engine: Arc<Mutex<HybridSearchEngine<S>>>,
    embedding_results: &[ChunkEmbeddingResult],
    file_label: &str,
    source_id: &str,
    content_hash: &str,
) -> Result<usize, FileProcessingError> {
    // Prepare documents outside the lock
    let timestamp = get_current_timestamp();
    let file_label_owned = file_label.to_string();
    let source_id_owned = source_id.to_string();
    let content_hash_owned = content_hash.to_string();

    let docs: Vec<(Document, Vec<f32>, usize)> = embedding_results
        .iter()
        .map(|chunk_result| {
            let doc = Document {
                text: chunk_result.text.clone(),
                metadata: DocumentMetadata {
                    filename: Some(format!(
                        "{} (chunk {})",
                        file_label_owned,
                        chunk_result.chunk_index + 1
                    )),
                    source: Some(source_id_owned.clone()),
                    created_at: timestamp,
                },
            };
            (
                doc,
                chunk_result.embedding.clone(),
                chunk_result.chunk_index,
            )
        })
        .collect();

    // Move all CPU-intensive indexing to blocking thread pool
    let index_start = Instant::now();
    let result = run_blocking(move || {
        // We need to block on the async lock from a sync context
        // Use futures::executor::block_on for the lock acquisition
        let mut search_engine = futures::executor::block_on(engine.lock());

        // Register the source before adding documents
        if let Err(e) = futures::executor::block_on(
            search_engine.register_source(&source_id_owned, content_hash_owned.clone()),
        ) {
            return Err(FileProcessingError::IndexingFailed(format!(
                "Failed to register source: {:?}",
                e
            )));
        }

        let mut indexed_count = 0;
        let mut total_index_time_ms = 0.0;

        for (doc, embedding, chunk_index) in docs {
            let insert_start = Instant::now();

            // add_document is async but doesn't do any actual async work
            // We can safely block_on it in the blocking thread
            match futures::executor::block_on(search_engine.add_document(doc, embedding)) {
                Ok(doc_id) => {
                    // Track this chunk as part of the source
                    if let Err(e) = futures::executor::block_on(
                        search_engine.add_doc_to_source(&source_id_owned, doc_id),
                    ) {
                        return Err(FileProcessingError::IndexingFailed(format!(
                            "Failed to track chunk {} in source: {:?}",
                            chunk_index, e
                        )));
                    }

                    let insert_duration_ms = insert_start.elapsed().as_secs_f64() * 1000.0;
                    total_index_time_ms += insert_duration_ms;
                    // Split metrics between HNSW (vector) and BM25 (keyword) indexing.
                    // 70/30 is an estimate - HNSW graph insertion is typically more expensive
                    // than BM25 term frequency updates. Actual ratio varies by document size.
                    global_metrics().record_hnsw_indexing(insert_duration_ms * 0.7);
                    global_metrics().record_bm25_indexing(insert_duration_ms * 0.3);
                    indexed_count += 1;
                }
                Err(e) => {
                    return Err(FileProcessingError::IndexingFailed(format!(
                        "Failed to index chunk {}: {:?}",
                        chunk_index, e
                    )));
                }
            }
        }

        // Mark the source as complete (all chunks successfully added)
        if let Err(e) = futures::executor::block_on(search_engine.complete_source(&source_id_owned))
        {
            return Err(FileProcessingError::IndexingFailed(format!(
                "Failed to complete source: {:?}",
                e
            )));
        }

        Ok((indexed_count, total_index_time_ms))
    })
    .await?;

    let (indexed_count, total_index_time_ms) = result;
    let total_elapsed_ms = index_start.elapsed().as_secs_f64() * 1000.0;

    info!(
        "✅ Added {} chunks to search engine ({:.1}ms total, {:.1}ms indexing)",
        indexed_count, total_elapsed_ms, total_index_time_ms
    );

    Ok(indexed_count)
}

// Web: Single-threaded, no blocking thread pool needed, no Send + Sync required
#[cfg(target_arch = "wasm32")]
pub async fn index_chunks<S: DocumentStore + 'static>(
    engine: Arc<Mutex<HybridSearchEngine<S>>>,
    embedding_results: &[ChunkEmbeddingResult],
    file_label: &str,
    source_id: &str,
    content_hash: &str,
) -> Result<usize, FileProcessingError> {
    // Prepare documents outside the lock
    let timestamp = get_current_timestamp();
    let file_label_owned = file_label.to_string();

    let docs: Vec<(Document, Vec<f32>, usize)> = embedding_results
        .iter()
        .map(|chunk_result| {
            let doc = Document {
                text: chunk_result.text.clone(),
                metadata: DocumentMetadata {
                    filename: Some(format!(
                        "{} (chunk {})",
                        file_label_owned,
                        chunk_result.chunk_index + 1
                    )),
                    source: Some(source_id.to_string()),
                    created_at: timestamp,
                },
            };
            (
                doc,
                chunk_result.embedding.clone(),
                chunk_result.chunk_index,
            )
        })
        .collect();

    // Web is single-threaded, so we can index directly without blocking thread pool
    let index_start = Instant::now();
    let mut search_engine = engine.lock().await;

    // Register the source before adding documents
    search_engine
        .register_source(source_id, content_hash.to_string())
        .await
        .map_err(|e| {
            FileProcessingError::IndexingFailed(format!("Failed to register source: {:?}", e))
        })?;

    let mut indexed_count = 0;
    let mut total_index_time_ms = 0.0;

    for (doc, embedding, chunk_index) in docs {
        let insert_start = Instant::now();

        match search_engine.add_document(doc, embedding).await {
            Ok(doc_id) => {
                // Track this chunk as part of the source
                search_engine
                    .add_doc_to_source(source_id, doc_id)
                    .await
                    .map_err(|e| {
                        FileProcessingError::IndexingFailed(format!(
                            "Failed to track chunk {} in source: {:?}",
                            chunk_index, e
                        ))
                    })?;

                let insert_duration_ms = insert_start.elapsed().as_secs_f64() * 1000.0;
                total_index_time_ms += insert_duration_ms;
                // Split metrics between HNSW (vector) and BM25 (keyword) indexing.
                // 70/30 is an estimate - HNSW graph insertion is typically more expensive
                // than BM25 term frequency updates. Actual ratio varies by document size.
                global_metrics().record_hnsw_indexing(insert_duration_ms * 0.7);
                global_metrics().record_bm25_indexing(insert_duration_ms * 0.3);
                indexed_count += 1;
            }
            Err(e) => {
                return Err(FileProcessingError::IndexingFailed(format!(
                    "Failed to index chunk {}: {:?}",
                    chunk_index, e
                )));
            }
        }
    }

    // Mark the source as complete (all chunks successfully added)
    search_engine
        .complete_source(source_id)
        .await
        .map_err(|e| {
            FileProcessingError::IndexingFailed(format!("Failed to complete source: {:?}", e))
        })?;

    let total_elapsed_ms = index_start.elapsed().as_secs_f64() * 1000.0;

    info!(
        "✅ Added {} chunks to search engine ({:.1}ms total, {:.1}ms indexing)",
        indexed_count, total_elapsed_ms, total_index_time_ms
    );

    Ok(indexed_count)
}

/// Recursively collects all files from a directory (desktop only).
///
/// On web platforms, directory traversal is handled automatically via
/// the `webkitdirectory` attribute on file inputs.
///
/// # Arguments
///
/// * `path` - Directory path to traverse
/// * `base_name` - Base name for constructing relative paths (empty string for root)
///
/// # Returns
///
/// Vector of (relative_path, absolute_path) tuples for all files found.
///
/// # Notes
///
/// This function is recursive and must be boxed to work with async recursion.
#[cfg(feature = "desktop")]
pub type BoxedFileCollector = Pin<Box<dyn Future<Output = Vec<(String, PathBuf)>> + Send>>;

#[cfg(feature = "desktop")]
pub fn collect_files_from_dir(path: PathBuf, base_name: String) -> BoxedFileCollector {
    Box::pin(async move {
        use tokio::fs;

        let mut files = Vec::new();

        match fs::read_dir(&path).await {
            Ok(mut entries) => {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let entry_path = entry.path();
                    let file_name = entry.file_name();
                    let file_name_str = file_name.to_string_lossy();

                    // Create relative path from base (e.g., "folder/subfolder/file.txt")
                    let relative_path = if base_name.is_empty() {
                        file_name_str.to_string()
                    } else {
                        format!("{}/{}", base_name, file_name_str)
                    };

                    match fs::metadata(&entry_path).await {
                        Ok(metadata) => {
                            if metadata.is_dir() {
                                // Recursively collect files from subdirectory
                                let mut subfiles =
                                    collect_files_from_dir(entry_path, relative_path.clone()).await;
                                files.append(&mut subfiles);
                            } else if metadata.is_file() {
                                // Add file to collection
                                files.push((relative_path, entry_path));
                            }
                        }
                        Err(e) => {
                            error!("❌ Failed to read metadata for {}: {}", relative_path, e);
                        }
                    }
                }
            }
            Err(e) => {
                error!("❌ Failed to read directory {}: {}", path.display(), e);
            }
        }

        files
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_likely_binary() {
        // Text files don't contain null bytes
        assert!(!is_likely_binary("Hello, world!"));
        assert!(!is_likely_binary("UTF-8 text with émojis 🚀"));
        assert!(!is_likely_binary(""));

        // Binary files contain null bytes
        assert!(is_likely_binary("binary\0data"));
        assert!(is_likely_binary("\0"));
    }

    #[test]
    fn test_compute_content_hash() {
        // Same content should produce same hash
        let hash1 = compute_content_hash("Hello, world!");
        let hash2 = compute_content_hash("Hello, world!");
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        let hash3 = compute_content_hash("Hello, world");
        assert_ne!(hash1, hash3);

        // Hash should be 64 hex characters (256 bits)
        assert_eq!(hash1.len(), 64);

        // Hash should be lowercase hex
        assert!(hash1
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn test_generate_source_id() {
        // Test source ID generation
        let source_id = generate_source_id("example.txt");

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Desktop: Uses full path as-is
            assert_eq!(source_id, "example.txt");

            let full_path = generate_source_id("/Users/test/docs/README.md");
            assert_eq!(full_path, "/Users/test/docs/README.md");
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Web: Prefixes with "web:"
            assert_eq!(source_id, "web:example.txt");

            // Should extract filename from path
            let with_path = generate_source_id("folder/subfolder/file.txt");
            assert_eq!(with_path, "web:file.txt");
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_generate_source_id_for_urls() {
        // URLs from the web crawler should be used as-is on desktop
        let url1 = generate_source_id("https://example.com/docs/intro");
        assert_eq!(url1, "https://example.com/docs/intro");

        let url2 = generate_source_id("https://example.com/docs/guide?section=auth#tokens");
        assert_eq!(url2, "https://example.com/docs/guide?section=auth#tokens");

        // Different URLs should produce different source_ids
        assert_ne!(
            generate_source_id("https://example.com/page1"),
            generate_source_id("https://example.com/page2")
        );

        // Same URL should produce same source_id (for update detection)
        assert_eq!(
            generate_source_id("https://example.com/docs"),
            generate_source_id("https://example.com/docs")
        );
    }

    // =========================================================================
    // Additional Tests for 1.0 Release
    // =========================================================================

    #[test]
    fn test_is_likely_binary_edge_cases() {
        // Long text file should not be binary
        let long_text = "Lorem ipsum ".repeat(1000);
        assert!(!is_likely_binary(&long_text));

        // Code-like content with special chars but no null bytes
        let code = r#"
            fn main() {
                let x = "hello\nworld";
                println!("{}", x);
            }
        "#;
        assert!(!is_likely_binary(code));

        // JSON content
        let json = r#"{"key": "value", "array": [1, 2, 3]}"#;
        assert!(!is_likely_binary(json));

        // XML/HTML content
        let xml = "<root><child attr=\"value\">text</child></root>";
        assert!(!is_likely_binary(xml));

        // Markdown with code blocks
        let markdown = r#"
# Title

```rust
fn example() {}
```

Some **bold** text.
        "#;
        assert!(!is_likely_binary(markdown));

        // Whitespace-only content
        assert!(!is_likely_binary("   \n\t\r\n   "));
    }

    #[test]
    fn test_compute_content_hash_determinism() {
        // Hash should be deterministic across multiple calls
        let content = "Test content for hashing";
        let hashes: Vec<String> = (0..5).map(|_| compute_content_hash(content)).collect();
        assert!(hashes.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn test_compute_content_hash_unicode() {
        // Unicode content should hash correctly
        let hash1 = compute_content_hash("日本語テキスト");
        let hash2 = compute_content_hash("日本語テキスト");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64);

        // Emoji content
        let emoji_hash = compute_content_hash("🎉🚀💻");
        assert_eq!(emoji_hash.len(), 64);
    }

    #[test]
    fn test_compute_content_hash_whitespace_sensitivity() {
        // Different whitespace should produce different hashes
        let hash1 = compute_content_hash("hello world");
        let hash2 = compute_content_hash("hello  world");
        let hash3 = compute_content_hash("hello\nworld");
        let hash4 = compute_content_hash("hello\tworld");

        assert_ne!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_ne!(hash1, hash4);
    }

    #[test]
    fn test_compute_content_hash_empty() {
        // Empty string should have a valid hash
        let hash = compute_content_hash("");
        assert_eq!(hash.len(), 64);

        // Different from whitespace
        let whitespace_hash = compute_content_hash(" ");
        assert_ne!(hash, whitespace_hash);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_generate_source_id_special_paths() {
        // Paths with spaces
        let path_with_spaces = generate_source_id("/Users/test/My Documents/file.txt");
        assert_eq!(path_with_spaces, "/Users/test/My Documents/file.txt");

        // Paths with unicode
        let unicode_path = generate_source_id("/Users/test/文档/readme.md");
        assert_eq!(unicode_path, "/Users/test/文档/readme.md");

        // Windows-style paths (if testing on Windows or for compatibility)
        let windows_path = generate_source_id("C:\\Users\\test\\docs\\file.txt");
        assert_eq!(windows_path, "C:\\Users\\test\\docs\\file.txt");
    }

    #[test]
    fn test_source_update_action_variants() {
        // Test that SourceUpdateAction enum covers expected cases
        let skip = SourceUpdateAction::Skip;
        let add = SourceUpdateAction::Add;
        let update = SourceUpdateAction::Update { old_chunk_count: 5 };

        // Ensure they're distinct
        match skip {
            SourceUpdateAction::Skip => {}
            _ => panic!("Expected Skip"),
        }

        match add {
            SourceUpdateAction::Add => {}
            _ => panic!("Expected Add"),
        }

        match update {
            SourceUpdateAction::Update { old_chunk_count } => {
                assert_eq!(old_chunk_count, 5);
            }
            _ => panic!("Expected Update"),
        }
    }
}
