//! File processing utilities for document indexing.
//!
//! This module provides utilities for processing uploaded files, including:
//! - Binary file detection
//! - Text chunking
//! - Search engine indexing
//! - Directory traversal (desktop only)

use crate::embedding::ChunkEmbeddingResult;
use crate::error::FileProcessingError;
use crate::metrics::global_metrics;
#[cfg(not(target_arch = "wasm32"))]
use crate::platform::run_blocking;
use crate::search::types::{get_current_timestamp, Document, DocumentMetadata};
use crate::search::HybridSearchEngine;
use crate::storage::DocumentStore;
#[cfg(not(target_arch = "wasm32"))]
use dioxus::logger::tracing::error;
use dioxus::logger::tracing::info;
use futures::lock::Mutex;
use instant::Instant;
use std::sync::Arc;

#[cfg(feature = "profile")]
use tracing::instrument;

#[cfg(not(target_arch = "wasm32"))]
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

/// Indexes embedding chunks in the search engine.
///
/// Takes embedding results (which include the decoded text for each chunk)
/// and indexes them in the search engine.
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
                    source: Some(file_label_owned.clone()),
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

        let mut indexed_count = 0;
        let mut total_index_time_ms = 0.0;

        for (doc, embedding, chunk_index) in docs {
            let insert_start = Instant::now();

            // add_document is async but doesn't do any actual async work
            // We can safely block_on it in the blocking thread
            match futures::executor::block_on(search_engine.add_document(doc, embedding)) {
                Ok(_) => {
                    let insert_duration_ms = insert_start.elapsed().as_secs_f64() * 1000.0;
                    total_index_time_ms += insert_duration_ms;
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
                    source: Some(file_label_owned.clone()),
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

    let mut indexed_count = 0;
    let mut total_index_time_ms = 0.0;

    for (doc, embedding, chunk_index) in docs {
        let insert_start = Instant::now();

        match search_engine.add_document(doc, embedding).await {
            Ok(_) => {
                let insert_duration_ms = insert_start.elapsed().as_secs_f64() * 1000.0;
                total_index_time_ms += insert_duration_ms;
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
#[cfg(not(target_arch = "wasm32"))]
pub type BoxedFileCollector = Pin<Box<dyn Future<Output = Vec<(String, PathBuf)>> + Send>>;

#[cfg(not(target_arch = "wasm32"))]
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
}
