//! File processing utilities for document indexing.
//!
//! This module provides utilities for processing uploaded files, including:
//! - Binary file detection
//! - Text chunking
//! - Search engine indexing
//! - Directory traversal (desktop only)

use crate::embedding::ChunkEmbeddingResult;
use crate::error::FileProcessingError;
use crate::search::types::{get_current_timestamp, Document, DocumentMetadata};
use crate::search::HybridSearchEngine;
use crate::storage::StorageBackend;
use dioxus::logger::tracing::{error, info};
use futures::lock::Mutex;
use std::sync::Arc;

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
/// and indexes them in the search engine with deferred index rebuilding.
///
/// # Type Parameters
///
/// * `S` - Storage backend type implementing `StorageBackend`
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
pub async fn index_chunks<S: StorageBackend>(
    engine: Arc<Mutex<HybridSearchEngine<S>>>,
    embedding_results: &[ChunkEmbeddingResult],
    file_label: &str,
) -> Result<usize, FileProcessingError> {
    let mut indexed_count = 0;

    // Lock the engine and add all documents (deferred index rebuild)
    {
        let mut search_engine = engine.lock().await;

        for chunk_result in embedding_results.iter() {
            let doc = Document {
                text: chunk_result.text.clone(),
                metadata: DocumentMetadata {
                    filename: Some(format!(
                        "{} (chunk {})",
                        file_label,
                        chunk_result.chunk_index + 1
                    )),
                    source: Some(file_label.to_string()),
                    created_at: get_current_timestamp(),
                },
            };

            match search_engine
                .add_document_deferred(doc, chunk_result.embedding.clone())
                .await
            {
                Ok(_) => indexed_count += 1,
                Err(e) => {
                    error!(
                        "❌ Failed to index chunk {}: {:?}",
                        chunk_result.chunk_index, e
                    );
                    return Err(FileProcessingError::IndexingFailed(format!(
                        "Failed to index chunk {}: {:?}",
                        chunk_result.chunk_index, e
                    )));
                }
            }
        }
    } // Release lock (index rebuild deferred until caller rebuilds)

    info!(
        "✅ Added {} chunks to search engine (rebuild pending)",
        indexed_count
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
