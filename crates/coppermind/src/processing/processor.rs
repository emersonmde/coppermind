//! File chunk processing and embedding.
//!
//! This module contains the core logic for processing text files:
//! 1. Split into chunks (semantic chunking with file type detection)
//! 2. Compute embeddings for each chunk (batched on desktop for efficiency)
//! 3. Track progress and metrics

use crate::components::FileMetrics;
use crate::embedding::{embed_text_chunks_auto, ChunkEmbeddingResult};
use crate::error::EmbeddingError;
use coppermind_core::config::MAX_CHUNK_TOKENS;
use instant::Instant;
#[cfg(feature = "profile")]
use tracing::instrument;

/// Result of processing all chunks from a file.
#[derive(Debug, Clone)]
pub struct ChunkProcessingResult {
    /// Successfully embedded chunks
    pub chunks: Vec<ChunkEmbeddingResult>,
    /// File processing metrics
    pub metrics: FileMetrics,
    /// Whether processing completed successfully
    pub success: bool,
    /// Error message if processing failed
    pub error: Option<String>,
}

/// Process text file chunks with progress tracking.
///
/// Uses semantic chunking (markdown/code/text aware) and mini-batch embedding
/// for efficiency. Processes chunks in batches of 8 for GPU efficiency while
/// still providing progress updates.
///
/// # Arguments
///
/// * `content` - File text content
/// * `filename` - Optional filename for file type detection (e.g., "doc.md" → markdown chunking)
/// * `progress_callback` - Called with progress updates (current, total, progress_pct, tokens_so_far, elapsed_ms)
///
/// # Returns
///
/// Processing result with embedded chunks and metrics, or error.
///
/// # Examples
///
/// ```ignore
/// let result = process_file_chunks(
///     &file_content,
///     Some("README.md"),
///     |current, total, pct, tokens, elapsed| {
///         println!("Progress: {}/{} ({:.1}%) - {} tokens in {}ms", current, total, pct, tokens, elapsed);
///     }
/// ).await?;
/// ```
#[cfg_attr(feature = "profile", instrument(skip_all, fields(content_len = content.len(), filename)))]
pub async fn process_file_chunks<F>(
    content: &str,
    filename: Option<&str>,
    mut progress_callback: F,
) -> Result<ChunkProcessingResult, EmbeddingError>
where
    F: FnMut(usize, usize, f64, usize, u64),
{
    let start_time = Instant::now();

    // Track progress reporting
    let mut _last_total = 0usize;

    // Use semantic chunking with mini-batch embedding (Background priority)
    // Mini-batches (8 chunks) balance GPU efficiency with progress updates
    let results =
        embed_text_chunks_auto(content, MAX_CHUNK_TOKENS, filename, |completed, total| {
            _last_total = total;
            let elapsed_ms = start_time.elapsed().as_millis() as u64;
            let pct = if total > 0 {
                (completed as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            // tokens_so_far not available during embedding, report 0 until final
            progress_callback(completed, total, pct, 0, elapsed_ms);
        })
        .await?;

    // Calculate final metrics
    let token_count: usize = results.iter().map(|r| r.token_count).sum();
    let elapsed_ms = start_time.elapsed().as_millis() as u64;
    let chunks_total = results.len();

    // Final progress update with accurate token count
    progress_callback(chunks_total, chunks_total, 100.0, token_count, elapsed_ms);

    let metrics = FileMetrics {
        tokens_processed: token_count,
        chunks_embedded: results.len(),
        chunks_total,
        elapsed_ms,
    };

    Ok(ChunkProcessingResult {
        chunks: results,
        metrics,
        success: true,
        error: None,
    })
}

// Tests removed: process_file_chunks() now delegates to embed_text_chunks_auto()
// which requires a real model to be loaded. Integration tests in the components
// module test the full file processing pipeline.
