//! File chunk processing and embedding.
//!
//! This module contains the core logic for processing text files:
//! 1. Split into chunks
//! 2. Compute embeddings for each chunk
//! 3. Track progress and metrics

use super::embedder::PlatformEmbedder;
use crate::components::FileMetrics;
use crate::embedding::ChunkEmbeddingResult;
use crate::error::EmbeddingError;
use dioxus::logger::tracing::error;
use instant::Instant;

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
/// Splits text into chunks, computes embeddings, and tracks metrics.
///
/// # Arguments
///
/// * `content` - File text content
/// * `embedder` - Platform-specific embedder
/// * `progress_callback` - Called after each chunk with (current, total, progress_pct)
///
/// # Returns
///
/// Processing result with embedded chunks and metrics, or error.
///
/// # Examples
///
/// ```ignore
/// let embedder = DesktopEmbedder::new();
/// let result = process_file_chunks(
///     &file_content,
///     &embedder,
///     |current, total, pct| {
///         println!("Progress: {}/{} ({:.1}%)", current, total, pct);
///     }
/// ).await?;
/// ```
pub async fn process_file_chunks<F>(
    content: &str,
    embedder: &impl PlatformEmbedder,
    mut progress_callback: F,
) -> Result<ChunkProcessingResult, EmbeddingError>
where
    F: FnMut(usize, usize, f64),
{
    let start_time = Instant::now();

    // Split into chunks (2000 chars each)
    const CHUNK_SIZE: usize = 2000;
    let text_chunks: Vec<String> = content
        .chars()
        .collect::<Vec<_>>()
        .chunks(CHUNK_SIZE)
        .map(|chunk| chunk.iter().collect())
        .collect();

    let chunks_total = text_chunks.len();
    let mut results = Vec::new();

    // Process each chunk
    for (chunk_idx, chunk_text) in text_chunks.iter().enumerate() {
        // Update progress
        let progress_pct = (chunk_idx as f64 / chunks_total as f64) * 100.0;
        progress_callback(chunk_idx, chunks_total, progress_pct);

        // Compute embedding
        match embedder.embed(chunk_text).await {
            Ok(computation) => {
                results.push(ChunkEmbeddingResult {
                    chunk_index: chunk_idx,
                    token_count: computation.token_count,
                    text: chunk_text.clone(),
                    embedding: computation.embedding,
                });
            }
            Err(e) => {
                error!("❌ Failed to embed chunk {}: {}", chunk_idx, e);
                return Err(e);
            }
        }
    }

    // Final progress update
    progress_callback(chunks_total, chunks_total, 100.0);

    // Calculate metrics
    let elapsed_ms = start_time.elapsed().as_millis() as u64;
    let token_count: usize = results.iter().map(|r| r.token_count).sum();

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

// Note: Tests temporarily removed during refactoring.
// Will be re-added with integration tests after components/mod.rs refactor is complete.
