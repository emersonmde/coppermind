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
/// * `progress_callback` - Called after each chunk with (current, total, progress_pct, tokens_so_far, elapsed_ms)
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
///     |current, total, pct, tokens, elapsed| {
///         println!("Progress: {}/{} ({:.1}%) - {} tokens in {}ms", current, total, pct, tokens, elapsed);
///     }
/// ).await?;
/// ```
pub async fn process_file_chunks<F>(
    content: &str,
    embedder: &impl PlatformEmbedder,
    mut progress_callback: F,
) -> Result<ChunkProcessingResult, EmbeddingError>
where
    F: FnMut(usize, usize, f64, usize, u64),
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
    let mut tokens_so_far = 0;

    // Process each chunk
    for (chunk_idx, chunk_text) in text_chunks.iter().enumerate() {
        // Compute embedding
        match embedder.embed(chunk_text).await {
            Ok(computation) => {
                tokens_so_far += computation.token_count;
                results.push(ChunkEmbeddingResult {
                    chunk_index: chunk_idx,
                    token_count: computation.token_count,
                    text: chunk_text.clone(),
                    embedding: computation.embedding,
                });

                // Update progress with partial metrics
                let progress_pct = ((chunk_idx + 1) as f64 / chunks_total as f64) * 100.0;
                let elapsed_ms = start_time.elapsed().as_millis() as u64;
                progress_callback(
                    chunk_idx + 1,
                    chunks_total,
                    progress_pct,
                    tokens_so_far,
                    elapsed_ms,
                );
            }
            Err(e) => {
                error!("❌ Failed to embed chunk {}: {}", chunk_idx, e);
                return Err(e);
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingComputation;
    use std::sync::{Arc, Mutex};

    /// Mock embedder for testing.
    ///
    /// Returns predictable embeddings with configurable behavior.
    struct MockEmbedder {
        /// Embedding dimension
        dim: usize,
        /// Whether to fail on specific chunk indices
        fail_on_chunks: Vec<usize>,
        /// Call counter
        call_count: Arc<Mutex<usize>>,
    }

    impl MockEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                fail_on_chunks: vec![],
                call_count: Arc::new(Mutex::new(0)),
            }
        }

        fn with_failures(dim: usize, fail_on_chunks: Vec<usize>) -> Self {
            Self {
                dim,
                fail_on_chunks,
                call_count: Arc::new(Mutex::new(0)),
            }
        }

        fn call_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    #[async_trait::async_trait(?Send)]
    impl PlatformEmbedder for MockEmbedder {
        async fn embed(&self, text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
            let count = {
                let mut c = self.call_count.lock().unwrap();
                let current = *c;
                *c += 1;
                current
            };

            // Check if we should fail on this chunk
            if self.fail_on_chunks.contains(&count) {
                return Err(EmbeddingError::InferenceFailed(format!(
                    "Mock failure on chunk {}",
                    count
                )));
            }

            // Calculate token count (approximate: whitespace tokenization)
            let token_count = text.split_whitespace().count();

            // Create deterministic embedding based on text length
            let embedding = vec![1.0; self.dim];

            Ok(EmbeddingComputation {
                token_count,
                embedding,
            })
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn status_message(&self) -> String {
            "Mock embedder ready".to_string()
        }
    }

    #[tokio::test]
    async fn test_process_file_chunks_success() {
        let embedder = MockEmbedder::new(512);
        let content = "This is a test file with some content.";

        let result = process_file_chunks(content, &embedder, |_, _, _, _, _| {}).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.error.is_none());
        assert_eq!(result.chunks.len(), 1); // Small content = 1 chunk
        assert_eq!(result.metrics.chunks_embedded, 1);
        assert_eq!(result.metrics.chunks_total, 1);
        assert!(result.metrics.tokens_processed > 0);
    }

    #[tokio::test]
    async fn test_process_file_chunks_progress_callback() {
        let embedder = MockEmbedder::new(512);
        // Create content that will be split into multiple chunks (2000 chars each)
        let content = "word ".repeat(1000); // ~5000 chars = 3 chunks

        let progress_calls = Arc::new(Mutex::new(Vec::new()));
        let progress_calls_clone = progress_calls.clone();

        let result = process_file_chunks(
            content.as_str(),
            &embedder,
            move |current, total, pct, tokens, elapsed| {
                progress_calls_clone
                    .lock()
                    .unwrap()
                    .push((current, total, pct, tokens, elapsed));
            },
        )
        .await;

        assert!(result.is_ok());

        // Verify progress callback was called for each chunk
        let calls = progress_calls.lock().unwrap();
        assert_eq!(calls.len(), 3, "Progress callback should be called 3 times");

        // Verify progress increases
        assert_eq!(calls[0].0, 1); // First chunk
        assert_eq!(calls[1].0, 2); // Second chunk
        assert_eq!(calls[2].0, 3); // Third chunk

        // Verify total is consistent
        assert_eq!(calls[0].1, 3);
        assert_eq!(calls[1].1, 3);
        assert_eq!(calls[2].1, 3);

        // Verify percentage increases
        assert!(calls[0].2 < calls[1].2);
        assert!(calls[1].2 < calls[2].2);
        assert!((calls[2].2 - 100.0).abs() < 0.01); // Last should be 100%

        // Verify tokens accumulate
        assert!(calls[0].3 > 0);
        assert!(calls[1].3 > calls[0].3);
        assert!(calls[2].3 > calls[1].3);

        // Verify elapsed time increases
        assert!(calls[1].4 >= calls[0].4);
        assert!(calls[2].4 >= calls[1].4);
    }

    #[tokio::test]
    async fn test_process_file_chunks_metrics_calculation() {
        let embedder = MockEmbedder::new(512);
        let content = "hello world test";

        let result = process_file_chunks(content, &embedder, |_, _, _, _, _| {})
            .await
            .unwrap();

        // Verify metrics
        assert_eq!(result.metrics.chunks_embedded, 1);
        assert_eq!(result.metrics.chunks_total, 1);
        assert_eq!(result.metrics.tokens_processed, 3); // "hello world test" = 3 tokens
                                                        // elapsed_ms can be 0 for very fast operations

        // Verify chunks
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunks[0].chunk_index, 0);
        assert_eq!(result.chunks[0].token_count, 3);
        assert_eq!(result.chunks[0].text, content);
        assert_eq!(result.chunks[0].embedding.len(), 512);
    }

    #[tokio::test]
    async fn test_process_file_chunks_empty_content() {
        let embedder = MockEmbedder::new(512);
        let content = "";

        let result = process_file_chunks(content, &embedder, |_, _, _, _, _| {})
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.chunks.len(), 0); // Empty content = no chunks
        assert_eq!(result.metrics.chunks_embedded, 0);
        assert_eq!(result.metrics.chunks_total, 0);
        assert_eq!(result.metrics.tokens_processed, 0);
    }

    #[tokio::test]
    async fn test_process_file_chunks_single_chunk() {
        let embedder = MockEmbedder::new(512);
        let content = "A small file that fits in one chunk.";

        let result = process_file_chunks(content, &embedder, |_, _, _, _, _| {})
            .await
            .unwrap();

        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.metrics.chunks_total, 1);
        assert_eq!(result.metrics.chunks_embedded, 1);
    }

    #[tokio::test]
    async fn test_process_file_chunks_multiple_chunks() {
        let embedder = MockEmbedder::new(512);
        // Create content larger than CHUNK_SIZE (2000 chars)
        let content = "x".repeat(5000); // 5000 chars = 3 chunks

        let result = process_file_chunks(content.as_str(), &embedder, |_, _, _, _, _| {})
            .await
            .unwrap();

        assert_eq!(result.chunks.len(), 3);
        assert_eq!(result.metrics.chunks_total, 3);
        assert_eq!(result.metrics.chunks_embedded, 3);

        // Verify chunk indices
        assert_eq!(result.chunks[0].chunk_index, 0);
        assert_eq!(result.chunks[1].chunk_index, 1);
        assert_eq!(result.chunks[2].chunk_index, 2);

        // Verify chunk sizes
        assert_eq!(result.chunks[0].text.len(), 2000);
        assert_eq!(result.chunks[1].text.len(), 2000);
        assert_eq!(result.chunks[2].text.len(), 1000);
    }

    #[tokio::test]
    async fn test_process_file_chunks_embedding_error() {
        // Mock embedder that fails on chunk 1 (second chunk)
        let embedder = MockEmbedder::with_failures(512, vec![1]);
        let content = "x".repeat(5000); // 3 chunks

        let result = process_file_chunks(content.as_str(), &embedder, |_, _, _, _, _| {}).await;

        // Should fail on the second chunk
        assert!(result.is_err());
        match result {
            Err(EmbeddingError::InferenceFailed(msg)) => {
                assert!(msg.contains("Mock failure on chunk 1"));
            }
            _ => panic!("Expected InferenceFailed error"),
        }
    }

    #[tokio::test]
    async fn test_process_file_chunks_token_accumulation() {
        let embedder = MockEmbedder::new(512);
        // Create content with known token counts
        let content = "one two three four five"; // 5 tokens

        let tokens_tracker = Arc::new(Mutex::new(Vec::new()));
        let tokens_tracker_clone = tokens_tracker.clone();

        let result = process_file_chunks(content, &embedder, move |_, _, _, tokens, _| {
            tokens_tracker_clone.lock().unwrap().push(tokens);
        })
        .await
        .unwrap();

        // Verify final token count
        assert_eq!(result.metrics.tokens_processed, 5);

        // Verify progress callback received correct token count
        let tokens = tokens_tracker.lock().unwrap();
        assert_eq!(tokens[0], 5); // Single chunk, all tokens at once
    }

    #[tokio::test]
    async fn test_process_file_chunks_elapsed_time_tracking() {
        let embedder = MockEmbedder::new(512);
        let content = "test";

        let start = std::time::Instant::now();
        let result = process_file_chunks(content, &embedder, |_, _, _, _, _| {})
            .await
            .unwrap();
        let end = start.elapsed().as_millis() as u64;

        // Elapsed time should be reasonable (can be 0 for very fast operations)
        assert!(result.metrics.elapsed_ms <= end);
    }

    #[tokio::test]
    async fn test_process_file_chunks_embedder_call_count() {
        let embedder = MockEmbedder::new(512);
        let content = "x".repeat(5000); // 3 chunks

        let _ = process_file_chunks(content.as_str(), &embedder, |_, _, _, _, _| {})
            .await
            .unwrap();

        // Verify embedder was called exactly 3 times
        assert_eq!(embedder.call_count(), 3);
    }

    #[tokio::test]
    async fn test_chunk_processing_result_structure() {
        let embedder = MockEmbedder::new(512);
        let content = "hello world";

        let result = process_file_chunks(content, &embedder, |_, _, _, _, _| {})
            .await
            .unwrap();

        // Verify result structure
        assert!(result.success);
        assert!(result.error.is_none());
        assert!(!result.chunks.is_empty());

        // Verify ChunkEmbeddingResult structure
        let chunk = &result.chunks[0];
        assert_eq!(chunk.chunk_index, 0);
        assert!(chunk.token_count > 0);
        assert!(!chunk.text.is_empty());
        assert_eq!(chunk.embedding.len(), 512);
    }
}
