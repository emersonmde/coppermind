//! Indexing pipeline for processing documents.
//!
//! The `IndexingPipeline` coordinates chunking, tokenization, and embedding
//! to process documents for indexing in the search engine.

use super::progress::{IndexingProgress, ProgressTimer};
use crate::chunking::{create_chunker, detect_file_type, TextChunk};
use crate::embedding::{Embedder, TokenizerHandle};
use crate::error::{ChunkingError, EmbeddingError};
use std::sync::Arc;
use tracing::debug;

/// Result of processing a single chunk.
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// The original text chunk
    pub chunk: TextChunk,
    /// Token IDs for this chunk
    pub token_ids: Vec<u32>,
    /// Number of tokens in this chunk
    pub token_count: usize,
    /// Computed embedding vector
    pub embedding: Vec<f32>,
}

/// Result of processing an entire document.
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Successfully processed chunks with embeddings
    pub chunks: Vec<ChunkResult>,
    /// Total tokens processed
    pub total_tokens: usize,
    /// Processing time in milliseconds
    pub elapsed_ms: u64,
}

impl ProcessingResult {
    /// Returns the number of chunks processed.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns true if any chunks were processed.
    pub fn has_chunks(&self) -> bool {
        !self.chunks.is_empty()
    }
}

/// Indexing pipeline for processing documents.
///
/// Coordinates chunking, tokenization, and embedding to produce indexed chunks
/// that can be added to a `HybridSearchEngine`.
///
/// # Thread Safety
///
/// The pipeline is `Send + Sync` and can be safely shared across threads.
/// The `Embedder` and `TokenizerHandle` are accessed through `Arc`.
///
/// # Example
///
/// ```ignore
/// use coppermind_core::processing::IndexingPipeline;
/// use std::sync::Arc;
///
/// let pipeline = IndexingPipeline::new(
///     Arc::new(embedder),
///     Arc::new(tokenizer),
/// );
///
/// let result = pipeline.process_text(
///     &content,
///     Some("README.md"),
///     512,
///     |progress| println!("{}%", progress.percent_complete()),
/// )?;
///
/// for chunk in result.chunks {
///     engine.add_document(doc_id, &chunk.chunk.text, chunk.embedding)?;
/// }
/// ```
pub struct IndexingPipeline {
    embedder: Arc<dyn Embedder>,
    tokenizer: Arc<TokenizerHandle>,
}

impl IndexingPipeline {
    /// Creates a new indexing pipeline.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The embedder to use for computing embeddings
    /// * `tokenizer` - The tokenizer to use for chunk tokenization
    pub fn new(embedder: Arc<dyn Embedder>, tokenizer: Arc<TokenizerHandle>) -> Self {
        Self {
            embedder,
            tokenizer,
        }
    }

    /// Returns a reference to the embedder.
    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    /// Returns a reference to the tokenizer.
    pub fn tokenizer(&self) -> &TokenizerHandle {
        self.tokenizer.as_ref()
    }

    /// Process text content into indexed chunks.
    ///
    /// Uses semantic chunking (file type detection based on filename) and
    /// computes embeddings for each chunk.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to process
    /// * `filename` - Optional filename for file type detection (e.g., "doc.md" for markdown)
    /// * `max_tokens_per_chunk` - Maximum tokens per chunk
    /// * `on_progress` - Callback for progress updates
    ///
    /// # Returns
    ///
    /// Processing result with embedded chunks, or an error.
    pub fn process_text<F>(
        &self,
        content: &str,
        filename: Option<&str>,
        max_tokens_per_chunk: usize,
        mut on_progress: F,
    ) -> Result<ProcessingResult, ProcessingError>
    where
        F: FnMut(IndexingProgress),
    {
        let timer = ProgressTimer::new();
        let content = content.trim();

        if content.is_empty() {
            return Ok(ProcessingResult {
                chunks: vec![],
                total_tokens: 0,
                elapsed_ms: timer.elapsed_ms(),
            });
        }

        // Detect file type and create appropriate chunker
        let file_type = filename.map_or(crate::chunking::FileType::Text, detect_file_type);

        // Get static tokenizer reference for chunker
        // Safety: The tokenizer is stored in an Arc and lives for the duration of the pipeline
        let tokenizer_ref: &'static tokenizers::Tokenizer =
            unsafe { std::mem::transmute(self.tokenizer.inner()) };

        let chunker = create_chunker(file_type, max_tokens_per_chunk, tokenizer_ref);

        debug!(
            "Processing text with {} chunker, {} chars",
            chunker.name(),
            content.len()
        );

        // Chunk the content
        let text_chunks = chunker.chunk(content)?;
        let total_chunks = text_chunks.len();

        if total_chunks == 0 {
            return Ok(ProcessingResult {
                chunks: vec![],
                total_tokens: 0,
                elapsed_ms: timer.elapsed_ms(),
            });
        }

        // Report initial progress
        on_progress(IndexingProgress::new(
            0,
            total_chunks,
            0,
            timer.elapsed_ms(),
        ));

        // Process chunks: tokenize and embed
        let mut results = Vec::with_capacity(total_chunks);
        let mut total_tokens = 0;

        for (i, chunk) in text_chunks.into_iter().enumerate() {
            // Tokenize
            let token_ids = self.tokenizer.tokenize(&chunk.text)?;
            let token_count = token_ids.len();
            total_tokens += token_count;

            // Embed
            let embedding = self.embedder.embed_tokens(token_ids.clone())?;

            results.push(ChunkResult {
                chunk,
                token_ids,
                token_count,
                embedding,
            });

            // Report progress
            on_progress(IndexingProgress::new(
                i + 1,
                total_chunks,
                total_tokens,
                timer.elapsed_ms(),
            ));
        }

        Ok(ProcessingResult {
            chunks: results,
            total_tokens,
            elapsed_ms: timer.elapsed_ms(),
        })
    }

    /// Process text with batched embedding for efficiency.
    ///
    /// Similar to `process_text` but uses batch embedding for improved throughput
    /// when processing many chunks. Processes chunks in batches of `batch_size`.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to process
    /// * `filename` - Optional filename for file type detection
    /// * `max_tokens_per_chunk` - Maximum tokens per chunk
    /// * `batch_size` - Number of chunks to embed in each batch
    /// * `on_progress` - Callback for progress updates
    ///
    /// # Returns
    ///
    /// Processing result with embedded chunks, or an error.
    pub fn process_text_batched<F>(
        &self,
        content: &str,
        filename: Option<&str>,
        max_tokens_per_chunk: usize,
        batch_size: usize,
        mut on_progress: F,
    ) -> Result<ProcessingResult, ProcessingError>
    where
        F: FnMut(IndexingProgress),
    {
        let timer = ProgressTimer::new();
        let content = content.trim();

        if content.is_empty() {
            return Ok(ProcessingResult {
                chunks: vec![],
                total_tokens: 0,
                elapsed_ms: timer.elapsed_ms(),
            });
        }

        // Detect file type and create chunker
        let file_type = filename.map_or(crate::chunking::FileType::Text, detect_file_type);

        let tokenizer_ref: &'static tokenizers::Tokenizer =
            unsafe { std::mem::transmute(self.tokenizer.inner()) };

        let chunker = create_chunker(file_type, max_tokens_per_chunk, tokenizer_ref);

        debug!(
            "Processing text with {} chunker (batched, batch_size={}), {} chars",
            chunker.name(),
            batch_size,
            content.len()
        );

        // Chunk the content
        let text_chunks = chunker.chunk(content)?;
        let total_chunks = text_chunks.len();

        if total_chunks == 0 {
            return Ok(ProcessingResult {
                chunks: vec![],
                total_tokens: 0,
                elapsed_ms: timer.elapsed_ms(),
            });
        }

        // Report initial progress
        on_progress(IndexingProgress::new(
            0,
            total_chunks,
            0,
            timer.elapsed_ms(),
        ));

        // Tokenize all chunks first
        let mut tokenized: Vec<(TextChunk, Vec<u32>)> = Vec::with_capacity(total_chunks);
        for chunk in text_chunks {
            let token_ids = self.tokenizer.tokenize(&chunk.text)?;
            tokenized.push((chunk, token_ids));
        }

        // Process in batches
        let mut results = Vec::with_capacity(total_chunks);
        let mut total_tokens = 0;
        let mut completed = 0;

        for batch in tokenized.chunks(batch_size) {
            let batch_token_ids: Vec<Vec<u32>> = batch.iter().map(|(_, t)| t.clone()).collect();
            let embeddings = self.embedder.embed_batch_tokens(batch_token_ids)?;

            for ((chunk, token_ids), embedding) in batch.iter().zip(embeddings) {
                let token_count = token_ids.len();
                total_tokens += token_count;
                completed += 1;

                results.push(ChunkResult {
                    chunk: chunk.clone(),
                    token_ids: token_ids.clone(),
                    token_count,
                    embedding,
                });

                // Report progress after each chunk in batch
                on_progress(IndexingProgress::new(
                    completed,
                    total_chunks,
                    total_tokens,
                    timer.elapsed_ms(),
                ));
            }
        }

        Ok(ProcessingResult {
            chunks: results,
            total_tokens,
            elapsed_ms: timer.elapsed_ms(),
        })
    }
}

/// Errors that can occur during document processing.
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    /// Chunking failed
    #[error("Chunking failed: {0}")]
    Chunking(#[from] ChunkingError),
    /// Tokenization failed
    #[error("Tokenization failed: {0}")]
    Tokenization(#[from] EmbeddingError),
    /// Embedding failed
    #[error("Embedding failed: {0}")]
    Embedding(String),
}

impl From<String> for ProcessingError {
    fn from(s: String) -> Self {
        ProcessingError::Embedding(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests require a real embedder and tokenizer, which need model files.
    // Integration tests should be added when testing infrastructure is available.

    #[test]
    fn test_processing_result_empty() {
        let result = ProcessingResult {
            chunks: vec![],
            total_tokens: 0,
            elapsed_ms: 0,
        };
        assert_eq!(result.chunk_count(), 0);
        assert!(!result.has_chunks());
    }
}
