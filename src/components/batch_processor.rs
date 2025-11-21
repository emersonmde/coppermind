//! Batch processing orchestration for file indexing.
//!
//! This module handles the high-level batch processing workflow:
//! 1. Create batches from file lists
//! 2. Process files using platform-specific embedders
//! 3. Update batch/file status as processing progresses
//! 4. Index embedded chunks in the search engine

use super::{Batch, BatchMetrics, BatchStatus, FileMetrics, FileStatus};
use crate::components::file_processing::{index_chunks, is_likely_binary};
use crate::processing::embedder::PlatformEmbedder;
use crate::processing::processor::process_file_chunks;
use crate::search::HybridSearchEngine;
use crate::storage::StorageBackend;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures::lock::Mutex;
use instant::Instant;
use std::sync::Arc;

/// Process a batch of files with embeddings and indexing.
///
/// This is the main orchestration function that coordinates file processing,
/// using the platform-specific embedder and updating UI state via signals.
///
/// # Arguments
///
/// * `batch_idx` - Index of the batch in the batches vector
/// * `file_list` - List of (filename, contents) pairs to process
/// * `batches_signal` - Signal for batch state updates
/// * `embedder` - Platform-specific embedder implementation
/// * `engine` - Search engine for indexing
/// * `engine_status` - Signal for updating engine status
///
/// # Returns
///
/// Batch metrics on success, or error message on failure.
pub async fn process_batch<S: StorageBackend>(
    batch_idx: usize,
    file_list: Vec<(String, String)>,
    mut batches_signal: Signal<Vec<Batch>>,
    embedder: &impl PlatformEmbedder,
    engine: Option<Arc<Mutex<HybridSearchEngine<S>>>>,
    mut engine_status: Signal<super::SearchEngineStatus>,
) -> Result<BatchMetrics, String> {
    // Update batch status to Processing
    batches_signal.write()[batch_idx].status = BatchStatus::Processing;

    // Track batch timing and stats
    let batch_start = Instant::now();
    let mut total_chunks: usize = 0;
    let mut total_tokens: usize = 0;
    let mut processed_count = 0;

    // Process each file
    for (file_idx, (file_name, contents)) in file_list.iter().enumerate() {
        // Check if binary
        if is_likely_binary(contents) {
            info!("⚠️ Skipped '{}': binary file", file_name);
            batches_signal.write()[batch_idx].files[file_idx].status =
                FileStatus::Failed("Binary file".to_string());
            continue;
        }

        processed_count += 1;

        // Process file chunks with progress tracking and live metrics updates
        let result = process_file_chunks(
            contents,
            embedder,
            |current, total, pct, tokens, elapsed_ms| {
                // Use .write() directly for better reactivity from async contexts
                let mut batches = batches_signal.write();
                batches[batch_idx].files[file_idx].status =
                    FileStatus::Processing { current, total };
                batches[batch_idx].files[file_idx].progress_pct = pct;
                // Update metrics with partial results as chunks complete
                batches[batch_idx].files[file_idx].metrics = Some(FileMetrics {
                    tokens_processed: tokens,
                    chunks_embedded: current,
                    chunks_total: total,
                    elapsed_ms,
                });
                // Write guard is dropped here, triggering reactivity
            },
        )
        .await;

        match result {
            Ok(chunk_result) => {
                total_chunks += chunk_result.chunks.len();
                total_tokens += chunk_result.metrics.tokens_processed;

                // Update file status to completed
                {
                    let mut batches = batches_signal.write();
                    batches[batch_idx].files[file_idx].status = FileStatus::Completed;
                    batches[batch_idx].files[file_idx].progress_pct = 100.0;
                    batches[batch_idx].files[file_idx].metrics = Some(chunk_result.metrics);
                }

                // Index chunks in search engine
                if let Some(engine_lock) = &engine {
                    match index_chunks(engine_lock.clone(), &chunk_result.chunks, file_name).await {
                        Ok(indexed_count) => {
                            info!("✅ Indexed {} chunks from {}", indexed_count, file_name);
                        }
                        Err(e) => {
                            error!("❌ Failed to index chunks: {}", e);
                            batches_signal.write()[batch_idx].files[file_idx].status =
                                FileStatus::Failed(format!("Indexing failed: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                error!("❌ Failed to process file {}: {}", file_name, e);
                batches_signal.write()[batch_idx].files[file_idx].status =
                    FileStatus::Failed(e.to_string());
                continue;
            }
        }
    }

    // Finalize vector index (no rebuild needed - incremental HNSW)
    if let Some(engine_lock) = &engine {
        let doc_count = {
            let search_engine = engine_lock.lock().await;
            search_engine.len()
        };

        {
            let mut search_engine = engine_lock.lock().await;
            // rebuild_vector_index() is a no-op with rust-cv/hnsw (supports incremental updates)
            // Index is already up-to-date from incremental insertions
            if let Err(e) = search_engine.rebuild_vector_index().await {
                error!("❌ Failed to finalize vector index: {}", e);
                return Err(format!("Index finalization failed: {}", e));
            }
        }

        // Update search engine status
        engine_status.set(super::SearchEngineStatus::Ready { doc_count });
        info!("✅ Search index ready with {} documents", doc_count);
    }

    // Calculate final batch metrics
    let duration_ms = batch_start.elapsed().as_millis() as u64;
    let metrics = BatchMetrics {
        file_count: processed_count,
        chunk_count: total_chunks,
        token_count: total_tokens,
        duration_ms,
    };

    // Mark batch as completed
    {
        let mut batches = batches_signal.write();
        batches[batch_idx].status = BatchStatus::Completed;
        batches[batch_idx].metrics = Some(metrics.clone());
    }

    info!(
        "✅ Batch complete: {} files, {} chunks, {} tokens in {:.1}s",
        processed_count,
        total_chunks,
        total_tokens,
        duration_ms as f64 / 1000.0
    );

    Ok(metrics)
}
