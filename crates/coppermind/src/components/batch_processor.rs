//! Batch processing orchestration for file indexing.
//!
//! This module handles the high-level batch processing workflow:
//! 1. Create batches from file lists
//! 2. Check if files need to be processed (hash-based update detection)
//! 3. Process files with semantic chunking and batch embedding
//! 4. Update batch/file status as processing progresses
//! 5. Index embedded chunks in the search engine with source tracking

use super::{Batch, BatchMetrics, BatchStatus, FileMetrics, FileStatus};
use crate::components::file_processing::{
    check_source_update, compute_content_hash, generate_source_id, index_chunks, is_likely_binary,
    SourceUpdateAction,
};
use crate::processing::processor::process_file_chunks;
use crate::search::HybridSearchEngine;
use crate::storage::DocumentStore;
use dioxus::logger::tracing::{debug, error, info, warn};
use dioxus::prelude::*;
use futures::lock::Mutex;
use instant::Instant;
use std::sync::Arc;

#[cfg(feature = "profile")]
use tracing::instrument;

/// Process a batch of files with embeddings and indexing.
///
/// This is the main orchestration function that coordinates file processing,
/// using semantic chunking and batch embedding, updating UI state via signals.
///
/// # Arguments
///
/// * `batch_idx` - Index of the batch in the batches vector
/// * `file_list` - List of (filename, contents) pairs to process
/// * `batches_signal` - Signal for batch state updates
/// * `engine` - Search engine for indexing
/// * `engine_status` - Signal for updating engine status
///
/// # Returns
///
/// Batch metrics on success, or error message on failure.
// Desktop requires Send + Sync for threading; web is single-threaded so doesn't need them
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(feature = "profile", instrument(skip_all, fields(batch_idx, file_count = file_list.len())))]
pub async fn process_batch<S: DocumentStore + Send + Sync + 'static>(
    batch_idx: usize,
    file_list: Vec<(String, String)>,
    batches_signal: Signal<Vec<Batch>>,
    engine: Option<Arc<Mutex<HybridSearchEngine<S>>>>,
    engine_status: Signal<super::SearchEngineStatus>,
) -> Result<BatchMetrics, String> {
    process_batch_impl(batch_idx, file_list, batches_signal, engine, engine_status).await
}

#[cfg(target_arch = "wasm32")]
pub async fn process_batch<S: DocumentStore + 'static>(
    batch_idx: usize,
    file_list: Vec<(String, String)>,
    batches_signal: Signal<Vec<Batch>>,
    engine: Option<Arc<Mutex<HybridSearchEngine<S>>>>,
    engine_status: Signal<super::SearchEngineStatus>,
) -> Result<BatchMetrics, String> {
    process_batch_impl(batch_idx, file_list, batches_signal, engine, engine_status).await
}

/// Desktop implementation - requires Send + Sync for threading
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(feature = "profile", instrument(skip_all, fields(batch_idx, file_count = file_list.len())))]
async fn process_batch_impl<S: DocumentStore + Send + Sync + 'static>(
    batch_idx: usize,
    file_list: Vec<(String, String)>,
    batches_signal: Signal<Vec<Batch>>,
    engine: Option<Arc<Mutex<HybridSearchEngine<S>>>>,
    engine_status: Signal<super::SearchEngineStatus>,
) -> Result<BatchMetrics, String> {
    process_batch_inner(batch_idx, file_list, batches_signal, engine, engine_status).await
}

/// Web implementation - single-threaded, no Send + Sync required
#[cfg(target_arch = "wasm32")]
async fn process_batch_impl<S: DocumentStore + 'static>(
    batch_idx: usize,
    file_list: Vec<(String, String)>,
    batches_signal: Signal<Vec<Batch>>,
    engine: Option<Arc<Mutex<HybridSearchEngine<S>>>>,
    engine_status: Signal<super::SearchEngineStatus>,
) -> Result<BatchMetrics, String> {
    process_batch_inner(batch_idx, file_list, batches_signal, engine, engine_status).await
}

/// Shared batch processing logic - Desktop version.
///
/// NOTE: This implementation is duplicated for web (see below) due to different
/// trait bounds required by each platform (Send + Sync for desktop threading).
/// When modifying this function, ensure you update BOTH versions.
/// The only intentional difference is the save() comment on line ~313 vs ~574.
#[cfg(not(target_arch = "wasm32"))]
async fn process_batch_inner<S: DocumentStore + Send + Sync + 'static>(
    batch_idx: usize,
    file_list: Vec<(String, String)>,
    mut batches_signal: Signal<Vec<Batch>>,
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
    let mut skipped_count = 0;
    let mut updated_count = 0;

    // Process each file
    for (file_idx, (file_name, contents)) in file_list.iter().enumerate() {
        // Check if binary
        if is_likely_binary(contents) {
            info!("⚠️ Skipped '{}': binary file", file_name);
            batches_signal.write()[batch_idx].files[file_idx].status =
                FileStatus::Failed("Binary file".to_string());
            continue;
        }

        // Generate source ID and content hash for update detection
        let source_id = generate_source_id(file_name);
        let content_hash = compute_content_hash(contents);

        // Check if this source needs to be processed
        let update_action = if let Some(engine_lock) = &engine {
            let search_engine = engine_lock.lock().await;
            match check_source_update(&search_engine, &source_id, &content_hash).await {
                Ok(action) => action,
                Err(e) => {
                    warn!(
                        "⚠️ Failed to check source update for '{}': {}",
                        file_name, e
                    );
                    // Default to Add on error - safer to re-index than skip
                    SourceUpdateAction::Add
                }
            }
        } else {
            SourceUpdateAction::Add
        };

        match update_action {
            SourceUpdateAction::Skip => {
                info!("⏭️ Skipped '{}': no changes detected", file_name);
                skipped_count += 1;
                batches_signal.write()[batch_idx].files[file_idx].status = FileStatus::Completed; // Mark as done (no work needed)
                batches_signal.write()[batch_idx].files[file_idx].progress_pct = 100.0;
                continue;
            }
            SourceUpdateAction::Update { old_chunk_count } => {
                info!(
                    "🔄 Updating '{}': content changed (removing {} old chunks)",
                    file_name, old_chunk_count
                );
                // Delete old chunks before re-indexing
                if let Some(engine_lock) = &engine {
                    let mut search_engine = engine_lock.lock().await;
                    match search_engine.delete_source(&source_id).await {
                        Ok(deleted) => {
                            info!("🗑️ Deleted {} old chunks from '{}'", deleted, file_name);
                        }
                        Err(e) => {
                            error!(
                                "❌ Failed to delete old chunks for '{}': {:?}",
                                file_name, e
                            );
                            batches_signal.write()[batch_idx].files[file_idx].status =
                                FileStatus::Failed(format!("Failed to delete old chunks: {:?}", e));
                            continue;
                        }
                    }
                }
                updated_count += 1;
            }
            SourceUpdateAction::Add => {
                info!("➕ Adding new source '{}'", file_name);
            }
        }

        processed_count += 1;

        // Process file chunks with progress tracking and live metrics updates
        let result = process_file_chunks(
            contents,
            Some(file_name), // Pass filename for semantic chunking
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

                // Index chunks in search engine with source tracking
                if let Some(engine_lock) = &engine {
                    match index_chunks(
                        engine_lock.clone(),
                        &chunk_result.chunks,
                        file_name,
                        &source_id,
                        &content_hash,
                    )
                    .await
                    {
                        Ok(indexed_count) => {
                            info!("✅ Indexed {} chunks from {}", indexed_count, file_name);
                        }
                        Err(e) => {
                            error!("❌ Failed to index chunks: {}", e);
                            // Clean up incomplete source record to avoid orphaned data
                            let mut search_engine = engine_lock.lock().await;
                            if let Err(cleanup_err) = search_engine.delete_source(&source_id).await
                            {
                                warn!(
                                    "⚠️ Failed to cleanup source '{}' after indexing error: {:?}",
                                    source_id, cleanup_err
                                );
                            }
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

    // Log summary of update detection
    info!(
        "📊 Batch summary: {} new, {} updated, {} skipped (unchanged)",
        processed_count - updated_count,
        updated_count,
        skipped_count
    );

    // Finalize vector index (no rebuild needed - incremental HNSW)
    if let Some(engine_lock) = &engine {
        let (engine_doc_count, engine_total_tokens) = {
            let mut search_engine = engine_lock.lock().await;
            // rebuild_vector_index() is a no-op with rust-cv/hnsw (supports incremental updates)
            // Index is already up-to-date from incremental insertions
            if let Err(e) = search_engine.rebuild_vector_index().await {
                error!("❌ Failed to finalize vector index: {}", e);
                return Err(format!("Index finalization failed: {}", e));
            }

            // Check if compaction is needed (tombstone ratio > 30%)
            let (tombstone_count, total_count, ratio) = search_engine.compaction_stats();
            if tombstone_count > 0 {
                debug!(
                    "📊 Tombstone stats: {}/{} entries ({:.1}% tombstoned)",
                    tombstone_count,
                    total_count,
                    ratio * 100.0
                );
            }

            // Run compaction if tombstone ratio exceeds threshold
            match search_engine.compact_if_needed().await {
                Ok(Some(compacted)) => {
                    info!(
                        "🧹 Compaction complete: removed {} tombstoned entries",
                        tombstone_count
                    );
                    debug!("📊 Index size after compaction: {} entries", compacted);
                }
                Ok(None) => {
                    // No compaction needed - only log at debug level if there are tombstones
                    if tombstone_count > 0 {
                        debug!(
                            "📊 Compaction not needed yet (threshold: 30%, current: {:.1}%)",
                            ratio * 100.0
                        );
                    }
                }
                Err(e) => {
                    // Compaction failure is non-fatal - search still works with tombstones
                    error!("⚠️ Compaction failed: {:?}", e);
                }
            }

            // Save index to persistent storage
            if let Err(e) = search_engine.save().await {
                error!("❌ Failed to save index: {}", e);
                // Don't fail the batch - data is in memory, just not persisted
                // User will lose data on app close, but can still use the app
            } else {
                info!("💾 Index saved to persistent storage");
            }

            // Get engine metrics for status update (while lock is held)
            let (doc_count, tokens, _) = search_engine.get_index_metrics_sync();
            (doc_count, tokens)
        };

        // Update search engine status
        engine_status.set(super::SearchEngineStatus::Ready {
            doc_count: engine_doc_count,
            total_tokens: engine_total_tokens,
        });
        info!("✅ Search index ready with {} documents", engine_doc_count);
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

/// Shared batch processing logic - Web version.
///
/// NOTE: This implementation is duplicated for desktop (see above) due to different
/// trait bounds required by each platform (Send + Sync for desktop threading).
/// When modifying this function, ensure you update BOTH versions.
/// The only intentional difference is the save() comment on line ~313 vs ~574.
#[cfg(target_arch = "wasm32")]
async fn process_batch_inner<S: DocumentStore + 'static>(
    batch_idx: usize,
    file_list: Vec<(String, String)>,
    mut batches_signal: Signal<Vec<Batch>>,
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
    let mut skipped_count = 0;
    let mut updated_count = 0;

    // Process each file
    for (file_idx, (file_name, contents)) in file_list.iter().enumerate() {
        // Check if binary
        if is_likely_binary(contents) {
            info!("⚠️ Skipped '{}': binary file", file_name);
            batches_signal.write()[batch_idx].files[file_idx].status =
                FileStatus::Failed("Binary file".to_string());
            continue;
        }

        // Generate source ID and content hash for update detection
        let source_id = generate_source_id(file_name);
        let content_hash = compute_content_hash(contents);

        // Check if this source needs to be processed
        let update_action = if let Some(engine_lock) = &engine {
            let search_engine = engine_lock.lock().await;
            match check_source_update(&search_engine, &source_id, &content_hash).await {
                Ok(action) => action,
                Err(e) => {
                    warn!(
                        "⚠️ Failed to check source update for '{}': {}",
                        file_name, e
                    );
                    // Default to Add on error - safer to re-index than skip
                    SourceUpdateAction::Add
                }
            }
        } else {
            SourceUpdateAction::Add
        };

        match update_action {
            SourceUpdateAction::Skip => {
                info!("⏭️ Skipped '{}': no changes detected", file_name);
                skipped_count += 1;
                batches_signal.write()[batch_idx].files[file_idx].status = FileStatus::Completed; // Mark as done (no work needed)
                batches_signal.write()[batch_idx].files[file_idx].progress_pct = 100.0;
                continue;
            }
            SourceUpdateAction::Update { old_chunk_count } => {
                info!(
                    "🔄 Updating '{}': content changed (removing {} old chunks)",
                    file_name, old_chunk_count
                );
                // Delete old chunks before re-indexing
                if let Some(engine_lock) = &engine {
                    let mut search_engine = engine_lock.lock().await;
                    match search_engine.delete_source(&source_id).await {
                        Ok(deleted) => {
                            info!("🗑️ Deleted {} old chunks from '{}'", deleted, file_name);
                        }
                        Err(e) => {
                            error!(
                                "❌ Failed to delete old chunks for '{}': {:?}",
                                file_name, e
                            );
                            batches_signal.write()[batch_idx].files[file_idx].status =
                                FileStatus::Failed(format!("Failed to delete old chunks: {:?}", e));
                            continue;
                        }
                    }
                }
                updated_count += 1;
            }
            SourceUpdateAction::Add => {
                info!("➕ Adding new source '{}'", file_name);
            }
        }

        processed_count += 1;

        // Process file chunks with progress tracking and live metrics updates
        let result = process_file_chunks(
            contents,
            Some(file_name), // Pass filename for semantic chunking
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

                // Index chunks in search engine with source tracking
                if let Some(engine_lock) = &engine {
                    match index_chunks(
                        engine_lock.clone(),
                        &chunk_result.chunks,
                        file_name,
                        &source_id,
                        &content_hash,
                    )
                    .await
                    {
                        Ok(indexed_count) => {
                            info!("✅ Indexed {} chunks from {}", indexed_count, file_name);
                        }
                        Err(e) => {
                            error!("❌ Failed to index chunks: {}", e);
                            // Clean up incomplete source record to avoid orphaned data
                            let mut search_engine = engine_lock.lock().await;
                            if let Err(cleanup_err) = search_engine.delete_source(&source_id).await
                            {
                                warn!(
                                    "⚠️ Failed to cleanup source '{}' after indexing error: {:?}",
                                    source_id, cleanup_err
                                );
                            }
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

    // Log summary of update detection
    info!(
        "📊 Batch summary: {} new, {} updated, {} skipped (unchanged)",
        processed_count - updated_count,
        updated_count,
        skipped_count
    );

    // Finalize vector index (no rebuild needed - incremental HNSW)
    if let Some(engine_lock) = &engine {
        let (engine_doc_count, engine_total_tokens) = {
            let mut search_engine = engine_lock.lock().await;
            // rebuild_vector_index() is a no-op with rust-cv/hnsw (supports incremental updates)
            // Index is already up-to-date from incremental insertions
            if let Err(e) = search_engine.rebuild_vector_index().await {
                error!("❌ Failed to finalize vector index: {}", e);
                return Err(format!("Index finalization failed: {}", e));
            }

            // Check if compaction is needed (tombstone ratio > 30%)
            let (tombstone_count, total_count, ratio) = search_engine.compaction_stats();
            if tombstone_count > 0 {
                debug!(
                    "📊 Tombstone stats: {}/{} entries ({:.1}% tombstoned)",
                    tombstone_count,
                    total_count,
                    ratio * 100.0
                );
            }

            // Run compaction if tombstone ratio exceeds threshold
            match search_engine.compact_if_needed().await {
                Ok(Some(compacted)) => {
                    info!(
                        "🧹 Compaction complete: removed {} tombstoned entries",
                        tombstone_count
                    );
                    debug!("📊 Index size after compaction: {} entries", compacted);
                }
                Ok(None) => {
                    // No compaction needed - only log at debug level if there are tombstones
                    if tombstone_count > 0 {
                        debug!(
                            "📊 Compaction not needed yet (threshold: 30%, current: {:.1}%)",
                            ratio * 100.0
                        );
                    }
                }
                Err(e) => {
                    // Compaction failure is non-fatal - search still works with tombstones
                    error!("⚠️ Compaction failed: {:?}", e);
                }
            }

            // Save index to persistent storage
            if let Err(e) = search_engine.save().await {
                error!("❌ Failed to save index: {}", e);
                // Don't fail the batch - data is in memory, just not persisted
                // User will lose data on app close, but can still use the app
            } else {
                info!("💾 Index saved to persistent storage");
            }

            // Get engine metrics for status update (while lock is held)
            let (doc_count, tokens, _) = search_engine.get_index_metrics_sync();
            (doc_count, tokens)
        };

        // Update search engine status
        engine_status.set(super::SearchEngineStatus::Ready {
            doc_count: engine_doc_count,
            total_tokens: engine_total_tokens,
        });
        info!("✅ Search index ready with {} documents", engine_doc_count);
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
