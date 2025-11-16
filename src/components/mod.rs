//! UI components for the Coppermind application.
//!
//! This module contains all Dioxus components that make up the user interface.
//!
//! # New Component Architecture (UI Migration)
//!
//! - [`app_shell`]: AppBar, StatusStrip, MetricsPane, Footer
//! - [`search`]: SearchView, SearchCard, ResultCard, EmptyState (Phase 2+)
//! - [`index`]: IndexView, UploadCard, FileRow, BatchCard (Phase 3+)
//!
//! # Legacy Components (to be removed in Phase 6)
//!
//! - [`Hero`]: Old landing section
//! - [`FileUpload`]: Old file upload interface
//! - [`Search`]: Old search interface
//! - [`DeveloperTesting`]: Developer tools
//! - `file_processing`: Utilities (will be kept)
//!
//! # Context Providers
//!
//! Components use Dioxus context for shared state:
//!
//! ```ignore
//! // Access search engine from any component
//! let engine = use_search_engine();
//!
//! // Check engine status
//! let status = use_search_engine_status();
//! match status.read().clone() {
//!     SearchEngineStatus::Ready { doc_count } => { /* ... */ }
//!     SearchEngineStatus::Pending => { /* ... */ }
//!     SearchEngineStatus::Failed(err) => { /* ... */ }
//! }
//! ```

// New component modules
mod app_shell;
mod index;
pub mod search; // Public for SearchView re-export

// Legacy components (Phase 6: remove)
mod file_processing;
mod testing;
pub mod worker; // Worker state management (web only)

// Re-export new components
pub use app_shell::{AppBar, Footer, MetricsPane, View};
pub use index::{Batch, BatchMetrics, BatchStatus, FileInBatch, IndexView};
pub use search::SearchView;

// Re-export testing component
pub use testing::DeveloperTesting;

// Main app component that composes all sections
use crate::search::HybridSearchEngine;
use dioxus::logger::tracing::{error, info};
use dioxus::prelude::*;
use futures::lock::Mutex;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;
use instant::Instant;
use std::sync::Arc;

// ============================================================================
// Platform-specific imports and types
// ============================================================================

// Web platform (WASM)
#[cfg(target_arch = "wasm32")]
use crate::storage::OpfsStorage;
#[cfg(target_arch = "wasm32")]
use worker::provide_worker_state;
#[cfg(target_arch = "wasm32")]
type PlatformStorage = OpfsStorage;

// Desktop platform (Native)
#[cfg(not(target_arch = "wasm32"))]
use crate::storage::NativeStorage;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
type PlatformStorage = NativeStorage;

// ============================================================================
// File processing for indexing
// ============================================================================

use file_processing::{index_chunks, is_likely_binary};
use index::file_row::{FileMetrics, FileStatus};

/// Messages for file processing coroutine
pub enum ProcessingMessage {
    ProcessFiles(Vec<(String, String)>), // Vec of (filename, contents)
}

/// Type alias for the search engine signal used throughout components.
///
/// This complex generic type appears in multiple function signatures and provides
/// shared mutable access to the search engine across the component tree.
type SearchEngineSignal = Signal<Option<Arc<Mutex<HybridSearchEngine<PlatformStorage>>>>>;

// Search engine status for UI display
#[derive(Clone)]
pub enum SearchEngineStatus {
    Pending,
    Ready { doc_count: usize },
    Failed(String),
}

// Search engine context provider (platform-agnostic via type alias)
pub fn use_search_engine() -> SearchEngineSignal {
    use_context::<SearchEngineSignal>()
}

// Search engine status context provider
pub fn use_search_engine_status() -> Signal<SearchEngineStatus> {
    use_context::<Signal<SearchEngineStatus>>()
}

// ============================================================================
// Indexing state context (shared across views)
// ============================================================================

/// Context provider for all batches (pending, processing, completed)
pub fn use_batches() -> Signal<Vec<Batch>> {
    use_context::<Signal<Vec<Batch>>>()
}

/// Context provider for sending files to the processing coroutine
pub fn use_processing_sender() -> Coroutine<ProcessingMessage> {
    use_context::<Coroutine<ProcessingMessage>>()
}

// ============================================================================
// Engine metrics (aggregate statistics from all batches)
// ============================================================================

/// Aggregate engine statistics calculated from all completed batches
#[derive(Clone, Default, PartialEq)]
pub struct EngineMetrics {
    /// Total number of documents indexed
    pub total_docs: usize,
    /// Total number of chunks created
    pub total_chunks: usize,
    /// Total number of tokens processed
    pub total_tokens: usize,
    /// Average tokens per chunk
    pub avg_tokens_per_chunk: usize,
}

/// Live indexing metrics from currently active batch
#[derive(Clone, Default, PartialEq)]
pub struct LiveIndexingMetrics {
    /// Tokens processed per second
    pub tokens_per_sec: f64,
    /// Chunks processed per second
    pub chunks_per_sec: f64,
    /// Average time per chunk in milliseconds
    pub avg_chunk_time_ms: f64,
    /// Whether these metrics are from an active batch (true) or last completed batch (false)
    pub is_live: bool,
}

/// Calculate aggregate engine metrics from all completed batches
pub fn calculate_engine_metrics(batches: &[Batch]) -> EngineMetrics {
    let mut total_docs = 0;
    let mut total_chunks = 0;
    let mut total_tokens = 0;

    for batch in batches {
        if let Some(metrics) = &batch.metrics {
            total_docs += metrics.file_count;
            total_chunks += metrics.chunk_count;
            total_tokens += metrics.token_count;
        }
    }

    let avg_tokens_per_chunk = if total_chunks > 0 {
        total_tokens / total_chunks
    } else {
        0
    };

    EngineMetrics {
        total_docs,
        total_chunks,
        total_tokens,
        avg_tokens_per_chunk,
    }
}

/// Calculate live indexing metrics from currently processing batch or last completed batch
pub fn calculate_live_metrics(batches: &[Batch]) -> Option<LiveIndexingMetrics> {
    // First, try to find a batch that's currently processing
    if let Some(processing_batch) = batches
        .iter()
        .find(|b| matches!(b.status, BatchStatus::Processing))
    {
        // Calculate from all files that have metrics (completed + currently processing)
        let mut total_chunks = 0;
        let mut total_tokens = 0;
        let mut total_elapsed_ms = 0u64;

        for file in &processing_batch.files {
            // Include both completed files AND currently processing files with metrics
            if let Some(metrics) = &file.metrics {
                total_chunks += metrics.chunks_embedded;
                total_tokens += metrics.tokens_processed;
                total_elapsed_ms += metrics.elapsed_ms;
            }
        }

        if total_chunks > 0 {
            let elapsed_secs = total_elapsed_ms as f64 / 1000.0;
            let tokens_per_sec = if elapsed_secs > 0.0 {
                total_tokens as f64 / elapsed_secs
            } else {
                0.0
            };
            let chunks_per_sec = if elapsed_secs > 0.0 {
                total_chunks as f64 / elapsed_secs
            } else {
                0.0
            };
            let avg_chunk_time_ms = if total_chunks > 0 {
                total_elapsed_ms as f64 / total_chunks as f64
            } else {
                0.0
            };

            return Some(LiveIndexingMetrics {
                tokens_per_sec,
                chunks_per_sec,
                avg_chunk_time_ms,
                is_live: true,
            });
        }
    }

    // No active batch, find the most recently completed batch
    let last_completed = batches
        .iter()
        .rev() // Reverse to get newest first
        .find(|b| matches!(b.status, BatchStatus::Completed) && b.metrics.is_some())?;

    // Calculate metrics from the batch metadata
    if let Some(batch_metrics) = &last_completed.metrics {
        let elapsed_secs = batch_metrics.duration_ms as f64 / 1000.0;
        let tokens_per_sec = if elapsed_secs > 0.0 {
            batch_metrics.token_count as f64 / elapsed_secs
        } else {
            0.0
        };
        let chunks_per_sec = if elapsed_secs > 0.0 {
            batch_metrics.chunk_count as f64 / elapsed_secs
        } else {
            0.0
        };
        let avg_chunk_time_ms = if batch_metrics.chunk_count > 0 {
            batch_metrics.duration_ms as f64 / batch_metrics.chunk_count as f64
        } else {
            0.0
        };

        return Some(LiveIndexingMetrics {
            tokens_per_sec,
            chunks_per_sec,
            avg_chunk_time_ms,
            is_live: false,
        });
    }

    None
}

// ============================================================================
// Platform-specific initialization helpers
// ============================================================================

/// Initialize platform-specific storage backend.
#[cfg(target_arch = "wasm32")]
async fn create_platform_storage() -> Result<PlatformStorage, String> {
    OpfsStorage::new()
        .await
        .map_err(|e| format!("Storage error: {:?}", e))
}

/// Initialize platform-specific storage backend.
#[cfg(not(target_arch = "wasm32"))]
async fn create_platform_storage() -> Result<PlatformStorage, String> {
    NativeStorage::new(PathBuf::from("./coppermind-storage"))
        .map_err(|e| format!("Storage error: {:?}", e))
}

/// Setup search engine with initialized storage.
async fn initialize_search_engine(
    storage: PlatformStorage,
) -> Result<HybridSearchEngine<PlatformStorage>, String> {
    let engine = HybridSearchEngine::new(storage, 512)
        .await
        .map_err(|e| format!("{:?}", e))?;

    Ok(engine)
}

#[component]
pub fn App() -> Element {
    // Provide worker state context for child components (web only)
    #[cfg(target_arch = "wasm32")]
    {
        provide_worker_state();
    }

    // Initialize search engine status
    let search_engine_status = use_signal(|| SearchEngineStatus::Pending);
    use_context_provider(|| search_engine_status);

    // Initialize search engine with platform-specific storage (unified logic)
    let search_engine = use_signal(|| None);
    use_context_provider(|| search_engine);

    let mut engine_signal = search_engine;
    let mut status_signal = search_engine_status;
    use_effect(move || {
        if engine_signal.read().is_none() {
            spawn(async move {
                match create_platform_storage().await {
                    Ok(storage) => match initialize_search_engine(storage).await {
                        Ok(engine) => {
                            let doc_count = engine.len();
                            // Arc is single-threaded on WASM (no Send/Sync needed)
                            #[cfg(target_arch = "wasm32")]
                            #[allow(clippy::arc_with_non_send_sync)]
                            let arc_engine = Arc::new(Mutex::new(engine));
                            #[cfg(not(target_arch = "wasm32"))]
                            let arc_engine = Arc::new(Mutex::new(engine));

                            engine_signal.set(Some(arc_engine));
                            status_signal.set(SearchEngineStatus::Ready { doc_count });
                        }
                        Err(e) => {
                            error!("Failed to initialize search engine: {}", e);
                            status_signal.set(SearchEngineStatus::Failed(e));
                        }
                    },
                    Err(e) => {
                        error!("Failed to initialize storage: {}", e);
                        status_signal.set(SearchEngineStatus::Failed(e));
                    }
                }
            });
        }
    });

    // Initialize batches state (persists across view switches)
    let batches = use_signal(Vec::<Batch>::new);
    let batch_counter = use_signal(|| 0usize);
    use_context_provider(|| batches);

    // File processing coroutine (runs in background, persists across view switches)
    #[cfg(target_arch = "wasm32")]
    let worker_state = worker::use_worker_state();

    let processing_coroutine = use_coroutine({
        let mut batches_signal = batches;
        let mut counter_signal = batch_counter;
        let engine = search_engine;
        let engine_status = search_engine_status;

        move |mut rx: UnboundedReceiver<ProcessingMessage>| async move {
            while let Some(msg) = rx.next().await {
                match msg {
                    ProcessingMessage::ProcessFiles(file_list) => {
                        // Create a new pending batch
                        let batch_number = counter_signal() + 1;
                        counter_signal.set(batch_number);

                        // Initialize all files as Queued
                        let files: Vec<FileInBatch> = file_list
                            .iter()
                            .map(|(name, _)| FileInBatch {
                                name: name.clone(),
                                status: FileStatus::Queued,
                                progress_pct: 0.0,
                                metrics: None,
                            })
                            .collect();

                        // Create pending batch and add to queue
                        let new_batch = Batch {
                            batch_number,
                            status: BatchStatus::Pending,
                            files: files.clone(),
                            metrics: None,
                        };

                        let mut all_batches = batches_signal();
                        all_batches.push(new_batch);
                        batches_signal.set(all_batches.clone());

                        // Find the index of this batch
                        let batch_idx = all_batches.len() - 1;

                        // Spawn processing as a separate async task (doesn't block message handler)
                        // This allows the UI to render the pending batch before processing starts
                        let mut batches_signal_clone = batches_signal;
                        let engine_clone = engine;
                        let mut engine_status_clone = engine_status;
                        #[cfg(target_arch = "wasm32")]
                        let worker_state_clone = worker_state;

                        spawn(async move {
                            // Update batch status to Processing
                            let mut all_batches = batches_signal_clone();
                            all_batches[batch_idx].status = BatchStatus::Processing;
                            batches_signal_clone.set(all_batches.clone());

                            // Track batch timing and stats
                            let batch_start = Instant::now();
                            let mut total_chunks: usize = 0;
                            let mut total_tokens: usize = 0;
                            let mut processed_count = 0;

                            for (file_idx, (file_name, contents)) in file_list.iter().enumerate() {
                                // Check if binary
                                if is_likely_binary(contents) {
                                    info!("⚠️ Skipped '{}': binary file", file_name);
                                    let mut all_batches = batches_signal_clone();
                                    all_batches[batch_idx].files[file_idx].status =
                                        FileStatus::Failed("Binary file".to_string());
                                    batches_signal_clone.set(all_batches.clone());
                                    continue;
                                }

                                processed_count += 1;
                                let start_time = Instant::now();

                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    // Desktop: Process chunks individually for real-time metric updates
                                    // Split into chunks (2000 chars each)
                                    let chunk_size = 2000;
                                    let text_chunks: Vec<String> = contents
                                        .chars()
                                        .collect::<Vec<_>>()
                                        .chunks(chunk_size)
                                        .map(|chunk| chunk.iter().collect())
                                        .collect();

                                    let chunks_in_file = text_chunks.len();
                                    let mut results = Vec::new();

                                    // Initialize metrics before processing
                                    let mut all_batches = batches_signal_clone();
                                    all_batches[batch_idx].files[file_idx].metrics =
                                        Some(FileMetrics {
                                            tokens_processed: 0,
                                            chunks_embedded: 0,
                                            chunks_total: chunks_in_file,
                                            elapsed_ms: 0,
                                        });
                                    batches_signal_clone.set(all_batches.clone());

                                    // Process each chunk and update metrics
                                    for (chunk_idx, chunk_text) in text_chunks.iter().enumerate() {
                                        // Update progress (read latest state first)
                                        let progress =
                                            (chunk_idx as f64 / chunks_in_file as f64) * 100.0;
                                        let mut all_batches = batches_signal_clone();
                                        all_batches[batch_idx].files[file_idx].status =
                                            FileStatus::Processing {
                                                current: chunk_idx,
                                                total: chunks_in_file,
                                            };
                                        all_batches[batch_idx].files[file_idx].progress_pct =
                                            progress;
                                        batches_signal_clone.set(all_batches.clone());

                                        match crate::embedding::compute_embedding(chunk_text).await
                                        {
                                            Ok(computation) => {
                                                results.push(
                                                    crate::embedding::ChunkEmbeddingResult {
                                                        chunk_index: chunk_idx,
                                                        token_count: computation.token_count,
                                                        text: chunk_text.clone(),
                                                        embedding: computation.embedding,
                                                    },
                                                );

                                                // Update metrics after each chunk (read latest state first)
                                                let elapsed_ms =
                                                    start_time.elapsed().as_millis() as u64;
                                                let mut all_batches = batches_signal_clone();
                                                all_batches[batch_idx].files[file_idx].metrics =
                                                    Some(FileMetrics {
                                                        tokens_processed: results
                                                            .iter()
                                                            .map(|c| c.token_count)
                                                            .sum(),
                                                        chunks_embedded: results.len(),
                                                        chunks_total: chunks_in_file,
                                                        elapsed_ms,
                                                    });
                                                batches_signal_clone.set(all_batches.clone());
                                            }
                                            Err(e) => {
                                                error!(
                                                    "❌ Failed to embed chunk {}: {}",
                                                    chunk_idx, e
                                                );
                                                let mut all_batches = batches_signal_clone();
                                                all_batches[batch_idx].files[file_idx].status =
                                                    FileStatus::Failed(e.to_string());
                                                batches_signal_clone.set(all_batches.clone());
                                                break;
                                            }
                                        }
                                    }

                                    let chunk_count = results.len();
                                    let file_tokens: usize =
                                        results.iter().map(|c| c.token_count).sum();
                                    let elapsed_ms = start_time.elapsed().as_millis() as u64;

                                    // Accumulate batch totals
                                    total_chunks += chunk_count;
                                    total_tokens += file_tokens;

                                    // Update file status to completed (read latest state first)
                                    let mut all_batches = batches_signal_clone();
                                    all_batches[batch_idx].files[file_idx].status =
                                        FileStatus::Completed;
                                    all_batches[batch_idx].files[file_idx].progress_pct = 100.0;
                                    all_batches[batch_idx].files[file_idx].metrics =
                                        Some(FileMetrics {
                                            tokens_processed: file_tokens,
                                            chunks_embedded: chunk_count,
                                            chunks_total: chunk_count,
                                            elapsed_ms,
                                        });
                                    batches_signal_clone.set(all_batches.clone());

                                    // Index chunks
                                    let engine_arc = engine_clone.read().clone();
                                    if let Some(engine_lock) = engine_arc {
                                        match index_chunks(engine_lock, &results, file_name).await {
                                            Ok(indexed_count) => {
                                                info!(
                                                    "✅ Indexed {} chunks from {}",
                                                    indexed_count, file_name
                                                );
                                            }
                                            Err(e) => {
                                                error!("❌ Failed to index chunks: {}", e);
                                            }
                                        }
                                    }
                                }

                                #[cfg(target_arch = "wasm32")]
                                {
                                    // Web: Use embedding worker
                                    let worker_snapshot = worker_state_clone.read().clone();
                                    match worker_snapshot {
                                        worker::WorkerStatus::Pending => {
                                            let mut all_batches = batches_signal_clone();
                                            all_batches[batch_idx].files[file_idx].status =
                                                FileStatus::Failed("Worker starting".to_string());
                                            batches_signal_clone.set(all_batches.clone());
                                            continue;
                                        }
                                        worker::WorkerStatus::Failed(err) => {
                                            let mut all_batches = batches_signal_clone();
                                            all_batches[batch_idx].files[file_idx].status =
                                                FileStatus::Failed(err);
                                            batches_signal_clone.set(all_batches.clone());
                                            continue;
                                        }
                                        worker::WorkerStatus::Ready(client) => {
                                            // Split into chunks (2000 chars each)
                                            let chunk_size = 2000;
                                            let text_chunks: Vec<String> = contents
                                                .chars()
                                                .collect::<Vec<_>>()
                                                .chunks(chunk_size)
                                                .map(|chunk| chunk.iter().collect())
                                                .collect();

                                            let chunks_in_file = text_chunks.len();
                                            let mut results = Vec::new();

                                            // Initialize metrics before processing
                                            let mut all_batches = batches_signal_clone();
                                            all_batches[batch_idx].files[file_idx].metrics =
                                                Some(FileMetrics {
                                                    tokens_processed: 0,
                                                    chunks_embedded: 0,
                                                    chunks_total: chunks_in_file,
                                                    elapsed_ms: 0,
                                                });
                                            batches_signal_clone.set(all_batches.clone());

                                            for (chunk_idx, chunk_text) in
                                                text_chunks.iter().enumerate()
                                            {
                                                // Update progress (read latest state first)
                                                let progress = (chunk_idx as f64
                                                    / chunks_in_file as f64)
                                                    * 100.0;
                                                let mut all_batches = batches_signal_clone();
                                                all_batches[batch_idx].files[file_idx].status =
                                                    FileStatus::Processing {
                                                        current: chunk_idx,
                                                        total: chunks_in_file,
                                                    };
                                                all_batches[batch_idx].files[file_idx]
                                                    .progress_pct = progress;
                                                batches_signal_clone.set(all_batches.clone());

                                                match client.embed(chunk_text.clone()).await {
                                                    Ok(computation) => {
                                                        results.push(
                                                        crate::embedding::ChunkEmbeddingResult {
                                                            chunk_index: chunk_idx,
                                                            token_count: computation.token_count,
                                                            text: chunk_text.clone(),
                                                            embedding: computation.embedding,
                                                        },
                                                    );

                                                        // Update metrics after each chunk (read latest state first)
                                                        let elapsed_ms =
                                                            start_time.elapsed().as_millis() as u64;
                                                        let mut all_batches =
                                                            batches_signal_clone();
                                                        all_batches[batch_idx].files[file_idx]
                                                            .metrics = Some(FileMetrics {
                                                            tokens_processed: results
                                                                .iter()
                                                                .map(|c| c.token_count)
                                                                .sum(),
                                                            chunks_embedded: results.len(),
                                                            chunks_total: chunks_in_file,
                                                            elapsed_ms,
                                                        });
                                                        batches_signal_clone
                                                            .set(all_batches.clone());
                                                    }
                                                    Err(e) => {
                                                        error!(
                                                            "❌ Failed to embed chunk {}: {}",
                                                            chunk_idx, e
                                                        );
                                                        let mut all_batches =
                                                            batches_signal_clone();
                                                        all_batches[batch_idx].files[file_idx]
                                                            .status =
                                                            FileStatus::Failed(e.to_string());
                                                        batches_signal_clone
                                                            .set(all_batches.clone());
                                                        break;
                                                    }
                                                }
                                            }

                                            let chunk_count = results.len();
                                            let file_tokens: usize =
                                                results.iter().map(|c| c.token_count).sum();
                                            let elapsed_ms =
                                                start_time.elapsed().as_millis() as u64;

                                            // Accumulate batch totals
                                            total_chunks += chunk_count;
                                            total_tokens += file_tokens;

                                            // Update file status to completed (read latest state first)
                                            let mut all_batches = batches_signal_clone();
                                            all_batches[batch_idx].files[file_idx].status =
                                                FileStatus::Completed;
                                            all_batches[batch_idx].files[file_idx].progress_pct =
                                                100.0;
                                            all_batches[batch_idx].files[file_idx].metrics =
                                                Some(FileMetrics {
                                                    tokens_processed: file_tokens,
                                                    chunks_embedded: chunk_count,
                                                    chunks_total: chunk_count,
                                                    elapsed_ms,
                                                });
                                            batches_signal_clone.set(all_batches.clone());

                                            // Index chunks
                                            let engine_arc = engine_clone.read().clone();
                                            if let Some(engine_lock) = engine_arc {
                                                match index_chunks(engine_lock, &results, file_name)
                                                    .await
                                                {
                                                    Ok(indexed_count) => {
                                                        info!(
                                                            "✅ Indexed {} chunks from {}",
                                                            indexed_count, file_name
                                                        );
                                                    }
                                                    Err(e) => {
                                                        error!("❌ Failed to index chunks: {}", e);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Rebuild vector index once after all files
                            let engine_arc = engine_clone.read().clone();
                            if let Some(engine_lock) = engine_arc {
                                let doc_count = {
                                    let search_engine = engine_lock.lock().await;
                                    search_engine.len()
                                };

                                info!("🔨 Rebuilding HNSW index for {} documents... (this may take a few minutes for large batches)", doc_count);

                                {
                                    let mut search_engine = engine_lock.lock().await;
                                    search_engine.rebuild_vector_index().await;
                                }

                                // Update search engine status
                                engine_status_clone.set(SearchEngineStatus::Ready { doc_count });
                                info!("✅ Search index rebuilt with {} documents", doc_count);
                            }

                            // Mark batch as completed with metrics (read latest state first)
                            let duration_ms = batch_start.elapsed().as_millis() as u64;
                            let mut all_batches = batches_signal_clone();
                            all_batches[batch_idx].status = BatchStatus::Completed;
                            all_batches[batch_idx].metrics = Some(BatchMetrics {
                                file_count: processed_count,
                                chunk_count: total_chunks,
                                token_count: total_tokens,
                                duration_ms,
                            });
                            batches_signal_clone.set(all_batches.clone());

                            info!(
                                "✅ Batch #{} complete: {} files, {} chunks, {} tokens in {:.1}s",
                                batch_number,
                                processed_count,
                                total_chunks,
                                total_tokens,
                                duration_ms as f64 / 1000.0
                            );
                        });
                    }
                }
            }
        }
    });

    use_context_provider(|| processing_coroutine);

    // View state management
    let mut current_view = use_signal(|| View::Search);
    let mut metrics_collapsed = use_signal(|| true);

    rsx! {
        div { class: "cm-app",
            // App shell components
            AppBar {
                current_view,
                on_view_change: move |view| current_view.set(view),
                on_metrics_toggle: move |_| {
                    metrics_collapsed.set(!metrics_collapsed());
                },
                metrics_collapsed
            }

            MetricsPane {
                collapsed: metrics_collapsed
            }

            // Main content area with view routing
            main { class: "cm-main",
                if current_view() == View::Search {
                    SearchView {
                        on_navigate: move |view| current_view.set(view)
                    }
                } else {
                    IndexView {}
                }
            }

            Footer {}
        }
    }
}
