//! UI components for the Coppermind application.
//!
//! This module contains all Dioxus components that make up the user interface.
//!
//! # New Component Architecture (UI Migration)
//!
//! - `app_shell`: AppBar, StatusStrip, MetricsPane, Footer
//! - `search`: SearchView, SearchCard, ResultCard, EmptyState (Phase 2+)
//! - `index`: IndexView, UploadCard, FileRow, BatchCard (Phase 3+)
//!
//! # Legacy Components (to be removed in Phase 6)
//!
//! - Old landing section (removed)
//! - Old file upload interface (removed)
//! - Old search interface (removed)
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
mod batch_processor; // Batch processing orchestration
mod index;
pub mod search; // Public for SearchView re-export

// Legacy components (Phase 6: remove)
mod file_processing;
mod testing;
pub mod worker; // Worker state management (web only)

// Re-export new components
pub use app_shell::{AppBar, Footer, MetricsPane, View};
pub use index::{
    Batch, BatchMetrics, BatchStatus, FileInBatch, FileMetrics, FileStatus, IndexView,
};
pub use search::SearchView;

// Re-export testing component
pub use testing::DeveloperTesting;

// Main app component that composes all sections
use crate::search::HybridSearchEngine;
use dioxus::logger::tracing::error;
use dioxus::prelude::*;
use futures::lock::Mutex;
use futures_channel::mpsc::UnboundedReceiver;
use futures_util::StreamExt;
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

use crate::processing::embedder::PlatformEmbedder;
use crate::utils::SignalExt;
// FileMetrics and FileStatus are now re-exported at the module level (line 48)

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
        #[cfg(target_arch = "wasm32")]
        let worker_state_for_coroutine = worker_state;

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

                        // Create pending batch and add to queue (using SignalExt::mutate)
                        let new_batch = Batch {
                            batch_number,
                            status: BatchStatus::Pending,
                            files,
                            metrics: None,
                        };

                        batches_signal.mutate(|batches| {
                            batches.push(new_batch);
                        });

                        let batch_idx = batches_signal.read().len() - 1;

                        // Spawn processing as a separate async task
                        let mut batches_for_spawn = batches_signal;
                        let engine_for_spawn = engine;
                        let engine_status_for_spawn = engine_status;

                        #[cfg(target_arch = "wasm32")]
                        let worker_state_for_spawn = worker_state_for_coroutine;

                        spawn(async move {
                            // Create platform-specific embedder
                            #[cfg(target_arch = "wasm32")]
                            let embedder = {
                                use crate::processing::embedder::web::WebEmbedder;
                                WebEmbedder::from_worker_state(worker_state_for_spawn)
                            };

                            #[cfg(not(target_arch = "wasm32"))]
                            let embedder = {
                                use crate::processing::embedder::desktop::DesktopEmbedder;
                                DesktopEmbedder::new()
                            };

                            // Check if embedder is ready
                            if !embedder.is_ready() {
                                error!("Embedder not ready: {}", embedder.status_message());
                                batches_for_spawn.mutate(|batches| {
                                    batches[batch_idx].status = BatchStatus::Completed;
                                    for file in &mut batches[batch_idx].files {
                                        file.status = FileStatus::Failed(embedder.status_message());
                                    }
                                });
                                return;
                            }

                            // Process the batch using our clean abstraction
                            let engine_clone = engine_for_spawn.read().clone();
                            let result = batch_processor::process_batch(
                                batch_idx,
                                file_list,
                                batches_for_spawn,
                                &embedder,
                                engine_clone,
                                engine_status_for_spawn,
                            )
                            .await;

                            if let Err(e) = result {
                                error!("Batch processing failed: {}", e);
                            }
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
