//! UI components for the Coppermind application.
//!
//! This module contains all Dioxus components that make up the user interface:
//!
//! - [`Hero`]: Landing section with app description
//! - [`FileUpload`]: File upload interface for indexing documents
//! - [`Search`]: Search interface with query input and results
//! - [`DeveloperTesting`]: Developer tools for testing embeddings
//! - [`file_processing`]: Utilities for processing uploaded files
//!
//! # Component Architecture
//!
//! The main [`App`] component orchestrates all sections and provides:
//! - Search engine initialization (platform-specific storage)
//! - Context providers for search engine and status
//! - Layout composition
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
//!
//! # Platform Differences
//!
//! This module uses **type aliases** and **helper functions** to minimize
//! `cfg` directive clutter while maintaining platform-specific behavior:
//!
//! **Pattern Used:**
//! - `PlatformStorage` type alias resolves to `OpfsStorage` (web) or `NativeStorage` (desktop)
//! - `create_platform_storage()` helper encapsulates platform-specific initialization
//! - Main component logic is unified, with platform differences isolated to helpers
//!
//! **Web (WASM)**:
//! - Uses `OpfsStorage` for persistence
//! - Web Worker support for background processing
//! - Single-threaded (Arc without Send/Sync)
//!
//! **Desktop**:
//! - Uses `NativeStorage` for filesystem storage
//! - Multi-threaded (Arc with Send/Sync)
//! - `tokio::spawn_blocking` for CPU-intensive operations

mod file_processing;
mod file_upload;
mod hero;
mod search;
mod testing;

pub use file_upload::FileUpload;
pub use hero::Hero;
pub use search::Search;
pub use testing::DeveloperTesting;

// Main app component that composes all sections
use crate::search::HybridSearchEngine;
use dioxus::logger::tracing::error;
use dioxus::prelude::*;
use futures::lock::Mutex;
use std::sync::Arc;

// ============================================================================
// Platform-specific imports and types
// ============================================================================

// Web platform (WASM)
#[cfg(target_arch = "wasm32")]
use crate::storage::OpfsStorage;
#[cfg(target_arch = "wasm32")]
use hero::provide_worker_state;
#[cfg(target_arch = "wasm32")]
type PlatformStorage = OpfsStorage;

// Desktop platform (Native)
#[cfg(not(target_arch = "wasm32"))]
use crate::storage::NativeStorage;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
type PlatformStorage = NativeStorage;

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
    HybridSearchEngine::new(storage, 512)
        .await
        .map_err(|e| format!("{:?}", e))
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

    rsx! {
        div { class: "app-layout",
            Hero {}
            FileUpload {}
            div { class: "divider" }
            Search {}
            div { class: "divider" }
            DeveloperTesting {}
        }
    }
}
