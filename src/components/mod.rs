// Component module organization
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
use dioxus::prelude::*;
use futures::lock::Mutex;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use crate::storage::OpfsStorage;
#[cfg(target_arch = "wasm32")]
use hero::provide_worker_state;

#[cfg(not(target_arch = "wasm32"))]
use crate::storage::NativeStorage;

// Search engine status for UI display
#[derive(Clone)]
pub enum SearchEngineStatus {
    Pending,
    Ready { doc_count: usize },
    Failed(String),
}

// Search engine context provider
#[cfg(target_arch = "wasm32")]
pub fn use_search_engine() -> Signal<Option<Arc<Mutex<HybridSearchEngine<OpfsStorage>>>>> {
    use_context::<Signal<Option<Arc<Mutex<HybridSearchEngine<OpfsStorage>>>>>>()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn use_search_engine() -> Signal<Option<Arc<Mutex<HybridSearchEngine<NativeStorage>>>>> {
    use_context::<Signal<Option<Arc<Mutex<HybridSearchEngine<NativeStorage>>>>>>()
}

// Search engine status context provider
pub fn use_search_engine_status() -> Signal<SearchEngineStatus> {
    use_context::<Signal<SearchEngineStatus>>()
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

    // Initialize search engine with platform-specific storage
    #[cfg(target_arch = "wasm32")]
    {
        let search_engine = use_signal(|| None);
        use_context_provider(|| search_engine);

        // Initialize engine asynchronously
        let mut engine_signal = search_engine;
        let mut status_signal = search_engine_status;
        use_effect(move || {
            if engine_signal.read().is_none() {
                spawn(async move {
                    match OpfsStorage::new().await {
                        Ok(storage) => match HybridSearchEngine::new(storage, 512).await {
                            Ok(engine) => {
                                let doc_count = engine.len();
                                engine_signal.set(Some(Arc::new(Mutex::new(engine))));
                                status_signal.set(SearchEngineStatus::Ready { doc_count });
                            }
                            Err(e) => {
                                dioxus::logger::tracing::error!(
                                    "Failed to initialize search engine: {:?}",
                                    e
                                );
                                status_signal.set(SearchEngineStatus::Failed(format!("{:?}", e)));
                            }
                        },
                        Err(e) => {
                            dioxus::logger::tracing::error!(
                                "Failed to initialize OPFS storage: {:?}",
                                e
                            );
                            status_signal.set(SearchEngineStatus::Failed(format!(
                                "Storage error: {:?}",
                                e
                            )));
                        }
                    }
                });
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let search_engine = use_signal(|| None);
        use_context_provider(|| search_engine);

        // Initialize engine asynchronously
        let mut engine_signal = search_engine;
        let mut status_signal = search_engine_status;
        use_effect(move || {
            if engine_signal.read().is_none() {
                spawn(async move {
                    let storage =
                        NativeStorage::new(std::path::PathBuf::from("./coppermind-storage"))
                            .expect("Failed to create storage");
                    match HybridSearchEngine::new(storage, 512).await {
                        Ok(engine) => {
                            let doc_count = engine.len();
                            engine_signal.set(Some(Arc::new(Mutex::new(engine))));
                            status_signal.set(SearchEngineStatus::Ready { doc_count });
                        }
                        Err(e) => {
                            eprintln!("Failed to initialize search engine: {:?}", e);
                            status_signal.set(SearchEngineStatus::Failed(format!("{:?}", e)));
                        }
                    }
                });
            }
        });
    }

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
