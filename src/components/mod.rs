// Component module organization
mod file_upload;
mod hero;
mod testing;

pub use file_upload::FileUpload;
pub use hero::Hero;
pub use testing::DeveloperTesting;

// Main app component that composes all sections
use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use hero::provide_worker_state;

#[component]
pub fn App() -> Element {
    // Provide worker state context for child components (web only)
    #[cfg(target_arch = "wasm32")]
    {
        provide_worker_state();
    }

    rsx! {
        div { class: "app-layout",
            Hero {}
            FileUpload {}
            div { class: "divider" }
            DeveloperTesting {}
        }
    }
}
