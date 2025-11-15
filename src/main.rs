use coppermind::components::App as CoppermindApp;
use dioxus::prelude::*;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        let window = web_sys::window();
        let has_document = window.as_ref().and_then(|w| w.document()).is_some();

        if window.is_none() || !has_document {
            // Running inside a Web Worker — skip mounting the UI.
            return;
        }
    }

    // Initialize cross-platform logger (web console + desktop stdout)
    dioxus::logger::init(dioxus::logger::tracing::Level::INFO).expect("logger failed to init");

    // Platform-specific launch configuration
    #[cfg(not(target_arch = "wasm32"))]
    {
        use dioxus::desktop::{Config, LogicalSize, WindowBuilder};

        let config = Config::default().with_window(
            WindowBuilder::new()
                .with_title("Coppermind")
                .with_resizable(true)
                .with_inner_size(LogicalSize::new(1200.0, 900.0))
                .with_min_inner_size(LogicalSize::new(800.0, 600.0))
                // Set dark background to prevent white flash on overscroll
                .with_transparent(false),
        );

        dioxus::LaunchBuilder::desktop()
            .with_cfg(config)
            .launch(App);
    }

    #[cfg(target_arch = "wasm32")]
    {
        dioxus::launch(App);
    }
}

#[component]
fn App() -> Element {
    rsx! {
        document::Link { rel: "icon", href: FAVICON }

        // CSS loading: asset! macro has issues on desktop, use include_str! as workaround
        if cfg!(target_arch = "wasm32") {
            document::Stylesheet { href: MAIN_CSS }
        } else {
            style { {include_str!("../assets/main.css")} }
        }

        // COEP Service Worker only needed for web (SharedArrayBuffer support)
        if cfg!(target_arch = "wasm32") {
            document::Script {
                r#"window.coi = {{ coepCredentialless: true, quiet: false }};"#
            }
            // In dev mode, public/ files are served at /assets/, but in release they're at root
            if cfg!(debug_assertions) {
                document::Script { src: "/coppermind/assets/coi-serviceworker.min.js" }
            } else {
                document::Script { src: "/coppermind/coi-serviceworker.min.js" }
            }
        }

        div { class: "container",
            CoppermindApp {}
        }
    }
}
