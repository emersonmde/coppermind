mod components;
mod cpu;
mod embedding;
mod search;
mod storage;
mod wgpu;

use components::TestControls;
use dioxus::prelude::*;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    // Initialize cross-platform logger (web console + desktop stdout)
    dioxus::logger::init(dioxus::logger::tracing::Level::INFO).expect("logger failed to init");
    dioxus::launch(App);
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
            document::Script { src: "/coppermind/assets/coi-serviceworker.min.js" }
        }

        div { class: "container",
            h1 { "Coppermind" }
            TestControls {}
        }
    }
}
