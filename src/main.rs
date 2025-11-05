mod components;
mod cpu;
mod embedding;
mod wgpu;

use components::TestControls;
use dioxus::prelude::*;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        document::Script {
            r#"window.coi = {{ coepCredentialless: true, quiet: false }};"#
        }
        // Load the local SW from the root (copied from public/)
        document::Script { src: "/coppermind/assets/coi-serviceworker.min.js" }

        div { class: "container",
            h1 { "Coppermind" }
            TestControls {}
        }
    }
}
