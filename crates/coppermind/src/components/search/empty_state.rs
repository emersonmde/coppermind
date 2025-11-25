use dioxus::prelude::*;

use crate::components::View;

/// Empty state shown when no documents are indexed
#[component]
pub fn EmptyState(on_navigate_to_index: EventHandler<View>) -> Element {
    rsx! {
        section { class: "cm-empty-state",
            div { class: "cm-empty-card",
                h2 { class: "cm-empty-title", "No documents indexed yet" }
                p { class: "cm-empty-text",
                    "Start by indexing your local files to power Coppermind's hybrid search engine."
                }
                button {
                    class: "cm-btn cm-btn--primary",
                    onclick: move |_| on_navigate_to_index.call(View::Index),
                    "Index documents"
                }
            }
        }
    }
}
