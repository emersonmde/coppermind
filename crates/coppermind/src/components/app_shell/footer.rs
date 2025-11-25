use dioxus::prelude::*;

/// Footer with local-first messaging
#[component]
pub fn Footer() -> Element {
    rsx! {
        footer { class: "cm-footer",
            span { class: "cm-footer-text",
                "Local-first • All processing happens on your device."
            }
        }
    }
}
