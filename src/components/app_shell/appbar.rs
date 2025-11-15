use dioxus::prelude::*;

/// View selection enum for navigation
#[derive(Clone, Copy, PartialEq)]
pub enum View {
    Search,
    Index,
}

/// Global app bar with logo, navigation, and metrics toggle
#[component]
pub fn AppBar(
    current_view: ReadSignal<View>,
    on_view_change: EventHandler<View>,
    on_metrics_toggle: EventHandler<()>,
) -> Element {
    rsx! {
        header { class: "cm-appbar",
            div { class: "cm-appbar-left",
                div { class: "cm-logo",
                    span { class: "cm-logo-word", "Copper" }
                    span { class: "cm-logo-word cm-logo-word--accent", "mind" }
                }
            }
            nav { class: "cm-appbar-center",
                button {
                    class: if current_view() == View::Search {
                        "cm-nav-link cm-nav-link--active"
                    } else {
                        "cm-nav-link"
                    },
                    onclick: move |_| on_view_change.call(View::Search),
                    "Search"
                }
                button {
                    class: if current_view() == View::Index {
                        "cm-nav-link cm-nav-link--active"
                    } else {
                        "cm-nav-link"
                    },
                    onclick: move |_| on_view_change.call(View::Index),
                    "Index"
                }
            }
            div { class: "cm-appbar-right",
                button {
                    class: "cm-icon-button",
                    onclick: move |_| on_metrics_toggle.call(()),
                    "aria-label": "Toggle metrics panel",
                    span { class: "cm-icon cm-icon-metrics", "</>"}
                }
            }
        }
    }
}
