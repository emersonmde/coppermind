use dioxus::prelude::*;

/// Search mode options
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    #[default]
    Hybrid,
    Semantic,
    Keyword,
}

impl SearchMode {
    pub fn label(&self) -> &'static str {
        match self {
            SearchMode::Hybrid => "Hybrid",
            SearchMode::Semantic => "Semantic",
            SearchMode::Keyword => "Keyword",
        }
    }
}

/// Available result count options (chunks to retrieve)
const RESULT_COUNTS: [usize; 4] = [10, 20, 50, 100];

/// Search card with input, search button, and inline controls.
///
/// Search is always enabled - users can submit queries even while the model/index
/// is loading. The search will wait for initialization to complete, showing
/// skeleton loading cards in the meantime.
#[component]
pub fn SearchCard(
    search_query: Signal<String>,
    on_search: EventHandler<String>,
    searching: ReadSignal<bool>,
    result_count: Signal<usize>,
    search_mode: Signal<SearchMode>,
) -> Element {
    let handle_keypress = move |evt: KeyboardEvent| {
        if evt.key() == Key::Enter && !searching() {
            let query = search_query.read().clone();
            if !query.trim().is_empty() {
                on_search.call(query);
            }
        }
    };

    rsx! {
        section { class: "cm-search-card",
            // Main search input row
            div { class: "cm-search-input-row",
                input {
                    class: "cm-search-input",
                    r#type: "text",
                    placeholder: "Search your knowledge base…",
                    value: "{search_query}",
                    disabled: searching(),
                    oninput: move |evt| search_query.set(evt.value()),
                    onkeypress: handle_keypress,
                }
                button {
                    class: "cm-btn cm-btn--primary",
                    disabled: searching(),
                    onclick: move |_| {
                        let query = search_query.read().clone();
                        if !query.trim().is_empty() {
                            on_search.call(query);
                        }
                    },
                    "Search"
                }
            }

            // Inline controls row - subtle hints per UX spec
            div { class: "cm-search-controls",
                // Max results selector (controls chunk retrieval depth)
                div { class: "cm-search-control",
                    label { class: "cm-search-control-label", "Max" }
                    select {
                        class: "cm-search-select",
                        value: "{result_count}",
                        disabled: searching(),
                        onchange: move |evt| {
                            if let Ok(count) = evt.value().parse::<usize>() {
                                result_count.set(count);
                            }
                        },
                        for count in RESULT_COUNTS {
                            option { value: "{count}", "{count}" }
                        }
                    }
                }

                // Search mode selector
                div { class: "cm-search-control",
                    label { class: "cm-search-control-label", "Mode" }
                    select {
                        class: "cm-search-select",
                        value: "{search_mode.read().label()}",
                        disabled: searching(),
                        onchange: move |evt| {
                            let mode = match evt.value().as_str() {
                                "Semantic" => SearchMode::Semantic,
                                "Keyword" => SearchMode::Keyword,
                                _ => SearchMode::Hybrid,
                            };
                            search_mode.set(mode);
                        },
                        option { value: "Hybrid", "Hybrid" }
                        option { value: "Semantic", "Semantic" }
                        option { value: "Keyword", "Keyword" }
                    }
                }
            }
        }
    }
}
