use dioxus::prelude::*;

/// Upload card with tabs and dropzone for file selection
#[component]
pub fn UploadCard(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    rsx! {
        section { class: "cm-upload-card",
            div { class: "cm-upload-header",
                div { class: "cm-upload-tabs",
                    button { class: "cm-upload-tab cm-upload-tab--active", "Files" }
                    button { class: "cm-upload-tab cm-upload-tab--disabled", "Web URLs (coming soon)" }
                }
            }

            div { class: "cm-upload-body",
                Dropzone {
                    on_files_selected
                }
            }
        }
    }
}

/// Dropzone component for file/folder selection
#[component]
pub fn Dropzone(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    rsx! {
        label { class: "cm-dropzone",
            input {
                r#type: "file",
                class: "cm-dropzone-input",
                multiple: true,
                directory: true,
                onchange: move |evt: dioxus::events::FormEvent| {
                    spawn(async move {
                        let files = evt.files();
                        if files.is_empty() {
                            return;
                        }

                        let mut file_contents = Vec::new();

                        // Read all files
                        for file in files {
                            let file_name = file.name().to_string();
                            match file.read_string().await {
                                Ok(contents) => {
                                    file_contents.push((file_name, contents));
                                }
                                Err(e) => {
                                    dioxus::logger::tracing::error!("❌ Failed to read {}: {}", file_name, e);
                                }
                            }
                        }

                        if !file_contents.is_empty() {
                            on_files_selected.call(file_contents);
                        }
                    });
                }
            }
            div { class: "cm-dropzone-inner",
                div { class: "cm-dropzone-icon", "📁" }
                div { class: "cm-dropzone-title", "Drop in files or folders to index them locally" }
                div { class: "cm-dropzone-subtitle",
                    "Coppermind will chunk, embed, and index them on your machine."
                }
            }
        }
    }
}
