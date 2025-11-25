use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use web_sys;

#[cfg(not(target_arch = "wasm32"))]
use crate::components::file_processing::collect_files_from_dir;

#[cfg(not(target_arch = "wasm32"))]
use tokio::fs;

/// Upload card with tabs and dropzone for file selection
#[component]
pub fn UploadCard(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    let mut mode = use_signal(|| "files"); // "files" or "folder"

    // Set webkitdirectory attribute on folder input when in folder mode (web only)
    #[cfg(target_arch = "wasm32")]
    use_effect(move || {
        use dioxus::logger::tracing::info;
        use wasm_bindgen::JsCast;

        // Only set attribute when in folder mode (element exists)
        if mode() != "folder" {
            return;
        }

        // Small delay to ensure DOM is ready
        spawn(async move {
            gloo_timers::future::TimeoutFuture::new(50).await;

            if let Some(window) = web_sys::window() {
                if let Some(document) = window.document() {
                    if let Some(input) = document.get_element_by_id("cm-folder-input") {
                        if let Some(input_element) = input.dyn_ref::<web_sys::HtmlInputElement>() {
                            match input_element.set_attribute("webkitdirectory", "") {
                                Ok(_) => info!("✓ Set webkitdirectory attribute"),
                                Err(e) => info!("❌ Failed to set webkitdirectory: {:?}", e),
                            }
                            let _ = input_element.set_attribute("directory", "");
                        } else {
                            info!("❌ Element is not an input");
                        }
                    } else {
                        info!("❌ Could not find element with id 'cm-folder-input'");
                    }
                } else {
                    info!("❌ Could not get document");
                }
            }
        });
    });

    rsx! {
        section { class: "cm-upload-card",
            div { class: "cm-upload-header",
                div { class: "cm-upload-tabs",
                    button {
                        class: if mode() == "files" { "cm-upload-tab cm-upload-tab--active" } else { "cm-upload-tab" },
                        onclick: move |_| mode.set("files"),
                        "Files"
                    }
                    button {
                        class: if mode() == "folder" { "cm-upload-tab cm-upload-tab--active" } else { "cm-upload-tab" },
                        onclick: move |_| mode.set("folder"),
                        "Folder"
                    }
                }
            }

            div { class: "cm-upload-body",
                if mode() == "files" {
                    FileDropzone {
                        on_files_selected
                    }
                } else {
                    FolderDropzone {
                        on_files_selected
                    }
                }
            }
        }
    }
}

/// Dropzone for selecting individual files
#[component]
pub fn FileDropzone(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    rsx! {
        label { class: "cm-dropzone",
            input {
                r#type: "file",
                class: "cm-dropzone-input",
                multiple: true,
                accept: "*",
                onchange: move |evt: dioxus::events::FormEvent| {
                    spawn(async move {
                        let files = evt.files();
                        if files.is_empty() {
                            return;
                        }

                        let mut file_contents = Vec::new();

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
                div { class: "cm-dropzone-icon", "📄" }
                div { class: "cm-dropzone-title", "Select individual files to index" }
                div { class: "cm-dropzone-subtitle",
                    "Choose one or more text files from your device."
                }
            }
        }
    }
}

/// Dropzone for selecting a folder
#[component]
pub fn FolderDropzone(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    rsx! {
        label { class: "cm-dropzone",
            input {
                id: "cm-folder-input",
                r#type: "file",
                class: "cm-dropzone-input",
                multiple: true,
                accept: "*",
                // Web: webkitdirectory set via JavaScript (see use_effect in UploadCard)
                // Desktop: No directory attribute (doesn't work), use parent directory detection instead
                onchange: move |evt: dioxus::events::FormEvent| {
                    spawn(async move {
                        let files = evt.files();
                        if files.is_empty() {
                            return;
                        }

                        let mut file_contents = Vec::new();

                        #[cfg(target_arch = "wasm32")]
                        {
                            // Web: webkitdirectory flattens all files from the folder
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
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            use dioxus::logger::tracing::info;

                            // Desktop: Get parent directory from first selected file and index entire directory
                            if let Some(first_file) = files.first() {
                                let file_path = first_file.path();

                                if let Some(parent_dir) = file_path.parent() {
                                    let dir_name = parent_dir
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("folder")
                                        .to_string();

                                    info!("📂 Indexing entire directory: {:?}", parent_dir);

                                    // Recursively collect all files from parent directory
                                    let discovered_files = collect_files_from_dir(
                                        parent_dir.to_path_buf(),
                                        dir_name.clone()
                                    ).await;

                                    info!("📂 Found {} files in {}", discovered_files.len(), dir_name);

                                    // Read contents of all discovered files
                                    for (relative_path, file_path) in discovered_files {
                                        match fs::read_to_string(&file_path).await {
                                            Ok(contents) => {
                                                file_contents.push((relative_path, contents));
                                            }
                                            Err(e) => {
                                                dioxus::logger::tracing::error!("❌ Failed to read {}: {}", relative_path, e);
                                            }
                                        }
                                    }
                                } else {
                                    // Fallback: just read the selected file
                                    let file_name = first_file.name().to_string();
                                    match first_file.read_string().await {
                                        Ok(contents) => {
                                            file_contents.push((file_name, contents));
                                        }
                                        Err(e) => {
                                            dioxus::logger::tracing::error!("❌ Failed to read {}: {}", file_name, e);
                                        }
                                    }
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
                div { class: "cm-dropzone-title",
                    if cfg!(target_arch = "wasm32") {
                        "Select a folder to index"
                    } else {
                        "Index a folder"
                    }
                }
                div { class: "cm-dropzone-subtitle",
                    if cfg!(target_arch = "wasm32") {
                        "All text files in the folder will be recursively indexed."
                    } else {
                        "Select any file from the folder you want to index. The entire parent directory will be indexed recursively."
                    }
                }
            }
        }
    }
}
