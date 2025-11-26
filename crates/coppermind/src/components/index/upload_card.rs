use dioxus::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use crate::components::file_processing::collect_files_from_dir;

#[cfg(not(target_arch = "wasm32"))]
use tokio::fs;

/// Upload card with two buttons: Select Files and Select Folder
/// Uses native rfd dialogs on desktop, hidden file inputs on web
#[component]
pub fn UploadCard(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    if cfg!(target_arch = "wasm32") {
        rsx! { WebUploadCard { on_files_selected } }
    } else {
        rsx! { DesktopUploadCard { on_files_selected } }
    }
}

// =============================================================================
// Desktop Implementation (using rfd native dialogs)
// =============================================================================

#[cfg(not(target_arch = "wasm32"))]
#[component]
fn DesktopUploadCard(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    let mut is_loading = use_signal(|| false);

    let handle_select_files = move |_| {
        spawn(async move {
            use dioxus::logger::tracing::info;
            use rfd::AsyncFileDialog;

            is_loading.set(true);

            let files = AsyncFileDialog::new()
                .set_title("Select files to index")
                .pick_files()
                .await;

            if let Some(files) = files {
                let mut file_contents = Vec::new();

                for file in files {
                    let path = file.path().to_path_buf();
                    let file_name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    match fs::read_to_string(&path).await {
                        Ok(contents) => {
                            file_contents.push((file_name, contents));
                        }
                        Err(e) => {
                            info!("Failed to read {}: {}", file_name, e);
                        }
                    }
                }

                if !file_contents.is_empty() {
                    on_files_selected.call(file_contents);
                }
            }

            is_loading.set(false);
        });
    };

    let handle_select_folder = move |_| {
        spawn(async move {
            use dioxus::logger::tracing::info;
            use rfd::AsyncFileDialog;

            is_loading.set(true);

            let folder = AsyncFileDialog::new()
                .set_title("Select folder to index")
                .pick_folder()
                .await;

            if let Some(folder) = folder {
                let folder_path = folder.path().to_path_buf();
                let folder_name = folder_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("folder")
                    .to_string();

                info!("📂 Indexing folder: {:?}", folder_path);

                let discovered_files =
                    collect_files_from_dir(folder_path.clone(), folder_name.clone()).await;

                info!(
                    "📂 Found {} files in {}",
                    discovered_files.len(),
                    folder_name
                );

                let mut file_contents = Vec::new();

                for (relative_path, file_path) in discovered_files {
                    match fs::read_to_string(&file_path).await {
                        Ok(contents) => {
                            file_contents.push((relative_path, contents));
                        }
                        Err(e) => {
                            info!("Failed to read {}: {}", relative_path, e);
                        }
                    }
                }

                if !file_contents.is_empty() {
                    on_files_selected.call(file_contents);
                }
            }

            is_loading.set(false);
        });
    };

    rsx! {
        section { class: "cm-upload-card",
            div { class: "cm-upload-body",
                div { class: "cm-upload-content",
                    div { class: "cm-dropzone-icon", "📂" }
                    div { class: "cm-dropzone-title", "Add files to your index" }
                    div { class: "cm-dropzone-subtitle",
                        "Select files or a folder to index. All text files will be processed."
                    }
                    div { class: "cm-upload-buttons",
                        button {
                            class: "cm-upload-button",
                            disabled: is_loading(),
                            onclick: handle_select_files,
                            if is_loading() { "Loading..." } else { "Select Files" }
                        }
                        button {
                            class: "cm-upload-button cm-upload-button--secondary",
                            disabled: is_loading(),
                            onclick: handle_select_folder,
                            if is_loading() { "Loading..." } else { "Select Folder" }
                        }
                    }
                }
            }
        }
    }
}

// Stub for WASM compilation (never actually called)
#[cfg(target_arch = "wasm32")]
#[component]
fn DesktopUploadCard(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    let _ = on_files_selected;
    rsx! {}
}

// =============================================================================
// Web Implementation (using hidden file inputs triggered by buttons)
// =============================================================================

#[cfg(target_arch = "wasm32")]
#[component]
fn WebUploadCard(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    use wasm_bindgen::JsCast;

    let mut is_loading = use_signal(|| false);

    // Set webkitdirectory on the folder input after mount
    use_effect(|| {
        spawn(async {
            gloo_timers::future::TimeoutFuture::new(50).await;

            if let Some(window) = web_sys::window() {
                if let Some(document) = window.document() {
                    if let Some(input) = document.get_element_by_id("cm-folder-input") {
                        if let Some(el) = input.dyn_ref::<web_sys::HtmlInputElement>() {
                            let _ = el.set_attribute("webkitdirectory", "");
                            let _ = el.set_attribute("directory", "");
                        }
                    }
                }
            }
        });
    });

    // Click the hidden file input
    let trigger_file_input = move |_| {
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                if let Some(input) = document.get_element_by_id("cm-file-input") {
                    if let Some(el) = input.dyn_ref::<web_sys::HtmlInputElement>() {
                        el.click();
                    }
                }
            }
        }
    };

    // Click the hidden folder input
    let trigger_folder_input = move |_| {
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                if let Some(input) = document.get_element_by_id("cm-folder-input") {
                    if let Some(el) = input.dyn_ref::<web_sys::HtmlInputElement>() {
                        el.click();
                    }
                }
            }
        }
    };

    // Handle file selection (shared logic)
    let handle_files = move |evt: FormEvent| {
        spawn(async move {
            let files = evt.files();
            if files.is_empty() {
                return;
            }

            is_loading.set(true);
            let mut file_contents = Vec::new();

            for file in files {
                let file_name = file.name().to_string();
                match file.read_string().await {
                    Ok(contents) => {
                        file_contents.push((file_name, contents));
                    }
                    Err(e) => {
                        dioxus::logger::tracing::error!("Failed to read {}: {}", file_name, e);
                    }
                }
            }

            if !file_contents.is_empty() {
                on_files_selected.call(file_contents);
            }

            is_loading.set(false);
        });
    };

    rsx! {
        section { class: "cm-upload-card",
            // Hidden file inputs
            input {
                id: "cm-file-input",
                r#type: "file",
                class: "cm-hidden-input",
                multiple: true,
                accept: "*",
                onchange: handle_files,
            }
            input {
                id: "cm-folder-input",
                r#type: "file",
                class: "cm-hidden-input",
                multiple: true,
                accept: "*",
                onchange: handle_files,
            }

            div { class: "cm-upload-body",
                div { class: "cm-upload-content",
                    div { class: "cm-dropzone-icon", "📂" }
                    div { class: "cm-dropzone-title", "Add files to your index" }
                    div { class: "cm-dropzone-subtitle",
                        "Select files or a folder to index. All text files will be processed."
                    }
                    div { class: "cm-upload-buttons",
                        button {
                            class: "cm-upload-button",
                            disabled: is_loading(),
                            onclick: trigger_file_input,
                            if is_loading() { "Loading..." } else { "Select Files" }
                        }
                        button {
                            class: "cm-upload-button cm-upload-button--secondary",
                            disabled: is_loading(),
                            onclick: trigger_folder_input,
                            if is_loading() { "Loading..." } else { "Select Folder" }
                        }
                    }
                }
            }
        }
    }
}

// Stub for non-WASM compilation
#[cfg(not(target_arch = "wasm32"))]
#[component]
fn WebUploadCard(on_files_selected: EventHandler<Vec<(String, String)>>) -> Element {
    let _ = on_files_selected;
    rsx! {}
}
