//! Asset loading for embedding models.
//!
//! This module handles fetching model weights and tokenizer files from either
//! the network (web platform) or filesystem (desktop platform).

use crate::error::EmbeddingError;
use dioxus::logger::tracing::debug;

#[cfg(target_arch = "wasm32")]
use dioxus::logger::tracing::error;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Fetches asset bytes from the network (web) or filesystem (desktop).
///
/// # Platform-Specific Behavior
///
/// - **Web**: Uses `fetch()` API to download from server. Resolves base path
///   from `DIOXUS_ASSET_ROOT` meta tag or worker global `__COPPERMIND_ASSET_BASE`.
/// - **Desktop**: Reads from filesystem, checking multiple locations:
///   1. `../Resources/assets/` (macOS app bundle)
///   2. `./assets` (same directory as executable)
///   3. `assets` (current working directory)
///
/// # Arguments
///
/// * `url` - Asset path (e.g., "/assets/models/model.safetensors")
///
/// # Returns
///
/// Byte vector containing the asset data, or an error if fetching failed.
///
/// # Examples
///
/// ```ignore
/// let model_bytes = fetch_asset_bytes("/assets/models/model.safetensors").await?;
/// ```
pub async fn fetch_asset_bytes(url: &str) -> Result<Vec<u8>, EmbeddingError> {
    #[cfg(target_arch = "wasm32")]
    {
        fetch_asset_bytes_web(url).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        fetch_asset_bytes_desktop(url).await
    }
}

/// Web implementation: Fetch assets via HTTP
#[cfg(target_arch = "wasm32")]
async fn fetch_asset_bytes_web(url: &str) -> Result<Vec<u8>, EmbeddingError> {
    use js_sys::{Function, Promise, Reflect};
    use wasm_bindgen::JsValue;
    use wasm_bindgen_futures::JsFuture;

    let global = js_sys::global();
    let fetch_fn = Reflect::get(&global, &JsValue::from_str("fetch"))
        .map_err(|_| EmbeddingError::AssetFetch("fetch API unavailable".to_string()))?
        .dyn_into::<Function>()
        .map_err(|_| EmbeddingError::AssetFetch("fetch is not callable".to_string()))?;

    // Resolve the URL - in dev mode, the URL may already have the base path
    // from Dioxus asset! macro, so we shouldn't double-prepend
    let resolved_url = if url.starts_with("/coppermind") {
        // Already has base path (from Dioxus asset macro in dev mode)
        url.to_string()
    } else if cfg!(debug_assertions) {
        // Dev mode without base path - prepend it
        format!("/coppermind{}", url)
    } else {
        // Production - use proper resolution
        resolve_asset_url(url)
    };

    debug!(
        "📥 Fetching asset (raw: {}, resolved: {})...",
        url, resolved_url
    );

    let promise = fetch_fn
        .call1(&global, &JsValue::from_str(&resolved_url))
        .map_err(|e| EmbeddingError::AssetFetch(format!("Fetch call failed: {:?}", e)))?;

    let resp_value = JsFuture::from(Promise::from(promise)).await.map_err(|e| {
        EmbeddingError::AssetFetch(format!(
            "Fetch failed: {:?} (resolved: {})",
            e, resolved_url
        ))
    })?;

    debug!("✓ Fetch completed");

    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| EmbeddingError::AssetFetch("Failed to cast to Response".to_string()))?;

    if !resp.ok() {
        return Err(EmbeddingError::AssetFetch(format!(
            "HTTP {} fetching {} (raw path: {})",
            resp.status(),
            resolved_url,
            url
        )));
    }

    debug!("✓ Response OK, reading array buffer...");

    let array_buffer =
        JsFuture::from(resp.array_buffer().map_err(|e| {
            EmbeddingError::AssetFetch(format!("Failed to get array buffer: {:?}", e))
        })?)
        .await
        .map_err(|e| {
            EmbeddingError::AssetFetch(format!("Failed to await array buffer: {:?}", e))
        })?;

    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let bytes = uint8_array.to_vec();

    debug!(
        "✓ Asset fetched successfully ({} bytes, {:.2}MB)",
        bytes.len(),
        bytes.len() as f64 / 1_000_000.0
    );

    Ok(bytes)
}

/// Resolves asset URLs for web platform, handling base paths and worker contexts.
#[cfg(target_arch = "wasm32")]
fn resolve_asset_url(input: &str) -> String {
    let trimmed = input.trim();

    // Already absolute URL
    if trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || trimmed.starts_with("blob:")
    {
        return trimmed.to_string();
    }

    // Try worker-specific base path
    if let Some(resolved) = resolve_with_worker_base(trimmed) {
        return resolved;
    }

    // Try main thread base path from meta tag
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            match document.query_selector("meta[name=\"DIOXUS_ASSET_ROOT\"]") {
                Ok(Some(meta)) => {
                    if let Some(content) = meta.get_attribute("content") {
                        if let Some(resolved) = join_base_path(&content, trimmed) {
                            return resolved;
                        }
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    error!("Failed to read DIOXUS_ASSET_ROOT meta tag: {:?}", err);
                }
            }
        }
    }

    trimmed.to_string()
}

/// Joins base path with relative path, handling various edge cases.
#[cfg(target_arch = "wasm32")]
fn join_base_path(base: &str, path: &str) -> Option<String> {
    // Don't modify absolute URLs
    if path.starts_with("http://") || path.starts_with("https://") || path.starts_with("blob:") {
        return None;
    }

    let normalized_base = if base == "/" {
        String::new()
    } else {
        base.trim_end_matches('/').to_string()
    };

    if normalized_base.is_empty() {
        return Some(format!("/{}", path.trim_start_matches('/')));
    }

    // Check if path already starts with the base path
    if path.starts_with(&normalized_base) {
        return Some(path.to_string());
    }

    if path.starts_with('/') {
        Some(format!("{}{}", normalized_base, path))
    } else {
        Some(format!("{}/{}", normalized_base, path))
    }
}

/// Resolves asset URL using worker global `__COPPERMIND_ASSET_BASE`.
#[cfg(target_arch = "wasm32")]
fn resolve_with_worker_base(path: &str) -> Option<String> {
    use js_sys::Reflect;
    use wasm_bindgen::JsValue;

    let global = js_sys::global();
    let base_value = Reflect::get(&global, &JsValue::from_str("__COPPERMIND_ASSET_BASE")).ok()?;
    let base = base_value.as_string()?;
    join_base_path(&base, path)
}

/// Desktop implementation: Read assets from filesystem
#[cfg(not(target_arch = "wasm32"))]
async fn fetch_asset_bytes_desktop(asset_path: &str) -> Result<Vec<u8>, EmbeddingError> {
    use std::path::PathBuf;

    debug!("📥 Reading asset from {}...", asset_path);

    // Get the current executable directory
    let exe_path = std::env::current_exe()
        .map_err(|e| EmbeddingError::AssetFetch(format!("Failed to get exe path: {}", e)))?;
    let exe_dir = exe_path.parent().ok_or_else(|| {
        EmbeddingError::AssetFetch("Failed to get exe parent directory".to_string())
    })?;

    debug!("📂 Executable directory: {:?}", exe_dir);
    debug!("📂 Current directory: {:?}", std::env::current_dir());

    // Search locations for bundled assets
    let asset_locations = vec![
        // macOS app bundle: ../Resources/assets/
        exe_dir.join("..").join("Resources").join("assets"),
        // Same directory as executable
        exe_dir.join("assets"),
        // Current working directory
        PathBuf::from("assets"),
    ];

    // Extract filename from asset path (may have hash like jina-bert-dxhXXX.safetensors)
    let filename = asset_path
        .trim_start_matches('/')
        .trim_start_matches("assets/");

    for base_dir in &asset_locations {
        let full_path = base_dir.join(filename);
        debug!("  Trying: {:?}", full_path);

        // Use spawn_blocking for file I/O to prevent UI freezing with large files (62MB model)
        let path_clone = full_path.clone();
        let read_result = tokio::task::spawn_blocking(move || std::fs::read(&path_clone))
            .await
            .map_err(|e| EmbeddingError::AssetFetch(format!("Task join failed: {}", e)))?;

        if let Ok(bytes) = read_result {
            debug!(
                "✓ Asset loaded from {:?}: {:.2}MB ({} bytes)",
                full_path,
                bytes.len() as f64 / 1_000_000.0,
                bytes.len()
            );
            return Ok(bytes);
        }
    }

    Err(EmbeddingError::AssetFetch(format!(
        "Failed to find asset {} in any of the expected locations",
        asset_path
    )))
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "wasm32")]
    use super::*;

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_join_base_path() {
        // Root base path
        assert_eq!(
            join_base_path("/", "assets/model.bin"),
            Some("/assets/model.bin".to_string())
        );

        // Non-root base path
        assert_eq!(
            join_base_path("/coppermind", "/assets/model.bin"),
            Some("/coppermind/assets/model.bin".to_string())
        );

        // Path already contains base
        assert_eq!(
            join_base_path("/coppermind", "/coppermind/assets/model.bin"),
            Some("/coppermind/assets/model.bin".to_string())
        );

        // Absolute URLs shouldn't be modified
        assert_eq!(
            join_base_path("/coppermind", "https://example.com/model.bin"),
            None
        );
    }
}
