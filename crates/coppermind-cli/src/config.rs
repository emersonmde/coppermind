//! Configuration and path resolution for the CLI.
//!
//! Handles finding model files and data directories across different environments:
//! - Development: workspace assets directory
//! - Distribution: relative to executable
//! - Custom: environment variables

use anyhow::{anyhow, Context, Result};
use directories::ProjectDirs;
use std::path::PathBuf;

/// Model file names
const MODEL_FILENAME: &str = "jina-bert.safetensors";
const TOKENIZER_FILENAME: &str = "jina-bert-tokenizer.json";

/// Database file name (shared with desktop app)
const DATABASE_FILENAME: &str = "documents.redb";

/// Environment variable for custom model directory
const MODEL_DIR_ENV: &str = "COPPERMIND_MODEL_DIR";

/// Finds the model directory containing JinaBERT weights and tokenizer.
///
/// Search order:
/// 1. `$COPPERMIND_MODEL_DIR` environment variable
/// 2. Workspace `assets/models/` directory (development)
/// 3. Executable-relative `../assets/models/` (bundled distribution)
pub fn find_model_dir() -> Result<PathBuf> {
    // 1. Environment variable
    if let Ok(dir) = std::env::var(MODEL_DIR_ENV) {
        let path = PathBuf::from(dir);
        if path.join(MODEL_FILENAME).exists() {
            return Ok(path);
        }
    }

    // 2. Workspace assets (development)
    // CARGO_MANIFEST_DIR points to crates/coppermind-cli
    let workspace_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("crates/coppermind/assets/models"));

    if let Some(ref path) = workspace_path {
        if path.join(MODEL_FILENAME).exists() {
            return Ok(path.clone());
        }
    }

    // 3. Relative to executable (distribution)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            // Try ../assets/models (bundled layout)
            let dist_path = exe_dir.join("../assets/models");
            if dist_path.join(MODEL_FILENAME).exists() {
                return Ok(dist_path);
            }

            // Also try assets/models directly next to executable
            let alt_path = exe_dir.join("assets/models");
            if alt_path.join(MODEL_FILENAME).exists() {
                return Ok(alt_path);
            }
        }
    }

    Err(anyhow!(
        "Model files not found. Please run ./download-models.sh from the workspace root.\n\
         Searched locations:\n\
         - ${} environment variable\n\
         - {}\n\
         - Relative to executable",
        MODEL_DIR_ENV,
        workspace_path
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<workspace>/crates/coppermind/assets/models".to_string())
    ))
}

/// Returns the path to the model weights file.
pub fn model_path() -> Result<PathBuf> {
    let dir = find_model_dir()?;
    let path = dir.join(MODEL_FILENAME);
    if !path.exists() {
        return Err(anyhow!("Model file not found: {}", path.display()));
    }
    Ok(path)
}

/// Returns the path to the tokenizer JSON file.
pub fn tokenizer_path() -> Result<PathBuf> {
    let dir = find_model_dir()?;
    let path = dir.join(TOKENIZER_FILENAME);
    if !path.exists() {
        return Err(anyhow!("Tokenizer file not found: {}", path.display()));
    }
    Ok(path)
}

/// Returns the data directory (shared with desktop app).
///
/// Uses the same bundle identifier as the desktop app for shared index access:
/// - macOS: `~/Library/Application Support/dev.errorsignal.Coppermind/`
/// - Linux: `~/.local/share/dev.errorsignal.Coppermind/`
/// - Windows: `%APPDATA%\errorsignal\Coppermind\data\`
pub fn get_data_dir(custom_dir: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(dir) = custom_dir {
        return Ok(dir.clone());
    }

    // Use same bundle identifier as desktop app: dev.errorsignal.Coppermind
    ProjectDirs::from("dev", "errorsignal", "Coppermind")
        .map(|dirs| dirs.data_dir().to_path_buf())
        .ok_or_else(|| anyhow!("Could not determine data directory"))
}

/// Returns the path to the database file.
pub fn database_path(custom_dir: Option<&PathBuf>) -> Result<PathBuf> {
    let data_dir = get_data_dir(custom_dir)?;
    Ok(data_dir.join(DATABASE_FILENAME))
}

/// Loads model bytes from disk.
pub fn load_model_bytes() -> Result<Vec<u8>> {
    let path = model_path()?;
    std::fs::read(&path).with_context(|| format!("Failed to read model file: {}", path.display()))
}

/// Loads tokenizer bytes from disk.
pub fn load_tokenizer_bytes() -> Result<Vec<u8>> {
    let path = tokenizer_path()?;
    std::fs::read(&path)
        .with_context(|| format!("Failed to read tokenizer file: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_model_dir() {
        // This test assumes download-models.sh has been run
        let result = find_model_dir();
        // Don't fail the test if models aren't downloaded
        if let Ok(dir) = result {
            assert!(dir.join(MODEL_FILENAME).exists());
            assert!(dir.join(TOKENIZER_FILENAME).exists());
        }
    }

    #[test]
    fn test_get_data_dir() {
        let dir = get_data_dir(None).unwrap();
        // Should contain the bundle identifier
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains("errorsignal") || dir_str.contains("Coppermind"),
            "Data dir should use bundle identifier: {}",
            dir_str
        );
    }

    #[test]
    fn test_custom_data_dir() {
        let custom = PathBuf::from("/tmp/custom-data");
        let dir = get_data_dir(Some(&custom)).unwrap();
        assert_eq!(dir, custom);
    }
}
