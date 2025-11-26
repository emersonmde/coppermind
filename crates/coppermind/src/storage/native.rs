// Native filesystem storage implementation for desktop
//
// Uses platform-idiomatic directories:
// - macOS: ~/Library/Application Support/dev.errorsignal.Coppermind/
// - Linux: ~/.local/share/coppermind/
// - Windows: C:\Users\<user>\AppData\Roaming\errorsignal\Coppermind\

use super::{StorageBackend, StorageError};
use directories::ProjectDirs;
use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
};

/// Native filesystem storage backend for desktop platforms.
///
/// Data is stored in the platform-idiomatic application data directory.
/// The directory structure supports nested paths (e.g., "index/documents.json").
pub struct NativeStorage {
    base_path: PathBuf,
}

impl NativeStorage {
    /// Creates a new NativeStorage using the platform-idiomatic data directory.
    ///
    /// On macOS: ~/Library/Application Support/dev.errorsignal.Coppermind/
    /// On Linux: ~/.local/share/coppermind/
    /// On Windows: C:\Users\<user>\AppData\Roaming\errorsignal\Coppermind\
    pub fn new() -> Result<Self, StorageError> {
        let proj_dirs = ProjectDirs::from("dev", "errorsignal", "Coppermind").ok_or_else(|| {
            StorageError::IoError("Failed to determine application data directory".to_string())
        })?;

        let base_path = proj_dirs.data_dir().to_path_buf();

        // Create base directory if it doesn't exist
        std::fs::create_dir_all(&base_path)
            .map_err(|e| StorageError::IoError(format!("Failed to create directory: {}", e)))?;

        Ok(Self { base_path })
    }

    /// Creates a NativeStorage with a custom base path.
    /// Useful for testing or when a specific location is needed.
    #[allow(dead_code)]
    pub fn with_path(base_path: PathBuf) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&base_path)
            .map_err(|e| StorageError::IoError(format!("Failed to create directory: {}", e)))?;

        Ok(Self { base_path })
    }

    /// Returns the base path where data is stored.
    pub fn base_path(&self) -> &PathBuf {
        &self.base_path
    }

    /// Resolves a key to an absolute path, creating parent directories as needed.
    fn get_path(&self, key: &str) -> PathBuf {
        self.base_path.join(key)
    }

    /// Ensures parent directories exist for a given path.
    async fn ensure_parent_dirs(&self, path: &Path) -> Result<(), StorageError> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| StorageError::IoError(format!("Failed to create directory: {}", e)))?;
        }
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl StorageBackend for NativeStorage {
    async fn save(&self, key: &str, data: &[u8]) -> Result<(), StorageError> {
        let path = self.get_path(key);

        // Ensure parent directories exist (for nested keys like "index/documents.json")
        self.ensure_parent_dirs(&path).await?;

        tokio::fs::write(&path, data)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

    async fn load(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        let path = self.get_path(key);
        tokio::fs::read(&path).await.map_err(|e| {
            if e.kind() == ErrorKind::NotFound {
                StorageError::NotFound(key.to_string())
            } else {
                StorageError::IoError(format!("Failed to read file: {}", e))
            }
        })
    }

    async fn exists(&self, key: &str) -> Result<bool, StorageError> {
        let path = self.get_path(key);
        Ok(path.exists())
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let path = self.get_path(key);
        if path.exists() {
            if path.is_dir() {
                tokio::fs::remove_dir_all(&path).await.map_err(|e| {
                    StorageError::IoError(format!("Failed to delete directory: {}", e))
                })?;
            } else {
                tokio::fs::remove_file(&path)
                    .await
                    .map_err(|e| StorageError::IoError(format!("Failed to delete file: {}", e)))?;
            }
        }
        Ok(())
    }

    async fn list_keys(&self) -> Result<Vec<String>, StorageError> {
        let mut keys = Vec::new();
        self.collect_keys_recursive(&self.base_path, &self.base_path, &mut keys)
            .await?;
        Ok(keys)
    }

    async fn clear(&self) -> Result<(), StorageError> {
        let mut entries = tokio::fs::read_dir(&self.base_path)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to read directory: {}", e)))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to read entry: {}", e)))?
        {
            let path = entry.path();
            if path.is_dir() {
                tokio::fs::remove_dir_all(&path).await.map_err(|e| {
                    StorageError::IoError(format!("Failed to delete directory: {}", e))
                })?;
            } else {
                tokio::fs::remove_file(&path)
                    .await
                    .map_err(|e| StorageError::IoError(format!("Failed to delete file: {}", e)))?;
            }
        }

        Ok(())
    }
}

impl NativeStorage {
    /// Recursively collects all file keys relative to the base path.
    async fn collect_keys_recursive(
        &self,
        current_dir: &PathBuf,
        base: &PathBuf,
        keys: &mut Vec<String>,
    ) -> Result<(), StorageError> {
        let mut entries = tokio::fs::read_dir(current_dir)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to read directory: {}", e)))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to read entry: {}", e)))?
        {
            let path = entry.path();
            if path.is_dir() {
                // Recursively collect keys from subdirectories
                Box::pin(self.collect_keys_recursive(&path, base, keys)).await?;
            } else if let Ok(relative) = path.strip_prefix(base) {
                if let Some(key) = relative.to_str() {
                    keys.push(key.to_string());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_storage() -> (NativeStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = NativeStorage::with_path(temp_dir.path().to_path_buf()).unwrap();
        (storage, temp_dir)
    }

    #[tokio::test]
    async fn test_save_and_load() {
        let (storage, _temp) = create_test_storage().await;

        storage.save("test.txt", b"hello world").await.unwrap();
        let data = storage.load("test.txt").await.unwrap();

        assert_eq!(data, b"hello world");
    }

    #[tokio::test]
    async fn test_nested_keys() {
        let (storage, _temp) = create_test_storage().await;

        // Save with nested path
        storage.save("index/documents.json", b"{}").await.unwrap();
        storage
            .save("index/embeddings.bin", b"\x00\x00")
            .await
            .unwrap();

        // Load should work
        let data = storage.load("index/documents.json").await.unwrap();
        assert_eq!(data, b"{}");

        // List should return nested keys
        let keys = storage.list_keys().await.unwrap();
        assert!(keys.contains(&"index/documents.json".to_string()));
        assert!(keys.contains(&"index/embeddings.bin".to_string()));
    }

    #[tokio::test]
    async fn test_exists() {
        let (storage, _temp) = create_test_storage().await;

        assert!(!storage.exists("missing.txt").await.unwrap());

        storage.save("exists.txt", b"data").await.unwrap();
        assert!(storage.exists("exists.txt").await.unwrap());
    }

    #[tokio::test]
    async fn test_delete() {
        let (storage, _temp) = create_test_storage().await;

        storage.save("to_delete.txt", b"data").await.unwrap();
        assert!(storage.exists("to_delete.txt").await.unwrap());

        storage.delete("to_delete.txt").await.unwrap();
        assert!(!storage.exists("to_delete.txt").await.unwrap());
    }

    #[tokio::test]
    async fn test_clear() {
        let (storage, _temp) = create_test_storage().await;

        storage.save("file1.txt", b"data1").await.unwrap();
        storage.save("file2.txt", b"data2").await.unwrap();
        storage.save("subdir/file3.txt", b"data3").await.unwrap();

        let keys = storage.list_keys().await.unwrap();
        assert_eq!(keys.len(), 3);

        storage.clear().await.unwrap();

        let keys = storage.list_keys().await.unwrap();
        assert!(keys.is_empty());
    }

    #[tokio::test]
    async fn test_load_not_found() {
        let (storage, _temp) = create_test_storage().await;

        let result = storage.load("nonexistent.txt").await;
        assert!(matches!(result, Err(StorageError::NotFound(_))));
    }
}
