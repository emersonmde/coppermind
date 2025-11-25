// Native filesystem storage implementation for desktop

use super::{StorageBackend, StorageError};
use std::{io::ErrorKind, path::PathBuf};

#[allow(dead_code)] // Public API for future use
pub struct NativeStorage {
    base_path: PathBuf,
}

impl NativeStorage {
    #[allow(dead_code)] // Public API
    pub fn new(base_path: PathBuf) -> Result<Self, StorageError> {
        // Create base directory if it doesn't exist
        std::fs::create_dir_all(&base_path)
            .map_err(|e| StorageError::IoError(format!("Failed to create directory: {}", e)))?;

        Ok(Self { base_path })
    }

    #[allow(dead_code)] // Helper method for storage implementation
    fn get_path(&self, key: &str) -> PathBuf {
        self.base_path.join(key)
    }
}

#[async_trait::async_trait(?Send)]
impl StorageBackend for NativeStorage {
    async fn save(&self, key: &str, data: &[u8]) -> Result<(), StorageError> {
        let path = self.get_path(key);
        tokio::fs::write(path, data)
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
            tokio::fs::remove_file(path)
                .await
                .map_err(|e| StorageError::IoError(format!("Failed to delete file: {}", e)))?;
        }
        Ok(())
    }

    async fn list_keys(&self) -> Result<Vec<String>, StorageError> {
        let mut keys = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.base_path)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to read directory: {}", e)))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to read entry: {}", e)))?
        {
            if let Some(name) = entry.file_name().to_str() {
                keys.push(name.to_string());
            }
        }

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
            tokio::fs::remove_file(entry.path())
                .await
                .map_err(|e| StorageError::IoError(format!("Failed to delete file: {}", e)))?;
        }

        Ok(())
    }
}
