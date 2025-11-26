// OPFS (Origin Private File System) storage implementation for web
//
// Stores index data in the browser's Origin Private File System,
// which persists across page refreshes and browser restarts.

use super::{StorageBackend, StorageError};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{FileSystemDirectoryHandle, FileSystemGetDirectoryOptions, FileSystemGetFileOptions};

/// OPFS storage backend for web platform.
///
/// Data is stored in the browser's Origin Private File System.
/// Supports nested paths (e.g., "index/documents.json") by creating subdirectories.
pub struct OpfsStorage {
    root: FileSystemDirectoryHandle,
}

impl OpfsStorage {
    pub async fn new() -> Result<Self, StorageError> {
        // Get navigator.storage.getDirectory()
        let window = web_sys::window().ok_or(StorageError::BrowserApiUnavailable)?;
        let navigator = window.navigator();
        let storage = navigator.storage();

        // Get OPFS root directory
        let root_promise = storage.get_directory();

        let root_value = JsFuture::from(root_promise)
            .await
            .map_err(|_| StorageError::OpfsUnavailable)?;

        let root: FileSystemDirectoryHandle = root_value
            .dyn_into()
            .map_err(|_| StorageError::OpfsUnavailable)?;

        Ok(Self { root })
    }

    /// Get or create a directory handle for the given path.
    /// Supports nested paths like "index" or "index/subdir".
    async fn get_directory(
        &self,
        path: &str,
        create: bool,
    ) -> Result<FileSystemDirectoryHandle, StorageError> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        let mut current_dir = self.root.clone();

        for part in parts {
            let options = FileSystemGetDirectoryOptions::new();
            options.set_create(create);

            let dir_promise = current_dir.get_directory_handle_with_options(part, &options);

            let dir_handle = JsFuture::from(dir_promise).await.map_err(|e| {
                if create {
                    StorageError::IoError(format!("Failed to create directory '{}': {:?}", part, e))
                } else {
                    StorageError::NotFound(path.to_string())
                }
            })?;

            current_dir = dir_handle.dyn_into().map_err(|_| {
                StorageError::IoError("Failed to cast directory handle".to_string())
            })?;
        }

        Ok(current_dir)
    }

    /// Parse a key into (directory_path, filename).
    /// E.g., "index/documents.json" -> (Some("index"), "documents.json")
    fn parse_key(key: &str) -> (Option<&str>, &str) {
        if let Some(pos) = key.rfind('/') {
            (Some(&key[..pos]), &key[pos + 1..])
        } else {
            (None, key)
        }
    }

    async fn write_file(&self, key: &str, data: &[u8]) -> Result<(), StorageError> {
        let (dir_path, filename) = Self::parse_key(key);

        // Get the directory (create if needed)
        let dir = if let Some(path) = dir_path {
            self.get_directory(path, true).await?
        } else {
            self.root.clone()
        };

        // Get or create file handle
        let options = FileSystemGetFileOptions::new();
        options.set_create(true);

        let file_handle_promise = dir.get_file_handle_with_options(filename, &options);

        let file_handle = JsFuture::from(file_handle_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to get file handle: {:?}", e)))?;

        let file_handle: web_sys::FileSystemFileHandle = file_handle
            .dyn_into()
            .map_err(|_| StorageError::IoError("Failed to cast file handle".to_string()))?;

        // Create writable stream
        let writable_promise = file_handle.create_writable();

        let writable = JsFuture::from(writable_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to create writable: {:?}", e)))?;

        let writable: web_sys::FileSystemWritableFileStream = writable
            .dyn_into()
            .map_err(|_| StorageError::IoError("Failed to cast writable".to_string()))?;

        // Write data
        let uint8_array = js_sys::Uint8Array::from(data);
        let write_promise = writable
            .write_with_buffer_source(&uint8_array)
            .map_err(|e| StorageError::IoError(format!("Failed to write: {:?}", e)))?;

        JsFuture::from(write_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to await write: {:?}", e)))?;

        // Close stream
        let close_promise = writable.close();

        JsFuture::from(close_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to close: {:?}", e)))?;

        Ok(())
    }

    async fn read_file(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        let (dir_path, filename) = Self::parse_key(key);

        // Get the directory (don't create)
        let dir = if let Some(path) = dir_path {
            self.get_directory(path, false).await?
        } else {
            self.root.clone()
        };

        // Get file handle (don't create)
        let file_handle_promise = dir.get_file_handle(filename);

        let file_handle = JsFuture::from(file_handle_promise)
            .await
            .map_err(|_| StorageError::NotFound(key.to_string()))?;

        let file_handle: web_sys::FileSystemFileHandle = file_handle
            .dyn_into()
            .map_err(|_| StorageError::IoError("Failed to cast file handle".to_string()))?;

        // Get File object
        let file_promise = file_handle.get_file();

        let file = JsFuture::from(file_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to get file: {:?}", e)))?;

        let file: web_sys::File = file
            .dyn_into()
            .map_err(|_| StorageError::IoError("Failed to cast file".to_string()))?;

        // Read as array buffer
        let array_buffer_promise = file.array_buffer();

        let array_buffer = JsFuture::from(array_buffer_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to read array buffer: {:?}", e)))?;

        // Convert to Vec<u8>
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let data = uint8_array.to_vec();

        Ok(data)
    }

    async fn file_exists(&self, key: &str) -> Result<bool, StorageError> {
        let (dir_path, filename) = Self::parse_key(key);

        // Try to get the directory
        let dir = if let Some(path) = dir_path {
            match self.get_directory(path, false).await {
                Ok(d) => d,
                Err(StorageError::NotFound(_)) => return Ok(false),
                Err(e) => return Err(e),
            }
        } else {
            self.root.clone()
        };

        // Try to get file handle without creating
        let file_handle_promise = dir.get_file_handle(filename);

        match JsFuture::from(file_handle_promise).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn delete_file(&self, key: &str) -> Result<(), StorageError> {
        let (dir_path, filename) = Self::parse_key(key);

        // Get the directory
        let dir = if let Some(path) = dir_path {
            self.get_directory(path, false).await?
        } else {
            self.root.clone()
        };

        let remove_promise = dir.remove_entry(filename);

        JsFuture::from(remove_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to delete: {:?}", e)))?;

        Ok(())
    }

    /// Recursively delete a directory and all its contents.
    async fn delete_directory_recursive(
        &self,
        parent: &FileSystemDirectoryHandle,
        name: &str,
    ) -> Result<(), StorageError> {
        // Use removeEntry with recursive option
        let options = web_sys::FileSystemRemoveOptions::new();
        options.set_recursive(true);

        let remove_promise = parent.remove_entry_with_options(name, &options);

        JsFuture::from(remove_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to delete directory: {:?}", e)))?;

        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl StorageBackend for OpfsStorage {
    async fn save(&self, key: &str, data: &[u8]) -> Result<(), StorageError> {
        self.write_file(key, data).await
    }

    async fn load(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        self.read_file(key).await
    }

    async fn exists(&self, key: &str) -> Result<bool, StorageError> {
        self.file_exists(key).await
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        self.delete_file(key).await
    }

    async fn list_keys(&self) -> Result<Vec<String>, StorageError> {
        // OPFS doesn't have a simple "list all" API in the standard web API.
        // The FileSystemDirectoryHandle.entries() method exists but requires
        // async iteration which is complex in wasm-bindgen.
        //
        // For our use case, we know the exact keys we use (manifest, documents, embeddings),
        // so we don't need full enumeration. Return empty and rely on exists() checks.
        Ok(vec![])
    }

    async fn clear(&self) -> Result<(), StorageError> {
        // Delete the "index" directory which contains all our data
        // This is more reliable than trying to enumerate all files
        match self.delete_directory_recursive(&self.root, "index").await {
            Ok(()) => Ok(()),
            Err(StorageError::IoError(_)) => {
                // Directory might not exist, that's fine
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}
