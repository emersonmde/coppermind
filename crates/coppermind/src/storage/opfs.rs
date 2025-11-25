// OPFS (Origin Private File System) storage implementation for web

use super::{StorageBackend, StorageError};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{FileSystemDirectoryHandle, FileSystemGetFileOptions};

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

    async fn write_file(&self, key: &str, data: &[u8]) -> Result<(), StorageError> {
        // Get or create file handle
        let options = FileSystemGetFileOptions::new();
        options.set_create(true);

        let file_handle_promise = self.root.get_file_handle_with_options(key, &options);

        let file_handle = JsFuture::from(file_handle_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to await file handle: {:?}", e)))?;

        let file_handle: web_sys::FileSystemFileHandle = file_handle
            .dyn_into()
            .map_err(|_| StorageError::IoError("Failed to cast file handle".to_string()))?;

        // Create writable stream
        let writable_promise = file_handle.create_writable();

        let writable = JsFuture::from(writable_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to await writable: {:?}", e)))?;

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
            .map_err(|e| StorageError::IoError(format!("Failed to await close: {:?}", e)))?;

        Ok(())
    }

    async fn read_file(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        // Get file handle (don't create)
        let file_handle_promise = self.root.get_file_handle(key);

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
            .map_err(|e| StorageError::IoError(format!("Failed to await file: {:?}", e)))?;

        let file: web_sys::File = file
            .dyn_into()
            .map_err(|_| StorageError::IoError("Failed to cast file".to_string()))?;

        // Read as array buffer
        let array_buffer_promise = file.array_buffer();

        let array_buffer = JsFuture::from(array_buffer_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to await array buffer: {:?}", e)))?;

        // Convert to Vec<u8>
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let data = uint8_array.to_vec();

        Ok(data)
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
        // Try to get file handle without creating
        let file_handle_promise = self.root.get_file_handle(key);

        // Await the promise to see if the file actually exists
        match JsFuture::from(file_handle_promise).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let remove_promise = self.root.remove_entry(key);

        JsFuture::from(remove_promise)
            .await
            .map_err(|e| StorageError::IoError(format!("Failed to await remove: {:?}", e)))?;

        Ok(())
    }

    async fn list_keys(&self) -> Result<Vec<String>, StorageError> {
        // OPFS doesn't have a simple "list all" API
        // For now, return empty vec - we can track keys separately if needed
        // TODO: Implement key tracking if needed
        web_sys::console::warn_1(
            &"OPFS list_keys() not fully implemented - returning empty vec".into(),
        );
        Ok(vec![])
    }

    async fn clear(&self) -> Result<(), StorageError> {
        // Since we can't easily list all keys, we can't clear easily
        // This would require tracking keys separately
        // TODO: Implement clear if needed
        web_sys::console::warn_1(&"OPFS clear() not fully implemented".into());
        Ok(())
    }
}
