/// Storage backend abstraction for cross-platform persistence
///
/// This trait provides a generic key-value storage interface that can be
/// implemented using different backends (OPFS for web, native filesystem for desktop).
#[async_trait::async_trait(?Send)]
#[allow(dead_code)] // Public API trait
pub trait StorageBackend {
    /// Save binary data to storage with a key
    async fn save(&self, key: &str, data: &[u8]) -> Result<(), StorageError>;

    /// Load binary data from storage by key
    async fn load(&self, key: &str) -> Result<Vec<u8>, StorageError>;

    /// Check if a key exists in storage
    async fn exists(&self, key: &str) -> Result<bool, StorageError>;

    /// Delete data by key
    async fn delete(&self, key: &str) -> Result<(), StorageError>;

    /// List all keys in storage (useful for debugging/management)
    async fn list_keys(&self) -> Result<Vec<String>, StorageError>;

    /// Clear all stored data
    async fn clear(&self) -> Result<(), StorageError>;
}

/// Storage error types
#[derive(Debug)]
#[allow(dead_code)] // Public API enum
pub enum StorageError {
    NotFound(String),
    IoError(String),
    SerializationError(String),
    BrowserApiUnavailable,
    OpfsUnavailable,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::NotFound(key) => write!(f, "Key not found: {}", key),
            StorageError::IoError(e) => write!(f, "IO error: {}", e),
            StorageError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            StorageError::BrowserApiUnavailable => write!(f, "Browser API unavailable"),
            StorageError::OpfsUnavailable => {
                write!(
                    f,
                    "OPFS unavailable - please use a modern browser in standard mode"
                )
            }
        }
    }
}

impl std::error::Error for StorageError {}

// Platform-specific implementations
#[cfg(target_arch = "wasm32")]
pub mod opfs;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

// Re-export the platform-specific storage (public API)
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
pub use opfs::OpfsStorage;

#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
pub use native::NativeStorage;
