//! Extension traits for adding context to errors.
//!
//! # Problem
//!
//! Manual error context is verbose and repetitive:
//!
//! ```ignore
//! something()
//!     .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to load model: {}", e)))?
//! ```
//!
//! This pattern appears throughout the codebase, making error propagation noisy.
//!
//! # Solution
//!
//! The `ResultExt` trait provides a `context()` method for cleaner error handling:
//!
//! ```ignore
//! something().context("Failed to load model")?
//! ```
//!
//! # Examples
//!
//! ## Before: Manual error formatting
//!
//! ```ignore
//! let model = VarBuilder::from_buffered_safetensors(bytes, DType::F32, device)
//!     .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to create VarBuilder: {}", e)))?;
//!
//! let embeddings = model.forward(&tokens)
//!     .map_err(|e| EmbeddingError::InferenceFailed(format!("Forward pass failed: {}", e)))?;
//! ```
//!
//! ## After: Clean context
//!
//! ```ignore
//! let model = VarBuilder::from_buffered_safetensors(bytes, DType::F32, device)
//!     .context("Failed to create VarBuilder")?;
//!
//! let embeddings = model.forward(&tokens)
//!     .context("Forward pass failed")?;
//! ```

use crate::error::{EmbeddingError, FileProcessingError};
use crate::search::types::SearchError;
use crate::storage::StorageError;

/// Extension trait for adding context to Result types.
///
/// Automatically implemented for `Result<T, E>` where `E` has a constructor
/// that accepts a String message.
pub trait ResultExt<T, E> {
    /// Add context to an error.
    ///
    /// If the Result is Ok, returns the value unchanged.
    /// If the Result is Err, wraps the error with additional context.
    ///
    /// # Arguments
    ///
    /// * `context` - Context message to prepend to the error
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Simple context
    /// let file = fs::read("model.bin")
    ///     .context("Failed to read model file")?;
    ///
    /// // Chained contexts (inner → outer)
    /// let result = do_something()
    ///     .context("Step 1 failed")?
    ///     .do_next()
    ///     .context("Step 2 failed")?;
    /// ```
    fn context(self, context: &str) -> Result<T, E>;

    /// Add context using a closure (lazy evaluation).
    ///
    /// Only evaluates the context message if there's an error.
    /// Useful when context construction is expensive.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that produces the context message
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Expensive string formatting only on error
    /// let result = operation()
    ///     .with_context(|| format!("Failed for user {} at {}", user_id, timestamp))?;
    /// ```
    fn with_context<F>(self, f: F) -> Result<T, E>
    where
        F: FnOnce() -> String;
}

// Implementation for EmbeddingError
impl<T> ResultExt<T, EmbeddingError> for Result<T, EmbeddingError> {
    fn context(self, context: &str) -> Result<T, EmbeddingError> {
        self.map_err(|e| match e {
            EmbeddingError::ModelLoad(msg) => {
                EmbeddingError::ModelLoad(format!("{}: {}", context, msg))
            }
            EmbeddingError::TensorCreation(msg) => {
                EmbeddingError::TensorCreation(format!("{}: {}", context, msg))
            }
            EmbeddingError::InferenceFailed(msg) => {
                EmbeddingError::InferenceFailed(format!("{}: {}", context, msg))
            }
            EmbeddingError::TokenizationFailed(msg) => {
                EmbeddingError::TokenizationFailed(format!("{}: {}", context, msg))
            }
            EmbeddingError::AssetFetch(msg) => {
                EmbeddingError::AssetFetch(format!("{}: {}", context, msg))
            }
            EmbeddingError::InvalidConfig(msg) => {
                EmbeddingError::InvalidConfig(format!("{}: {}", context, msg))
            }
            EmbeddingError::TokenizerUnavailable(msg) => {
                EmbeddingError::TokenizerUnavailable(format!("{}: {}", context, msg))
            }
            EmbeddingError::ModelUnavailable(msg) => {
                EmbeddingError::ModelUnavailable(format!("{}: {}", context, msg))
            }
            EmbeddingError::ChunkingFailed(msg) => {
                EmbeddingError::ChunkingFailed(format!("{}: {}", context, msg))
            }
        })
    }

    fn with_context<F>(self, f: F) -> Result<T, EmbeddingError>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let context = f();
            match e {
                EmbeddingError::ModelLoad(msg) => {
                    EmbeddingError::ModelLoad(format!("{}: {}", context, msg))
                }
                EmbeddingError::TensorCreation(msg) => {
                    EmbeddingError::TensorCreation(format!("{}: {}", context, msg))
                }
                EmbeddingError::InferenceFailed(msg) => {
                    EmbeddingError::InferenceFailed(format!("{}: {}", context, msg))
                }
                EmbeddingError::TokenizationFailed(msg) => {
                    EmbeddingError::TokenizationFailed(format!("{}: {}", context, msg))
                }
                EmbeddingError::AssetFetch(msg) => {
                    EmbeddingError::AssetFetch(format!("{}: {}", context, msg))
                }
                EmbeddingError::InvalidConfig(msg) => {
                    EmbeddingError::InvalidConfig(format!("{}: {}", context, msg))
                }
                EmbeddingError::TokenizerUnavailable(msg) => {
                    EmbeddingError::TokenizerUnavailable(format!("{}: {}", context, msg))
                }
                EmbeddingError::ModelUnavailable(msg) => {
                    EmbeddingError::ModelUnavailable(format!("{}: {}", context, msg))
                }
                EmbeddingError::ChunkingFailed(msg) => {
                    EmbeddingError::ChunkingFailed(format!("{}: {}", context, msg))
                }
            }
        })
    }
}

// Implementation for FileProcessingError
impl<T> ResultExt<T, FileProcessingError> for Result<T, FileProcessingError> {
    fn context(self, context: &str) -> Result<T, FileProcessingError> {
        self.map_err(|e| match e {
            FileProcessingError::FileRead(msg) => {
                FileProcessingError::FileRead(format!("{}: {}", context, msg))
            }
            FileProcessingError::BinaryFile(msg) => {
                FileProcessingError::BinaryFile(format!("{}: {}", context, msg))
            }
            FileProcessingError::EmbeddingFailed(msg) => {
                FileProcessingError::EmbeddingFailed(format!("{}: {}", context, msg))
            }
            FileProcessingError::IndexingFailed(msg) => {
                FileProcessingError::IndexingFailed(format!("{}: {}", context, msg))
            }
            FileProcessingError::DirectoryTraversal(msg) => {
                FileProcessingError::DirectoryTraversal(format!("{}: {}", context, msg))
            }
        })
    }

    fn with_context<F>(self, f: F) -> Result<T, FileProcessingError>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let context = f();
            match e {
                FileProcessingError::FileRead(msg) => {
                    FileProcessingError::FileRead(format!("{}: {}", context, msg))
                }
                FileProcessingError::BinaryFile(msg) => {
                    FileProcessingError::BinaryFile(format!("{}: {}", context, msg))
                }
                FileProcessingError::EmbeddingFailed(msg) => {
                    FileProcessingError::EmbeddingFailed(format!("{}: {}", context, msg))
                }
                FileProcessingError::IndexingFailed(msg) => {
                    FileProcessingError::IndexingFailed(format!("{}: {}", context, msg))
                }
                FileProcessingError::DirectoryTraversal(msg) => {
                    FileProcessingError::DirectoryTraversal(format!("{}: {}", context, msg))
                }
            }
        })
    }
}

// Implementation for SearchError
impl<T> ResultExt<T, SearchError> for Result<T, SearchError> {
    fn context(self, context: &str) -> Result<T, SearchError> {
        self.map_err(|e| match e {
            SearchError::StorageError(msg) => {
                SearchError::StorageError(format!("{}: {}", context, msg))
            }
            SearchError::EmbeddingError(msg) => {
                SearchError::EmbeddingError(format!("{}: {}", context, msg))
            }
            SearchError::IndexError(msg) => {
                SearchError::IndexError(format!("{}: {}", context, msg))
            }
            SearchError::NotFound => SearchError::NotFound,
        })
    }

    fn with_context<F>(self, f: F) -> Result<T, SearchError>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let context = f();
            match e {
                SearchError::StorageError(msg) => {
                    SearchError::StorageError(format!("{}: {}", context, msg))
                }
                SearchError::EmbeddingError(msg) => {
                    SearchError::EmbeddingError(format!("{}: {}", context, msg))
                }
                SearchError::IndexError(msg) => {
                    SearchError::IndexError(format!("{}: {}", context, msg))
                }
                SearchError::NotFound => SearchError::NotFound,
            }
        })
    }
}

// Implementation for StorageError
impl<T> ResultExt<T, StorageError> for Result<T, StorageError> {
    fn context(self, context: &str) -> Result<T, StorageError> {
        self.map_err(|e| match e {
            StorageError::NotFound(msg) => StorageError::NotFound(format!("{}: {}", context, msg)),
            StorageError::IoError(msg) => StorageError::IoError(format!("{}: {}", context, msg)),
            StorageError::SerializationError(msg) => {
                StorageError::SerializationError(format!("{}: {}", context, msg))
            }
            StorageError::BrowserApiUnavailable => StorageError::BrowserApiUnavailable,
            StorageError::OpfsUnavailable => StorageError::OpfsUnavailable,
        })
    }

    fn with_context<F>(self, f: F) -> Result<T, StorageError>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let context = f();
            match e {
                StorageError::NotFound(msg) => {
                    StorageError::NotFound(format!("{}: {}", context, msg))
                }
                StorageError::IoError(msg) => {
                    StorageError::IoError(format!("{}: {}", context, msg))
                }
                StorageError::SerializationError(msg) => {
                    StorageError::SerializationError(format!("{}: {}", context, msg))
                }
                StorageError::BrowserApiUnavailable => StorageError::BrowserApiUnavailable,
                StorageError::OpfsUnavailable => StorageError::OpfsUnavailable,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_error_context() {
        let result: Result<(), EmbeddingError> =
            Err(EmbeddingError::ModelLoad("original error".to_string()));

        let with_context = result.context("Failed to initialize");

        match with_context {
            Err(EmbeddingError::ModelLoad(msg)) => {
                assert_eq!(msg, "Failed to initialize: original error");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_with_context_lazy() {
        let expensive_call_count = std::cell::Cell::new(0);

        let result: Result<i32, EmbeddingError> = Ok(42);

        let _ok = result.with_context(|| {
            expensive_call_count.set(expensive_call_count.get() + 1);
            "expensive context".to_string()
        });

        // Context closure should not be called for Ok
        assert_eq!(expensive_call_count.get(), 0);
    }

    #[test]
    fn test_context_ok_passthrough() {
        let result: Result<i32, EmbeddingError> = Ok(42);
        let with_context = result.context("This shouldn't matter");

        assert_eq!(with_context.unwrap(), 42);
    }
}
