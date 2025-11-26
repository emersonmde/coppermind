//! Platform-specific execution utilities.
//!
//! This module provides abstractions for CPU-intensive operations that need
//! different execution strategies on web (WASM) vs. desktop platforms:
//!
//! - **Desktop**: Uses `tokio::spawn_blocking` to offload to thread pool
//! - **Web**: Executes directly (web workers handle parallelism)
//!
//! # Motivation
//!
//! This pattern appears 10+ times across the codebase with identical cfg blocks.
//! Extracting it here eliminates duplication and provides a single place to
//! adjust platform-specific execution strategy.
//!
//! # Examples
//!
//! ```ignore
//! use crate::platform::run_blocking;
//! use crate::error::EmbeddingError;
//!
//! // Before (with duplication):
//! #[cfg(not(target_arch = "wasm32"))]
//! {
//!     tokio::task::spawn_blocking(move || heavy_computation())
//!         .await
//!         .map_err(|e| EmbeddingError::InferenceFailed(format!("Join: {}", e)))??
//! }
//! #[cfg(target_arch = "wasm32")]
//! {
//!     heavy_computation()?
//! }
//!
//! // After (clean):
//! run_blocking(|| heavy_computation()).await?
//! ```

use std::future::Future;

/// Execute a CPU-intensive operation on the appropriate thread pool.
///
/// # Platform Behavior
///
/// - **Desktop**: Runs `f` on tokio's blocking thread pool to prevent UI freezing
/// - **Web**: Runs `f` directly (web workers handle parallelism externally)
///
/// # Type Parameters
///
/// - `F`: Closure that performs the CPU-intensive work
/// - `T`: Return type (must be Send on desktop for thread transfer)
/// - `E`: Error type (must be Send on desktop, must support string conversion)
///
/// # Arguments
///
/// * `f` - Closure to execute (will be moved to blocking thread on desktop)
///
/// # Returns
///
/// Result from executing `f`, with join errors converted to `E::from(String)`.
///
/// # Examples
///
/// ```ignore
/// // Embedding inference
/// let embedding = run_blocking(move || model.embed_tokens(token_ids)).await?;
///
/// // Model loading
/// let model = run_blocking(move || create_heavy_model(bytes, config)).await?;
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub async fn run_blocking<F, T, E>(f: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E> + Send + 'static,
    T: Send + 'static,
    E: From<String> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| E::from(format!("Task join failed: {}", e)))?
}

/// Execute a CPU-intensive operation on the appropriate thread pool.
///
/// Web version: Executes directly since web workers handle parallelism.
#[cfg(target_arch = "wasm32")]
pub async fn run_blocking<F, T, E>(f: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    f()
}

/// Execute an async CPU-intensive operation with platform-appropriate strategy.
///
/// Similar to `run_blocking`, but for async closures that perform CPU work.
/// This is useful when the operation itself needs to await (e.g., for
/// progress updates or yielding), but the core work is CPU-intensive.
///
/// # Platform Behavior
///
/// - **Desktop**: Spawns as a separate tokio task
/// - **Web**: Executes directly
///
/// # Examples
///
/// ```ignore
/// // Chunk processing with progress updates
/// let results = run_async(async move {
///     let mut results = Vec::new();
///     for chunk in chunks {
///         let embedding = compute_embedding(chunk).await?;
///         results.push(embedding);
///         // Yield to event loop
///         tokio::task::yield_now().await;
///     }
///     Ok(results)
/// }).await?;
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub async fn run_async<F, T, E>(f: F) -> Result<T, E>
where
    F: Future<Output = Result<T, E>> + Send + 'static,
    T: Send + 'static,
    E: From<String> + Send + 'static,
{
    tokio::task::spawn(f)
        .await
        .map_err(|e| E::from(format!("Task join failed: {}", e)))?
}

/// Execute an async CPU-intensive operation with platform-appropriate strategy.
///
/// Web version: Executes directly.
#[cfg(target_arch = "wasm32")]
pub async fn run_async<F, T, E>(f: F) -> Result<T, E>
where
    F: Future<Output = Result<T, E>>,
{
    f.await
}

/// Yield control back to the async runtime.
///
/// Use this between batches of CPU-intensive work to allow the UI to update.
///
/// # Platform Behavior
///
/// - **Desktop**: Uses `tokio::task::yield_now()` to yield to the tokio runtime
/// - **Web**: No-op (WASM is single-threaded, yielding doesn't help UI responsiveness)
///
/// # Examples
///
/// ```ignore
/// for batch in chunks.chunks(32) {
///     process_batch(batch).await;
///     yield_now().await; // Let UI update
/// }
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub async fn yield_now() {
    tokio::task::yield_now().await;
}

/// Yield control back to the async runtime.
///
/// Web version: No-op since WASM is single-threaded.
#[cfg(target_arch = "wasm32")]
pub async fn yield_now() {
    // No-op on WASM - single threaded, yielding doesn't help
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock error type for testing
    #[derive(Debug, PartialEq)]
    struct TestError(String);

    impl From<String> for TestError {
        fn from(s: String) -> Self {
            TestError(s)
        }
    }

    #[tokio::test]
    async fn test_run_blocking_success() {
        let result: Result<i32, TestError> = run_blocking(|| Ok(42)).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_run_blocking_error() {
        let result: Result<i32, TestError> =
            run_blocking(|| Err(TestError("fail".to_string()))).await;
        assert_eq!(result, Err(TestError("fail".to_string())));
    }

    #[tokio::test]
    async fn test_run_async_success() {
        let result: Result<i32, TestError> = run_async(async { Ok(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_run_async_error() {
        let result: Result<i32, TestError> =
            run_async(async { Err(TestError("fail".to_string())) }).await;
        assert_eq!(result, Err(TestError("fail".to_string())));
    }
}
