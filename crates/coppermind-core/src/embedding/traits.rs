//! Traits for embedding operations.
//!
//! This module defines the core abstractions for embedding models, asset loading,
//! and model configuration. These traits allow different implementations to be
//! swapped without changing dependent code.

use crate::error::{AssetError, EmbeddingError};
use async_trait::async_trait;

/// Trait for loading model assets (weights, tokenizer).
///
/// Implementations provide platform-specific loading:
/// - **Web (Dioxus)**: Fetch API with Dioxus asset macro base path resolution
/// - **Desktop (Dioxus)**: Filesystem with bundle path resolution
/// - **CLI**: Direct filesystem path
///
/// # Examples
///
/// ```ignore
/// // Dioxus implementation
/// struct DioxusAssetLoader { /* asset URLs */ }
///
/// impl AssetLoader for DioxusAssetLoader {
///     async fn load_model_bytes(&self) -> Result<Vec<u8>, AssetError> {
///         fetch_asset_bytes(&self.model_url).await
///     }
///     // ...
/// }
///
/// // CLI implementation
/// struct FileAssetLoader { model_path: PathBuf, tokenizer_path: PathBuf }
///
/// impl AssetLoader for FileAssetLoader {
///     async fn load_model_bytes(&self) -> Result<Vec<u8>, AssetError> {
///         tokio::fs::read(&self.model_path).await
///             .map_err(|e| AssetError::LoadFailed(e.to_string()))
///     }
///     // ...
/// }
/// ```
#[async_trait(?Send)]
pub trait AssetLoader: Send + Sync {
    /// Load model weights as raw bytes (safetensors format).
    async fn load_model_bytes(&self) -> Result<Vec<u8>, AssetError>;

    /// Load tokenizer configuration as raw bytes (JSON format).
    async fn load_tokenizer_bytes(&self) -> Result<Vec<u8>, AssetError>;
}

/// Trait for embedding model operations.
///
/// This trait defines the interface for embedding models, allowing different
/// architectures (JinaBERT, OpenAI, etc.) to be swapped without changing
/// dependent code.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow sharing across threads.
/// The GPU scheduler may call embedding methods from a dedicated worker thread.
///
/// # Examples
///
/// ```ignore
/// let embedder: Arc<dyn Embedder> = Arc::new(JinaBertEmbedder::from_bytes(...)?);
///
/// let token_ids = vec![101, 2023, 2003, 1037, 3231, 102];
/// let embedding = embedder.embed_tokens(token_ids)?;
/// assert_eq!(embedding.len(), embedder.embedding_dim());
/// ```
pub trait Embedder: Send + Sync {
    /// Returns the maximum number of position embeddings (sequence length).
    ///
    /// Tokens beyond this limit will be truncated by the tokenizer.
    fn max_position_embeddings(&self) -> usize;

    /// Returns the embedding dimension (vector size).
    ///
    /// All embeddings from this model will have this length.
    fn embedding_dim(&self) -> usize;

    /// Generates an embedding from token IDs.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Vector of token IDs from the tokenizer
    ///
    /// # Returns
    ///
    /// Embedding vector of dimension `embedding_dim()`
    fn embed_tokens(&self, token_ids: Vec<u32>) -> Result<Vec<f32>, EmbeddingError>;

    /// Generates embeddings for a batch of token sequences.
    ///
    /// More efficient than calling `embed_tokens` multiple times when
    /// processing multiple texts.
    ///
    /// # Arguments
    ///
    /// * `batch_token_ids` - Vector of token ID vectors
    ///
    /// # Returns
    ///
    /// Vector of embeddings, one per input sequence
    fn embed_batch_tokens(
        &self,
        batch_token_ids: Vec<Vec<u32>>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

/// Trait for embedding model configurations.
///
/// This trait allows different model types to provide their own configuration
/// while maintaining a common interface for model initialization.
pub trait ModelConfig: Clone + Send + Sync {
    /// Returns the model identifier (e.g., "jinaai/jina-embeddings-v2-small-en").
    fn model_id(&self) -> &str;

    /// Returns the output embedding dimension.
    fn embedding_dim(&self) -> usize;

    /// Returns the maximum sequence length the model can handle.
    fn max_sequence_length(&self) -> usize;

    /// Whether embeddings should be L2 normalized (unit vectors).
    fn normalize_embeddings(&self) -> bool;
}
