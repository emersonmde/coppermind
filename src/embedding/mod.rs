//! Embedding model inference and text processing.
//!
//! This module provides embedding generation for semantic search. It supports
//! multiple platforms (web via WASM and desktop) with appropriate optimizations
//! for each environment.
//!
//! # Architecture
//!
//! The module is organized into sub-modules for maintainability and testability:
//!
//! - [`config`]: Model configuration (JinaBertConfig, etc.)
//! - [`model`]: Model implementations and inference (JinaBertEmbedder, Embedder trait)
//! - [`tokenizer`]: Tokenization and chunking utilities
//! - [`assets`]: Asset loading (network/filesystem)
//!
//! # Usage
//!
//! ```ignore
//! use coppermind::embedding::{compute_embedding, embed_text_chunks};
//!
//! // Generate embedding for a single text
//! let result = compute_embedding("hello world").await?;
//! println!("Embedding: {:?}", result.embedding);
//!
//! // Process long document in chunks
//! let chunks = embed_text_chunks("long document...", 512).await?;
//! for chunk in chunks {
//!     println!("Chunk {}: {} tokens", chunk.chunk_index, chunk.token_count);
//! }
//! ```
//!
//! # Platform-Specific Behavior
//!
//! ## Web (WASM)
//! - CPU-only inference (WebGPU not yet supported in Candle)
//! - Memory-constrained (512MB WASM limit)
//! - Direct async/await (no spawn_blocking needed)
//!
//! ## Desktop
//! - GPU acceleration via CUDA or Metal (falls back to CPU with MKL/Accelerate)
//! - CPU-intensive operations use `tokio::spawn_blocking` to prevent UI freezing
//! - Higher memory limits allow longer sequences
//!
//! # Future Extensions
//!
//! When adding support for additional models:
//!
//! 1. Create a new config type implementing `ModelConfig` trait
//! 2. Create a new model type implementing `Embedder` trait
//! 3. Update `get_or_load_model()` to support model selection
//! 4. Consider replacing singleton with model registry pattern

pub mod assets;
pub mod chunking;
pub mod config;
pub mod model;
pub mod tokenizer;

// Re-export key types for convenience
pub use chunking::ChunkingStrategy;
pub use config::{JinaBertConfig, ModelConfig};
pub use model::{Embedder, JinaBertEmbedder};

use crate::error::EmbeddingError;
use crate::platform::run_blocking;
use dioxus::logger::tracing::{debug, info};
use dioxus::prelude::*; // Includes asset! macro and Asset type
use std::sync::Arc;

// Asset declarations - loaded via Dioxus asset system
const MODEL_FILE: Asset = asset!("/assets/models/jina-bert.safetensors");
const TOKENIZER_FILE: Asset = asset!("/assets/models/jina-bert-tokenizer.json");

/// Result of embedding a single text chunk.
///
/// Contains the chunk index, token count, text content, and embedding vector.
#[derive(Clone, Debug)]
pub struct ChunkEmbeddingResult {
    /// Index of this chunk in the original document (0-based)
    pub chunk_index: usize,
    /// Number of tokens in this chunk (including special tokens)
    pub token_count: usize,
    /// Text content of this chunk (decoded from tokens)
    pub text: String,
    /// Embedding vector (dimension matches model config)
    pub embedding: Vec<f32>,
}

/// Result of computing an embedding.
///
/// Contains the token count and embedding vector, used by both
/// single-text and chunked embedding workflows.
#[derive(Clone)]
pub struct EmbeddingComputation {
    /// Number of tokens processed
    pub token_count: usize,
    /// Embedding vector (dimension matches model config)
    pub embedding: Vec<f32>,
}

/// Gets or loads the embedding model (singleton pattern).
///
/// On first call, downloads model and tokenizer assets and initializes
/// the model. Subsequent calls return the cached instance.
///
/// Uses spawn_blocking on desktop to prevent UI freezing during model load.
///
/// # Returns
///
/// Arc-wrapped model for thread-safe sharing.
///
/// # Errors
///
/// Returns `EmbeddingError` if asset fetching or model initialization fails.
pub async fn get_or_load_model() -> Result<Arc<dyn Embedder>, EmbeddingError> {
    // NOTE: This singleton will be replaced with model registry when
    // supporting multiple models. For now, hard-coded to JinaBERT.

    // Check if model is already cached (avoid fetching 65MB asset unnecessarily)
    if let Some(cached) = model::get_cached_model() {
        return Ok(cached);
    }

    let model_url = MODEL_FILE.to_string();
    info!("📦 Loading embedding model...");

    let model_bytes = assets::fetch_asset_bytes(&model_url).await?;
    let config = JinaBertConfig::default();

    // Model creation is CPU-intensive - run in blocking thread pool on desktop
    let model = run_blocking(move || model::get_or_load_model(model_bytes, 30528, config)).await?;

    Ok(model)
}

/// Ensures the tokenizer is initialized and configured.
///
/// Downloads tokenizer on first call and caches it. Configures for
/// the specified max sequence length.
///
/// # Arguments
///
/// * `max_positions` - Maximum sequence length for truncation
///
/// # Returns
///
/// Static reference to the tokenizer (valid for program lifetime).
///
/// # Errors
///
/// Returns `EmbeddingError` if download or initialization fails.
async fn ensure_tokenizer(
    max_positions: usize,
) -> Result<&'static tokenizers::Tokenizer, EmbeddingError> {
    // Check cache first to avoid fetching 466KB asset on every call
    if let Some(tokenizer) = tokenizer::get_cached_tokenizer() {
        return Ok(tokenizer);
    }

    let tokenizer_url = TOKENIZER_FILE.to_string();
    debug!("📚 Loading tokenizer from {}", tokenizer_url);

    let tokenizer_bytes = assets::fetch_asset_bytes(&tokenizer_url).await?;
    tokenizer::ensure_tokenizer(tokenizer_bytes, max_positions)
}

/// Generates an embedding for a single text.
///
/// This is the high-level API for embedding generation. It handles model
/// loading, tokenization, and inference.
///
/// # Arguments
///
/// * `text` - Input text to embed
///
/// # Returns
///
/// `EmbeddingComputation` containing token count and embedding vector.
///
/// # Platform-Specific Behavior
///
/// - **Desktop**: Uses `spawn_blocking` for CPU-intensive tokenization and inference
/// - **Web**: Runs directly (should be called from a worker to avoid blocking UI)
///
/// # Examples
///
/// ```ignore
/// let computation = compute_embedding("semantic search query").await?;
/// println!("Generated {}-dim embedding from {} tokens",
///          computation.embedding.len(),
///          computation.token_count);
/// ```
#[must_use = "Embedding computation results should be used or errors handled"]
pub async fn compute_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    let model = get_or_load_model().await?;
    let max_positions = model.max_position_embeddings();
    let tokenizer = ensure_tokenizer(max_positions).await?;

    let token_ids = tokenizer::tokenize_text_async(tokenizer, text).await?;
    let token_count = token_ids.len();

    debug!("🧾 Tokenized into {} tokens", token_count);
    debug!("Generating embedding vector...");

    // On desktop: Use spawn_blocking to prevent UI freezing
    // On WASM: Run directly (web worker handles parallelism)
    let model_clone = model.clone();
    let embedding = run_blocking(move || model_clone.embed_tokens(token_ids)).await?;

    debug!("✓ Generated {}-dimensional embedding", embedding.len());

    Ok(EmbeddingComputation {
        token_count,
        embedding,
    })
}

/// Embeds long text by splitting into chunks and processing each.
///
/// Uses sentence-based chunking to preserve semantic coherence. Splits text at
/// sentence boundaries, then groups sentences into chunks that respect token limits.
///
/// # Chunking Strategy
///
/// - **Semantic boundaries**: Chunks split at sentence boundaries, not mid-thought
/// - **Overlap**: 2 sentences overlap between chunks to preserve context
/// - **Coherence**: Produces readable search results, better embedding quality
///
/// # Arguments
///
/// * `text` - Input text to chunk and embed
/// * `chunk_tokens` - Target tokens per chunk (capped at model's max_position_embeddings)
///
/// # Returns
///
/// Vector of `ChunkEmbeddingResult`, one per chunk, in order. Each result includes
/// the original text, enabling direct indexing without re-splitting.
///
/// # Platform-Specific Behavior
///
/// - **Desktop**: Processes batches with `spawn_blocking`, yields between batches
/// - **Web**: Processes sequentially with explicit yields to keep UI responsive
///
/// # Examples
///
/// ```ignore
/// let results = embed_text_chunks("Long document with many paragraphs...", 512).await?;
/// for chunk in &results {
///     println!("Chunk {}: {} tokens, text: {:.50}...",
///              chunk.chunk_index,
///              chunk.token_count,
///              chunk.text);
/// }
/// ```
/// Embed text chunks with automatic chunking strategy selection based on file type.
///
/// This is a convenience wrapper around `embed_text_chunks` that automatically
/// selects the appropriate chunking strategy based on the filename extension:
/// - `.md`, `.markdown` → Markdown-aware chunking
/// - Everything else → Generic text chunking
///
/// # Arguments
///
/// * `text` - Text content to embed
/// * `chunk_tokens` - Target tokens per chunk
/// * `filename` - Optional filename to detect file type (if None, uses text chunking)
///
/// # Returns
///
/// Vector of `ChunkEmbeddingResult` with embeddings, token counts, and text for each chunk.
///
/// # Examples
///
/// ```ignore
/// // Markdown file - will use MarkdownSplitter
/// let chunks = embed_text_chunks_auto(markdown_content, 512, Some("README.md")).await?;
///
/// // Plain text - will use TextSplitter
/// let chunks = embed_text_chunks_auto(text_content, 512, Some("document.txt")).await?;
///
/// // No filename - defaults to TextSplitter
/// let chunks = embed_text_chunks_auto(content, 512, None).await?;
/// ```
pub async fn embed_text_chunks_auto(
    text: &str,
    chunk_tokens: usize,
    filename: Option<&str>,
) -> Result<Vec<ChunkEmbeddingResult>, EmbeddingError> {
    let model = get_or_load_model().await?;
    let max_positions = model.max_position_embeddings();
    let tokenizer = ensure_tokenizer(max_positions).await?;

    let effective_chunk = chunk_tokens.min(max_positions);

    // Detect file type and select appropriate chunking strategy
    let file_type = filename
        .map(chunking::detect_file_type)
        .unwrap_or(chunking::FileType::Text);

    let text_chunks = match file_type {
        chunking::FileType::Markdown => {
            let chunker = chunking::markdown_splitter_adapter::MarkdownSplitterAdapter::new(
                effective_chunk,
                tokenizer,
            );
            chunker.chunk(text)?
        }
        #[cfg(not(target_arch = "wasm32"))]
        chunking::FileType::Code(language) => {
            let chunker = chunking::code_splitter_adapter::CodeSplitterAdapter::new(
                effective_chunk,
                language,
                tokenizer,
            );
            chunker.chunk(text)?
        }
        chunking::FileType::Text => {
            let chunker = chunking::text_splitter_adapter::TextSplitterAdapter::new(
                effective_chunk,
                tokenizer,
            );
            chunker.chunk(text)?
        }
    };

    if text_chunks.is_empty() {
        return Ok(vec![]);
    }

    info!(
        "🧩 Embedding {} chunks ({} tokens max per chunk, {:?} chunking)",
        text_chunks.len(),
        effective_chunk,
        file_type
    );

    // Process each chunk
    let total_chunks = text_chunks.len();
    let mut results = Vec::with_capacity(total_chunks);

    for (idx, text_chunk) in text_chunks.iter().enumerate() {
        info!(
            "📝 Chunk {}/{}: {} chars",
            idx + 1,
            total_chunks,
            text_chunk.text.len()
        );

        let computation = compute_embedding(&text_chunk.text).await?;

        results.push(ChunkEmbeddingResult {
            chunk_index: text_chunk.index,
            token_count: computation.token_count,
            text: text_chunk.text.clone(),
            embedding: computation.embedding,
        });
    }

    info!("✅ Embedded all {} chunks successfully", total_chunks);

    Ok(results)
}

/// Embed text chunks using generic text chunking strategy.
///
/// This function always uses the generic TextSplitter regardless of content type.
/// For automatic strategy selection based on file type, use `embed_text_chunks_auto`.
///
/// # Arguments
///
/// * `text` - Text content to embed
/// * `chunk_tokens` - Target tokens per chunk
///
/// # Returns
///
/// Vector of `ChunkEmbeddingResult` with embeddings, token counts, and text for each chunk.
///
/// # Examples
///
/// ```ignore
/// let results = embed_text_chunks("Long document with many paragraphs...", 512).await?;
/// for chunk in results {
///     println!("Chunk {} has {} tokens: {}",
///              chunk.chunk_index, chunk.token_count, chunk.text);
/// }
/// ```
pub async fn embed_text_chunks(
    text: &str,
    chunk_tokens: usize,
) -> Result<Vec<ChunkEmbeddingResult>, EmbeddingError> {
    embed_text_chunks_auto(text, chunk_tokens, None).await
}

/// Generates an embedding and formats it for display.
///
/// Convenience function for testing and debugging. Embeds text and
/// returns a formatted string with embedding details.
///
/// # Arguments
///
/// * `text` - Text to embed
///
/// # Returns
///
/// Formatted string containing token count, dimension, and first 10 values.
pub async fn run_embedding(text: &str) -> Result<String, EmbeddingError> {
    info!("🔮 Generating embedding for: '{}'", text);
    let computation = compute_embedding(text).await?;
    Ok(format_embedding_summary(
        text,
        computation.token_count,
        &computation.embedding,
    ))
}

/// Formats embedding details as a human-readable string.
///
/// # Arguments
///
/// * `text` - Original input text
/// * `token_count` - Number of tokens processed
/// * `embedding` - Embedding vector
///
/// # Returns
///
/// Multi-line formatted string with embedding statistics.
pub fn format_embedding_summary(text: &str, token_count: usize, embedding: &[f32]) -> String {
    let mut preview = [0.0f32; 10];
    for (dest, src) in preview.iter_mut().zip(embedding.iter()) {
        *dest = *src;
    }

    format!(
        "✓ Embedding Generated Successfully!\n\n\
        Input: '{}'\n\
        Tokens used: {}\n\
        Dimension: {} (normalized)\n\
        First 10 values: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]\n\n\
        Model: jinaai/jina-embeddings-v2-small-en\n\
        Config: 512-dim, 4 layers, 8 heads\n\
        Normalization: L2 (unit vector)",
        text,
        token_count,
        embedding.len(),
        preview[0],
        preview[1],
        preview[2],
        preview[3],
        preview[4],
        preview[5],
        preview[6],
        preview[7],
        preview[8],
        preview[9]
    )
}

/// Computes cosine similarity between two embedding vectors.
///
/// Returns a value in [-1, 1] where:
/// - 1.0 = identical vectors
/// - 0.0 = orthogonal vectors
/// - -1.0 = opposite vectors
///
/// For L2-normalized embeddings, this is equivalent to dot product.
///
/// # Arguments
///
/// * `e1` - First embedding vector
/// * `e2` - Second embedding vector
///
/// # Returns
///
/// Cosine similarity score, or 0.0 if vectors have different dimensions or zero norm.
///
/// # Examples
///
/// ```
/// use coppermind::embedding::cosine_similarity;
///
/// let e1 = vec![1.0, 0.0, 0.0];
/// let e2 = vec![1.0, 0.0, 0.0];
/// assert_eq!(cosine_similarity(&e1, &e2), 1.0);
///
/// let e3 = vec![0.0, 1.0, 0.0];
/// assert_eq!(cosine_similarity(&e1, &e3), 0.0);
/// ```
pub fn cosine_similarity(e1: &[f32], e2: &[f32]) -> f32 {
    if e1.len() != e2.len() {
        return 0.0;
    }

    let dot_product: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = e1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = e2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    dot_product / (norm1 * norm2)
}

// WASM bindings for JavaScript interop
#[cfg(target_arch = "wasm32")]
pub mod wasm {
    //! WASM bindings for JavaScript interop.
    //!
    //! These bindings allow the embedding model to be used from JavaScript
    //! in a Web Worker context.

    use super::*;
    use wasm_bindgen::prelude::*;

    /// WASM wrapper for the embedding model.
    #[wasm_bindgen]
    pub struct WasmEmbeddingModel {
        model: JinaBertEmbedder,
    }

    #[wasm_bindgen]
    impl WasmEmbeddingModel {
        /// Creates a new model from model bytes.
        ///
        /// # Arguments
        ///
        /// * `model_bytes` - Safetensors model weights
        /// * `vocab_size` - Tokenizer vocabulary size
        #[wasm_bindgen(constructor)]
        pub fn new(model_bytes: Vec<u8>, vocab_size: usize) -> Result<WasmEmbeddingModel, JsValue> {
            console_error_panic_hook::set_once();

            let config = JinaBertConfig::default();
            let model = JinaBertEmbedder::from_bytes(model_bytes, vocab_size, config)?;

            Ok(WasmEmbeddingModel { model })
        }

        /// Creates model with custom configuration.
        #[wasm_bindgen(js_name = newWithConfig)]
        pub fn new_with_config(
            model_bytes: Vec<u8>,
            vocab_size: usize,
            hidden_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            max_position_embeddings: usize,
        ) -> Result<WasmEmbeddingModel, JsValue> {
            console_error_panic_hook::set_once();

            let config = JinaBertConfig::new(
                "custom".to_string(),
                hidden_size,
                num_hidden_layers,
                num_attention_heads,
                max_position_embeddings,
            );

            let model = JinaBertEmbedder::from_bytes(model_bytes, vocab_size, config)?;

            Ok(WasmEmbeddingModel { model })
        }

        /// Generates embedding from token IDs.
        #[wasm_bindgen(js_name = embedTokens)]
        pub fn embed_tokens(&self, token_ids: Vec<u32>) -> Result<Vec<f32>, JsValue> {
            self.model.embed_tokens(token_ids).map_err(|e| e.into())
        }

        /// Computes cosine similarity between two embeddings.
        #[wasm_bindgen(js_name = cosineSimilarity)]
        pub fn cosine_similarity_js(e1: &[f32], e2: &[f32]) -> f32 {
            cosine_similarity(e1, e2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let e1 = vec![1.0, 0.0, 0.0];
        let e2 = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&e1, &e2);
        assert!((similarity - 1.0).abs() < 1e-6);

        let e3 = vec![0.0, 1.0, 0.0];
        let similarity = cosine_similarity(&e1, &e3);
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let e1 = vec![1.0, 0.0];
        let e2 = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&e1, &e2), 0.0);
    }
}
