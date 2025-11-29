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
//! - Memory-constrained (4GB WASM limit)
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
use dioxus::logger::tracing::{debug, info};
use dioxus::prelude::*; // Includes asset! macro and Asset type
#[cfg(feature = "profile")]
use tracing::instrument;

// Platform-specific imports
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::{
    get_scheduler, init_scheduler, is_scheduler_initialized, BatchEmbedRequest, EmbedRequest,
    ModelId, ModelLoadConfig, Priority,
};

// Metrics - available on all platforms via coppermind-core
use coppermind_core::metrics::global_metrics;

use crate::platform::run_blocking;
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

/// Initialize the embedding system with the GPU scheduler.
///
/// On desktop, this initializes the GPU scheduler and loads the default
/// JinaBERT model. Must be called once at application startup.
///
/// On WASM, this is a no-op (scheduler not used).
///
/// # Errors
///
/// Returns `EmbeddingError` if:
/// - Scheduler initialization fails
/// - Model loading fails
/// - Assets cannot be fetched
///
/// # Example
///
/// ```ignore
/// // In main.rs or app initialization
/// embedding::init_embedding_system().await?;
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub async fn init_embedding_system() -> Result<(), EmbeddingError> {
    // Initialize scheduler if not already done
    if !is_scheduler_initialized() {
        init_scheduler().map_err(|e| EmbeddingError::SchedulerError(e.to_string()))?;
        info!("✓ GPU scheduler initialized");
    }

    let scheduler = get_scheduler().map_err(|e| EmbeddingError::SchedulerError(e.to_string()))?;

    // Skip if model already loaded
    if scheduler.is_model_loaded(&ModelId::JinaBert) {
        debug!("Model already loaded, skipping initialization");
        return Ok(());
    }

    // Fetch model and tokenizer assets
    let model_url = MODEL_FILE.to_string();
    let tokenizer_url = TOKENIZER_FILE.to_string();

    info!("📦 Loading embedding model for GPU scheduler...");
    let model_bytes = assets::fetch_asset_bytes(&model_url).await?;
    let tokenizer_bytes = assets::fetch_asset_bytes(&tokenizer_url).await?;

    // Load model into scheduler
    let config = ModelLoadConfig::new(model_bytes, tokenizer_bytes, 30528);
    scheduler
        .load_model(ModelId::JinaBert, config)
        .await
        .map_err(|e| EmbeddingError::SchedulerError(e.to_string()))?;

    // Also ensure tokenizer is initialized for local use
    let _ = ensure_tokenizer(8192).await?;

    info!("✓ Embedding system initialized with GPU scheduler");
    Ok(())
}

/// Initialize the embedding system (WASM version).
///
/// On WASM, this loads the model directly without a scheduler.
#[cfg(target_arch = "wasm32")]
pub async fn init_embedding_system() -> Result<(), EmbeddingError> {
    // On WASM, just ensure model is loaded
    let _ = get_or_load_model().await?;
    info!("✓ Embedding system initialized (WASM)");
    Ok(())
}

/// Check if the embedding system is ready.
#[cfg(not(target_arch = "wasm32"))]
pub fn is_embedding_ready() -> bool {
    is_scheduler_initialized() && get_scheduler().map(|s| s.is_ready()).unwrap_or(false)
}

/// Check if the embedding system is ready (WASM version).
#[cfg(target_arch = "wasm32")]
pub fn is_embedding_ready() -> bool {
    model::get_cached_model().is_some()
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
/// - **Desktop**: Uses GPU scheduler for thread-safe Metal/CUDA inference
/// - **Web**: Runs directly (no threading issues on CPU)
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
#[cfg(not(target_arch = "wasm32"))]
pub async fn compute_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    compute_embedding_impl(text, Priority::Interactive).await
}

/// Generates an embedding for a single text (WASM version).
#[must_use = "Embedding computation results should be used or errors handled"]
#[cfg(target_arch = "wasm32")]
pub async fn compute_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    compute_embedding_wasm(text).await
}

/// Generates an embedding for a search query (highest priority).
///
/// Uses `Priority::Immediate` to ensure search queries are processed
/// before background work like bulk indexing.
///
/// # Arguments
///
/// * `text` - Search query text to embed
///
/// # Returns
///
/// `EmbeddingComputation` containing token count and embedding vector.
#[must_use = "Embedding computation results should be used or errors handled"]
#[cfg(not(target_arch = "wasm32"))]
pub async fn compute_search_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    compute_embedding_impl(text, Priority::Immediate).await
}

/// Generates an embedding for a search query (WASM version - same as regular embedding).
#[must_use = "Embedding computation results should be used or errors handled"]
#[cfg(target_arch = "wasm32")]
pub async fn compute_search_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    compute_embedding_wasm(text).await
}

/// Desktop implementation using GPU scheduler.
#[cfg(not(target_arch = "wasm32"))]
async fn compute_embedding_impl(
    text: &str,
    priority: Priority,
) -> Result<EmbeddingComputation, EmbeddingError> {
    // Ensure scheduler is initialized
    if !is_scheduler_initialized() {
        init_embedding_system().await?;
    }

    let scheduler = get_scheduler().map_err(|e| EmbeddingError::SchedulerError(e.to_string()))?;
    let tokenizer = ensure_tokenizer(8192).await?;

    let token_ids = tokenizer::tokenize_text_async(tokenizer, text).await?;
    let token_count = token_ids.len();

    debug!(
        "🧾 Tokenized into {} tokens (priority: {:?})",
        token_count, priority
    );

    // Submit to GPU scheduler
    let stats = scheduler.stats();
    debug!(
        "📤 Submitting to scheduler (priority: {:?}, queue_depth: {})",
        priority, stats.queue_depth
    );

    let response = scheduler
        .embed(EmbedRequest::new(token_ids).with_priority(priority))
        .await
        .map_err(|e| EmbeddingError::SchedulerError(e.to_string()))?;

    debug!(
        "✓ Generated {}-dimensional embedding",
        response.embedding.len()
    );

    Ok(EmbeddingComputation {
        token_count,
        embedding: response.embedding,
    })
}

/// WASM implementation - direct model calls (no scheduler needed).
#[cfg(target_arch = "wasm32")]
async fn compute_embedding_wasm(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    // Log cache status to diagnose performance issues
    // (dioxus logger may not be initialized in worker context, so use web_sys::console)
    let cache_status = if model::get_cached_model().is_some() {
        "cache hit"
    } else {
        "cache miss - will load 65MB model"
    };
    web_sys::console::log_1(&format!("[compute_embedding_wasm] Model {}", cache_status).into());

    let model = get_or_load_model().await?;
    let max_positions = model.max_position_embeddings();
    let tokenizer = ensure_tokenizer(max_positions).await?;

    // Time tokenization for metrics
    let tokenize_start = instant::Instant::now();
    let token_ids = tokenizer::tokenize_text_async(tokenizer, text).await?;
    let tokenize_duration_ms = tokenize_start.elapsed().as_secs_f64() * 1000.0;
    global_metrics().record_tokenization(tokenize_duration_ms);

    let token_count = token_ids.len();

    web_sys::console::log_1(
        &format!(
            "[compute_embedding_wasm] Tokenized into {} tokens ({:.1}ms)",
            token_count, tokenize_duration_ms
        )
        .into(),
    );

    let model_clone = model.clone();
    let embedding = run_blocking(move || model_clone.embed_tokens(token_ids)).await?;

    web_sys::console::log_1(
        &format!(
            "[compute_embedding_wasm] Generated {}-dim embedding",
            embedding.len()
        )
        .into(),
    );

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
/// let chunks = embed_text_chunks_auto(content, 512, None, |_, _| {}).await?;
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(feature = "profile", instrument(skip_all, fields(text_len = text.len(), chunk_tokens, filename)))]
pub async fn embed_text_chunks_auto<F>(
    text: &str,
    chunk_tokens: usize,
    filename: Option<&str>,
    mut progress_callback: F,
) -> Result<Vec<ChunkEmbeddingResult>, EmbeddingError>
where
    F: FnMut(usize, usize), // (completed, total)
{
    // Ensure scheduler is initialized
    if !is_scheduler_initialized() {
        init_embedding_system().await?;
    }

    let scheduler = get_scheduler().map_err(|e| EmbeddingError::SchedulerError(e.to_string()))?;
    let tokenizer = ensure_tokenizer(8192).await?;
    let effective_chunk = chunk_tokens.min(8192);

    // Detect file type and select appropriate chunking strategy
    let file_type = filename
        .map(chunking::detect_file_type)
        .unwrap_or(chunking::FileType::Text);

    let chunker = chunking::create_chunker(file_type, effective_chunk, tokenizer);

    // Chunking is CPU-intensive (ICU4X sentence detection + repeated tokenization)
    // Move to blocking thread pool to prevent UI freezing
    let text_owned = text.to_string();
    let chunk_start = instant::Instant::now();
    let text_chunks = run_blocking(move || chunker.chunk(&text_owned)).await?;
    let chunk_duration_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;
    global_metrics().record_chunking(chunk_duration_ms);

    if text_chunks.is_empty() {
        return Ok(vec![]);
    }

    let total_chunks = text_chunks.len();
    info!(
        "🧩 Embedding {} chunks ({} tokens max per chunk, {:?} chunking, {:.1}ms)",
        total_chunks, effective_chunk, file_type, chunk_duration_ms
    );

    // Report initial progress
    progress_callback(0, total_chunks);

    // Tokenize all chunks in a single blocking call to avoid:
    // 1. Cloning 466KB tokenizer for each chunk
    // 2. spawn_blocking overhead per chunk
    // 3. Sequential awaits blocking the async runtime
    debug!("🔤 Tokenizing {} chunks...", total_chunks);
    let chunk_texts: Vec<String> = text_chunks.iter().map(|c| c.text.clone()).collect();
    let tokenize_start = instant::Instant::now();
    let (token_batches, token_counts) = run_blocking(move || {
        let mut batches = Vec::with_capacity(chunk_texts.len());
        let mut counts = Vec::with_capacity(chunk_texts.len());
        for text in &chunk_texts {
            let tokens = tokenizer::tokenize_text(tokenizer, text)?;
            counts.push(tokens.len());
            batches.push(tokens);
        }
        Ok::<_, EmbeddingError>((batches, counts))
    })
    .await?;
    let tokenize_duration_ms = tokenize_start.elapsed().as_secs_f64() * 1000.0;
    global_metrics().record_tokenization(tokenize_duration_ms);

    // Process in mini-batches to balance GPU efficiency with progress updates
    // True GPU batching is faster (single forward pass, better utilization)
    // but we want progress feedback, so we use small batches (8 chunks)
    const MINI_BATCH_SIZE: usize = 8;

    let mut results = Vec::with_capacity(total_chunks);
    let mut completed = 0;

    for batch_start in (0..total_chunks).step_by(MINI_BATCH_SIZE) {
        let batch_end = (batch_start + MINI_BATCH_SIZE).min(total_chunks);
        let batch_tokens: Vec<Vec<u32>> = token_batches[batch_start..batch_end].to_vec();
        let batch_size = batch_tokens.len();

        let batch_request =
            BatchEmbedRequest::new(batch_tokens).with_priority(Priority::Background);

        let embed_start = instant::Instant::now();
        let embeddings = scheduler
            .embed_batch(batch_request)
            .await
            .map_err(|e| EmbeddingError::SchedulerError(e.to_string()))?;
        let embed_duration_ms = embed_start.elapsed().as_secs_f64() * 1000.0;
        global_metrics().record_embedding(embed_duration_ms);

        // Build results for this batch
        for (i, embed_response) in embeddings.into_iter().enumerate() {
            let idx = batch_start + i;
            let text_chunk = &text_chunks[idx];
            let token_count = token_counts[idx];

            debug!(
                "✓ Chunk {}/{}: {} tokens, {}-dim embedding",
                idx + 1,
                total_chunks,
                token_count,
                embed_response.embedding.len()
            );

            results.push(ChunkEmbeddingResult {
                chunk_index: text_chunk.index,
                token_count,
                text: text_chunk.text.clone(),
                embedding: embed_response.embedding,
            });
        }

        completed += batch_size;
        progress_callback(completed, total_chunks);
    }

    info!("✅ Embedded all {} chunks successfully", total_chunks);

    Ok(results)
}

/// Embed text chunks with automatic chunking strategy selection (WASM version).
///
/// Uses the web worker if available for better UI responsiveness.
#[cfg(target_arch = "wasm32")]
pub async fn embed_text_chunks_auto<F>(
    text: &str,
    chunk_tokens: usize,
    filename: Option<&str>,
    mut progress_callback: F,
) -> Result<Vec<ChunkEmbeddingResult>, EmbeddingError>
where
    F: FnMut(usize, usize), // (completed, total)
{
    use crate::workers::get_global_worker;

    // Use known max positions for JinaBERT to avoid loading 65MB model on main thread
    // The worker has its own model instance for actual embedding
    const JINA_BERT_MAX_POSITIONS: usize = 2048;
    let max_positions = JINA_BERT_MAX_POSITIONS;
    let tokenizer = ensure_tokenizer(max_positions).await?;

    let effective_chunk = chunk_tokens.min(max_positions);

    // Detect file type and select appropriate chunking strategy
    let file_type = filename
        .map(chunking::detect_file_type)
        .unwrap_or(chunking::FileType::Text);

    let chunker = chunking::create_chunker(file_type, effective_chunk, tokenizer);

    // Time chunking operation
    let chunk_start = instant::Instant::now();
    let text_chunks = chunker.chunk(text)?;
    let chunk_duration_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;
    global_metrics().record_chunking(chunk_duration_ms);

    if text_chunks.is_empty() {
        return Ok(vec![]);
    }

    let total_chunks = text_chunks.len();

    // Check if worker is available for off-main-thread embedding
    let worker = get_global_worker();
    let using_worker = worker.is_some();

    info!(
        "🧩 Embedding {} chunks ({} tokens max per chunk, {:?} chunking, {:.1}ms, worker: {})",
        total_chunks,
        effective_chunk,
        file_type,
        chunk_duration_ms,
        if using_worker { "yes" } else { "no" }
    );

    // Report initial progress
    progress_callback(0, total_chunks);

    // Process each chunk
    let mut results = Vec::with_capacity(total_chunks);

    for (idx, text_chunk) in text_chunks.iter().enumerate() {
        debug!(
            "📝 Chunk {}/{}: {} chars",
            idx + 1,
            total_chunks,
            text_chunk.text.len()
        );

        // Time embedding operation (includes worker round-trip on web)
        let embed_start = instant::Instant::now();

        // Use worker if available, otherwise fall back to main thread
        let computation = if let Some(ref w) = worker {
            w.embed(text_chunk.text.clone())
                .await
                .map_err(EmbeddingError::InferenceFailed)?
        } else {
            compute_embedding(&text_chunk.text).await?
        };

        let embed_duration_ms = embed_start.elapsed().as_secs_f64() * 1000.0;
        global_metrics().record_embedding(embed_duration_ms);

        results.push(ChunkEmbeddingResult {
            chunk_index: text_chunk.index,
            token_count: computation.token_count,
            text: text_chunk.text.clone(),
            embedding: computation.embedding,
        });

        // Report progress after each chunk
        progress_callback(idx + 1, total_chunks);
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
    embed_text_chunks_auto(text, chunk_tokens, None, |_, _| {}).await
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
