use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::jina_bert::{BertModel, Config, PositionEmbeddingType};
use dioxus::prelude::*;
use once_cell::sync::OnceCell;
use std::sync::Arc;
use tokenizers::tokenizer::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Logging macros that work in both main thread and Web Worker contexts
// Use web_sys::console directly instead of Dioxus logger (which requires initialization)
#[cfg(target_arch = "wasm32")]
macro_rules! info {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!($($t)*).into())
    }
}

#[cfg(target_arch = "wasm32")]
macro_rules! error {
    ($($t:tt)*) => {
        web_sys::console::error_1(&format!($($t)*).into())
    }
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! info {
    ($($t:tt)*) => {
        eprintln!("[INFO] {}", format!($($t)*))
    }
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! error {
    ($($t:tt)*) => {
        eprintln!("[ERROR] {}", format!($($t)*))
    }
}

const MODEL_FILE: Asset = asset!("/assets/models/jina-bert.safetensors");
const TOKENIZER_FILE: Asset = asset!("/assets/models/jina-bert-tokenizer.json");

/// Configuration for the JinaBERT embedding model
#[derive(Clone)]
pub struct JinaBertConfig {
    pub model_id: String,
    pub normalize_embeddings: bool,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
}

impl Default for JinaBertConfig {
    fn default() -> Self {
        // Default config for jinaai/jina-embeddings-v2-small-en
        // ALiBi bias memory scales as heads * seq_len^2 * 4 bytes
        // At 2048 tokens: 8 heads * 2048^2 * 4 = ~128MB (fits in 4GB WASM memory)
        // Model supports up to 8192 tokens via ALiBi positional embeddings
        Self {
            model_id: "jinaai/jina-embeddings-v2-small-en".to_string(),
            normalize_embeddings: true,
            hidden_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_position_embeddings: 2048,
        }
    }
}

/// Apply L2 normalization to a tensor
pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

/// Embedding model wrapper
pub struct EmbeddingModel {
    model: BertModel,
    config: JinaBertConfig,
    device: Device,
}

impl EmbeddingModel {
    /// Create a new model from safetensors bytes and vocab size
    /// Takes ownership of model_bytes to avoid cloning in WASM
    pub fn from_bytes(
        model_bytes: Vec<u8>,
        vocab_size: usize,
        config: JinaBertConfig,
    ) -> Result<Self, String> {
        info!("📦 Loading embedding model '{}'", config.model_id.as_str());
        info!(
            "📊 Model bytes length: {} bytes ({:.2}MB)",
            model_bytes.len(),
            model_bytes.len() as f64 / 1_000_000.0
        );

        // Use CPU device for WASM
        let device = Device::cuda_if_available(0).unwrap();
        if device.is_cpu() {
            info!("✓ Initialized CPU device");
        } else {
            info!("✓ Initialized GPU device");
        }

        // Create model config for JinaBert
        info!(
            "⚙️  Config: {}d hidden, {} layers, {} heads",
            config.hidden_size, config.num_hidden_layers, config.num_attention_heads
        );

        let model_config = Config::new(
            vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.intermediate_size,
            Activation::Gelu,
            config.max_position_embeddings,
            2,     // type_vocab_size
            0.02,  // initializer_range
            1e-12, // layer_norm_eps
            0,     // pad_token_id
            PositionEmbeddingType::Alibi,
        );
        info!(
            "✓ Created model config (max positions: {})",
            config.max_position_embeddings
        );

        // Check safetensors header
        if model_bytes.len() < 8 {
            return Err("Model file too small".to_string());
        }
        let header_size = u64::from_le_bytes([
            model_bytes[0],
            model_bytes[1],
            model_bytes[2],
            model_bytes[3],
            model_bytes[4],
            model_bytes[5],
            model_bytes[6],
            model_bytes[7],
        ]);
        info!("📋 Safetensors header size: {} bytes", header_size);

        // Load model weights from bytes
        // Use F32 for WASM (converts F16 weights on load, following candle-wasm-examples pattern)
        // Pass ownership directly to avoid cloning 62MB in WASM
        info!("🔄 Loading VarBuilder from safetensors (converting to F32)...");
        let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device).map_err(
            |e| {
                let err_msg = format!("Failed to create VarBuilder: {}", e);
                error!("{}", err_msg);
                err_msg
            },
        )?;
        info!("✓ VarBuilder created successfully");

        info!("🔄 Creating BertModel...");
        let model = BertModel::new(vb, &model_config).map_err(|e| {
            let err_msg = format!("Failed to create BertModel: {}", e);
            error!("{}", err_msg);
            err_msg
        })?;
        info!("✓ BertModel created successfully");

        Ok(Self {
            model,
            config,
            device,
        })
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    /// Generate embedding from token IDs
    /// token_ids should be a 1D or 2D array of token IDs
    pub fn embed_tokens(&self, token_ids: Vec<u32>) -> Result<Vec<f32>, String> {
        // Convert token IDs to tensor
        let token_ids = Tensor::from_vec(token_ids.clone(), token_ids.len(), &self.device)
            .map_err(|e| format!("Failed to create tensor: {}", e))?
            .unsqueeze(0)
            .map_err(|e| format!("Failed to unsqueeze: {}", e))?;

        // Forward pass
        let embeddings = self
            .model
            .forward(&token_ids)
            .map_err(|e| format!("Forward pass failed: {}", e))?;

        // Apply mean pooling across tokens
        let (_n_sentence, n_tokens, _hidden_size) = embeddings
            .dims3()
            .map_err(|e| format!("Failed to get dims: {}", e))?;

        let embeddings = embeddings
            .sum(1)
            .map_err(|e| format!("Failed to sum: {}", e))?
            .affine(1.0 / n_tokens as f64, 0.0)
            .map_err(|e| format!("Failed to affine: {}", e))?;

        // Apply L2 normalization if enabled
        let embeddings = if self.config.normalize_embeddings {
            normalize_l2(&embeddings).map_err(|e| format!("Failed to normalize: {}", e))?
        } else {
            embeddings
        };

        // Convert to Vec<f32>
        let embeddings_vec = embeddings
            .squeeze(0)
            .map_err(|e| format!("Failed to squeeze: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert to vec: {}", e))?;

        Ok(embeddings_vec)
    }

    /// Generate embeddings for a batch of token sequences
    /// batch_token_ids should be a vector of token ID vectors
    #[allow(dead_code)]
    pub fn embed_batch_tokens(
        &self,
        batch_token_ids: Vec<Vec<u32>>,
    ) -> Result<Vec<Vec<f32>>, String> {
        if batch_token_ids.is_empty() {
            return Ok(vec![]);
        }

        // Find max length for padding
        let max_len = batch_token_ids
            .iter()
            .map(|ids| ids.len())
            .max()
            .unwrap_or(0);

        // Pad all sequences to max length
        let padded: Vec<Vec<u32>> = batch_token_ids
            .iter()
            .map(|ids| {
                let mut padded = ids.clone();
                padded.resize(max_len, 0); // Pad with 0
                padded
            })
            .collect();

        // Stack into 2D tensor
        let flat: Vec<u32> = padded.into_iter().flatten().collect();
        let batch_size = batch_token_ids.len();

        let token_ids = Tensor::from_vec(flat, (batch_size, max_len), &self.device)
            .map_err(|e| format!("Failed to create batch tensor: {}", e))?;

        // Forward pass
        let embeddings = self
            .model
            .forward(&token_ids)
            .map_err(|e| format!("Forward pass failed: {}", e))?;

        // Apply mean pooling across tokens
        let (_n_sentence, n_tokens, _hidden_size) = embeddings
            .dims3()
            .map_err(|e| format!("Failed to get dims: {}", e))?;

        let embeddings = embeddings
            .sum(1)
            .map_err(|e| format!("Failed to sum: {}", e))?
            .affine(1.0 / n_tokens as f64, 0.0)
            .map_err(|e| format!("Failed to affine: {}", e))?;

        // Apply L2 normalization if enabled
        let embeddings = if self.config.normalize_embeddings {
            normalize_l2(&embeddings).map_err(|e| format!("Failed to normalize: {}", e))?
        } else {
            embeddings
        };

        // Convert to Vec<Vec<f32>>
        let mut result = Vec::new();
        for i in 0..batch_size {
            let embedding = embeddings
                .get(i)
                .map_err(|e| format!("Failed to get embedding {}: {}", i, e))?
                .to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert embedding to vec: {}", e))?;
            result.push(embedding);
        }

        Ok(result)
    }
}

/// Compute cosine similarity between two embedding vectors
#[allow(dead_code)]
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

// WASM bindings for JavaScript
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmEmbeddingModel {
    model: EmbeddingModel,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmEmbeddingModel {
    /// Create a new model from model bytes
    /// vocab_size should match the tokenizer vocabulary size
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: Vec<u8>, vocab_size: usize) -> Result<WasmEmbeddingModel, JsValue> {
        console_error_panic_hook::set_once();

        let config = JinaBertConfig::default();
        let model = EmbeddingModel::from_bytes(model_bytes, vocab_size, config)
            .map_err(|e| JsValue::from_str(&e))?;

        Ok(WasmEmbeddingModel { model })
    }

    /// Create model with custom configuration
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

        let config = JinaBertConfig {
            model_id: "custom".to_string(),
            normalize_embeddings: true,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size: hidden_size * 4,
            max_position_embeddings,
        };

        let model = EmbeddingModel::from_bytes(model_bytes, vocab_size, config)
            .map_err(|e| JsValue::from_str(&e))?;

        Ok(WasmEmbeddingModel { model })
    }

    /// Generate embedding from token IDs
    /// Returns a Float32Array containing the embedding
    #[wasm_bindgen(js_name = embedTokens)]
    pub fn embed_tokens(&self, token_ids: Vec<u32>) -> Result<Vec<f32>, JsValue> {
        self.model
            .embed_tokens(token_ids)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Compute cosine similarity between two embeddings
    #[wasm_bindgen(js_name = cosineSimilarity)]
    pub fn cosine_similarity_js(e1: &[f32], e2: &[f32]) -> f32 {
        cosine_similarity(e1, e2)
    }
}

/// Fetch asset bytes from the server (web version)
#[cfg(target_arch = "wasm32")]
async fn fetch_asset_bytes(url: &str) -> Result<Vec<u8>, String> {
    use js_sys::{Function, Promise, Reflect};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let global = js_sys::global();
    let fetch_fn = Reflect::get(&global, &JsValue::from_str("fetch"))
        .map_err(|_| "fetch API unavailable".to_string())?
        .dyn_into::<Function>()
        .map_err(|_| "fetch is not callable".to_string())?;

    let resolved_url = resolve_asset_url(url);
    info!(
        "📥 Fetching asset (raw: {}, resolved: {})...",
        url, resolved_url
    );

    let promise = fetch_fn
        .call1(&global, &JsValue::from_str(&resolved_url))
        .map_err(|e| format!("Fetch call failed: {:?}", e))?;

    let resp_value = JsFuture::from(Promise::from(promise)).await.map_err(|e| {
        let err = format!("Fetch failed: {:?} (resolved: {})", e, resolved_url);
        error!("{}", err);
        err
    })?;

    info!("✓ Fetch completed");

    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| "Failed to cast to Response")?;

    if !resp.ok() {
        let err = format!(
            "HTTP {} fetching {} (raw path: {})",
            resp.status(),
            resolved_url,
            url
        );
        error!("{}", err);
        return Err(err);
    }

    info!("✓ Response OK, reading array buffer...");

    let array_buffer = JsFuture::from(
        resp.array_buffer()
            .map_err(|e| format!("Failed to get array buffer: {:?}", e))?,
    )
    .await
    .map_err(|e| {
        let err = format!("Failed to await array buffer: {:?}", e);
        error!("{}", err);
        err
    })?;

    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let bytes = uint8_array.to_vec();

    info!(
        "✓ Asset fetched successfully ({} bytes, {:.2}MB)",
        bytes.len(),
        bytes.len() as f64 / 1_000_000.0
    );

    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
fn resolve_asset_url(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || trimmed.starts_with("blob:")
    {
        return trimmed.to_string();
    }

    if let Some(resolved) = resolve_with_worker_base(trimmed) {
        return resolved;
    }

    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            match document.query_selector("meta[name=\"DIOXUS_ASSET_ROOT\"]") {
                Ok(Some(meta)) => {
                    if let Some(content) = meta.get_attribute("content") {
                        if let Some(resolved) = join_base_path(&content, trimmed) {
                            return resolved;
                        }
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    error!("Failed to read DIOXUS_ASSET_ROOT meta tag: {:?}", err);
                }
            }
        }
    }

    trimmed.to_string()
}

#[cfg(target_arch = "wasm32")]
fn join_base_path(base: &str, path: &str) -> Option<String> {
    if path.starts_with("http://") || path.starts_with("https://") || path.starts_with("blob:") {
        return None;
    }

    let normalized_base = if base == "/" {
        String::new()
    } else {
        base.trim_end_matches('/').to_string()
    };

    if normalized_base.is_empty() {
        return Some(format!("/{}", path.trim_start_matches('/')));
    }

    // Check if path already starts with the base path
    if path.starts_with(&normalized_base) {
        return Some(path.to_string());
    }

    if path.starts_with('/') {
        Some(format!("{}{}", normalized_base, path))
    } else {
        Some(format!("{}/{}", normalized_base, path))
    }
}

#[cfg(target_arch = "wasm32")]
fn resolve_with_worker_base(path: &str) -> Option<String> {
    use js_sys::Reflect;

    let global = js_sys::global();
    let base_value = Reflect::get(&global, &JsValue::from_str("__COPPERMIND_ASSET_BASE")).ok()?;
    let base = base_value.as_string()?;
    join_base_path(&base, path)
}

/// Read asset bytes from filesystem (desktop version)
#[cfg(not(target_arch = "wasm32"))]
async fn fetch_asset_bytes(asset_path: &str) -> Result<Vec<u8>, String> {
    info!("📥 Reading asset from {}...", asset_path);

    // On desktop, Dioxus bundles assets into the app
    // The asset! macro returns a manganis URL, but we need the actual file path
    // Let's just read from the source assets directory directly
    use std::path::PathBuf;

    // Get the current executable directory
    let exe_path = std::env::current_exe().map_err(|e| format!("Failed to get exe path: {}", e))?;
    let exe_dir = exe_path
        .parent()
        .ok_or("Failed to get exe parent directory")?;

    info!("📂 Executable directory: {:?}", exe_dir);
    info!("📂 Current directory: {:?}", std::env::current_dir());

    // The bundled assets are typically in the Resources folder on macOS
    // Or in the same directory as the executable on other platforms
    let asset_locations = vec![
        // Try ../Resources/assets/ (macOS app bundle)
        exe_dir.join("..").join("Resources").join("assets"),
        // Try ./assets (same directory as exe)
        exe_dir.join("assets"),
        // Try current working directory
        PathBuf::from("assets"),
    ];

    // Extract just the filename from the asset path (it has a hash like jina-bert-dxhXXX.safetensors)
    let filename = asset_path
        .trim_start_matches('/')
        .trim_start_matches("assets/");

    for base_dir in &asset_locations {
        let full_path = base_dir.join(filename);
        info!("  Trying: {:?}", full_path);

        if let Ok(bytes) = tokio::fs::read(&full_path).await {
            info!(
                "✓ Asset loaded from {:?}: {:.2}MB ({} bytes)",
                full_path,
                bytes.len() as f64 / 1_000_000.0,
                bytes.len()
            );
            return Ok(bytes);
        }
    }

    let err = format!(
        "Failed to find asset {} in any of the expected locations",
        asset_path
    );
    error!("{}", err);
    Err(err)
}

static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();
static MODEL_CACHE: OnceCell<Arc<EmbeddingModel>> = OnceCell::new();

/// Download and initialize the tokenizer once per session
pub async fn ensure_tokenizer(max_positions: usize) -> Result<&'static Tokenizer, String> {
    if let Some(tokenizer) = TOKENIZER.get() {
        return Ok(tokenizer);
    }

    let tokenizer_url = TOKENIZER_FILE.to_string();
    info!("📚 Tokenizer URL: {}", tokenizer_url);
    let tokenizer_bytes = fetch_asset_bytes(&tokenizer_url).await?;

    let mut tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|e| format!("Failed to deserialize tokenizer: {}", e))?;

    configure_tokenizer(&mut tokenizer, max_positions)?;
    TOKENIZER
        .set(tokenizer)
        .map_err(|_| "Tokenizer already initialized".to_string())?;

    TOKENIZER
        .get()
        .ok_or_else(|| "Tokenizer unavailable after initialization".to_string())
}

fn configure_tokenizer(tokenizer: &mut Tokenizer, max_positions: usize) -> Result<(), String> {
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_positions,
            stride: 0,
            strategy: TruncationStrategy::OnlyFirst,
            direction: TruncationDirection::Right,
        }))
        .map_err(|e| format!("Failed to configure tokenizer truncation: {}", e))?;

    Ok(())
}

fn tokenize_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| format!("Tokenization failed: {}", e))?;

    let ids = encoding.get_ids();
    if ids.is_empty() {
        return Err("Tokenizer returned no tokens".to_string());
    }

    Ok(ids.to_vec())
}

#[derive(Clone)]
pub struct ChunkEmbeddingResult {
    pub chunk_index: usize,
    pub token_count: usize,
    pub embedding: Vec<f32>,
}

/// Embedding payload returned from worker-friendly APIs
#[derive(Clone)]
pub struct EmbeddingComputation {
    pub token_count: usize,
    pub embedding: Vec<f32>,
}

pub async fn get_or_load_model() -> Result<Arc<EmbeddingModel>, String> {
    if let Some(existing) = MODEL_CACHE.get() {
        return Ok(existing.clone());
    }

    let model_url = MODEL_FILE.to_string();
    info!("📦 Loading embedding model (cold start)...");
    let model_bytes = fetch_asset_bytes(&model_url).await?;
    let config = JinaBertConfig::default();
    let model = EmbeddingModel::from_bytes(model_bytes, 30528, config)?;
    let model = Arc::new(model);

    // Try to set the model in the cache (may fail if another thread beat us to it)
    let _ = MODEL_CACHE.set(model.clone());

    // Return the cached model (in case another thread set it first)
    Ok(MODEL_CACHE.get().unwrap().clone())
}

pub fn tokenize_into_chunks(
    tokenizer: &Tokenizer,
    text: &str,
    max_tokens: usize,
) -> Result<Vec<Vec<u32>>, String> {
    if max_tokens < 2 {
        return Err("chunk size must be at least 2 tokens to account for special tokens".into());
    }

    let mut chunk_tokenizer = tokenizer.clone();
    chunk_tokenizer
        .with_truncation(None)
        .map_err(|e| format!("Failed to disable truncation: {}", e))?;
    let encoding = chunk_tokenizer
        .encode(text, false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let raw_ids: Vec<u32> = encoding.get_ids().to_vec();

    if raw_ids.is_empty() {
        return Ok(vec![]);
    }

    let cls_id = tokenizer
        .token_to_id("[CLS]")
        .ok_or_else(|| "Tokenizer missing [CLS] token".to_string())?;
    let sep_id = tokenizer
        .token_to_id("[SEP]")
        .ok_or_else(|| "Tokenizer missing [SEP] token".to_string())?;

    let body_len = max_tokens - 2;
    let mut chunks = Vec::new();

    for body in raw_ids.chunks(body_len) {
        let mut ids = Vec::with_capacity(body.len() + 2);
        ids.push(cls_id);
        ids.extend_from_slice(body);
        ids.push(sep_id);
        chunks.push(ids);
    }

    Ok(chunks)
}

/// Embed text by chunking it into smaller pieces and processing in batches
///
/// On WASM targets, processes chunks in batches of 10 with periodic yields
/// to keep the UI responsive during large embedding operations.
///
/// # Arguments
/// * `text` - The text to embed
/// * `chunk_tokens` - Maximum tokens per chunk (will be capped at model's max_position_embeddings)
///
/// # Returns
/// Vector of chunk embedding results, each containing the chunk index, token count, and embedding
pub async fn embed_text_chunks(
    text: &str,
    chunk_tokens: usize,
) -> Result<Vec<ChunkEmbeddingResult>, String> {
    let model = get_or_load_model().await?;
    let max_positions = model.max_position_embeddings();
    let tokenizer = ensure_tokenizer(max_positions).await?;

    let effective_chunk = chunk_tokens.min(max_positions);
    let token_chunks = tokenize_into_chunks(tokenizer, text, effective_chunk)?;

    if token_chunks.is_empty() {
        return Ok(vec![]);
    }

    // Store token counts before processing
    let token_counts: Vec<usize> = token_chunks.iter().map(|ids| ids.len()).collect();

    info!(
        "🧩 Embedding {} chunks ({} tokens max per chunk)",
        token_chunks.len(),
        effective_chunk
    );

    // Process one chunk at a time with yields to keep UI responsive
    const BATCH_SIZE: usize = 1;
    let mut all_embeddings = Vec::with_capacity(token_chunks.len());

    for (batch_idx, chunk_batch) in token_chunks.chunks(BATCH_SIZE).enumerate() {
        let batch_start = batch_idx * BATCH_SIZE;
        let batch_end = batch_start + chunk_batch.len();

        info!(
            "🚀 Processing batch {}/{} (chunks {}-{})",
            batch_idx + 1,
            token_chunks.len().div_ceil(BATCH_SIZE),
            batch_start,
            batch_end - 1
        );

        // Process this batch together for maximum performance
        let batch_embeddings = model
            .embed_batch_tokens(chunk_batch.to_vec())
            .map_err(|e| format!("Batch embedding failed: {}", e))?;

        all_embeddings.extend(batch_embeddings);

        info!(
            "✅ Batch {}/{} complete",
            batch_idx + 1,
            token_chunks.len().div_ceil(BATCH_SIZE)
        );

        // Yield to event loop between batches on WASM to keep UI responsive
        #[cfg(target_arch = "wasm32")]
        if batch_end < token_chunks.len() {
            use gloo_timers::future::TimeoutFuture;
            TimeoutFuture::new(0).await;
        }
    }

    // Build results with stored token counts
    let results = all_embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| ChunkEmbeddingResult {
            chunk_index: index,
            token_count: token_counts[index],
            embedding,
        })
        .collect();

    Ok(results)
}

/// Main embedding function - loads model and generates embeddings
pub async fn run_embedding(text: &str) -> Result<String, String> {
    info!("🔮 Generating embedding for: '{}'", text);
    let computation = compute_embedding(text).await?;
    Ok(format_embedding_summary(
        text,
        computation.token_count,
        &computation.embedding,
    ))
}

/// Produce the raw embedding vector (used by Web Worker and native flows)
pub async fn compute_embedding(text: &str) -> Result<EmbeddingComputation, String> {
    let model = get_or_load_model().await?;
    let max_positions = model.max_position_embeddings();
    let tokenizer = ensure_tokenizer(max_positions).await?;
    let token_ids = tokenize_text(tokenizer, text)?;
    let token_count = token_ids.len();
    info!("🧾 Tokenized into {} tokens", token_count);

    info!("Generating embedding vector...");

    let embedding = model
        .embed_tokens(token_ids)
        .map_err(|e| format!("Embedding failed: {}", e))?;

    info!("✓ Generated {}-dimensional embedding", embedding.len());

    Ok(EmbeddingComputation {
        token_count,
        embedding,
    })
}

/// Format the embedding details for UI display
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

/// Example function to demonstrate usage from Rust with actual model
/// Note: Requires model bytes and tokenization
#[allow(dead_code)]
pub async fn run_embedding_example(
    model_bytes: Vec<u8>,
    token_ids: Vec<u32>,
) -> Result<Vec<f32>, String> {
    info!("Loading JinaBert model...");

    let config = JinaBertConfig::default();
    let model = EmbeddingModel::from_bytes(model_bytes, 30522, config)?;

    info!("Generating embedding...");
    let embedding = model.embed_tokens(token_ids)?;

    info!("Embedding dimension: {}", embedding.len());

    Ok(embedding)
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
    fn test_normalize_l2() {
        let device = Device::Cpu;
        let v = Tensor::from_vec(vec![3.0f32, 4.0f32], (1, 2), &device).unwrap();
        let normalized = normalize_l2(&v).unwrap();
        let result = normalized.to_vec2::<f32>().unwrap();

        // 3,4 normalized should be 0.6, 0.8
        assert!((result[0][0] - 0.6).abs() < 1e-6);
        assert!((result[0][1] - 0.8).abs() < 1e-6);
    }
}
