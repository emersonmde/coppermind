use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::jina_bert::{BertModel, Config, PositionEmbeddingType};
use dioxus::prelude::*;
use futures_channel::mpsc;
use once_cell::sync::OnceCell;
use std::cell::RefCell;
use std::rc::Rc;
use tokenizers::tokenizer::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};
use wasm_bindgen::prelude::*;

const MODEL_FILE: Asset = asset!("/assets/models/jina-bert.safetensors");
const TOKENIZER_FILE: Asset = asset!("/assets/models/jina-bert-tokenizer.json");

/// Configuration for the JinaBert embedding model
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
        // Limit max_position_embeddings so WASM does not allocate multi-GB ALiBi tensors
        // (ALiBi bias size is heads * seq_len^2). 1024 keeps memory < ~32MB for 8 heads.
        Self {
            model_id: "jinaai/jina-embeddings-v2-small-en".to_string(),
            normalize_embeddings: true,
            hidden_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_position_embeddings: 1024,
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
        web_sys::console::log_1(
            &format!("📦 Loading embedding model '{}'", config.model_id.as_str()).into(),
        );
        web_sys::console::log_1(
            &format!(
                "📊 Model bytes length: {} bytes ({:.2}MB)",
                model_bytes.len(),
                model_bytes.len() as f64 / 1_000_000.0
            )
            .into(),
        );

        // Use CPU device for WASM
        let device = Device::cuda_if_available(0).unwrap();
        if device.is_cpu() {
            web_sys::console::log_1(&"✓ Initialized CPU device".into());
        } else {
            web_sys::console::log_1(&"✓ Initialized GPU device".into());
        }

        // Create model config for JinaBert
        web_sys::console::log_1(
            &format!(
                "⚙️  Config: {}d hidden, {} layers, {} heads",
                config.hidden_size, config.num_hidden_layers, config.num_attention_heads
            )
            .into(),
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
        web_sys::console::log_1(
            &format!(
                "✓ Created model config (max positions: {})",
                config.max_position_embeddings
            )
            .into(),
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
        web_sys::console::log_1(
            &format!("📋 Safetensors header size: {} bytes", header_size).into(),
        );

        // Load model weights from bytes
        // Use F32 for WASM (converts F16 weights on load, following candle-wasm-examples pattern)
        // Pass ownership directly to avoid cloning 62MB in WASM
        web_sys::console::log_1(
            &"🔄 Loading VarBuilder from safetensors (converting to F32)...".into(),
        );
        let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device).map_err(
            |e| {
                let err_msg = format!("Failed to create VarBuilder: {}", e);
                web_sys::console::error_1(&err_msg.clone().into());
                err_msg
            },
        )?;
        web_sys::console::log_1(&"✓ VarBuilder created successfully".into());

        web_sys::console::log_1(&"🔄 Creating BertModel...".into());
        let model = BertModel::new(vb, &model_config).map_err(|e| {
            let err_msg = format!("Failed to create BertModel: {}", e);
            web_sys::console::error_1(&err_msg.clone().into());
            err_msg
        })?;
        web_sys::console::log_1(&"✓ BertModel created successfully".into());

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
#[wasm_bindgen]
pub struct WasmEmbeddingModel {
    model: EmbeddingModel,
}

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

/// Fetch asset bytes from the server
/// Used for both the model weights and tokenizer JSON
async fn fetch_asset_bytes(url: &str) -> Result<Vec<u8>, String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().ok_or("No window object")?;

    web_sys::console::log_1(&format!("📥 Fetching model from {}...", url).into());

    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| {
            let err = format!("Fetch failed: {:?}", e);
            web_sys::console::error_1(&err.clone().into());
            err
        })?;

    web_sys::console::log_1(&"✓ Fetch completed".into());

    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| "Failed to cast to Response")?;

    if !resp.ok() {
        let err = format!("HTTP {} fetching model", resp.status());
        web_sys::console::error_1(&err.clone().into());
        return Err(err);
    }

    web_sys::console::log_1(&"✓ Response OK, reading array buffer...".into());

    let array_buffer = JsFuture::from(
        resp.array_buffer()
            .map_err(|e| format!("Failed to get array buffer: {:?}", e))?,
    )
    .await
    .map_err(|e| {
        let err = format!("Failed to await array buffer: {:?}", e);
        web_sys::console::error_1(&err.clone().into());
        err
    })?;

    web_sys::console::log_1(&"✓ Array buffer received, converting to bytes...".into());

    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let bytes = uint8_array.to_vec();

    web_sys::console::log_1(
        &format!(
            "✓ Model downloaded: {:.2}MB ({} bytes)",
            bytes.len() as f64 / 1_000_000.0,
            bytes.len()
        )
        .into(),
    );

    Ok(bytes)
}

static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();
thread_local! {
    static MODEL_CACHE: RefCell<Option<Rc<EmbeddingModel>>> = const { RefCell::new(None) };
}

/// Download and initialize the tokenizer once per session
async fn ensure_tokenizer(max_positions: usize) -> Result<&'static Tokenizer, String> {
    if let Some(tokenizer) = TOKENIZER.get() {
        return Ok(tokenizer);
    }

    let tokenizer_url = TOKENIZER_FILE.to_string();
    web_sys::console::log_1(&format!("📚 Tokenizer URL: {}", tokenizer_url).into());
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

async fn get_or_load_model() -> Result<Rc<EmbeddingModel>, String> {
    if let Some(existing) = MODEL_CACHE.with(|cell| cell.borrow().clone()) {
        return Ok(existing);
    }

    let model_url = MODEL_FILE.to_string();
    web_sys::console::log_1(&"📦 Loading embedding model (cold start)...".into());
    let model_bytes = fetch_asset_bytes(&model_url).await?;
    let config = JinaBertConfig::default();
    let model = EmbeddingModel::from_bytes(model_bytes, 30528, config)?;
    let model = Rc::new(model);
    MODEL_CACHE.with(|cell| {
        cell.borrow_mut().replace(model.clone());
    });
    Ok(model)
}

fn tokenize_into_chunks(
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

    web_sys::console::log_1(
        &format!(
            "🧩 Embedding {} chunks ({} tokens max per chunk)",
            token_chunks.len(),
            effective_chunk
        )
        .into(),
    );

    let mut results = Vec::with_capacity(token_chunks.len());
    for (index, ids) in token_chunks.into_iter().enumerate() {
        let tokens = ids.len();
        web_sys::console::log_1(
            &format!("🚀 Embedding chunk {} ({} tokens)", index, tokens).into(),
        );
        let embedding = model
            .embed_tokens(ids)
            .map_err(|e| format!("Embedding chunk {} failed: {}", index, e))?;
        web_sys::console::log_1(&format!("✅ Chunk {} complete ({} tokens)", index, tokens).into());
        results.push(ChunkEmbeddingResult {
            chunk_index: index,
            token_count: tokens,
            embedding,
        });
    }

    Ok(results)
}

/// Process text chunks with streaming results to avoid UI blocking
/// Returns a receiver that yields chunks as they are processed
pub async fn embed_text_chunks_streaming(
    text: String,
    chunk_tokens: usize,
) -> Result<mpsc::UnboundedReceiver<Result<ChunkEmbeddingResult, String>>, String> {
    let model = get_or_load_model().await?;
    let max_positions = model.max_position_embeddings();
    let tokenizer = ensure_tokenizer(max_positions).await?;

    let effective_chunk = chunk_tokens.min(max_positions);
    let token_chunks = tokenize_into_chunks(tokenizer, &text, effective_chunk)?;

    if token_chunks.is_empty() {
        let (tx, rx) = mpsc::unbounded();
        tx.close_channel();
        return Ok(rx);
    }

    let chunk_count = token_chunks.len();
    web_sys::console::log_1(
        &format!(
            "🧩 Embedding {} chunks ({} tokens max per chunk)",
            chunk_count, effective_chunk
        )
        .into(),
    );

    let (tx, rx) = mpsc::unbounded();

    // Spawn a task to process chunks one at a time
    spawn(async move {
        for (index, ids) in token_chunks.into_iter().enumerate() {
            let tokens = ids.len();
            web_sys::console::log_1(
                &format!("🚀 Embedding chunk {} ({} tokens)", index, tokens).into(),
            );

            let result = model
                .embed_tokens(ids)
                .map(|embedding| {
                    web_sys::console::log_1(
                        &format!("✅ Chunk {} complete ({} tokens)", index, tokens).into(),
                    );
                    ChunkEmbeddingResult {
                        chunk_index: index,
                        token_count: tokens,
                        embedding,
                    }
                })
                .map_err(|e| format!("Embedding chunk {} failed: {}", index, e));

            // Send the result through the channel
            if tx.unbounded_send(result).is_err() {
                web_sys::console::error_1(
                    &"❌ Failed to send chunk result (receiver dropped)".into(),
                );
                break;
            }

            // Yield control to event loop after each chunk to prevent UI blocking
            // This is crucial for maintaining UI responsiveness
            gloo_timers::future::TimeoutFuture::new(0).await;
        }

        // Close the channel when done
        tx.close_channel();
    });

    Ok(rx)
}

/// Main embedding function - loads model and generates embeddings
pub async fn run_embedding(text: &str) -> Result<String, String> {
    web_sys::console::log_1(&format!("🔮 Generating embedding for: '{}'", text).into());

    let model = get_or_load_model().await?;
    let tokenizer = ensure_tokenizer(model.max_position_embeddings()).await?;
    let token_ids = tokenize_text(tokenizer, text)?;
    let token_count = token_ids.len();
    web_sys::console::log_1(&format!("🧾 Tokenized into {} tokens", token_count).into());

    web_sys::console::log_1(&"Generating embedding vector...".into());

    let embedding = model
        .embed_tokens(token_ids)
        .map_err(|e| format!("Embedding failed: {}", e))?;

    web_sys::console::log_1(
        &format!("✓ Generated {}-dimensional embedding", embedding.len()).into(),
    );

    Ok(format!(
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
        embedding[0],
        embedding[1],
        embedding[2],
        embedding[3],
        embedding[4],
        embedding[5],
        embedding[6],
        embedding[7],
        embedding[8],
        embedding[9]
    ))
}

/// Example function to demonstrate usage from Rust with actual model
/// Note: Requires model bytes and tokenization
#[allow(dead_code)]
pub async fn run_embedding_example(
    model_bytes: Vec<u8>,
    token_ids: Vec<u32>,
) -> Result<Vec<f32>, String> {
    web_sys::console::log_1(&"Loading JinaBert model...".into());

    let config = JinaBertConfig::default();
    let model = EmbeddingModel::from_bytes(model_bytes, 30522, config)?;

    web_sys::console::log_1(&"Generating embedding...".into());
    let embedding = model.embed_tokens(token_ids)?;

    web_sys::console::log_1(&format!("Embedding dimension: {}", embedding.len()).into());

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
