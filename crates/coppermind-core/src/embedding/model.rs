//! Embedding model implementations and inference.
//!
//! This module provides the JinaBERT embedding model implementation using
//! the Candle ML framework. The trait-based design allows adding new models
//! in the future with different architectures.

use super::config::JinaBertConfig;
use super::traits::Embedder;
use crate::error::EmbeddingError;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::jina_bert::{BertModel, Config, PositionEmbeddingType};
use tracing::info;

/// JinaBERT embedding model implementation.
///
/// JinaBERT is a BERT-based encoder model using ALiBi positional embeddings,
/// designed for semantic similarity tasks. This implementation uses Candle
/// for inference.
///
/// # Architecture
///
/// - Hidden size: 512 (default)
/// - Layers: 4 (default)
/// - Attention heads: 8 (default)
/// - Position embeddings: ALiBi (supports up to 8192 tokens)
/// - Normalization: L2 (unit vector)
///
/// # Examples
///
/// ```ignore
/// let model_bytes = load_model_bytes()?;
/// let config = JinaBertConfig::default();
/// let model = JinaBertEmbedder::from_bytes(model_bytes, 30528, config)?;
///
/// let token_ids = vec![101, 2023, 2003, 1037, 3231, 102]; // [CLS] this is a test [SEP]
/// let embedding = model.embed_tokens(token_ids)?;
/// assert_eq!(embedding.len(), 512);
/// ```
pub struct JinaBertEmbedder {
    model: BertModel,
    config: JinaBertConfig,
    device: Device,
}

impl JinaBertEmbedder {
    /// Creates a new model from safetensors bytes.
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Safetensors-format model weights
    /// * `vocab_size` - Size of tokenizer vocabulary
    /// * `config` - Model configuration
    ///
    /// # Returns
    ///
    /// Initialized model ready for inference.
    ///
    /// # Platform-Specific Behavior
    ///
    /// - **Desktop**: Attempts CUDA -> Metal -> CPU (with MKL/Accelerate)
    /// - **Web**: CPU only (WebGPU not yet supported in Candle)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::ModelLoad` if initialization fails.
    pub fn from_bytes(
        model_bytes: Vec<u8>,
        vocab_size: usize,
        config: JinaBertConfig,
    ) -> Result<Self, EmbeddingError> {
        info!("Loading embedding model '{}'", config.model_id);
        info!(
            "Model bytes length: {} bytes ({:.2}MB)",
            model_bytes.len(),
            model_bytes.len() as f64 / 1_000_000.0
        );

        let device = Self::select_device();
        let model = Self::create_model(model_bytes, vocab_size, &config, &device)?;

        Ok(Self {
            model,
            config,
            device,
        })
    }

    /// Creates a model from pre-built parts.
    ///
    /// Used by the GPU scheduler to create a model on its dedicated thread
    /// with an explicit device, rather than auto-detecting.
    ///
    /// # Arguments
    ///
    /// * `model` - Pre-built BertModel
    /// * `config` - Model configuration
    /// * `device` - Compute device (owned by caller)
    pub fn from_parts(model: BertModel, config: JinaBertConfig, device: Device) -> Self {
        Self {
            model,
            config,
            device,
        }
    }

    /// Returns a reference to the config.
    pub fn config(&self) -> &JinaBertConfig {
        &self.config
    }

    /// Returns a reference to the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Selects the best available compute device.
    pub fn select_device() -> Device {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Desktop: Try CUDA -> Metal -> CPU (with MKL/Accelerate)
            if let Ok(cuda_device) = Device::new_cuda(0) {
                #[cfg(feature = "candle-cudnn")]
                info!("Using CUDA GPU (with cuDNN)");
                #[cfg(not(feature = "candle-cudnn"))]
                info!("Using CUDA GPU");
                return cuda_device;
            }

            if let Ok(metal_device) = Device::new_metal(0) {
                info!("Using Metal GPU");
                return metal_device;
            }

            // CPU fallback with platform-specific optimizations
            #[cfg(all(
                not(any(target_os = "macos", target_os = "ios")),
                any(target_arch = "x86_64", target_arch = "x86")
            ))]
            info!("Using CPU (with Intel MKL)");

            #[cfg(any(target_os = "macos", target_os = "ios"))]
            info!("Using CPU (with Accelerate)");

            #[cfg(all(
                not(any(target_os = "macos", target_os = "ios")),
                not(any(target_arch = "x86_64", target_arch = "x86"))
            ))]
            info!("Using CPU");

            Device::Cpu
        }

        #[cfg(target_arch = "wasm32")]
        {
            // WASM: CPU only (WebGPU not yet supported in Candle)
            info!("Using CPU (WASM)");
            Device::Cpu
        }
    }

    /// Selects the data type for model weights.
    ///
    /// Currently uses F32 for all devices. The safetensors file stores weights as F16,
    /// which are converted to F32 at load time.
    ///
    /// # Why F32?
    ///
    /// Candle's `jina_bert.rs` hardcodes F32 for ALiBi positional bias:
    /// - `pub const DTYPE: DType = DType::F32;` (line 9)
    /// - `alibi_bias.to_dtype(DType::F32)?` in `build_alibi_bias()` (line 270)
    ///
    /// Loading model weights as F16 causes dtype mismatch errors:
    /// `dtype mismatch in add, lhs: F16, rhs: F32`
    ///
    /// This is an upstream limitation in candle-transformers, not fixable without
    /// forking. See: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/jina_bert.rs
    ///
    /// # Performance Impact
    ///
    /// The F16→F32 conversion happens at load time (once), not inference time.
    /// GPU acceleration (Metal/CUDA) still provides significant speedups for
    /// the matrix operations during inference.
    ///
    /// # Future Optimization
    ///
    /// If candle-transformers adds dtype flexibility, we could use:
    /// - **Metal GPU**: F16 (faster than BF16 on Apple Silicon)
    /// - **CUDA GPU**: BF16 (native support on modern NVIDIA GPUs)
    pub fn select_dtype(_device: &Device) -> DType {
        DType::F32
    }

    /// Creates the BertModel from bytes and configuration.
    pub fn create_model(
        model_bytes: Vec<u8>,
        vocab_size: usize,
        config: &JinaBertConfig,
        device: &Device,
    ) -> Result<BertModel, EmbeddingError> {
        info!(
            "Config: {}d hidden, {} layers, {} heads",
            config.hidden_size, config.num_hidden_layers, config.num_attention_heads
        );

        // Create Candle config for JinaBERT
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
            "Created model config (max positions: {})",
            config.max_position_embeddings
        );

        // Validate safetensors header
        if model_bytes.len() < 8 {
            return Err(EmbeddingError::ModelLoad(
                "Model file too small".to_string(),
            ));
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
        info!("Safetensors header size: {} bytes", header_size);

        // Select optimal dtype for the device
        let dtype = Self::select_dtype(device);

        // Load model weights with device-appropriate precision
        info!("Loading VarBuilder from safetensors...");
        let vb =
            VarBuilder::from_buffered_safetensors(model_bytes, dtype, device).map_err(|e| {
                EmbeddingError::ModelLoad(format!("Failed to create VarBuilder: {}", e))
            })?;
        info!("VarBuilder created successfully");

        info!("Creating BertModel...");
        let model = BertModel::new(vb, &model_config)
            .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to create BertModel: {}", e)))?;
        info!("BertModel created successfully");

        Ok(model)
    }

    /// Applies mean pooling across token dimension.
    fn mean_pool(embeddings: &Tensor, n_tokens: usize) -> Result<Tensor, EmbeddingError> {
        embeddings
            .sum(1)
            .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to sum: {}", e)))?
            .affine(1.0 / n_tokens as f64, 0.0)
            .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to affine: {}", e)))
    }

    /// Applies L2 normalization to create unit vectors.
    fn normalize_l2(v: &Tensor) -> Result<Tensor, EmbeddingError> {
        v.broadcast_div(
            &v.sqr()
                .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to square: {}", e)))?
                .sum_keepdim(1)
                .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to sum: {}", e)))?
                .sqrt()
                .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to sqrt: {}", e)))?,
        )
        .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to normalize: {}", e)))
    }
}

impl Embedder for JinaBertEmbedder {
    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn embedding_dim(&self) -> usize {
        self.config.hidden_size
    }

    fn embed_tokens(&self, token_ids: Vec<u32>) -> Result<Vec<f32>, EmbeddingError> {
        // Convert token IDs to tensor [1, seq_len]
        let token_ids_tensor = Tensor::from_vec(token_ids.clone(), token_ids.len(), &self.device)
            .map_err(|e| EmbeddingError::TensorCreation(format!("Failed to create tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| EmbeddingError::TensorCreation(format!("Failed to unsqueeze: {}", e)))?;

        // Forward pass: [1, seq_len] -> [1, seq_len, hidden_size]
        let embeddings = self
            .model
            .forward(&token_ids_tensor)
            .map_err(|e| EmbeddingError::InferenceFailed(format!("Forward pass failed: {}", e)))?;

        // Mean pooling: [1, seq_len, hidden_size] -> [1, hidden_size]
        let (_n_sentence, n_tokens, _hidden_size) = embeddings
            .dims3()
            .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to get dims: {}", e)))?;

        let pooled = Self::mean_pool(&embeddings, n_tokens)?;

        // L2 normalization (if enabled)
        let normalized = if self.config.normalize_embeddings {
            Self::normalize_l2(&pooled)?
        } else {
            pooled
        };

        // Convert to Vec<f32>
        let embeddings_vec = normalized
            .squeeze(0)
            .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to squeeze: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| {
                EmbeddingError::InferenceFailed(format!("Failed to convert to vec: {}", e))
            })?;

        Ok(embeddings_vec)
    }

    fn embed_batch_tokens(
        &self,
        batch_token_ids: Vec<Vec<u32>>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
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

        // Stack into 2D tensor [batch_size, seq_len]
        let flat: Vec<u32> = padded.into_iter().flatten().collect();
        let batch_size = batch_token_ids.len();

        let token_ids_tensor = Tensor::from_vec(flat, (batch_size, max_len), &self.device)
            .map_err(|e| {
                EmbeddingError::TensorCreation(format!("Failed to create batch tensor: {}", e))
            })?;

        // Forward pass: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        let embeddings = self
            .model
            .forward(&token_ids_tensor)
            .map_err(|e| EmbeddingError::InferenceFailed(format!("Forward pass failed: {}", e)))?;

        // Mean pooling: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        let (_n_sentence, n_tokens, _hidden_size) = embeddings
            .dims3()
            .map_err(|e| EmbeddingError::InferenceFailed(format!("Failed to get dims: {}", e)))?;

        let pooled = Self::mean_pool(&embeddings, n_tokens)?;

        // L2 normalization (if enabled)
        let normalized = if self.config.normalize_embeddings {
            Self::normalize_l2(&pooled)?
        } else {
            pooled
        };

        // Convert to Vec<Vec<f32>>
        let mut result = Vec::new();
        for i in 0..batch_size {
            let embedding = normalized
                .get(i)
                .map_err(|e| {
                    EmbeddingError::InferenceFailed(format!("Failed to get embedding {}: {}", i, e))
                })?
                .to_vec1::<f32>()
                .map_err(|e| {
                    EmbeddingError::InferenceFailed(format!(
                        "Failed to convert embedding to vec: {}",
                        e
                    ))
                })?;
            result.push(embedding);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_invalid_bytes() {
        let config = JinaBertConfig::default();
        let result = JinaBertEmbedder::from_bytes(vec![1, 2, 3], 30528, config);
        assert!(result.is_err());
        assert!(matches!(result, Err(EmbeddingError::ModelLoad(_))));
    }
}
