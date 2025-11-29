//! Configuration for embedding models.
//!
//! This module defines configuration structures for various embedding models.
//! The design supports multiple model types with different architectures and parameters.

use super::traits::ModelConfig;
use serde::{Deserialize, Serialize};

/// Configuration for JinaBERT embedding models.
///
/// JinaBERT uses ALiBi (Attention with Linear Biases) positional embeddings,
/// which allows extrapolation beyond the training sequence length.
///
/// # Memory Considerations
///
/// ALiBi bias memory scales as `heads * seq_len^2 * 4 bytes`:
/// - At 1024 tokens: 8 heads * 1024^2 * 4 = ~32MB
/// - At 2048 tokens: 8 heads * 2048^2 * 4 = ~128MB
/// - At 8192 tokens: 8 heads * 8192^2 * 4 = ~2GB
///
/// The model supports up to 8192 tokens, but WASM memory limits may require
/// lower values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JinaBertConfig {
    /// Model identifier (e.g., "jinaai/jina-embeddings-v2-small-en")
    pub model_id: String,

    /// Whether to apply L2 normalization to embeddings
    pub normalize_embeddings: bool,

    /// Hidden dimension size (embedding output dimension)
    pub hidden_size: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads per layer
    pub num_attention_heads: usize,

    /// Intermediate (FFN) dimension size
    pub intermediate_size: usize,

    /// Maximum position embeddings (sequence length limit)
    ///
    /// This should be set based on available memory:
    /// - Web (WASM): 512-2048 tokens (limited by 512MB WASM memory)
    /// - Desktop: 2048-8192 tokens (depending on available RAM)
    pub max_position_embeddings: usize,
}

impl Default for JinaBertConfig {
    fn default() -> Self {
        // Default config for jinaai/jina-embeddings-v2-small-en
        // Conservative max_position_embeddings for WASM compatibility
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

impl ModelConfig for JinaBertConfig {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embedding_dim(&self) -> usize {
        self.hidden_size
    }

    fn max_sequence_length(&self) -> usize {
        self.max_position_embeddings
    }

    fn normalize_embeddings(&self) -> bool {
        self.normalize_embeddings
    }
}

impl JinaBertConfig {
    /// Creates a new configuration with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier string
    /// * `hidden_size` - Embedding dimension (typically 512 for small models)
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads per layer
    /// * `max_positions` - Maximum sequence length
    pub fn new(
        model_id: String,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        max_positions: usize,
    ) -> Self {
        Self {
            model_id,
            normalize_embeddings: true,
            hidden_size,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            intermediate_size: hidden_size * 4, // Standard transformer ratio
            max_position_embeddings: max_positions,
        }
    }

    /// Estimates the ALiBi bias memory requirement in bytes.
    ///
    /// Formula: `heads * seq_len^2 * 4 bytes`
    pub fn estimate_alibi_memory_bytes(&self) -> usize {
        self.num_attention_heads * self.max_position_embeddings * self.max_position_embeddings * 4
    }

    /// Estimates the ALiBi bias memory requirement in megabytes.
    pub fn estimate_alibi_memory_mb(&self) -> f64 {
        self.estimate_alibi_memory_bytes() as f64 / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = JinaBertConfig::default();
        assert_eq!(config.embedding_dim(), 512);
        assert_eq!(config.max_sequence_length(), 2048);
        assert!(config.normalize_embeddings());
    }

    #[test]
    fn test_alibi_memory_calculation() {
        let config = JinaBertConfig::default();
        // 8 heads * 2048^2 * 4 bytes = 8 * 4194304 * 4 = 134,217,728 bytes = ~134.2MB
        let memory_mb = config.estimate_alibi_memory_mb();
        assert!((memory_mb - 134.2).abs() < 1.0);
    }

    #[test]
    fn test_custom_config() {
        let config = JinaBertConfig::new("test-model".to_string(), 256, 2, 4, 512);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.max_position_embeddings, 512);
        assert_eq!(config.intermediate_size, 1024); // 256 * 4
    }
}
