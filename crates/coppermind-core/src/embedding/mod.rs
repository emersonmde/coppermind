//! Embedding model abstractions and implementations.
//!
//! This module provides traits and implementations for text embedding models.
//! The design supports multiple model types with a common interface.
//!
//! ## Core Traits
//!
//! - [`AssetLoader`] - Platform-agnostic model/tokenizer loading
//! - [`Embedder`] - Embedding model inference interface
//! - [`ModelConfig`] - Model configuration parameters
//!
//! ## Implementations
//!
//! - [`JinaBertConfig`] - Configuration for JinaBERT models
//! - [`JinaBertEmbedder`] - JinaBERT implementation using Candle
//! - [`TokenizerHandle`] - Wrapper for HuggingFace tokenizers
//!
//! ## Example
//!
//! ```ignore
//! use coppermind_core::embedding::{Embedder, JinaBertConfig, JinaBertEmbedder, TokenizerHandle};
//!
//! // Load tokenizer
//! let tokenizer_bytes = std::fs::read("tokenizer.json")?;
//! let tokenizer = TokenizerHandle::from_bytes(tokenizer_bytes, 512)?;
//!
//! // Create embedder
//! let model_bytes = std::fs::read("model.safetensors")?;
//! let config = JinaBertConfig::default();
//! let embedder = JinaBertEmbedder::from_bytes(model_bytes, tokenizer.vocab_size(), config)?;
//!
//! // Generate embeddings
//! let tokens = tokenizer.tokenize("Hello, world!")?;
//! let embedding = embedder.embed_tokens(tokens)?;
//! ```

mod traits;

pub mod config;
pub mod model;
pub mod tokenizer;
pub mod types;

// Re-export traits
pub use traits::{AssetLoader, Embedder, ModelConfig};

// Re-export config
pub use config::JinaBertConfig;

// Re-export model
pub use model::JinaBertEmbedder;

// Re-export tokenizer
pub use tokenizer::TokenizerHandle;

// Re-export types
pub use types::{ChunkEmbeddingResult, EmbeddingResult};
