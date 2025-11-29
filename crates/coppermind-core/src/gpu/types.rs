//! Core types for the GPU scheduler.
//!
//! This module defines the request/response types, priority levels, and model
//! identifiers used by the GPU scheduler.

use std::cmp::Ordering;

/// Request priority levels.
///
/// Lower values indicate higher priority. The scheduler processes requests
/// in priority order, with FIFO ordering within each priority level.
///
/// # Priority Levels
///
/// - `Immediate` (P0): Search queries - user is waiting, must complete in <100ms
/// - `Interactive` (P1): Single file uploads, manual triggers - user expects feedback soon
/// - `Background` (P2): Bulk indexing, crawl batches - can be delayed for efficiency
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Priority {
    /// P0: Search queries - user is waiting, must complete in <100ms.
    /// Examples: semantic search, query embedding
    Immediate = 0,

    /// P1: Interactive operations - user expects feedback soon.
    /// Examples: single file upload, manual embedding trigger
    #[default]
    Interactive = 1,

    /// P2: Background operations - can be delayed for efficiency.
    /// Examples: crawl batches, bulk indexing, LLM extraction
    Background = 2,
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower value = higher priority (P0 > P1 > P2)
        (*self as u8).cmp(&(*other as u8))
    }
}

/// Identifies which model to use for a request.
///
/// Currently supports the built-in JinaBERT embedding model. Will be extended
/// to support custom embedding models and LLMs as the application grows.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ModelId {
    /// Built-in JinaBERT embedding model (jina-embeddings-v2-small-en)
    #[default]
    JinaBert,
    /// Custom embedding model by name (future)
    Embedding(String),
    /// LLM model by name (future)
    Llm(String),
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelId::JinaBert => write!(f, "jina-bert"),
            ModelId::Embedding(name) => write!(f, "embedding:{}", name),
            ModelId::Llm(name) => write!(f, "llm:{}", name),
        }
    }
}

/// Single embedding request.
///
/// Represents a request to generate an embedding from token IDs.
#[derive(Debug)]
pub struct EmbedRequest {
    /// Which model to use for embedding
    pub model_id: ModelId,
    /// Token IDs to embed
    pub tokens: Vec<u32>,
    /// Request priority
    pub priority: Priority,
}

impl EmbedRequest {
    /// Creates a new embedding request with default model and priority.
    pub fn new(tokens: Vec<u32>) -> Self {
        Self {
            model_id: ModelId::default(),
            tokens,
            priority: Priority::default(),
        }
    }

    /// Creates an immediate (P0) priority request for search queries.
    pub fn immediate(tokens: Vec<u32>) -> Self {
        Self {
            model_id: ModelId::default(),
            tokens,
            priority: Priority::Immediate,
        }
    }

    /// Creates a background (P2) priority request.
    pub fn background(tokens: Vec<u32>) -> Self {
        Self {
            model_id: ModelId::default(),
            tokens,
            priority: Priority::Background,
        }
    }

    /// Sets the model ID for this request.
    pub fn with_model(mut self, model_id: ModelId) -> Self {
        self.model_id = model_id;
        self
    }

    /// Sets the priority for this request.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
}

/// Batch embedding request - more efficient for background work.
///
/// When processing multiple texts (e.g., during crawling), batching allows
/// the GPU to process them in a single forward pass, which is more efficient
/// than processing individually.
#[derive(Debug)]
pub struct BatchEmbedRequest {
    /// Which model to use for embedding
    pub model_id: ModelId,
    /// Vector of token ID vectors, one per text
    pub token_batches: Vec<Vec<u32>>,
    /// Request priority
    pub priority: Priority,
}

impl BatchEmbedRequest {
    /// Creates a new batch embedding request with default model and background priority.
    pub fn new(token_batches: Vec<Vec<u32>>) -> Self {
        Self {
            model_id: ModelId::default(),
            token_batches,
            priority: Priority::Background,
        }
    }

    /// Sets the model ID for this request.
    pub fn with_model(mut self, model_id: ModelId) -> Self {
        self.model_id = model_id;
        self
    }

    /// Sets the priority for this request.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
}

/// Embedding response.
#[derive(Debug, Clone)]
pub struct EmbedResponse {
    /// Generated embedding vector
    pub embedding: Vec<f32>,
}

/// LLM generation request (future).
///
/// Placeholder for future LLM integration.
#[derive(Debug)]
pub struct GenerateRequest {
    /// Which LLM model to use
    pub model_id: ModelId,
    /// Input prompt tokens
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Request priority
    pub priority: Priority,
}

/// LLM generation response (future).
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    /// Generated token IDs
    pub tokens: Vec<u32>,
    /// Generated text (decoded from tokens)
    pub text: String,
}

/// Configuration for loading a model.
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    /// Model weights (safetensors bytes)
    pub model_bytes: Vec<u8>,
    /// Tokenizer configuration (JSON bytes)
    pub tokenizer_bytes: Vec<u8>,
    /// Vocabulary size for the model
    pub vocab_size: usize,
}

impl ModelLoadConfig {
    /// Creates a new model load configuration.
    pub fn new(model_bytes: Vec<u8>, tokenizer_bytes: Vec<u8>, vocab_size: usize) -> Self {
        Self {
            model_bytes,
            tokenizer_bytes,
            vocab_size,
        }
    }
}

/// Scheduler statistics for monitoring.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Current queue depth
    pub queue_depth: usize,
    /// Total requests completed since startup
    pub requests_completed: u64,
    /// Number of loaded models
    pub models_loaded: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        // Lower value = higher priority
        assert!(Priority::Immediate < Priority::Interactive);
        assert!(Priority::Interactive < Priority::Background);
        assert!(Priority::Immediate < Priority::Background);
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Interactive);
    }

    #[test]
    fn test_model_id_display() {
        assert_eq!(ModelId::JinaBert.to_string(), "jina-bert");
        assert_eq!(
            ModelId::Embedding("custom".to_string()).to_string(),
            "embedding:custom"
        );
        assert_eq!(ModelId::Llm("llama".to_string()).to_string(), "llm:llama");
    }

    #[test]
    fn test_embed_request_builders() {
        let tokens = vec![101, 2023, 102];

        let req = EmbedRequest::new(tokens.clone());
        assert_eq!(req.priority, Priority::Interactive);
        assert_eq!(req.model_id, ModelId::JinaBert);

        let req = EmbedRequest::immediate(tokens.clone());
        assert_eq!(req.priority, Priority::Immediate);

        let req = EmbedRequest::background(tokens.clone());
        assert_eq!(req.priority, Priority::Background);

        let req = EmbedRequest::new(tokens)
            .with_model(ModelId::Embedding("test".to_string()))
            .with_priority(Priority::Background);
        assert_eq!(req.model_id, ModelId::Embedding("test".to_string()));
        assert_eq!(req.priority, Priority::Background);
    }
}
