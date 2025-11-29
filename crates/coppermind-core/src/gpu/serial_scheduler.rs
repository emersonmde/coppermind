//! Serial GPU scheduler implementation.
//!
//! This module implements a serial GPU scheduler that works around Candle's
//! Metal threading bug by having a single dedicated thread own the GPU device
//! and process all requests serially with priority ordering.

use super::error::GpuError;
use super::scheduler::GpuScheduler;
use super::types::{
    BatchEmbedRequest, DeviceType, EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse,
    ModelId, ModelLoadConfig, Priority, SchedulerStats,
};
use crate::embedding::{Embedder, JinaBertConfig, JinaBertEmbedder, ModelConfig};
use async_trait::async_trait;
use candle_core::Device;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use tokio::sync::oneshot;
#[cfg(feature = "profile")]
use tracing::instrument;
use tracing::{debug, info, warn};

/// Serial GPU scheduler - single thread owns all GPU resources.
///
/// This implementation serializes all GPU access to work around Candle's
/// Metal threading bug. A dedicated OS thread owns the Metal device and
/// processes requests from a priority queue.
///
/// # Architecture
///
/// ```text
/// ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
/// │  Async Callers  │────▶│  MPSC Channel   │────▶│  Worker Thread  │
/// │  (tokio tasks)  │     │  (bounded)      │     │  (owns GPU)     │
/// └─────────────────┘     └─────────────────┘     └─────────────────┘
///                                                         │
///                                                         ▼
///                                                 ┌───────────────┐
///                                                 │ Priority Queue│
///                                                 │ P0 > P1 > P2  │
///                                                 └───────────────┘
/// ```
///
/// # Thread Safety
///
/// - Worker thread: Single owner of Device, models, and command buffers
/// - Main thread: Only sends messages via channel, receives via oneshot
/// - Model set tracking: RwLock for checking which models are loaded
///
/// # Example
///
/// ```ignore
/// let scheduler = SerialScheduler::new()?;
///
/// // Load model first
/// scheduler.load_model(ModelId::JinaBert, config).await?;
///
/// // Submit embedding request
/// let response = scheduler.embed(EmbedRequest::immediate(tokens)).await?;
/// ```
pub struct SerialScheduler {
    /// Channel to send requests to the worker thread
    tx: mpsc::Sender<SchedulerMessage>,
    /// Atomic flag indicating scheduler readiness (at least one model loaded)
    ready: Arc<AtomicBool>,
    /// Set of loaded model IDs (for is_model_loaded checks without blocking)
    loaded_models: Arc<RwLock<HashSet<ModelId>>>,
    /// Statistics for monitoring
    stats: Arc<SchedulerStatsInner>,
}

/// Internal message type for worker thread communication.
enum SchedulerMessage {
    Embed {
        request: EmbedRequest,
        response: oneshot::Sender<Result<EmbedResponse, GpuError>>,
    },
    EmbedBatch {
        request: BatchEmbedRequest,
        response: oneshot::Sender<Result<Vec<EmbedResponse>, GpuError>>,
    },
    Generate {
        request: GenerateRequest,
        response: oneshot::Sender<Result<GenerateResponse, GpuError>>,
    },
    LoadModel {
        model_id: ModelId,
        config: ModelLoadConfig,
        response: oneshot::Sender<Result<(), GpuError>>,
    },
    UnloadModel {
        model_id: ModelId,
        response: oneshot::Sender<Result<(), GpuError>>,
    },
    #[allow(dead_code)]
    Shutdown,
}

impl SchedulerMessage {
    /// Get the priority of this message.
    fn priority(&self) -> Priority {
        match self {
            SchedulerMessage::Embed { request, .. } => request.priority,
            SchedulerMessage::EmbedBatch { request, .. } => request.priority,
            SchedulerMessage::Generate { request, .. } => request.priority,
            // Model operations are immediate priority
            SchedulerMessage::LoadModel { .. } => Priority::Immediate,
            SchedulerMessage::UnloadModel { .. } => Priority::Immediate,
            SchedulerMessage::Shutdown => Priority::Immediate,
        }
    }
}

/// Wrapper for priority queue ordering.
struct PrioritizedMessage {
    priority: Priority,
    sequence: u64, // For FIFO within same priority
    message: SchedulerMessage,
}

impl PartialEq for PrioritizedMessage {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for PrioritizedMessage {}

impl PartialOrd for PrioritizedMessage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedMessage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap, so we reverse the comparison
        // We want: lower priority value = higher actual priority (processed first)
        // So P0 (value 0) should compare as "greater" than P1 (value 1)
        match other.priority.cmp(&self.priority) {
            std::cmp::Ordering::Equal => {
                // Same priority: earlier sequence wins (FIFO)
                // Lower sequence = earlier = should be processed first = "greater" in max-heap
                other.sequence.cmp(&self.sequence)
            }
            other => other,
        }
    }
}

/// Internal statistics tracking.
pub(crate) struct SchedulerStatsInner {
    pub device_type: RwLock<DeviceType>,
    pub queue_depth: AtomicUsize,
    pub requests_completed: AtomicU64,
    pub models_loaded: AtomicUsize,
}

impl SchedulerStatsInner {
    fn new() -> Self {
        Self {
            device_type: RwLock::new(DeviceType::Cpu),
            queue_depth: AtomicUsize::new(0),
            requests_completed: AtomicU64::new(0),
            models_loaded: AtomicUsize::new(0),
        }
    }

    fn to_stats(&self) -> SchedulerStats {
        SchedulerStats {
            scheduler_name: "SerialScheduler",
            device_type: *self.device_type.read().unwrap(),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            models_loaded: self.models_loaded.load(Ordering::Relaxed),
        }
    }
}

impl SerialScheduler {
    /// Create a new serial scheduler.
    ///
    /// Spawns a dedicated OS thread that owns all GPU resources.
    /// The thread runs until the scheduler is dropped.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::ThreadSpawnFailed` if thread creation fails.
    pub fn new() -> Result<Self, GpuError> {
        let (tx, rx) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let loaded_models = Arc::new(RwLock::new(HashSet::new()));
        let stats = Arc::new(SchedulerStatsInner::new());

        let ready_clone = ready.clone();
        let loaded_models_clone = loaded_models.clone();
        let stats_clone = stats.clone();

        // Spawn dedicated GPU thread
        thread::Builder::new()
            .name("gpu-scheduler".to_string())
            .spawn(move || {
                Self::worker_loop(rx, ready_clone, loaded_models_clone, stats_clone);
            })
            .map_err(|e| GpuError::ThreadSpawnFailed(e.to_string()))?;

        info!("GPU scheduler initialized with dedicated worker thread");

        Ok(Self {
            tx,
            ready,
            loaded_models,
            stats,
        })
    }

    /// Worker thread main loop.
    fn worker_loop(
        rx: mpsc::Receiver<SchedulerMessage>,
        ready: Arc<AtomicBool>,
        loaded_models: Arc<RwLock<HashSet<ModelId>>>,
        stats: Arc<SchedulerStatsInner>,
    ) {
        info!("GPU scheduler worker thread started");

        // Initialize compute device (owned by this thread only)
        let (device, device_type) = Self::select_device();

        // Record device type in stats for monitoring
        if let Ok(mut dt) = stats.device_type.write() {
            *dt = device_type;
        }

        // Model registry (owned by this thread)
        let mut models: HashMap<ModelId, Arc<dyn Embedder>> = HashMap::new();

        // Priority queue
        let mut queue: BinaryHeap<PrioritizedMessage> = BinaryHeap::new();
        let mut sequence: u64 = 0;

        loop {
            // Phase 1: Drain channel into priority queue (non-blocking)
            loop {
                match rx.try_recv() {
                    Ok(msg) => {
                        let priority = msg.priority();
                        queue.push(PrioritizedMessage {
                            priority,
                            sequence,
                            message: msg,
                        });
                        sequence = sequence.wrapping_add(1);
                        stats.queue_depth.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        info!("GPU scheduler channel disconnected, shutting down");
                        return;
                    }
                }
            }

            // Phase 2: Process highest priority request
            if let Some(PrioritizedMessage {
                message, priority, ..
            }) = queue.pop()
            {
                stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                let remaining = stats.queue_depth.load(Ordering::Relaxed);

                match message {
                    SchedulerMessage::Embed { request, response } => {
                        debug!(
                            "Processing embed request (priority: {:?}, model: {}, queue_depth: {})",
                            priority, request.model_id, remaining
                        );
                        let result = Self::do_embed(&models, &request);
                        let _ = response.send(result);
                        stats.requests_completed.fetch_add(1, Ordering::Relaxed);
                    }
                    SchedulerMessage::EmbedBatch { request, response } => {
                        debug!(
                            "Processing batch embed request ({} items, priority: {:?}, queue_depth: {})",
                            request.token_batches.len(),
                            priority,
                            remaining
                        );
                        let result = Self::do_embed_batch(&models, &request);
                        let _ = response.send(result);
                        stats.requests_completed.fetch_add(1, Ordering::Relaxed);
                    }
                    SchedulerMessage::Generate { request, response } => {
                        debug!(
                            "Processing generate request (priority: {:?}, queue_depth: {})",
                            priority, remaining
                        );
                        let result = Self::do_generate(&models, &request);
                        let _ = response.send(result);
                        stats.requests_completed.fetch_add(1, Ordering::Relaxed);
                    }
                    SchedulerMessage::LoadModel {
                        model_id,
                        config,
                        response,
                    } => {
                        info!("Loading model: {}", model_id);
                        let result =
                            Self::do_load_model(&mut models, &device, model_id.clone(), config);
                        if result.is_ok() {
                            if let Ok(mut set) = loaded_models.write() {
                                set.insert(model_id);
                            }
                            stats.models_loaded.fetch_add(1, Ordering::Relaxed);
                            ready.store(true, Ordering::Release);
                        }
                        let _ = response.send(result);
                    }
                    SchedulerMessage::UnloadModel { model_id, response } => {
                        info!("Unloading model: {}", model_id);
                        let result = Self::do_unload_model(&mut models, &model_id);
                        if result.is_ok() {
                            if let Ok(mut set) = loaded_models.write() {
                                set.remove(&model_id);
                            }
                            stats.models_loaded.fetch_sub(1, Ordering::Relaxed);
                            if models.is_empty() {
                                ready.store(false, Ordering::Release);
                            }
                        }
                        let _ = response.send(result);
                    }
                    SchedulerMessage::Shutdown => {
                        info!("GPU scheduler received shutdown signal");
                        return;
                    }
                }
            } else {
                // No work available, block waiting for next message
                match rx.recv() {
                    Ok(msg) => {
                        let priority = msg.priority();
                        queue.push(PrioritizedMessage {
                            priority,
                            sequence,
                            message: msg,
                        });
                        sequence = sequence.wrapping_add(1);
                        stats.queue_depth.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        info!("GPU scheduler channel disconnected, shutting down");
                        return; // Channel closed
                    }
                }
            }
        }
    }

    /// Select the best available compute device.
    fn select_device() -> (Device, DeviceType) {
        // Try CUDA first (NVIDIA GPUs)
        if let Ok(cuda_device) = Device::new_cuda(0) {
            info!("GPU scheduler using CUDA device");
            return (cuda_device, DeviceType::Cuda);
        }

        // Try Metal (Apple Silicon)
        if let Ok(metal_device) = Device::new_metal(0) {
            info!("GPU scheduler using Metal device");
            return (metal_device, DeviceType::Metal);
        }

        // Fallback to CPU
        info!("GPU scheduler using CPU device");
        (Device::Cpu, DeviceType::Cpu)
    }

    /// Execute a single embedding request.
    #[cfg_attr(feature = "profile", instrument(skip_all, fields(model = %request.model_id, tokens = request.tokens.len())))]
    fn do_embed(
        models: &HashMap<ModelId, Arc<dyn Embedder>>,
        request: &EmbedRequest,
    ) -> Result<EmbedResponse, GpuError> {
        let model = models
            .get(&request.model_id)
            .ok_or_else(|| GpuError::ModelNotLoaded(request.model_id.clone()))?;

        let embedding = model
            .embed_tokens(request.tokens.clone())
            .map_err(|e| GpuError::EmbeddingFailed(e.to_string()))?;

        Ok(EmbedResponse { embedding })
    }

    /// Execute a batch embedding request.
    #[cfg_attr(feature = "profile", instrument(skip_all, fields(model = %request.model_id, batch_size = request.token_batches.len())))]
    fn do_embed_batch(
        models: &HashMap<ModelId, Arc<dyn Embedder>>,
        request: &BatchEmbedRequest,
    ) -> Result<Vec<EmbedResponse>, GpuError> {
        if request.token_batches.is_empty() {
            return Err(GpuError::InvalidRequest("Empty batch".to_string()));
        }

        let model = models
            .get(&request.model_id)
            .ok_or_else(|| GpuError::ModelNotLoaded(request.model_id.clone()))?;

        let embeddings = model
            .embed_batch_tokens(request.token_batches.clone())
            .map_err(|e| GpuError::EmbeddingFailed(e.to_string()))?;

        Ok(embeddings
            .into_iter()
            .map(|embedding| EmbedResponse { embedding })
            .collect())
    }

    /// Execute an LLM generation request (placeholder).
    fn do_generate(
        _models: &HashMap<ModelId, Arc<dyn Embedder>>,
        _request: &GenerateRequest,
    ) -> Result<GenerateResponse, GpuError> {
        // LLM generation not yet implemented
        Err(GpuError::GenerationFailed(
            "LLM generation not yet implemented".to_string(),
        ))
    }

    /// Load a model into the registry.
    fn do_load_model(
        models: &mut HashMap<ModelId, Arc<dyn Embedder>>,
        device: &Device,
        model_id: ModelId,
        config: ModelLoadConfig,
    ) -> Result<(), GpuError> {
        if models.contains_key(&model_id) {
            return Err(GpuError::ModelAlreadyLoaded(model_id));
        }

        match &model_id {
            ModelId::JinaBert => {
                let jina_config = JinaBertConfig::default();
                info!(
                    "JinaBERT config: {}d hidden, {} layers, {} heads",
                    jina_config.hidden_size,
                    jina_config.num_hidden_layers,
                    jina_config.num_attention_heads
                );

                // Create model with explicit device (worker thread owns device)
                let model = create_jina_bert_model(
                    config.model_bytes,
                    config.vocab_size,
                    jina_config,
                    device,
                )
                .map_err(|e| GpuError::ModelLoadFailed(e.to_string()))?;

                models.insert(model_id.clone(), Arc::new(model));
                info!("Model {} loaded successfully", model_id);
                Ok(())
            }
            ModelId::Embedding(name) => {
                warn!(
                    "Custom embedding model '{}' not yet supported, using JinaBERT config",
                    name
                );
                // For now, treat custom embedding models as JinaBERT variants
                let jina_config = JinaBertConfig::default();
                let model = create_jina_bert_model(
                    config.model_bytes,
                    config.vocab_size,
                    jina_config,
                    device,
                )
                .map_err(|e| GpuError::ModelLoadFailed(e.to_string()))?;

                models.insert(model_id.clone(), Arc::new(model));
                info!("Model {} loaded successfully", model_id);
                Ok(())
            }
            ModelId::Llm(name) => Err(GpuError::ModelLoadFailed(format!(
                "LLM model '{}' not yet supported",
                name
            ))),
        }
    }

    /// Unload a model from the registry.
    fn do_unload_model(
        models: &mut HashMap<ModelId, Arc<dyn Embedder>>,
        model_id: &ModelId,
    ) -> Result<(), GpuError> {
        models
            .remove(model_id)
            .map(|_| ())
            .ok_or_else(|| GpuError::ModelNotLoaded(model_id.clone()))
    }
}

/// Create JinaBERT model with explicit device.
///
/// This is similar to `JinaBertEmbedder::from_bytes` but uses an explicit device
/// rather than auto-detecting, since the worker thread already owns the device.
fn create_jina_bert_model(
    model_bytes: Vec<u8>,
    vocab_size: usize,
    config: JinaBertConfig,
    device: &Device,
) -> Result<JinaBertEmbedder, String> {
    use candle_core::DType;
    use candle_nn::{Activation, VarBuilder};
    use candle_transformers::models::jina_bert::{BertModel, Config, PositionEmbeddingType};

    info!(
        "Loading embedding model '{}' on worker thread",
        config.model_id()
    );
    info!(
        "Model bytes length: {} bytes ({:.2}MB)",
        model_bytes.len(),
        model_bytes.len() as f64 / 1_000_000.0
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

    // Validate safetensors header
    if model_bytes.len() < 8 {
        return Err("Model file too small".to_string());
    }

    // Load model weights
    info!("Loading VarBuilder from safetensors (converting to F32)...");
    let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, device)
        .map_err(|e| format!("Failed to create VarBuilder: {}", e))?;
    info!("VarBuilder created successfully");

    info!("Creating BertModel...");
    let model = BertModel::new(vb, &model_config)
        .map_err(|e| format!("Failed to create BertModel: {}", e))?;
    info!("BertModel created successfully");

    // Create the embedder wrapper
    Ok(JinaBertEmbedder::from_parts(model, config, device.clone()))
}

#[async_trait]
impl GpuScheduler for SerialScheduler {
    #[cfg_attr(feature = "profile", instrument(skip_all, fields(priority = ?request.priority)))]
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, GpuError> {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .send(SchedulerMessage::Embed {
                request,
                response: response_tx,
            })
            .map_err(|_| GpuError::ChannelDisconnected)?;

        response_rx
            .await
            .map_err(|e| GpuError::ResponseFailed(e.to_string()))?
    }

    #[cfg_attr(feature = "profile", instrument(skip_all, fields(batch_size = request.token_batches.len(), priority = ?request.priority)))]
    async fn embed_batch(
        &self,
        request: BatchEmbedRequest,
    ) -> Result<Vec<EmbedResponse>, GpuError> {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .send(SchedulerMessage::EmbedBatch {
                request,
                response: response_tx,
            })
            .map_err(|_| GpuError::ChannelDisconnected)?;

        response_rx
            .await
            .map_err(|e| GpuError::ResponseFailed(e.to_string()))?
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, GpuError> {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .send(SchedulerMessage::Generate {
                request,
                response: response_tx,
            })
            .map_err(|_| GpuError::ChannelDisconnected)?;

        response_rx
            .await
            .map_err(|e| GpuError::ResponseFailed(e.to_string()))?
    }

    async fn load_model(&self, model_id: ModelId, config: ModelLoadConfig) -> Result<(), GpuError> {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .send(SchedulerMessage::LoadModel {
                model_id,
                config,
                response: response_tx,
            })
            .map_err(|_| GpuError::ChannelDisconnected)?;

        response_rx
            .await
            .map_err(|e| GpuError::ResponseFailed(e.to_string()))?
    }

    async fn unload_model(&self, model_id: &ModelId) -> Result<(), GpuError> {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .send(SchedulerMessage::UnloadModel {
                model_id: model_id.clone(),
                response: response_tx,
            })
            .map_err(|_| GpuError::ChannelDisconnected)?;

        response_rx
            .await
            .map_err(|e| GpuError::ResponseFailed(e.to_string()))?
    }

    fn is_model_loaded(&self, model_id: &ModelId) -> bool {
        self.loaded_models
            .read()
            .map(|set| set.contains(model_id))
            .unwrap_or(false)
    }

    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    fn stats(&self) -> SchedulerStats {
        self.stats.to_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_queue_ordering() {
        let mut heap: BinaryHeap<PrioritizedMessage> = BinaryHeap::new();

        // Add in reverse priority order
        heap.push(PrioritizedMessage {
            priority: Priority::Background,
            sequence: 0,
            message: SchedulerMessage::Shutdown,
        });
        heap.push(PrioritizedMessage {
            priority: Priority::Interactive,
            sequence: 1,
            message: SchedulerMessage::Shutdown,
        });
        heap.push(PrioritizedMessage {
            priority: Priority::Immediate,
            sequence: 2,
            message: SchedulerMessage::Shutdown,
        });

        // Should pop in priority order: Immediate, Interactive, Background
        assert_eq!(heap.pop().unwrap().priority, Priority::Immediate);
        assert_eq!(heap.pop().unwrap().priority, Priority::Interactive);
        assert_eq!(heap.pop().unwrap().priority, Priority::Background);
    }

    #[test]
    fn test_fifo_within_priority() {
        let mut heap: BinaryHeap<PrioritizedMessage> = BinaryHeap::new();

        // Add three requests at same priority
        heap.push(PrioritizedMessage {
            priority: Priority::Interactive,
            sequence: 0,
            message: SchedulerMessage::Shutdown,
        });
        heap.push(PrioritizedMessage {
            priority: Priority::Interactive,
            sequence: 1,
            message: SchedulerMessage::Shutdown,
        });
        heap.push(PrioritizedMessage {
            priority: Priority::Interactive,
            sequence: 2,
            message: SchedulerMessage::Shutdown,
        });

        // Should pop in FIFO order: 0, 1, 2
        assert_eq!(heap.pop().unwrap().sequence, 0);
        assert_eq!(heap.pop().unwrap().sequence, 1);
        assert_eq!(heap.pop().unwrap().sequence, 2);
    }

    #[test]
    fn test_priority_trumps_sequence() {
        let mut heap: BinaryHeap<PrioritizedMessage> = BinaryHeap::new();

        // Add background first (sequence 0), then immediate (sequence 1)
        heap.push(PrioritizedMessage {
            priority: Priority::Background,
            sequence: 0,
            message: SchedulerMessage::Shutdown,
        });
        heap.push(PrioritizedMessage {
            priority: Priority::Immediate,
            sequence: 1,
            message: SchedulerMessage::Shutdown,
        });

        // Immediate should pop first despite higher sequence number
        let first = heap.pop().unwrap();
        assert_eq!(first.priority, Priority::Immediate);
        assert_eq!(first.sequence, 1);
    }
}
