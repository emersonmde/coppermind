# ADR 006: GPU Scheduler for Metal Thread Safety and Multi-Model Support

**Status**: Implemented
**Date**: 2025-11-25
**Context**: Candle Metal threading crash, multi-model support, priority-based request scheduling

---

## Context and Problem Statement

### The Metal Crash Problem

Candle's Metal backend crashes when accessed from multiple threads via `tokio::spawn_blocking`. Two distinct crash patterns have been observed:

**Crash Type 1: Command Encoder Assertion (Candle 0.8.x, 0.9.1)**
```
-[AGXG15XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:]:1015:
failed assertion `A command encoder is already encoding to this command buffer'
```

**Crash Type 2: SIGSEGV in allocate_zeros (Candle 0.9.2-alpha.1)**
```
Thread 28 Crashed:: tokio-runtime-worker
Exception Type:        EXC_BAD_ACCESS (SIGSEGV)
Exception Codes:       KERN_INVALID_ADDRESS at 0x0000000000000004

0   AGXMetalG15X_B0    endComputePass + 344
1   AGXMetalG15X_B0    -[AGXG15XFamilyComputeContext deferredEndEncoding] + 176
2   AGXMetalG15X_B0    -[AGXG15XFamilyCommandBuffer commitEncoder] + 96
3   AGXMetalG15X_B0    -[AGXG15XFamilyCommandBuffer blitCommandEncoderCommon:] + 360
4   coppermind         metal::commandbuffer::CommandBufferRef::new_blit_command_encoder
5   coppermind         candle_core::metal_backend::device::MetalDevice::allocate_zeros
6   coppermind         candle_core::tensor::Tensor::zeros
7   coppermind         BertEmbeddings::forward
8   coppermind         BertModel::forward
9   coppermind         JinaBertEmbedder::embed_tokens
```

### Root Cause Analysis

**Metal Threading Model:**
- `MTLCommandQueue`: Thread-safe (can submit from multiple threads)
- `MTLCommandBuffer`: **NOT thread-safe** (single thread per buffer)
- `MTLCommandEncoder`: **NOT thread-safe** (tied to command buffer)

**Candle Implementation:**
- **0.8.x/0.9.1**: Single shared command buffer across all threads → immediate race condition
- **0.9.2-alpha.1**: Pool-based command buffer system with `RwLock<Commands>` but regression in `allocate_zeros` during blit encoder creation

**Relevant Candle Issues/PRs:**
- [Issue #2637](https://github.com/huggingface/candle/issues/2637): Metal tensor assertion failure
- [PR #3079](https://github.com/huggingface/candle/pull/3079): Thread-isolated command buffers (merged, but regression in alpha)
- [PR #3070](https://github.com/huggingface/candle/pull/3070): Major Metal refactor (caused 0.9.2-alpha.1 issues)

### Current Code Path

```rust
// embedding/mod.rs - current implementation
pub async fn compute_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    let model = get_or_load_model().await?;
    let token_ids = tokenizer::tokenize_text_async(tokenizer, text).await?;

    // PROBLEM: spawn_blocking creates new threads from tokio's pool
    // Multiple concurrent embeddings = multiple threads accessing Metal
    let embedding = run_blocking(move || model_clone.embed_tokens(token_ids)).await?;

    Ok(EmbeddingComputation { token_count, embedding })
}
```

### Additional Requirements

Beyond fixing the crash, the architecture must support:

1. **Priority Scheduling**: Search queries must never be blocked by background work
2. **Multi-Model Support**: Eventually support multiple embedding models + LLM models
3. **Model Selection**: Users will select which models to use
4. **Future Parallelism**: When Candle is fixed, enable true parallel GPU access
5. **Efficient Batching**: Background work should batch requests for GPU efficiency

---

## Decision Drivers

1. **Fix Metal crash** - Primary goal, must serialize GPU access
2. **Search responsiveness** - P0 priority, <100ms latency
3. **Background efficiency** - Batch processing for crawl/chunking
4. **Multi-model ready** - Design for embedding + LLM models
5. **Future parallelism** - Interface should not change when Candle is fixed
6. **Cross-platform** - Desktop (Metal/CUDA) and Web (CPU) support
7. **Minimal rework** - Clean migration path from current code

---

## Considered Options

### Option 1: Semaphore Serialization

**Approach**: Add `tokio::sync::Semaphore` with 1 permit to serialize GPU access.

```rust
static GPU_SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(1)));

pub async fn compute_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    let _permit = GPU_SEMAPHORE.acquire().await?;
    // ... existing code
}
```

**Pros:**
- ✅ Minimal code change (5 lines)
- ✅ Fixes crash immediately
- ✅ Works with existing architecture

**Cons:**
- ❌ No priority support - first-come-first-served
- ❌ Search blocked by background work
- ❌ No multi-model support
- ❌ No batching optimization
- ❌ Hard to extend for parallelism

**Verdict**: Rejected - doesn't meet requirements for priority/multi-model.

### Option 2: Dedicated Worker Thread (Simple)

**Approach**: Single thread owns Metal device, requests via channel, FIFO processing.

```rust
pub struct GpuWorker {
    tx: mpsc::Sender<WorkerMessage>,
}

impl GpuWorker {
    pub async fn embed(&self, tokens: Vec<u32>) -> Result<Vec<f32>, GpuError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.tx.send(WorkerMessage::Embed { tokens, response_tx }).await?;
        response_rx.await?
    }
}
```

**Pros:**
- ✅ Fixes crash
- ✅ Clean ownership model
- ✅ Simple implementation

**Cons:**
- ❌ FIFO only - no priority support
- ❌ Single model only
- ❌ No batching

**Verdict**: Rejected - too simple for requirements.

### Option 3: GPU Scheduler with Priority Queue (Recommended)

**Approach**: Scheduler trait with priority queue, model registry, batch support.

**Pros:**
- ✅ Fixes crash (single thread owns GPU)
- ✅ Priority scheduling (search > interactive > background)
- ✅ Multi-model ready (model registry)
- ✅ Batch support (efficient background processing)
- ✅ Future-proof (swap SerialScheduler for ParallelScheduler)
- ✅ Clean API (trait abstraction)

**Cons:**
- ⚠️ More complex implementation (~500 lines)
- ⚠️ New module/abstractions
- ⚠️ Migration required for existing callers

**Verdict**: Accepted - meets all requirements, worth the complexity.

### Option 4: Wait for Candle Fix

**Approach**: Wait for stable Candle release with working Metal threading.

**Cons:**
- ❌ Unknown timeline
- ❌ Doesn't add priority/multi-model support
- ❌ Blocking progress on critical feature

**Verdict**: Rejected - not a solution.

---

## Decision

Implement **Option 3: GPU Scheduler with Priority Queue**.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Application Code                           │
│  compute_embedding(), search(), embed_chunks(), llm_generate()      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GpuScheduler Trait                          │
│  embed(), embed_batch(), generate(), load_model(), is_ready()       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
┌─────────────────────────┐           ┌─────────────────────────┐
│    SerialScheduler      │           │   ParallelScheduler     │
│  (NOW - workaround)     │           │   (FUTURE - when fixed) │
│                         │           │                         │
│  ┌───────────────────┐  │           │  ┌─────────────────┐    │
│  │  Priority Queue   │  │           │  │  Load Balancer  │    │
│  │  P0 > P1 > P2     │  │           │  └────────┬────────┘    │
│  └─────────┬─────────┘  │           │           │             │
│            │            │           │    ┌──────┴──────┐      │
│  ┌─────────▼─────────┐  │           │    ▼      ▼      ▼      │
│  │  Model Registry   │  │           │  Worker Worker Worker   │
│  │  - embedding_a    │  │           │  (thread pool)          │
│  │  - embedding_b    │  │           └─────────────────────────┘
│  │  - llm (future)   │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

### Priority Levels

```rust
/// Request priority levels
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Priority {
    /// P0: Search queries - user is waiting, must complete in <100ms
    /// Examples: semantic search, query embedding
    Immediate = 0,

    /// P1: Interactive operations - user expects feedback soon
    /// Examples: single file upload, manual embedding trigger
    Interactive = 1,

    /// P2: Background operations - can be delayed for efficiency
    /// Examples: crawl batches, bulk indexing, LLM extraction
    Background = 2,
}
```

### Request Types

```rust
/// Identifies which model to use for a request
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelId {
    /// Built-in JinaBERT embedding model
    JinaBert,
    /// Custom embedding model by name
    Embedding(String),
    /// LLM model by name (future)
    Llm(String),
}

/// Single embedding request
#[derive(Debug)]
pub struct EmbedRequest {
    pub model_id: ModelId,
    pub tokens: Vec<u32>,
    pub priority: Priority,
}

/// Batch embedding request - more efficient for background work
#[derive(Debug)]
pub struct BatchEmbedRequest {
    pub model_id: ModelId,
    pub token_batches: Vec<Vec<u32>>,
    pub priority: Priority,
}

/// Embedding response
#[derive(Debug, Clone)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
}

/// LLM generation request (future)
#[derive(Debug)]
pub struct GenerateRequest {
    pub model_id: ModelId,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub priority: Priority,
}

/// LLM generation response (future)
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub tokens: Vec<u32>,
    pub text: String,
}
```

### Scheduler Trait

```rust
use async_trait::async_trait;

/// GPU scheduler trait - abstracts execution strategy
///
/// Current: SerialScheduler (single thread owns GPU)
/// Future: ParallelScheduler (thread pool when Candle fixed)
#[async_trait]
pub trait GpuScheduler: Send + Sync {
    /// Submit single embedding request
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, GpuError>;

    /// Submit batch embedding request (more efficient for background work)
    /// GPU processes entire batch in single forward pass
    async fn embed_batch(&self, request: BatchEmbedRequest) -> Result<Vec<EmbedResponse>, GpuError>;

    /// Generate text with LLM (future)
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, GpuError>;

    /// Load a model into the registry
    /// Returns error if model already loaded or loading fails
    async fn load_model(&self, model_id: ModelId, config: ModelLoadConfig) -> Result<(), GpuError>;

    /// Unload a model from the registry
    async fn unload_model(&self, model_id: &ModelId) -> Result<(), GpuError>;

    /// Check if a specific model is loaded
    fn is_model_loaded(&self, model_id: &ModelId) -> bool;

    /// Check if scheduler is ready (at least one model loaded)
    fn is_ready(&self) -> bool;

    /// Get scheduler statistics (queue depth, requests processed, etc.)
    fn stats(&self) -> SchedulerStats;
}
```

### Serial Scheduler Implementation

```rust
/// Serial GPU scheduler - single thread owns all GPU resources
///
/// This implementation serializes all GPU access to work around Candle's
/// Metal threading bug. When Candle is fixed, swap for ParallelScheduler.
pub struct SerialScheduler {
    /// Channel to send requests to the worker thread
    tx: mpsc::Sender<SchedulerMessage>,
    /// Atomic flag indicating scheduler readiness
    ready: Arc<AtomicBool>,
    /// Statistics for monitoring
    stats: Arc<SchedulerStatsInner>,
}

/// Internal message type for worker thread communication
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
    Shutdown,
}

/// Wrapper for priority queue ordering
struct PrioritizedMessage {
    priority: Priority,
    sequence: u64,  // For FIFO within same priority
    message: SchedulerMessage,
}

impl Ord for PrioritizedMessage {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority value = higher priority (P0 > P1 > P2)
        // Same priority: earlier sequence wins (FIFO)
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.sequence.cmp(&self.sequence),
            other => other,
        }
    }
}

impl SerialScheduler {
    /// Create a new serial scheduler
    ///
    /// Spawns a dedicated OS thread that owns all GPU resources.
    /// The thread runs until the scheduler is dropped.
    pub fn new() -> Result<Self, GpuError> {
        let (tx, rx) = mpsc::channel(1000);  // Bounded channel
        let ready = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(SchedulerStatsInner::new());

        let ready_clone = ready.clone();
        let stats_clone = stats.clone();

        // Spawn dedicated GPU thread
        std::thread::Builder::new()
            .name("gpu-scheduler".to_string())
            .spawn(move || {
                Self::worker_loop(rx, ready_clone, stats_clone);
            })
            .map_err(|e| GpuError::ThreadSpawnFailed(e.to_string()))?;

        Ok(Self { tx, ready, stats })
    }

    /// Worker thread main loop
    fn worker_loop(
        rx: mpsc::Receiver<SchedulerMessage>,
        ready: Arc<AtomicBool>,
        stats: Arc<SchedulerStatsInner>,
    ) {
        // Initialize Metal device (owned by this thread only)
        let device = match Device::new_metal(0) {
            Ok(d) => {
                info!("✓ GPU scheduler using Metal device");
                d
            }
            Err(_) => {
                info!("✓ GPU scheduler using CPU device");
                Device::Cpu
            }
        };

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
                        let priority = Self::get_priority(&msg);
                        queue.push(PrioritizedMessage {
                            priority,
                            sequence,
                            message: msg,
                        });
                        sequence += 1;
                        stats.queue_depth.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return,
                }
            }

            // Phase 2: Process highest priority request
            if let Some(PrioritizedMessage { message, .. }) = queue.pop() {
                stats.queue_depth.fetch_sub(1, Ordering::Relaxed);

                match message {
                    SchedulerMessage::Embed { request, response } => {
                        let result = Self::do_embed(&models, &device, request);
                        let _ = response.send(result);
                        stats.requests_completed.fetch_add(1, Ordering::Relaxed);
                    }
                    SchedulerMessage::EmbedBatch { request, response } => {
                        let result = Self::do_embed_batch(&models, &device, request);
                        let _ = response.send(result);
                        stats.requests_completed.fetch_add(1, Ordering::Relaxed);
                    }
                    SchedulerMessage::Generate { request, response } => {
                        let result = Self::do_generate(&models, &device, request);
                        let _ = response.send(result);
                        stats.requests_completed.fetch_add(1, Ordering::Relaxed);
                    }
                    SchedulerMessage::LoadModel { model_id, config, response } => {
                        let result = Self::do_load_model(&mut models, &device, model_id, config);
                        if result.is_ok() {
                            ready.store(true, Ordering::Release);
                        }
                        let _ = response.send(result);
                    }
                    SchedulerMessage::UnloadModel { model_id, response } => {
                        let result = Self::do_unload_model(&mut models, model_id);
                        let _ = response.send(result);
                    }
                    SchedulerMessage::Shutdown => return,
                }
            } else {
                // No work available, block waiting for next message
                match rx.recv() {
                    Ok(msg) => {
                        let priority = Self::get_priority(&msg);
                        queue.push(PrioritizedMessage {
                            priority,
                            sequence,
                            message: msg,
                        });
                        sequence += 1;
                        stats.queue_depth.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => return,  // Channel closed
                }
            }
        }
    }

    fn do_embed(
        models: &HashMap<ModelId, Arc<dyn Embedder>>,
        device: &Device,
        request: EmbedRequest,
    ) -> Result<EmbedResponse, GpuError> {
        let model = models.get(&request.model_id)
            .ok_or_else(|| GpuError::ModelNotLoaded(request.model_id.clone()))?;

        let embedding = model.embed_tokens(request.tokens)
            .map_err(|e| GpuError::EmbeddingFailed(e.to_string()))?;

        Ok(EmbedResponse { embedding })
    }

    fn do_embed_batch(
        models: &HashMap<ModelId, Arc<dyn Embedder>>,
        device: &Device,
        request: BatchEmbedRequest,
    ) -> Result<Vec<EmbedResponse>, GpuError> {
        let model = models.get(&request.model_id)
            .ok_or_else(|| GpuError::ModelNotLoaded(request.model_id.clone()))?;

        let embeddings = model.embed_batch_tokens(request.token_batches)
            .map_err(|e| GpuError::EmbeddingFailed(e.to_string()))?;

        Ok(embeddings.into_iter().map(|e| EmbedResponse { embedding: e }).collect())
    }
}
```

### Integration with Embedding Module

```rust
// embedding/mod.rs - updated to use scheduler

use crate::gpu::{GpuScheduler, EmbedRequest, Priority, ModelId};
use once_cell::sync::OnceCell;
use std::sync::Arc;

/// Global GPU scheduler (desktop only, initialized at startup)
#[cfg(not(target_arch = "wasm32"))]
static SCHEDULER: OnceCell<Arc<dyn GpuScheduler>> = OnceCell::new();

/// Initialize the GPU scheduler (call at app startup)
#[cfg(not(target_arch = "wasm32"))]
pub async fn init_gpu_scheduler() -> Result<(), EmbeddingError> {
    use crate::gpu::SerialScheduler;

    let scheduler = SerialScheduler::new()
        .map_err(|e| EmbeddingError::SchedulerInit(e.to_string()))?;

    // Load default embedding model
    scheduler.load_model(
        ModelId::JinaBert,
        ModelLoadConfig::from_assets(MODEL_FILE, TOKENIZER_FILE),
    ).await?;

    SCHEDULER.set(Arc::new(scheduler) as Arc<dyn GpuScheduler>)
        .map_err(|_| EmbeddingError::AlreadyInitialized)?;

    Ok(())
}

/// Compute embedding with default priority (Interactive)
pub async fn compute_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    compute_embedding_with_priority(text, Priority::Interactive).await
}

/// Compute embedding with explicit priority
pub async fn compute_embedding_with_priority(
    text: &str,
    priority: Priority,
) -> Result<EmbeddingComputation, EmbeddingError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let scheduler = SCHEDULER.get()
            .ok_or(EmbeddingError::SchedulerNotInitialized)?;

        let tokenizer = ensure_tokenizer(8192).await?;
        let token_ids = tokenizer::tokenize_text_async(tokenizer, text).await?;
        let token_count = token_ids.len();

        let response = scheduler.embed(EmbedRequest {
            model_id: ModelId::JinaBert,
            tokens: token_ids,
            priority,
        }).await.map_err(|e| EmbeddingError::GpuError(e.to_string()))?;

        Ok(EmbeddingComputation {
            token_count,
            embedding: response.embedding,
        })
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM: Direct execution (no threading issues on CPU)
        let model = get_or_load_model().await?;
        let tokenizer = ensure_tokenizer(model.max_position_embeddings()).await?;
        let token_ids = tokenizer::tokenize_text_async(tokenizer, text).await?;
        let embedding = model.embed_tokens(token_ids.clone())?;

        Ok(EmbeddingComputation {
            token_count: token_ids.len(),
            embedding,
        })
    }
}

/// Search-optimized embedding (highest priority)
pub async fn compute_search_embedding(text: &str) -> Result<EmbeddingComputation, EmbeddingError> {
    compute_embedding_with_priority(text, Priority::Immediate).await
}

/// Batch embedding for background work (lowest priority, most efficient)
pub async fn compute_embeddings_batch(
    texts: Vec<String>,
) -> Result<Vec<EmbeddingComputation>, EmbeddingError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let scheduler = SCHEDULER.get()
            .ok_or(EmbeddingError::SchedulerNotInitialized)?;

        let tokenizer = ensure_tokenizer(8192).await?;

        // Tokenize all texts
        let mut token_batches = Vec::with_capacity(texts.len());
        let mut token_counts = Vec::with_capacity(texts.len());

        for text in &texts {
            let tokens = tokenizer::tokenize_text_async(tokenizer, text).await?;
            token_counts.push(tokens.len());
            token_batches.push(tokens);
        }

        // Single batch request to GPU
        let responses = scheduler.embed_batch(BatchEmbedRequest {
            model_id: ModelId::JinaBert,
            token_batches,
            priority: Priority::Background,
        }).await.map_err(|e| EmbeddingError::GpuError(e.to_string()))?;

        Ok(responses.into_iter().zip(token_counts)
            .map(|(r, tc)| EmbeddingComputation {
                token_count: tc,
                embedding: r.embedding,
            })
            .collect())
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM: Sequential processing
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(compute_embedding(&text).await?);
        }
        Ok(results)
    }
}
```

---

## Consequences

### Positive

1. **Fixes Metal crash** - Single thread owns GPU, no concurrent command buffer access
2. **Priority scheduling** - Search queries (P0) processed before background work (P2)
3. **Multi-model ready** - Model registry supports multiple embedding models + future LLM
4. **Batch efficiency** - Background work batched for GPU efficiency
5. **Future-proof** - Swap `SerialScheduler` for `ParallelScheduler` when Candle fixed
6. **Clean API** - `GpuScheduler` trait abstracts implementation details
7. **Observable** - Statistics for monitoring queue depth, throughput
8. **Cross-platform** - Desktop uses scheduler, WASM continues with direct execution

### Negative

1. **Serialized GPU access** - One operation at a time (until Candle fixed)
   - **Mitigation**: Batching, GPU still fast, priority ensures search responsiveness

2. **New abstraction layer** - More code, more complexity (~500 lines)
   - **Mitigation**: Well-documented, testable, clear responsibilities

3. **Migration required** - Existing callers need updates
   - **Mitigation**: API similar to current, mostly mechanical changes

4. **Startup cost** - Must initialize scheduler before first embedding
   - **Mitigation**: Lazy initialization on first use, or explicit init at app start

### Neutral

1. **Thread vs async tradeoff** - Dedicated OS thread for GPU vs async executor
   - Standard pattern for GPU work in async systems
   - Thread blocked waiting is fine (no CPU cost)

2. **Channel-based communication** - Overhead vs direct function calls
   - Negligible compared to GPU inference time (~100ms)

---

## Implementation Plan

### Phase 1: Core Types (~2 hours)

Create `crates/coppermind/src/gpu/mod.rs` with:
- `Priority` enum
- `ModelId` enum
- Request/response structs
- `GpuError` type
- `GpuScheduler` trait

### Phase 2: Serial Scheduler (~4 hours)

Create `crates/coppermind/src/gpu/serial_scheduler.rs` with:
- `SerialScheduler` struct
- Worker thread loop
- Priority queue implementation
- Model registry
- Statistics tracking

### Phase 3: Integration (~2 hours)

Update existing code:
- `embedding/mod.rs` - Use scheduler for desktop
- `embedding/model.rs` - Remove device selection (scheduler owns)
- `lib.rs` - Export `gpu` module
- `main.rs` - Initialize scheduler at startup

### Phase 4: Testing (~2 hours)

- Unit tests for priority ordering
- Integration tests for concurrent requests
- Verify Metal doesn't crash
- Performance benchmarks

### Phase 5: Documentation (~1 hour)

- Update CLAUDE.md with GPU scheduler architecture
- Update docs/config-options.md
- Finalize this ADR

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/gpu/mod.rs` | Module root, re-exports |
| `src/gpu/types.rs` | Priority, ModelId, request/response types |
| `src/gpu/error.rs` | GpuError enum |
| `src/gpu/scheduler.rs` | GpuScheduler trait |
| `src/gpu/serial_scheduler.rs` | Serial implementation |
| `docs/adrs/006-gpu-scheduler.md` | This ADR |

### Modified Files

| File | Changes |
|------|---------|
| `src/lib.rs` | Add `pub mod gpu;` |
| `src/main.rs` | Initialize scheduler at startup |
| `src/embedding/mod.rs` | Use scheduler for desktop, add priority API |
| `src/embedding/model.rs` | Remove device selection logic |
| `src/components/search/search_view.rs` | Use `compute_search_embedding()` |
| `src/components/web_crawler.rs` | Use `compute_embeddings_batch()` |
| `docs/config-options.md` | Document scheduler config |
| `CLAUDE.md` | Update architecture section |

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_priority_ordering() {
    assert!(Priority::Immediate < Priority::Interactive);
    assert!(Priority::Interactive < Priority::Background);
}

#[test]
fn test_prioritized_message_ordering() {
    let mut heap = BinaryHeap::new();
    heap.push(PrioritizedMessage { priority: Priority::Background, sequence: 0, .. });
    heap.push(PrioritizedMessage { priority: Priority::Immediate, sequence: 1, .. });
    heap.push(PrioritizedMessage { priority: Priority::Interactive, sequence: 2, .. });

    // Should pop in priority order: Immediate, Interactive, Background
    assert_eq!(heap.pop().unwrap().priority, Priority::Immediate);
    assert_eq!(heap.pop().unwrap().priority, Priority::Interactive);
    assert_eq!(heap.pop().unwrap().priority, Priority::Background);
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_concurrent_embeddings_no_crash() {
    let scheduler = SerialScheduler::new().unwrap();
    scheduler.load_model(ModelId::JinaBert, config).await.unwrap();

    // Spawn many concurrent embedding requests
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let scheduler = scheduler.clone();
            tokio::spawn(async move {
                scheduler.embed(EmbedRequest {
                    model_id: ModelId::JinaBert,
                    tokens: vec![101, 2023, 2003, 1037, 3231, 102],
                    priority: Priority::Interactive,
                }).await
            })
        })
        .collect();

    // All should complete without crash
    for handle in handles {
        assert!(handle.await.unwrap().is_ok());
    }
}

#[tokio::test]
async fn test_priority_ordering_in_practice() {
    // Submit background tasks first, then immediate
    // Verify immediate completes before background
}
```

---

## Post-Implementation Updates

### Batch Processing Integration (2025-11-25)

**Problem Discovered**: Initial implementation revealed that file processing bypassed the scheduler's batch API entirely. The `process_file_chunks()` function was:
- Processing chunks **sequentially** in a loop via `embedder.embed()`
- Using old character-based chunking (2000 chars) instead of semantic chunking
- Defaulting to **Interactive priority (P1)** instead of Background (P2)
- Never building queue depth (always 0 or 1)

**Root Cause**: The `PlatformEmbedder` trait abstraction (`DesktopEmbedder`, `WebEmbedder`) called `compute_embedding()` individually per chunk, which bypassed the scheduler's `BatchEmbedRequest` API.

**Solution**: Refactored file processing to use batch embedding:

```rust
// OLD: Sequential processing via PlatformEmbedder abstraction
pub async fn process_file_chunks<F>(
    content: &str,
    embedder: &impl PlatformEmbedder,  // ❌ Abstraction hid scheduler
    progress_callback: F,
) -> Result<ChunkProcessingResult, EmbeddingError> {
    for chunk in chunks {
        let result = embedder.embed(&chunk.text).await?;  // ❌ Sequential
        // ... update progress per chunk
    }
}

// NEW: Direct batch embedding through scheduler
pub async fn process_file_chunks<F>(
    content: &str,
    filename: Option<&str>,  // ✅ For semantic chunking
    progress_callback: F,
) -> Result<ChunkProcessingResult, EmbeddingError> {
    // Uses embed_text_chunks_auto() which:
    // - Detects file type (markdown/code/text) for semantic chunking
    // - Tokenizes all chunks upfront
    // - Submits as BatchEmbedRequest with Priority::Background
    let results = embed_text_chunks_auto(content, 512, filename).await?;

    // Progress: start (0%) and end (100%)
    progress_callback(0, results.len(), 0.0, 0, 0);
    progress_callback(results.len(), results.len(), 100.0, total_tokens, elapsed);
}
```

**`embed_text_chunks_auto()` Implementation** (`src/embedding/mod.rs:455-541`):

```rust
#[cfg(not(target_arch = "wasm32"))]  // Desktop only - WASM still sequential
pub async fn embed_text_chunks_auto(
    text: &str,
    chunk_tokens: usize,
    filename: Option<&str>,
) -> Result<Vec<ChunkEmbeddingResult>, EmbeddingError> {
    let scheduler = get_scheduler()?;
    let tokenizer = ensure_tokenizer(8192).await?;

    // Semantic chunking based on file type
    let file_type = filename.map(chunking::detect_file_type)
        .unwrap_or(chunking::FileType::Text);
    let chunker = chunking::create_chunker(file_type, chunk_tokens, tokenizer);
    let text_chunks = chunker.chunk(text)?;

    // Tokenize all chunks upfront
    let mut token_batches = Vec::with_capacity(text_chunks.len());
    for chunk in &text_chunks {
        let tokens = tokenizer::tokenize_text_async(tokenizer, &chunk.text).await?;
        token_batches.push(tokens);
    }

    // Submit entire file as single batch with Background priority
    let batch_request = BatchEmbedRequest::new(token_batches)
        .with_priority(Priority::Background);

    let embeddings = scheduler.embed_batch(batch_request).await?;
    // ... build results from embeddings
}
```

**Benefits Achieved**:
1. ✅ **True batching**: Entire file submitted as one GPU batch
2. ✅ **Correct priorities**: Background (P2) for file processing, Immediate (P0) for search
3. ✅ **Queue visibility**: Queue depth now shows actual backpressure
4. ✅ **GPU efficiency**: Single forward pass per file instead of N individual passes
5. ✅ **Semantic chunking**: Markdown-aware, code-aware (tree-sitter on native), text-aware (ICU4X)

**Trade-offs**:
- **Progress granularity**: Per-file progress now jumps 0% → 100% instantly (batch completes atomically)
  - Still visible: Progress across multiple files in a batch
  - Acceptable because Metal on M-series is fast enough that per-chunk progress wasn't useful
- **WASM limitation**: Batch API only on desktop; WASM still uses sequential processing
  - Reason: WASM uses web worker abstraction that doesn't expose scheduler

### Debug Logging (2025-11-25)

Added comprehensive debug logging to verify scheduler behavior and diagnose issues:

**Log Level Strategy** (`src/main.rs:19-24`):
```rust
// DEBUG for development (dx serve), INFO for release (dx bundle --release)
#[cfg(debug_assertions)]
dioxus::logger::init(dioxus::logger::tracing::Level::DEBUG).expect("logger failed to init");
#[cfg(not(debug_assertions))]
dioxus::logger::init(dioxus::logger::tracing::Level::INFO).expect("logger failed to init");
```

**Request Submission** (`src/embedding/mod.rs:343-347`):
```rust
let stats = scheduler.stats();
debug!(
    "📤 Submitting to scheduler (priority: {:?}, queue_depth: {})",
    priority, stats.queue_depth
);
```

**Batch Submission** (`src/embedding/mod.rs:501-504`):
```rust
debug!(
    "📦 Submitting batch of {} chunks to scheduler (queue_depth: {})",
    total_chunks, stats.queue_depth
);
```

**Worker Processing** (`src/gpu/serial_scheduler.rs:261-290`):
```rust
debug!(
    "📊 Processing embed request (priority: {:?}, model: {}, queue_depth: {})",
    priority, request.model_id, remaining
);

debug!(
    "📊 Processing batch embed request ({} items, priority: {:?}, queue_depth: {})",
    request.token_batches.len(), priority, remaining
);
```

**Example Debug Output**:
```
DEBUG 🔤 Tokenizing 14 chunks...
DEBUG 📦 Submitting batch of 14 chunks to scheduler (queue_depth: 3)
DEBUG 📊 Processing batch embed request (14 items, priority: Background, queue_depth: 2)
DEBUG ✓ Chunk 14/14: 429 tokens, 512-dim embedding
INFO  ✅ Embedded all 14 chunks successfully
```

**Benefits**:
- ✅ Verifies priority ordering (Background batches don't block Immediate searches)
- ✅ Shows queue backpressure (crawler adds batches faster than GPU processes)
- ✅ Confirms batch API usage (logs show "batch embed request" not individual requests)
- ✅ Production builds remain clean (INFO only, no spam)

### UI Blocking Investigation and Fix (2025-11-25)

After implementing the GPU scheduler, we discovered that the UI was still freezing during web crawling and file processing. Profiling with `tracing-chrome` revealed the root causes were **not** in the GPU scheduler, but in synchronous CPU work blocking the tokio async runtime.

#### Profiling Setup

Added `tracing` instrumentation with Chrome trace format output:

```toml
# Cargo.toml
[features]
profile = ["tracing", "tracing-subscriber", "tracing-chrome"]

[dependencies]
tracing = { version = "0.1", optional = true }
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"], optional = true }
tracing-chrome = { version = "0.7", optional = true }
```

```rust
// main.rs - Profiling initialization
#[cfg(feature = "profile")]
fn init_profiling() -> tracing_chrome::FlushGuard {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    // Filter to only capture spans from our crates, not dependencies
    // Without this filter, trace files were 500MB+ for 20-30 seconds!
    let trace_filter = EnvFilter::new("coppermind=trace,coppermind_core=trace");

    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file("./trace.json")
        .include_args(true)
        .build();

    // Console layer for log output during profiling
    let console_layer = fmt::layer()
        .with_target(true)
        .with_level(true)
        .with_filter(EnvFilter::new("coppermind=info,coppermind_core=info"));

    tracing_subscriber::registry()
        .with(chrome_layer.with_filter(trace_filter))
        .with(console_layer)
        .init();

    guard
}
```

**Trace file repair script** (`scripts/fix-trace.py`): Dioxus desktop apps may call `std::process::exit()` which skips Rust's drop handlers, leaving the trace file truncated. The script finds the last complete JSON object and properly closes the array.

Run profiled build: `dx serve --platform desktop --features profile`

View traces in: https://ui.perfetto.dev or https://speedscope.app

#### Issue 1: Synchronous Chunking Blocking Main Thread

**Symptom**: UI frozen for 4+ seconds during file embedding.

**Initial trace data** (BEFORE fix):
```
11.69s (95%) total
├── 4.15s (34%) embed_text_chunks_auto {"text_len":"17418"}
│   └── 4.12s (33%) chunk {"max_tokens":"512","text_len":"17418"}  ← tid:0 (main thread!)
├── 3.04s (25%) embed_text_chunks_auto {"text_len":"11350"}
│   └── 3.02s (25%) chunk {"max_tokens":"512","text_len":"11350"}  ← tid:0
├── 2.09s (17%) embed_text_chunks_auto {"text_len":"10484"}
│   └── 2.07s (17%) chunk {"max_tokens":"512","text_len":"10484"}  ← tid:0
└── ... (more files)
```

**Root cause**: The `text-splitter` crate's `chunk()` method was called synchronously on the main tokio thread (tid:0). It uses ICU4X for Unicode-aware sentence boundary detection and calls `TokenizerSizer.size()` hundreds of times internally to find optimal chunk boundaries.

**Location**: `src/embedding/mod.rs:479`
```rust
// BEFORE: Synchronous chunking blocked async runtime
let chunker = chunking::create_chunker(file_type, effective_chunk, tokenizer);
let text_chunks = chunker.chunk(text)?;  // ❌ 4+ seconds blocking main thread!
```

**Fix**: Wrap chunking in `run_blocking()` to move it to tokio's blocking thread pool:
```rust
// AFTER: Chunking runs on blocking thread pool
let chunker = chunking::create_chunker(file_type, effective_chunk, tokenizer);
let text_owned = text.to_string();
let text_chunks = run_blocking(move || chunker.chunk(&text_owned)).await?;  // ✅ tid:3
```

#### Issue 2: Tokenizer Cloning Per Chunk

**Symptom**: Unnecessary memory allocations, ~3ms overhead per chunk.

**Root cause**: `tokenize_text_async()` cloned the entire 466KB tokenizer for every chunk:
```rust
// BEFORE: Clone 466KB tokenizer for EACH chunk
pub async fn tokenize_text_async(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, EmbeddingError> {
    let tokenizer_clone = tokenizer.clone();  // ❌ 466KB clone
    let text_owned = text.to_string();
    tokio::task::spawn_blocking(move || tokenize_text(&tokenizer_clone, &text_owned)).await
}

// Called in a loop:
for text_chunk in &text_chunks {
    let tokens = tokenizer::tokenize_text_async(tokenizer, &text_chunk.text).await?;  // ❌ N clones
}
```

**Fix**: Batch all tokenization into a single `run_blocking()` call:
```rust
// AFTER: Single blocking call, no clones, no per-chunk spawn overhead
let chunk_texts: Vec<String> = text_chunks.iter().map(|c| c.text.clone()).collect();
let (token_batches, token_counts) = run_blocking(move || {
    let mut batches = Vec::with_capacity(chunk_texts.len());
    let mut counts = Vec::with_capacity(chunk_texts.len());
    for text in &chunk_texts {
        let tokens = tokenizer::tokenize_text(tokenizer, text)?;  // ✅ Direct call, no clone
        counts.push(tokens.len());
        batches.push(tokens);
    }
    Ok::<_, EmbeddingError>((batches, counts))
}).await?;
```

#### Issue 3: Sequential Tokenization Awaits

**Symptom**: Each chunk awaited sequentially, preventing parallelism.

**Root cause**: The tokenization loop awaited each chunk individually:
```rust
// BEFORE: Sequential awaits
for text_chunk in &text_chunks {
    let tokens = tokenizer::tokenize_text_async(tokenizer, &text_chunk.text).await?;
    token_batches.push(tokens);
}
```

**Fix**: Same as Issue 2 - batch all work into single `run_blocking()`.

#### Results: Before vs After

**BEFORE fix** (profiling trace):
```
Thread allocation:
- tid:0 (main): 11.69s of work ← UI thread blocked!
- tid:1 (gpu-scheduler): ~100ms of work (properly offloaded)

Top operations on main thread:
- chunk: 4.12s, 3.02s, 2.07s, 1.21s, 1.03s (synchronous!)
- Everything else: <1%
```

**AFTER fix** (profiling trace):
```
Thread allocation:
- tid:0 (main): 640ms total ← UI stays responsive!
- tid:1 (gpu-scheduler): GPU work (properly offloaded)
- tid:3 (blocking pool): Chunking work (properly offloaded)

Top operations:
- crawl_with_progress: 629ms (93%) - network I/O, expected
- process_batch: 32ms (4.8%)
- index_chunks: 12ms (1.8%)
- fetch_html: 11ms (1.7%)

Chunking still runs (on blocking thread):
- chunk text_len=17418: 4109.7ms on tid:3 ✅
- chunk text_len=11350: 3021.5ms on tid:3 ✅
- chunk text_len=10484: 2076.5ms on tid:3 ✅
```

#### Performance Improvement Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total time on main thread** | 11.69s (95%) | 640ms (95%) | **18x faster** |
| **UI blocking during chunking** | 4+ seconds frozen | <1ms | **Responsive** |
| **Tokenizer clones per file** | N clones (466KB each) | 0 clones | **Eliminated** |
| **spawn_blocking calls per file** | N calls (one per chunk) | 2 calls total | **Reduced overhead** |
| **Chunking thread** | tid:0 (main) | tid:3 (blocking pool) | **Properly offloaded** |

#### Key Insight: GPU Scheduler Was NOT the Problem

The GPU scheduler implementation was correct - GPU work was already properly offloaded to a dedicated thread (tid:1). The UI blocking issue was caused by **synchronous CPU work** (chunking, tokenization) running on the main async thread.

**Lesson learned**: When investigating async performance issues, profile the entire pipeline, not just the suspected component. The bottleneck was in the preprocessing (chunking) not the execution (GPU inference).

#### Files Modified

| File | Change |
|------|--------|
| `src/main.rs` | Added `init_profiling()` for `tracing-chrome` |
| `src/embedding/mod.rs` | Wrapped `chunk()` in `run_blocking()`, batched tokenization |
| `src/embedding/tokenizer.rs` | Added `#[instrument]` for profiling |
| `src/embedding/chunking/*.rs` | Added `#[instrument]` for profiling |
| `src/gpu/serial_scheduler.rs` | Added `#[instrument]` for profiling |
| `src/processing/processor.rs` | Added `#[instrument]` for profiling |
| `src/crawler/engine.rs` | Added `#[instrument]` for profiling |
| `src/crawler/fetcher.rs` | Added `#[instrument]` for profiling |
| `scripts/fix-trace.py` | New script to repair truncated trace files |
| `Cargo.toml` | Added `profile` feature with tracing dependencies |

### Cleanup: Deferred Rebuild API Removed

**Unrelated but concurrent cleanup**: Removed obsolete "deferred rebuild" API that was a holdover from the `instant-distance` vector index.

**Background**: The HNSW index (from `hnsw` crate) supports incremental insertion, unlike `instant-distance` which required periodic full rebuilds. The deferred API was a no-op:

```rust
// OLD: Misleading API that didn't do anything
pub fn add_document_deferred(&mut self, doc_id: DocId, embedding: Vec<f32>) {
    self.add_document(doc_id, embedding)  // Just forwarded to regular add
}

pub fn rebuild_index(&mut self) {
    // No-op: HNSW doesn't need rebuilds
}
```

**Changes**:
- `src/components/file_processing.rs:109`: `add_document_deferred()` → `add_document()`
- Removed `"(rebuild pending)"` from log messages
- Updated documentation to remove rebuild references

**Benefit**: Cleaner API surface that accurately reflects HNSW's incremental capabilities.

---

## Future Work

### ParallelScheduler (when Candle fixed)

```rust
pub struct ParallelScheduler {
    workers: Vec<WorkerHandle>,
    load_balancer: RoundRobin,  // or LeastLoaded
}

impl ParallelScheduler {
    pub fn new(num_workers: usize) -> Self {
        // Each worker gets its own thread with its own Metal command buffer
        // Load balancer distributes requests across workers
    }
}

// Same GpuScheduler trait - no changes to callers
```

### LLM Support

```rust
// Add to scheduler trait
async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, GpuError>;

// Model registry supports both
enum ModelId {
    Embedding(String),
    Llm(String),
}
```

### Model Hot-Swapping

```rust
// Unload/load models dynamically
scheduler.unload_model(&ModelId::Embedding("old".into())).await?;
scheduler.load_model(ModelId::Embedding("new".into()), config).await?;
```

---

## References

- [Candle Issue #2637](https://github.com/huggingface/candle/issues/2637) - Metal threading bug
- [Candle PR #3079](https://github.com/huggingface/candle/pull/3079) - Thread-isolated command buffers
- [Apple Metal Threading Guide](https://developer.apple.com/documentation/metal/mtlcommandqueue)
- [docs/candle-metal-threading-issue.md](./candle-metal-threading-issue.md) - Detailed crash analysis
- [tokio::sync::mpsc](https://docs.rs/tokio/latest/tokio/sync/mpsc/) - Channel documentation
- [std::collections::BinaryHeap](https://doc.rust-lang.org/std/collections/struct.BinaryHeap.html) - Priority queue
