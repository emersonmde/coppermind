# Coppermind - Technical Design Document

This document provides a comprehensive technical overview of Coppermind's architecture, implementation details, and hard-earned lessons from building a local-first hybrid search engine in Rust that runs in browsers via WebAssembly.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Hybrid Search System](#hybrid-search-system)
3. [Browser ML with Candle](#browser-ml-with-candle)
4. [Web Worker Architecture](#web-worker-architecture)
5. [Cross-Platform Compilation](#cross-platform-compilation)
6. [Storage & Persistence](#storage--persistence)
7. [Technical Stack](#technical-stack)

---

## Architecture Overview

Coppermind is structured around three core subsystems that work together to provide local-first semantic search:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface (Dioxus)                │
│  Components: Hero, FileUpload, Search, Testing              │
└────────────────────┬────────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
┌─────────┐   ┌──────────────┐   ┌──────────────┐
│Embedding│   │Hybrid Search │   │   Storage    │
│ System  │   │    Engine    │   │   Backend    │
│         │   │              │   │              │
│JinaBERT │   │ Vector+BM25  │   │ OPFS/Native  │
│ Candle  │   │    + RRF     │   │              │
└─────────┘   └──────────────┘   └──────────────┘
```

### Module Structure

Coppermind uses a Cargo workspace with three crates:

#### `crates/coppermind-core/` - Platform-Independent Core Library
- **src/lib.rs** - Public API exports
- **src/search/** - Hybrid search implementation
  - `engine.rs` - `HybridSearchEngine` orchestrating vector + keyword search
  - `vector.rs` - HNSW vector search using `hnsw` crate (cosine distance)
  - `keyword.rs` - BM25 keyword search using `bm25` crate
  - `fusion.rs` - Reciprocal Rank Fusion algorithm
  - `aggregation.rs` - `aggregate_chunks_by_file()` for file-level grouping
  - `types.rs` - `DocId`, `Document`, `SearchResult`, `SearchError`
- **src/storage/** - `DocumentStore` trait and implementations
  - `document_store.rs` - `DocumentStore` trait, `InMemoryDocumentStore`
  - `redb_store.rs` - `RedbDocumentStore` for desktop (feature-gated)
- **src/embedding/** - ML model abstractions and implementations (Candle-based)
  - `traits.rs` - `AssetLoader`, `Embedder`, `ModelConfig` traits
  - `config.rs` - `JinaBertConfig` implementation
  - `model.rs` - `JinaBertEmbedder` (Candle-based inference)
  - `tokenizer.rs` - `TokenizerHandle` wrapper for HuggingFace tokenizers
  - `types.rs` - `EmbeddingResult`, `ChunkEmbeddingResult`
- **src/chunking/** - Text chunking strategies
  - `mod.rs` - `ChunkingStrategy` trait, `FileType` enum, `create_chunker()`
  - `text_splitter_adapter.rs` - ICU4X sentence-based chunking
  - `markdown_splitter_adapter.rs` - Markdown-aware chunking (pulldown-cmark)
  - `code_splitter_adapter.rs` - Syntax-aware chunking with tree-sitter (native only)
  - `tokenizer_sizer.rs` - Token-based chunk sizing
- **src/gpu/** - GPU scheduler for thread-safe Metal access (native only)
  - `scheduler.rs` - `GpuScheduler` trait
  - `serial_scheduler.rs` - `SerialScheduler` with dedicated worker thread
  - `types.rs` - `EmbedRequest`, `Priority`, `ModelId`
  - `error.rs` - `GpuError` types
- **src/processing/** - Document processing and indexing pipeline
  - `pipeline.rs` - `IndexingPipeline` for chunking → tokenization → embedding
  - `progress.rs` - `IndexingProgress`, `BatchProgress` for UI feedback
- **src/error.rs** - Error types (`EmbeddingError`, `ChunkingError`, `GpuError`)
- **src/config.rs** - Production configuration constants (`MAX_CHUNK_TOKENS`, `EMBEDDING_DIM`, etc.)
- **src/evaluation/** - IR evaluation framework
  - `mod.rs` - Two-tier evaluation approach (synthetic + real datasets)
  - `metrics.rs` - NDCG, MAP, MRR, Precision@k, Recall@k, F1@k
  - `stats.rs` - Bootstrap CI, paired t-test, Cohen's d effect size
  - `datasets/` - Dataset loaders (synthetic, Natural Questions)

#### `crates/coppermind-eval/` - Standalone Evaluation Tool
- **src/main.rs** - CLI for running evaluation benchmarks
- **src/datasets/` - Custom dataset definitions
- Uses `elinor` crate for additional IR metrics

#### `crates/coppermind/` - Application Crate
- **src/main.rs** - Entry point with platform-specific launch (desktop/mobile/web)
- **src/lib.rs** - Public API surface, module exports
- **src/error.rs** - Error types (`FileProcessingError`)

- **src/embedding/** - App-specific embedding utilities
  - `mod.rs` - High-level API (`compute_embedding`, `embed_text_chunks_auto`)
  - `assets.rs` - Platform-agnostic asset loading (Fetch API on web, tokio::fs on desktop)

- **src/crawler/** - Web page crawling (native only, CORS blocks web)
  - `engine.rs` - BFS crawl with depth limits and cycle detection
  - `fetcher.rs` - HTTP fetching with reqwest
  - `parser.rs` - HTML parsing and text extraction (scraper crate)

- **src/storage/** - Platform-specific `DocumentStore` implementations
  - `indexeddb_store.rs` - `IndexedDbDocumentStore` for web (WASM only)

- **src/workers/** - Web Worker for CPU-intensive embedding (WASM only)
  - `mod.rs` - `EmbeddingWorkerClient`, `start_embedding_worker()` export

- **src/processing/** - File processing pipeline
  - `embedder.rs` - `PlatformEmbedder` (worker on web, direct on desktop)
  - `processor.rs` - High-level `process_file_chunks()` pipeline

- **src/metrics.rs** - Performance metrics with rolling averages

- **src/components/** - Dioxus UI components
  - `mod.rs` - `App` component, context providers
  - `app_shell/` - Layout (appbar, footer, metrics_pane)
  - `search/` - Search UI (search_view, search_card, result_card, source_preview)
  - `index/` - Index management (index_view, upload_card, file_row, batch_list)
  - `web_crawler.rs` - Crawler UI (desktop only)
  - `worker.rs` - Platform-specific embedding coordination
  - `batch_processor.rs` - Queue management for file processing
  - `file_processing.rs` - File utilities (binary detection, directory traversal)

- **src/platform/** - Platform abstraction utilities
  - `mod.rs` - `run_blocking()`, `run_async()` (tokio on desktop, direct on web)

- **src/utils/** - General utilities
  - `formatting.rs` - Duration and timestamp formatting
  - `signal_ext.rs` - Dioxus signal utilities

---

## Hybrid Search System

Coppermind implements a hybrid search architecture that combines vector similarity search with keyword matching, merged using Reciprocal Rank Fusion. This approach delivers better results than either method alone by catching both semantic matches (paraphrases, synonyms) and exact keyword matches.

### Vector Search - HNSW Algorithm

Vector search uses the [hnsw](https://github.com/rust-cv/hnsw) crate, which implements the HNSW (Hierarchical Navigable Small World) algorithm from [Malkov & Yashunin (2018)](https://arxiv.org/abs/1603.09320). This crate was chosen for its support of incremental indexing - documents can be added one at a time without rebuilding the entire index.

**How HNSW Works:**
HNSW builds a multi-layer graph structure where each layer contains a subset of the data points. Navigation starts at the top layer (sparse) and progressively moves to denser layers, using greedy search to find approximate nearest neighbors. This hierarchical approach achieves logarithmic search complexity - O(log n) average case - making it practical for large-scale vector search.

**Implementation Details:**

The engine uses `rust-cv/hnsw` which supports incremental updates and is WASM-compatible:

```rust
pub struct VectorSearchEngine {
    /// HNSW index for semantic similarity search using cosine distance
    index: Hnsw<CosineDistance, Box<[f32]>, Pcg64, 16, 32>,
    /// Maps DocId to index position in HNSW
    doc_positions: HashMap<DocId, usize>,
    /// Embedding dimension (512 for JinaBERT)
    dimension: usize,
}
```

Distance metric: **Cosine distance** = `1 - cosine_similarity`
- Documents are represented as 512-dimensional embeddings from JinaBERT
- HNSW parameters: M=16 (bidirectional links per node), M0=32 (links at layer 0)
- Incremental indexing: documents added one at a time without full rebuild

### Keyword Search - BM25 Algorithm

Keyword search uses the [bm25](https://github.com/Michael-JB/bm25) crate, implementing the Okapi BM25 ranking function - the same algorithm used by Elasticsearch and Lucene.

**BM25 Formula:**
```
BM25(q, d) = Σ IDF(q_i) × (f(q_i, d) × (k1 + 1)) / (f(q_i, d) + k1 × (1 - b + b × |d| / avgdl))

Where:
- IDF(q_i) = log((N - df(q_i) + 0.5) / (df(q_i) + 0.5))
- f(q_i, d) = term frequency in document
- |d| = document length
- avgdl = average document length
- k1 = 1.5, b = 0.75 (standard tuning parameters)
```

**Why BM25:**
- Handles term frequency saturation (diminishing returns for repeated terms)
- Length normalization prevents bias toward longer documents
- Industry-standard algorithm with proven effectiveness

### Reciprocal Rank Fusion (RRF)

RRF merges the ranked results from vector and keyword search using a rank-based formula from [Cormack, Clarke & Buettcher (2009)](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).

**RRF Formula:**
```
RRF_score(d) = Σ 1/(k + rank_r(d))

Where:
- d = document
- r = ranker (vector search, keyword search)
- rank_r(d) = 1-indexed position of d in ranker r
- k = 60 (standard constant)
```

**Why RRF Works:**
RRF operates purely on document positions, **not raw scores**. This means you can combine vector similarity scores (0-1 range) with BM25 scores (unbounded) without normalizing their different scales. The constant k=60 reduces the impact of high rankings - the difference between rank 1 and rank 2 is small, but the difference between rank 1 and rank 100 is large.

**Example:**
```
Vector Search Results:          Keyword Search Results:
1. Doc A (0.95)  → rank 1       1. Doc C (12.5)  → rank 1
2. Doc B (0.87)  → rank 2       2. Doc A (8.3)   → rank 2
3. Doc C (0.81)  → rank 3       3. Doc D (5.1)   → rank 3

RRF Scores (k=60):
Doc A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
Doc B: 1/(60+2) = 0.0161
Doc C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
Doc D: 1/(60+3) = 0.0159

Final Ranking: Doc A > Doc C > Doc B > Doc D
```

### HybridSearchEngine Implementation

```rust
pub async fn search(
    &self,
    query_embedding: &[f32],
    query_text: &str,
    k: usize,
) -> Result<Vec<SearchResult>, SearchError> {
    // 1. Get top 2k results from vector search
    let vector_results = self.vector_engine.search(query_embedding, k * 2);

    // 2. Get top 2k results from keyword search
    let keyword_results = self.keyword_engine.search(query_text, k * 2);

    // 3. Fuse using RRF (k=60 constant)
    let fused = reciprocal_rank_fusion(&vector_results, &keyword_results, 60);

    // 4. Return top k with document metadata
    Ok(fused.into_iter().take(k).collect())
}
```

Why fetch 2k results for fusion: RRF needs sufficient overlap between result sets. Fetching more candidates ensures documents ranked highly in one method but lowly in another aren't prematurely excluded.

---

## Browser ML with Candle

Coppermind runs JinaBERT embedding inference directly in the browser using [Candle](https://github.com/huggingface/candle), a minimalist ML framework from Hugging Face designed for Rust-first workflows.

### Model: JinaBERT-v2-small-en

**Model Card:** [jinaai/jina-embeddings-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)

**Specifications:**
- Output: 512-dimensional embeddings
- Max sequence length: 8192 tokens (via ALiBi positional embeddings)
- Current configuration: 2048 tokens (configurable via `max_position_embeddings`)
- Model size: 65MB (safetensors format, F16 weights auto-converted to F32 for WASM)
- Tokenizer: 695KB (JSON format)

**ALiBi Positional Embeddings:**
ALiBi (Attention with Linear Biases) enables length extrapolation - the model can handle sequences longer than it was trained on. Memory usage scales as `heads × seq_len² × 4 bytes`:
- 2048 tokens: 8 heads × 2048² × 4 = ~134MB
- 4096 tokens: 8 heads × 4096² × 4 = ~537MB
- 8192 tokens: 8 heads × 8192² × 4 = ~2GB

Current 2048 token limit balances context window with memory constraints in WASM's 512MB heap.

### Candle vs JavaScript ML Frameworks

**Traditional approach:** TensorFlow.js, ONNX Runtime Web, Transformers.js
- Written in JavaScript with WASM backends
- Bundler overhead, runtime interpretation
- Limited type safety

**Candle approach:**
- Pure Rust implementation compiled to WASM
- Compile-time type safety
- Zero-cost abstractions
- Smaller bundle sizes
- Near-native performance

### Model Loading Strategy

**Lazy Loading with Caching:**
The embedding module (`src/embedding/`) uses a singleton pattern for efficient model reuse:

```rust
// src/embedding/model.rs
static MODEL_CACHE: OnceCell<Arc<dyn Embedder>> = OnceCell::new();

pub fn get_or_load_model(
    model_bytes: Vec<u8>,
    vocab_size: usize,
    config: JinaBertConfig,
) -> Result<Arc<dyn Embedder>, EmbeddingError> {
    if let Some(existing) = MODEL_CACHE.get() {
        return Ok(existing.clone());
    }

    let model = JinaBertEmbedder::from_bytes(model_bytes, vocab_size, config)?;
    MODEL_CACHE.set(Arc::new(model)).ok();

    Ok(MODEL_CACHE.get().unwrap().clone())
}
```

**Platform-Specific Asset Loading:**
The `src/embedding/assets.rs` module handles cross-platform asset fetching:
- **Web:** HTTP fetch with base path resolution (from `DIOXUS_ASSET_ROOT` meta tag or worker global)
- **Desktop:** Filesystem read from multiple search locations (app bundle resources, exe directory, working directory)

Both paths use `spawn_blocking` on desktop to prevent UI freezing when loading the 65MB model file.

### Tokenization

The `src/embedding/tokenizer.rs` module wraps [tokenizers-rs](https://github.com/huggingface/tokenizers):

```rust
// src/embedding/tokenizer.rs
pub fn tokenize_text(tokenizer: &Tokenizer, text: &str)
    -> Result<Vec<u32>, EmbeddingError>
{
    let encoding = tokenizer.encode(text, true)?;
    let ids = encoding.get_ids();
    Ok(ids.to_vec())
}

// Async wrapper that uses spawn_blocking on desktop
pub async fn tokenize_text_async(tokenizer: &Tokenizer, text: &str)
    -> Result<Vec<u32>, EmbeddingError>
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        tokio::task::spawn_blocking(/* ... */).await?
    }

    #[cfg(target_arch = "wasm32")]
    {
        tokenize_text(tokenizer, text)
    }
}
```

### Inference Pipeline

The public API in `src/embedding/mod.rs` provides high-level functions:

```rust
// src/embedding/mod.rs
pub async fn compute_embedding(text: &str)
    -> Result<EmbeddingComputation, EmbeddingError>
{
    let model = get_or_load_model().await?;
    let max_positions = model.max_position_embeddings();
    let tokenizer = ensure_tokenizer(max_positions).await?;

    let token_ids = tokenize_text_async(tokenizer, text).await?;
    let token_count = token_ids.len();

    // Platform-specific: spawn_blocking on desktop, direct on web
    let embedding = {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let model_clone = model.clone();
            tokio::task::spawn_blocking(move ||
                model_clone.embed_tokens(token_ids)
            ).await??
        }

        #[cfg(target_arch = "wasm32")]
        {
            model.embed_tokens(token_ids)?
        }
    };

    Ok(EmbeddingComputation { token_count, embedding })
}
```

Normalization (L2) happens inside the model's `embed_tokens()` method, ensuring embeddings are unit vectors for efficient cosine similarity via dot product.

### Modular Architecture for Multi-Model Support

The embedding module was refactored with extensibility in mind, anticipating support for multiple models with different architectures and parameters:

**Trait-Based Design:**
```rust
// src/embedding/config.rs
pub trait ModelConfig {
    fn model_id(&self) -> &str;
    fn embedding_dim(&self) -> usize;
    fn max_sequence_length(&self) -> usize;
    fn normalize_embeddings(&self) -> bool;
}

// src/embedding/model.rs
pub trait Embedder: Send + Sync {
    fn max_position_embeddings(&self) -> usize;
    fn embedding_dim(&self) -> usize;
    fn embed_tokens(&self, token_ids: Vec<u32>) -> Result<Vec<f32>, EmbeddingError>;
    fn embed_batch_tokens(&self, batch: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}
```

This design allows adding new models (e.g., sentence-transformers, OpenAI-style models, domain-specific encoders) by:
1. Implementing `ModelConfig` for model-specific configuration
2. Implementing `Embedder` for inference logic
3. Updating `get_or_load_model()` to support model selection

**Benefits:**
- Different embedding dimensions (256D, 512D, 768D, 1024D)
- Different sequence lengths (512, 1024, 2048, 8192 tokens)
- Different architectures (BERT, RoBERTa, T5, custom models)
- Specialized models (code embeddings, multilingual, domain-specific)

---

## Web Worker Architecture

Running ML inference in the main browser thread freezes the UI. Web Workers solve this by offloading CPU-intensive embedding computation to a background thread.

### Architecture Diagram

```
Main Thread                          Worker Thread
───────────                          ─────────────
App Component                        embedding-worker.js
    │                                        │
    ▼                                        ▼
EmbeddingWorkerClient               shimDomForWorker()
    │                                        │
    │  postMessage({                         ▼
    │    requestId: 1,              Load WASM module
    │    text: "..."                         │
    │  })                                    ▼
    │ ──────────────────────────────►  start_embedding_worker()
    │                                        │
    │                                        ▼
    │                                compute_embedding(text)
    │                                        │
    │  ◄────────────────────────────  postMessage({
    │       {                              requestId: 1,
    │         requestId: 1,                 embedding: [...]
    │         embedding: [...]              tokenCount: 42
    │       }                            })
    ▼
Update UI signal
```

### Main Thread: EmbeddingWorkerClient

```rust
pub struct EmbeddingWorkerClient {
    worker: Worker,
    pending: HashMap<u32, oneshot::Sender<Result<EmbeddingComputation, String>>>,
    next_id: Cell<u32>,
    ready_state: RefCell<ReadyState>,
}

impl EmbeddingWorkerClient {
    pub async fn embed(&self, text: String) -> Result<EmbeddingComputation, String> {
        self.wait_until_ready().await?;

        let request_id = self.next_id.get();
        self.next_id.set(request_id + 1);

        let (tx, rx) = oneshot::channel();
        self.pending.borrow_mut().insert(request_id, tx);

        // Send to worker
        self.worker.post_message(&JsValue::from_serde(&WorkerRequest {
            request_id,
            text,
        })?)?;

        // Wait for response
        rx.await?
    }
}
```

### Worker Thread: Bootstrap Script

The worker bootstrap (`assets/workers/embedding-worker.js`) solves a critical problem: the worker needs to load the WASM module at the correct path.

```javascript
// Wait for init message from main thread
const initPromise = new Promise(resolve => {
    self.addEventListener('message', (event) => {
        if (event.data.type === 'init') {
            resolve(event.data.wasmJsPath);
        }
    });
});

async function boot() {
    const relativeJsPath = await initPromise;
    const absoluteJsUrl = new URL(relativeJsPath, self.location.href).toString();

    // Shim DOM APIs (wasm-bindgen expects window, document, etc.)
    shimDomForWorker();

    // Dynamically import WASM module
    const module = await import(absoluteJsUrl);
    await module.default();  // Initialize WASM

    // Call Rust entry point
    module.start_embedding_worker();
}

boot();
```

### Worker Thread: Rust Entry Point

```rust
#[wasm_bindgen]
pub fn start_embedding_worker() -> Result<(), JsValue> {
    let scope: DedicatedWorkerGlobalScope = js_sys::global().dyn_into()?;

    let handler = Closure::wrap(Box::new(move |event: MessageEvent| {
        let request: WorkerRequest = serde_wasm_bindgen::from_value(event.data())?;

        spawn_local(async move {
            let result = compute_embedding(&request.text).await;

            let response = match result {
                Ok(embedding) => WorkerResponse {
                    request_id: request.request_id,
                    embedding: embedding.embedding,
                    token_count: embedding.token_count,
                },
                Err(err) => /* ... error response ... */,
            };

            scope.post_message(&serde_wasm_bindgen::to_value(&response)?)?;
        });
    }) as Box<dyn FnMut(_)>);

    scope.set_onmessage(Some(handler.as_ref().unchecked_ref()));
    handler.forget();

    // Signal ready to main thread
    scope.post_message(&JsValue::from_str("ready"))?;
    Ok(())
}
```

### Message Protocol

**Main → Worker:**
```javascript
{
    type: "init",
    wasmJsPath: "/coppermind/assets/coppermind-dxh...js"
}
// Then:
{
    requestId: 1,
    text: "document text to embed"
}
```

**Worker → Main:**
```javascript
// Ready signal
"ready"
// Then:
{
    requestId: 1,
    embedding: Float32Array(512),
    tokenCount: 42
}
```

### Desktop: No Worker Needed

On desktop, embeddings run directly in async context:

```rust
#[cfg(not(target_arch = "wasm32"))]
pub async fn embed_text(text: &str) -> Result<Vec<f32>, String> {
    // Direct call, no worker
    compute_embedding(text).await
}
```

Tokio's async runtime handles concurrency without blocking the UI thread.

---

## Cross-Platform Compilation

The entire codebase compiles to both WebAssembly (browsers) and native binaries (desktop) from a single Rust codebase.

### Conditional Compilation

Platform-specific code uses `#[cfg()]` attributes:

```rust
// Web-only imports
#[cfg(target_arch = "wasm32")]
use web_sys::{FileSystemDirectoryHandle, Worker};

// Desktop-only imports
#[cfg(not(target_arch = "wasm32"))]
use tokio::fs;

// Conditional logic
#[cfg(target_arch = "wasm32")]
{
    // Web: Use IndexedDB storage
    let store = IndexedDbDocumentStore::new("coppermind").await?;
}

#[cfg(not(target_arch = "wasm32"))]
{
    // Desktop: Use redb storage
    let store = RedbDocumentStore::new(data_dir.join("coppermind.redb"))?;
}
```

### Platform Differences

| Feature | Web | Desktop |
|---------|-----|---------|
| UI Rendering | DOM manipulation | Native window (webview) |
| Storage | IndexedDB | redb |
| Threading | Web Workers | Native threads |
| Asset Loading | HTTP fetch | Filesystem read |
| Model Inference | CPU (WASM) | CPU/GPU (CUDA/Metal) |

### Dioxus: React-like, Rust-native

[Dioxus](https://dioxuslabs.com/) provides a familiar component model with hooks, props, and reactive state management - but compiles to native code instead of running in a JavaScript runtime.

**Key advantages:**
- Statically typed at compile time
- Zero-cost abstractions
- No bundler/transpiler overhead
- Platform-specific rendering handled transparently

**Component example:**
```rust
#[component]
pub fn SearchBox(query: Signal<String>) -> Element {
    rsx! {
        input {
            r#type: "text",
            value: "{query}",
            oninput: move |evt| query.set(evt.value()),
            placeholder: "Search documents..."
        }
    }
}
```

Same component compiles to:
- **Web:** DOM `<input>` element with event handlers
- **Desktop:** Native text input widget
- **Mobile:** Platform-native input (iOS/Android)

### Build Commands

```bash
# Web (WASM)
dx serve                    # Dev server
dx bundle --release         # Production build

# Desktop (native)
dx serve --platform desktop
dx bundle --release --platform desktop

# Mobile (future)
dx bundle --platform ios
dx bundle --platform android
```

---

## Storage & Persistence

Documents and search indexes are persisted using platform-appropriate storage backends via a common trait.

**ADR:** [007-document-storage-reupload-handling.md](adrs/007-document-storage-reupload-handling.md)

### DocumentStore Trait

The `DocumentStore` trait provides a unified interface for document and embedding storage across platforms:

```rust
#[async_trait(?Send)]
pub trait DocumentStore: Send + Sync {
    // Document operations
    async fn store_document(&self, doc_id: DocId, record: &DocumentRecord) -> Result<(), StoreError>;
    async fn get_document(&self, doc_id: DocId) -> Result<Option<DocumentRecord>, StoreError>;
    async fn delete_document(&self, doc_id: DocId) -> Result<(), StoreError>;
    async fn list_documents(&self) -> Result<Vec<(DocId, DocumentRecord)>, StoreError>;

    // Embedding operations
    async fn store_embedding(&self, doc_id: DocId, embedding: &[f32]) -> Result<(), StoreError>;
    async fn get_embedding(&self, doc_id: DocId) -> Result<Option<Vec<f32>>, StoreError>;
    async fn delete_embedding(&self, doc_id: DocId) -> Result<(), StoreError>;
    async fn list_embeddings(&self) -> Result<Vec<(DocId, Vec<f32>)>, StoreError>;

    // Source tracking (for re-upload detection)
    async fn store_source(&self, source_id: &str, record: &SourceRecord) -> Result<(), StoreError>;
    async fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>, StoreError>;
    async fn delete_source(&self, source_id: &str) -> Result<(), StoreError>;
    async fn list_sources(&self) -> Result<Vec<String>, StoreError>;

    // Bulk operations
    async fn clear(&self) -> Result<(), StoreError>;
}
```

### Platform Implementations

| Platform | Store | Lookup | Characteristics |
|----------|-------|--------|-----------------|
| Desktop/Mobile | [redb](https://github.com/cberner/redb) | O(log n) | Pure Rust B-tree, ACID transactions, no external deps |
| Web | [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) | O(1) | Browser-native, zero bundle cost |
| Testing | `InMemoryDocumentStore` | O(1) | In-memory HashMap for unit tests |

### Web: IndexedDB

IndexedDB is the browser-native key-value store, providing excellent performance for structured data with zero bundle cost.

**Implementation:** `crates/coppermind/src/storage/indexeddb_store.rs`

### Desktop: redb

[redb](https://github.com/cberner/redb) is a pure Rust embedded database providing ACID transactions and efficient B-tree lookups.

**Implementation:** `crates/coppermind-core/src/storage/redb_store.rs`

**Storage location:** Platform-specific app data directory (e.g., `~/Library/Application Support/com.coppermind.app/` on macOS).

### Source Tracking and Re-Upload Detection

The storage layer tracks document sources with SHA-256 content hashes to enable intelligent re-upload handling:

```rust
pub struct SourceRecord {
    pub content_hash: String,     // SHA-256 of file contents
    pub doc_ids: Vec<DocId>,      // Chunks belonging to this source
    pub complete: bool,           // Indexing completed successfully
    pub indexed_at: u64,          // Unix timestamp
}
```

**Update detection logic:**
- `source_id exists + hash matches` → **SKIP** (unchanged)
- `source_id exists + hash differs` → **UPDATE** (delete old chunks, re-index)
- `source_id not found` → **ADD** (new source)

### Tombstone-Based HNSW Deletion

Since rust-cv/hnsw doesn't support deletion, we implement soft-delete with background compaction:

- `mark_deleted(doc_id)` - O(1) tombstone marking
- Search filters tombstones with oversampling (fetch 2x candidates, filter, return top-k)
- Background compaction rebuilds index when tombstone ratio > 30%

This is the industry-standard approach used by Weaviate, Milvus, and hnswlib.

---

## Technical Stack

### Core Framework
- **[Dioxus 0.7](https://dioxuslabs.com/)** - Reactive UI framework (React-like component model, cross-platform rendering)
- **[Candle 0.8](https://github.com/huggingface/candle)** - ML inference framework (Hugging Face Rust ML library)

### Machine Learning
- **[JinaBERT-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)** - Embedding model (512-dimensional, ALiBi positional embeddings, 8192 token capacity)
- **[tokenizers-rs 0.20](https://github.com/huggingface/tokenizers)** - Tokenization (Hugging Face Transformers tokenizer in Rust)

### Search Infrastructure
- **[hnsw 0.11](https://github.com/rust-cv/hnsw)** - Vector search (HNSW approximate nearest neighbor from Malkov & Yashunin 2018, incremental indexing)
- **[space 0.17](https://github.com/rust-cv/space)** - Distance metrics (cosine distance for HNSW)
- **[bm25 2.3](https://github.com/Michael-JB/bm25)** - Keyword search (Okapi BM25 ranking with TF-IDF)
- **Reciprocal Rank Fusion** - Result fusion (rank-based merging from Cormack et al. SIGIR 2009)

### Storage & Persistence
- **[IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)** - Web storage (browser-native key-value store, O(1) lookups, zero bundle cost)
- **[redb 2.4](https://github.com/cberner/redb)** - Desktop/Mobile storage (pure Rust B-tree database, ACID transactions, O(log n) lookups)

### Browser Integration
- **Web Workers** - Background processing (ML inference off main thread, message passing with WASM binary)

### Build & Tooling
- **[Dioxus CLI](https://dioxuslabs.com/)** - Build tool (dx serve, dx bundle, platform targeting)
- **[wasm-bindgen](https://github.com/rustwasm/wasm-bindgen)** - WASM/JS interop (zero-cost bindings)

---

## References

### Academic Papers
- Malkov, Y. A., & Yashunin, D. A. (2018). [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320). IEEE TPAMI.
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). [Reciprocal rank fusion outperforms condorcet and individual rank learning methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf). SIGIR.

### Web Standards & Specifications
- [Origin Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) - MDN Web Docs
- [Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API) - MDN Web Docs

### Rust Ecosystem
- [Dioxus Documentation](https://dioxuslabs.com/learn/0.6/)
- [Candle Examples](https://github.com/huggingface/candle/tree/main/candle-examples)
- [wasm-bindgen Book](https://rustwasm.github.io/wasm-bindgen/)

### Performance Benchmarks
- [LocalStorage vs IndexedDB vs OPFS Performance](https://rxdb.info/articles/localstorage-indexeddb-cookies-opfs-sqlite-wasm.html) - RxDB
