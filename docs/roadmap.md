# Coppermind - Roadmap & Implementation Plan

## Vision

Build a cross-platform (Web, Desktop, Mobile) semantic search application where:
1. Users select and process files locally
2. Text is chunked and embedded using local ML models
3. Embeddings stored in local vector database
4. Fast semantic search across all processed documents

All processing happens client-side without server dependencies.

---

## Platform Strategy

### Current: Web (WASM)
```bash
dx serve                    # Browser-based development
dx bundle --release         # Deploy to GitHub Pages
```

**Status:** ✅ Working POC with JinaBERT embeddings

### Near-term: Desktop
```bash
dx serve --platform desktop
```

**Desktop-Specific Advantages:**
- Native threading (rayon) instead of Web Workers
- Direct file system access (no streaming needed)
- Native SQLite (rusqlite) instead of wa-sqlite/IndexedDB
- More memory available (>4GB)
- SIMD optimizations available
- Potential GPU acceleration (CUDA/Metal)

**Desktop Implementation Notes:**
- Use `cfg(not(target_arch = "wasm32"))` for desktop-only code
- Model loading: From local file system instead of assets
- Storage: rusqlite with standard SQLite file
- Threading: rayon for parallel embedding

### Future: Mobile
```bash
dx serve --platform mobile
```

**Mobile Considerations:**
- Smaller models needed (memory constraints)
- Battery/thermal management
- Touch UI optimizations
- Storage limits more restrictive

---

## Milestone Completion Checklist

**This checklist MUST be completed before marking any milestone as done:**

1. **Write Tests (Preferred):**
   - [ ] Write unit/integration tests for new functionality
   - [ ] Prefer automated tests over manual UAT when possible

2. **Quality Checks:**
   - [ ] Run `.githooks/pre-commit` successfully
     - Covers: fmt, clippy, tests, cargo audit, web build

3. **Manual UAT (if automated tests insufficient):**
   - [ ] Ask user to run `dx serve` with specific UAT checklist
   - [ ] Ask user to run `dx serve --platform desktop` with specific UAT checklist
   - **IMPORTANT:** Do NOT run interactive commands yourself

4. **Documentation:**
   - [ ] Update `CLAUDE.md` if architecture or module structure changed
   - [ ] Update relevant `docs/*.md` files with new patterns/implementations
   - [ ] Update `docs/roadmap.md` to mark milestone complete
   - [ ] Add new documentation if introducing new concepts

**Only after ALL items checked: Milestone is complete ✅**

---

## Current Implementation (Completed ✅)

### Milestone 1: Browser ML Inference POC ✅
**Status:** Complete

**Implemented:**
- ✅ Dioxus web app scaffolding
- ✅ JinaBERT model loading via Candle (262MB safetensors)
- ✅ Tokenization with tokenizers-rs
- ✅ Single-text embedding generation
- ✅ File upload and chunked embedding
- ✅ Cross-Origin Isolation (COOP/COEP) via Service Worker
- ✅ Web Workers for parallel CPU tasks (demo)
- ✅ WebGPU compute shader demo
- ✅ Cosine similarity computation

**Tech Stack:**
- **ML Framework:** Candle (Rust-native, WASM-first)
- **Model:** jinaai/jina-embeddings-v2-small-en (512-dim, 8192 token support)
- **Platform:** Dioxus 0.6 (web + desktop support)
- **Deployment:** GitHub Pages

**Key Files:**
- `src/main.rs` - App entry + COI setup
- `src/embedding.rs` - JinaBERT inference
- `src/components.rs` - UI components
- `src/cpu.rs` - Web Worker demo
- `src/wgpu.rs` - WebGPU demo

**See:** `docs/` for technical deep-dives

---

### Milestone 2: Hybrid Search System ✅
**Status:** Complete (December 2024)

**Implemented:**
- ✅ **Vector Search**: instant-distance HNSW for semantic similarity
  - Cosine distance metric for 512D embeddings
  - Efficient nearest neighbor retrieval
  - Automatic index rebuilding on document addition
- ✅ **Keyword Search**: BM25 for exact keyword matching
  - Term frequency-inverse document frequency scoring
  - Fast full-text search over document corpus
  - English language support via bm25 crate
- ✅ **RRF Fusion**: Reciprocal Rank Fusion algorithm
  - Merges vector and keyword rankings (k=60)
  - Robust to score scale differences
  - Combines semantic understanding + exact matches
- ✅ **Storage Backend**: Cross-platform persistence layer
  - `StorageBackend` trait for key-value abstraction
  - OPFS implementation for web (large binary data)
  - Native filesystem implementation for desktop (tokio::fs)
- ✅ **Cross-Platform Logging**: Dioxus logger integration
  - Unified logging via `tracing` crate
  - Browser console on web, stdout on desktop
  - No manual platform-specific code required
- ✅ **Test UI**: Hybrid search test button with detailed logging
  - Shows vector search results (semantic scores)
  - Shows keyword search results (BM25 scores)
  - Shows final RRF fused rankings
  - Validates search quality on both platforms

**Tech Stack:**
- **Vector Search:** instant-distance 0.6 (pure Rust HNSW)
- **Keyword Search:** bm25 2.3 (TF-IDF implementation)
- **Storage:** OPFS (web), tokio::fs (desktop)
- **Serialization:** bincode 1.3
- **Platform:** Both web and desktop verified working

**Key Files:**
- `src/search/engine.rs` - HybridSearchEngine orchestration
- `src/search/vector.rs` - HNSW vector search
- `src/search/keyword.rs` - BM25 keyword search
- `src/search/fusion.rs` - RRF algorithm
- `src/search/types.rs` - Shared types (DocId, SearchResult, etc.)
- `src/storage/opfs.rs` - Web storage implementation
- `src/storage/native.rs` - Desktop storage implementation

**Architecture Decision:**
See `docs/adr-0001-hybrid-search-architecture.md` for detailed rationale on choosing instant-distance + BM25 over alternatives (USearch, tantivy, etc.)

**Test Results:**
Query: "machine learning neural networks"
- Vector: Ranks by semantic similarity (0.9949, 0.9947, 0.9509)
- Keyword: Ranks by term frequency (3.27, 3.27, N/A)
- RRF Fusion: Balanced ranking (0.0325, 0.0325, 0.0159) ✓

---

## UI Architecture

### Current: Single View with POC Tests
The application currently has one view (`TestControls` component) containing:
- CPU Workers demo (parallel computation test)
- WebGPU compute shader demo
- Embedding generation demo (file upload + chunking)

**Purpose:** Validate that Rust + WASM can handle computationally expensive tasks in browsers.

### Future: Semantic Search View
Once storage and search are implemented, add a new view for actual semantic search:
- Document upload and indexing
- Search input
- Results display with ranking

**Eventually:** Remove POC test view once semantic search is production-ready.

---

## Next Milestones

> **NOTE:** Milestones 2-3 below are superseded by **ADR-0001: Hybrid Search Architecture**
> See `docs/adr-0001-hybrid-search-architecture.md` for the updated architecture using USearch + BM25.

### Milestone 1.5: Non-Blocking Compute Operations (IMMEDIATE) 🎯
**Goal:** Ensure ALL computationally expensive operations run off the main thread

**Current Problem:**
- **Embedding** (components.rs:109-128, 155-200): Blocks main thread despite using `spawn()`
  - `run_embedding()` and `embed_text_chunks()` are CPU-intensive synchronous work
  - UI freezes for seconds during processing
- **WebGPU** (components.rs:86-102): `test_webgpu()` may block during shader compilation
- **General Issue:** `spawn()` creates async task but doesn't move compute off main thread
- Poor user experience, violates browser responsiveness best practices

**Current Status Check:**
- ✅ **CPU Workers Test** (components.rs:32-77): Already uses Web Workers correctly
- ⚠️ **GPU Test**: May block during shader compilation/execution
- ❌ **Embedding Test**: Blocks main thread
- ❌ **File Embedding**: Blocks main thread (major issue)

**Solution: Web Workers for All Heavy Compute**

Move ALL CPU/GPU-intensive operations to dedicated Web Workers:

**Implementation Tasks:**
- [ ] **Audit All Compute Operations**
  - Review `src/wgpu.rs`, `src/embedding.rs`, `src/components.rs`
  - Identify any synchronous CPU/GPU-intensive functions
  - Ensure all run in Web Workers (web) or spawn_blocking (desktop)

- [ ] **Create Embedding Worker** (Priority 1 - biggest UX issue)
  - New file: `src/worker/embedding_worker.rs`
  - Initialize JinaBERT model inside worker
  - Listen for embedding requests via `postMessage`
  - Return embeddings via message passing
  - Handle errors gracefully

- [ ] **Verify WebGPU Non-Blocking** (Priority 2)
  - Check if `test_webgpu()` blocks during shader compilation
  - If blocking: Move WebGPU compute to worker
  - Ensure shader compilation happens asynchronously
  - Test with large compute workloads (verify UI stays responsive)

- [ ] **Worker Message Protocol**
  ```rust
  // Message types (pseudo-code)
  enum WorkerMessage {
      // From main → worker
      InitModel { model_bytes: Vec<u8> },
      EmbedText { request_id: u64, text: String, chunk_tokens: usize },
      EmbedBatch { request_id: u64, texts: Vec<String> },

      // From worker → main
      ModelReady,
      Progress { request_id: u64, chunk_index: usize, total_chunks: usize },
      ChunkComplete { request_id: u64, chunk: ChunkEmbeddingResult },
      Complete { request_id: u64, chunks: Vec<ChunkEmbeddingResult> },
      Error { request_id: u64, error: String },
  }
  ```

- [ ] **Update Components**
  - Modify `TestControls` file upload handler
  - Send text to worker instead of calling `embed_text_chunks` directly
  - Handle progress updates (show per-chunk progress)
  - Update UI reactively as chunks complete
  - Show loading indicator while worker is busy

- [ ] **Worker Lifecycle Management**
  - Initialize worker on app startup
  - Keep worker alive for session (avoid reload overhead)
  - Clean shutdown on app close
  - Handle worker crashes gracefully

- [ ] **Future: Add Rayon for Batch Parallelism**
  - Use `wasm-bindgen-rayon` inside worker
  - Parallel process multiple chunks with Rayon
  - Requires nightly Rust (already acceptable)
  - Leverages existing COEP/COIP setup
  ```rust
  // Inside worker, with rayon
  let embeddings: Vec<Vec<f32>> = chunks
      .par_iter()  // Rayon parallel iterator
      .map(|chunk| model.embed_tokens(chunk))
      .collect()?;
  ```

**Platform Differences:**
```rust
#[cfg(target_arch = "wasm32")]
{
    // Web: Use Web Worker for task isolation
    let worker = EmbeddingWorker::new().await?;
    worker.embed_text(text).await?
}

#[cfg(not(target_arch = "wasm32"))]
{
    // Desktop: Use tokio::spawn or rayon directly (already non-blocking)
    tokio::task::spawn_blocking(|| {
        embed_text_chunks(text, chunk_tokens)
    }).await?
}
```

**Architecture Diagram:**
```
┌─────────────────────────────────────────────────────────┐
│                      Main Thread                        │
│  ┌──────────────┐        ┌──────────────┐              │
│  │  UI (Dioxus) │◄──────►│ Worker Proxy │              │
│  └──────────────┘        └───────┬──────┘              │
└───────────────────────────────────┼──────────────────────┘
                                    │ postMessage
                                    │
┌───────────────────────────────────▼──────────────────────┐
│                   Web Worker Thread                      │
│  ┌────────────────────────────────────────────┐          │
│  │ Embedding Worker                           │          │
│  │  ├─ JinaBERT Model (262MB)                │          │
│  │  ├─ Tokenizer                              │          │
│  │  ├─ embed_text_chunks() logic             │          │
│  │  └─ Optional: Rayon thread pool            │          │
│  │     (for parallel batch processing)        │          │
│  └────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

**Performance Expectations:**
- **UI Responsiveness:** Main thread stays responsive (<16ms frame time)
- **Throughput:** Same or better (1 worker = baseline, worker + rayon = 2-4x speedup)
- **Memory:** +~10MB overhead for worker (model shared or copied)
- **Latency:** +5-10ms overhead for message passing (negligible vs 50-500ms inference)

**Success Criteria:**
- ✅ UI never freezes during ANY compute operation (embedding, GPU, future ML tasks)
  - Can scroll, click, interact at all times
  - Browser devtools Performance tab shows main thread <16ms per frame
- ✅ Progress updates show in real-time during processing
- ✅ Can cancel long-running operations mid-process
- ✅ Workers survive errors and can process subsequent requests
- ✅ Works on web platform (desktop already non-blocking via tokio::spawn_blocking)
- ✅ Establish **pattern/guideline** for future compute operations:
  - "If operation takes >50ms, move to Web Worker"
  - Document worker creation template
  - Add to CLAUDE.md or new doc
- ✅ Passes all items in **Milestone Completion Checklist** (see above)

**References:**
- [Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
- [wasm-bindgen Web Worker example](https://rustwasm.github.io/wasm-bindgen/examples/wasm-in-web-worker.html)
- [Dioxus async patterns](https://dioxuslabs.com/learn/0.5/reference/async)

---

### Milestone 2: Vector Storage (SUPERSEDED) ⚠️
> **Status:** Superseded by ADR-0001
>
> This milestone proposed separate IndexedDB (web) + SQLite (desktop) implementations.
> **New approach:** See ADR-0001 for unified hybrid search with USearch + BM25 + OPFS storage.

**Original Goal:** Persist embeddings locally with search capability

<details>
<summary>Original Implementation Tasks (click to expand)</summary>

**Implementation Tasks:**
- [ ] **Web:** Implement IndexedDB storage
  - Schema: documents, chunks, embeddings tables
  - Use `rexie` crate for IndexedDB access
  - Store chunk text + 512-dim F32 vectors
- [ ] **Desktop:** Implement SQLite storage
  - Use `rusqlite` with bundled SQLite
  - Same schema as web version
  - Consider FTS5 for text search
- [ ] **Both:** Unified VectorStore trait with insert, search, and clear methods
- [ ] Add "Clear Database" button for testing
- [ ] Write unit tests for VectorStore implementations

**Platform Differences:**
```rust
#[cfg(target_arch = "wasm32")]
use IndexedDBStore;

#[cfg(not(target_arch = "wasm32"))]
use SqliteStore;
```

**Success Criteria:**
- ✅ Process file → chunks → embeddings → stored
- ✅ Reload app, data persists
- ✅ Can retrieve and display stored embeddings
- ✅ Works on both web and desktop platforms
- ✅ Passes all items in **Milestone Completion Checklist** (see above)

</details>

---

### Milestone 3: Semantic Search UI 🎯
**Goal:** Query stored embeddings and display results

**Implementation Tasks:**
- [ ] Search input component
- [ ] Embed query text using same model
- [ ] Brute-force cosine similarity search (OK for <10K vectors)
- [ ] Display top-k results with:
  - Similarity score
  - Source document name
  - Chunk text (with highlighting)
- [ ] Click result → show context (surrounding chunks)
- [ ] Write tests for search functionality

**Performance Target:**
- Search across 1K chunks: <100ms (web), <50ms (desktop)

**Success Criteria:**
- ✅ User can enter search query
- ✅ Results display ranked by similarity
- ✅ Results are accurate and relevant
- ✅ Performance meets targets
- ✅ Passes all items in **Milestone Completion Checklist** (see above)

**Future Optimization:**
- For >10K vectors: Consider HNSW indexing (usearch, hnswlib)

---

### Milestone 4: Multi-File Processing
**Goal:** Batch file processing with progress tracking

**Implementation Tasks:**
- [ ] File list management (add/remove files)
- [ ] Process files sequentially or parallel (desktop: parallel)
- [ ] Progress bars:
  - Per-file progress (chunks processed)
  - Overall progress (N of M files)
- [ ] Statistics display:
  - Total files processed
  - Total chunks
  - Total embeddings stored
  - Time elapsed
- [ ] Error handling:
  - Skip failed files
  - Show error messages
  - Continue processing
- [ ] Write tests for file processing logic

**Platform Differences:**
```rust
// Web: Sequential (avoid memory pressure)
for file in files {
    process_file(file).await?;
}

// Desktop: Parallel with rayon
files.par_iter()
    .try_for_each(|file| process_file(file))?;
```

**Success Criteria:**
- ✅ Can process 10+ files in one session
- ✅ Progress accurately reflects status
- ✅ Errors don't crash the app
- ✅ Works on both platforms
- ✅ Passes all items in **Milestone Completion Checklist** (see above)

---

### Milestone 5: WASM Multi-Threading (Web Parallel Processing) 🎯
**Goal:** Parallel embedding inference in browser using WASM threads + SharedArrayBuffer

**Why This Is Core:**
- This is exactly what this project is about: client-side semantic search more performant than JS
- Demonstrates Rust WASM threading superiority
- Makes multi-file processing practical on web

**Implementation Tasks:**
- [ ] Build with `target-feature=+atomics,+bulk-memory`
  ```toml
  # .cargo/config.toml additions for wasm32-unknown-unknown
  rustflags = [
    # ... existing flags ...
    "-C", "target-feature=+atomics,+bulk-memory,+mutable-globals",
  ]
  ```
- [ ] Test rayon in WASM with simple workload
  - Verify SharedArrayBuffer works (COOP/COEP already enabled ✅)
  - Benchmark thread pool overhead
- [ ] Implement model weight sharing:
  - Load model in main thread
  - Share read-only weights via SharedArrayBuffer
  - Alternative: Each worker loads own model copy (simpler but more memory)
- [ ] Distribute chunks to worker pool for parallel embedding:
  ```rust
  // Conceptual API
  let chunks = chunk_text(&content, config.max_position_embeddings);
  let embeddings: Vec<Vec<f32>> = chunks
      .par_iter()  // rayon parallel iterator
      .map(|chunk| embed_chunk(chunk))
      .collect()?;
  ```
- [ ] Benchmark scaling (1 vs 2 vs 4 vs 8 cores)
- [ ] Write tests for parallel processing
- [ ] Compare memory usage: shared weights vs per-worker models

**Challenges:**
- Rust wasm32 threading support is experimental
- Need to manage worker pool lifecycle
- Memory pressure: Each worker needs model copy (262MB × N)
  - With shared weights: ~262MB + overhead
  - Without shared weights: ~262MB × N workers
- Coordination overhead between main thread and workers

**Platform Differences:**
```rust
#[cfg(target_arch = "wasm32")]
{
    // Web: WASM threads + rayon (after this milestone)
    use rayon::prelude::*;
    chunks.par_iter().map(|c| embed(c)).collect()
}

#[cfg(not(target_arch = "wasm32"))]
{
    // Desktop: Native rayon (already fast)
    use rayon::prelude::*;
    chunks.par_iter().map(|c| embed(c)).collect()
}
```

**Expected Outcome:**
- Processing 100 files: 10 minutes → 2-3 minutes (4x speedup on quad-core)
- Scales with CPU core count
- Web performance approaches desktop performance for CPU-bound tasks

**Success Criteria:**
- ✅ WASM threads + rayon working in browser
- ✅ Parallel embedding shows measurable speedup (2-4x)
- ✅ Memory usage acceptable (<1GB for 4 workers)
- ✅ Worker pool lifecycle managed correctly (no leaks)
- ✅ Benchmark results documented
- ✅ Passes all items in **Milestone Completion Checklist** (see above)

**References:**
- [WASM Threads Proposal](https://github.com/WebAssembly/threads)
- [Rayon WASM Support](https://github.com/rayon-rs/rayon/issues/685)

---

### Milestone 6: File Streaming (Web) / Direct Access (Desktop)
**Goal:** Handle large files efficiently on each platform

**Web Approach:**
- Use `wasm_streams` for streaming file reads
- Decode incrementally with `encoding_rs`
- Chunk on-the-fly (don't load entire file)
- Process chunks as they arrive

**Desktop Approach:**
- Direct file system access (std::fs)
- Memory-map large files
- Or read in larger chunks (OS handles buffering)

**Why Different:**
- Web: No file system, must stream from Blob
- Desktop: Native FS, can use memory mapping

**Implementation Tasks:**
- [ ] Web: Implement Blob streaming
- [ ] Desktop: Implement efficient file reading
- [ ] Unified chunking logic (works for both)
- [ ] Write tests for streaming logic
- [ ] Test with files >100MB

**Success Criteria:**
- ✅ Can process files >100MB without memory issues
- ✅ UI remains responsive during processing
- ✅ Works on both platforms
- ✅ Passes all items in **Milestone Completion Checklist** (see above)

---

### Milestone 7: Model Configuration & Optimization
**Goal:** Optimize for each platform's capabilities

**Implementation Tasks:**
- [ ] ✅ **DONE:** WASM memory increased to 4GB (`.cargo/config.toml`)
- [ ] ✅ **DONE:** Sequence length increased to 2048 (`src/embedding.rs`)
- [ ] Add sequence length preset selector in UI:
  - Short (512 tokens) - Fast, less context
  - Medium (2048 tokens) - Balanced (current default)
  - Long (4096 tokens) - More context, slower
  - Max (8192 tokens) - Full model capability
- [ ] Desktop-specific optimizations:
  - Use F16 precision if supported (smaller memory)
  - Enable SIMD
  - Multi-threaded inference with rayon
- [ ] Model download/caching:
  - Web: Bundle with app or CDN
  - Desktop: Download to app data folder, cache locally
- [ ] Write tests for configuration management

**Configuration Options:**
```rust
pub struct ModelConfig {
    sequence_length: SequenceLengthPreset,  // Short/Medium/Long/Max
    platform: Platform,                      // Web/Desktop/Mobile
    #[cfg(not(target_arch = "wasm32"))]
    device: DeviceType,                     // CPU/CUDA/Metal
}
```

**Success Criteria:**
- ✅ Users can select sequence length preset
- ✅ Performance improves with optimizations
- ✅ Works on both platforms
- ✅ Passes all items in **Milestone Completion Checklist** (see above)

---

## Architecture Evolution

### Current: Single-Threaded Inference
```
User → UI → Tokenizer → Candle Model → Embedding → Display
```

### After Milestone 5: Parallel Processing
- **Web:** WASM threads + rayon with SharedArrayBuffer
- **Desktop:** Native rayon parallelism
- **Goal:** Process multiple chunks concurrently

### Future: GPU Acceleration
- **Desktop:** CUDA/Metal via Candle
- **Web:** Candle WebGPU backend (when available)

---

## Performance Targets

### Web (Current Hardware: M1 MacBook)
- **Cold start:** 3-7s (model download + init)
- **Warm start:** Instant (model cached)
- **Embedding (512 tokens):** ~50-200ms
- **Embedding (2048 tokens):** ~200-500ms (projected)
- **Search (1K vectors):** <100ms
- **Memory:** ~600MB (with 2048 token config)

### Desktop (Projected)
- **Cold start:** 1-2s (model load from disk)
- **Embedding (512 tokens):** ~20-50ms (native)
- **Embedding (2048 tokens):** ~50-150ms
- **Search (10K vectors):** <50ms
- **Memory:** ~800MB (F32), ~400MB (F16 if supported)

### Mobile (Future)
- Use smaller model (e.g., MiniLM-L6, ~25MB)
- Sequence length: 512 max (memory constraints)
- Embedding: ~100-300ms per chunk

---

## Open Questions

Cross-cutting decisions that affect multiple milestones:

- **Model selection:** JinaBERT vs alternatives (MiniLM, larger models)
- **Chunking strategy:** Token-based vs sentence-aware vs semantic chunking
- **Storage limits:** IndexedDB quota handling, F16 vs F32 storage
- **Search UX:** Real-time vs button, filters, combined semantic + text search

Specific implementation questions should be tracked in GitHub issues or milestone tasks.

---

## Technical Debt & Future Improvements

### High Priority
1. ✅ **COMPLETED:** Increase WASM memory limit (512MB → 4GB)
   - Implemented in `.cargo/config.toml`
   - See: `docs/model-optimization.md`
2. ✅ **COMPLETED:** Increase sequence length (1024 → 2048)
   - Implemented in `src/embedding.rs`
   - Unlocks 2-4x more context per chunk
3. **Add vector storage** → Milestone 2
4. **Add search UI** → Milestone 3

### Medium Priority
5. **Desktop platform testing**
   - Ensure `dx serve --platform desktop` works
   - Implement desktop-specific optimizations
6. **Quantization**
   - F16 or INT8 model weights
   - Reduce model size and memory

### Low Priority
7. **Mobile platform**
   - Smaller model selection
   - Touch UI optimizations
8. **HNSW indexing** (for >10K vectors)
9. **Model switching UI**
   - Let users choose different models
   - Re-embed documents when switching

---

## Dependencies Roadmap

### Current (Web)
```toml
dioxus = { version = "0.6", features = ["web"] }
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"
tokenizers = { version = "0.20", features = ["unstable_wasm"] }
```

### Add for Storage
```toml
# Web - IndexedDB (Rust wrapper)
[target.'cfg(target_arch = "wasm32")'.dependencies]
rexie = "0.6"  # Pure Rust IndexedDB wrapper

# Desktop - SQLite (native)
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
rusqlite = { version = "0.31", features = ["bundled"] }
```

### Add for Advanced Features
```toml
# Streaming (web)
wasm-streams = "0.4"
encoding_rs = "0.8"

# Chunking
unicode-segmentation = "1.11"

# Parallel (desktop)
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
rayon = "1.8"
```

---

## Development Workflow

### Iteration Cycle
1. Implement feature for **web first** (faster iteration)
2. Test in browser with `dx serve`
3. Add desktop support with `cfg` attributes
4. Test desktop with `dx serve --platform desktop`
5. Ensure both platforms work before committing

### Platform Testing
```bash
# Web (primary development)
dx serve

# Desktop (ensure compatibility)
dx serve --platform desktop

# Production builds
dx bundle --release              # Web
dx bundle --release --platform desktop  # Desktop app
```

### Before Committing
```bash
# Pre-commit hook runs these automatically:
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --verbose
cargo doc --no-deps
cargo audit
dx build --release --platform web
```

---

## References

**New Technical Docs:**
- `docs/model-optimization.md` - WASM memory and sequence length optimization
- `docs/browser-ml-architecture.md` - Browser ML patterns, COOP/COEP, WebGPU
- `docs/ecosystem-and-limitations.md` - Ecosystem, alternatives, resources

**Dioxus:**
- Main docs: https://dioxuslabs.com/
- Cross-platform guide: https://dioxuslabs.com/learn/0.5/getting_started/desktop

**Candle:**
- GitHub: https://github.com/huggingface/candle
- WASM examples: https://github.com/huggingface/candle/tree/main/candle-wasm-examples

**Model:**
- JinaBERT: https://huggingface.co/jinaai/jina-embeddings-v2-small-en
- Paper: https://arxiv.org/abs/2310.19923
