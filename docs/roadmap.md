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

### Milestone 1.5: Parallel Embedding with wasm-bindgen-rayon (NEXT) 🎯
**Goal:** Implement true parallel CPU-bound embedding using Rayon in WASM

**Current Problem:**
- **Web Platform:** Embedding blocks main thread, UI freezes during inference
- **Attempted Solutions:**
  - ❌ Web Worker with separate WASM instance: Failed (full app module includes DOM APIs)
  - ❌ Dioxus logger incompatibility in worker context
- **Root Cause:** Need true multi-threading, not just async task isolation
- **Best Solution:** wasm-bindgen-rayon for automatic parallelism

**Why wasm-bindgen-rayon:**
- ✅ True multi-threading with SharedArrayBuffer (COOP/COEP already configured)
- ✅ Automatic work distribution across CPU cores
- ✅ Proven to work with Candle (3x speedup in Candle PR #3063)
- ✅ Minimal code changes (`.par_iter()` instead of `.iter()`)
- ✅ No separate worker binary needed
- ✅ Scales with CPU core count

**Implementation Plan:**

### Phase 1: Prerequisites & Setup ⚙️

- [ ] **1.1 Install nightly Rust toolchain**
  ```bash
  rustup toolchain install nightly-2024-08-02
  rustup component add rust-src --toolchain nightly-2024-08-02
  ```

- [ ] **1.2 Create rust-toolchain.toml**
  ```toml
  [toolchain]
  channel = "nightly-2024-08-02"
  components = ["rust-src"]
  targets = ["wasm32-unknown-unknown"]
  ```

- [ ] **1.3 Update .cargo/config.toml**
  ```toml
  [target.wasm32-unknown-unknown]
  rustflags = [
    "--cfg", "getrandom_backend=\"wasm_js\"",
    "-C", "link-arg=--initial-memory=268435456",      # 128MB initial
    "-C", "link-arg=--max-memory=4294967296",          # 4GB max
    "-C", "target-feature=+atomics,+bulk-memory,+mutable-globals",  # ADD THIS
  ]

  [unstable]
  build-std = ["panic_abort", "std"]  # ADD THIS - rebuild stdlib with atomics
  ```

- [ ] **1.4 Add dependencies**
  ```toml
  [dependencies]
  rayon = "1.8"
  wasm-bindgen-rayon = "1.2"

  [target.'cfg(target_arch = "wasm32")'.dependencies]
  # Already have: gloo-timers, getrandom
  ```

- [ ] **1.5 Update Dioxus.toml (if needed)**
  Check if Dioxus CLI supports custom WASM flags, or if manual build script needed

### Phase 2: JavaScript Initialization 📜

- [ ] **2.1 Create worker pool init script**
  **File:** `public/init-rayon.js`
  ```javascript
  import init, { initThreadPool } from './wasm/coppermind.js';

  export async function initializeRayon() {
    await init();

    // Initialize thread pool with number of CPU cores
    const threads = navigator.hardwareConcurrency || 4;
    console.log(`🧵 Initializing Rayon thread pool with ${threads} threads`);

    await initThreadPool(threads);
    console.log('✓ Rayon thread pool ready');
  }
  ```

- [ ] **2.2 Update main.rs to export initThreadPool**
  ```rust
  #[cfg(target_arch = "wasm32")]
  pub use wasm_bindgen_rayon::init_thread_pool;
  ```

- [ ] **2.3 Call init from main thread**
  Update `main.rs` or components to call `initializeRayon()` on app startup

### Phase 3: Rust Code Changes 🦀

- [ ] **3.1 Update embedding.rs for parallel batch processing**
  ```rust
  use rayon::prelude::*;

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

      info!(
          "🧩 Embedding {} chunks in parallel with Rayon ({} tokens max per chunk)",
          token_chunks.len(),
          effective_chunk
      );

      // PARALLEL PROCESSING HERE
      #[cfg(target_arch = "wasm32")]
      let embeddings: Result<Vec<Vec<f32>>, String> = token_chunks
          .par_iter()  // Rayon parallel iterator
          .enumerate()
          .map(|(index, ids)| {
              info!("🚀 Processing chunk {} on thread", index);
              model.embed_tokens(ids.clone())
          })
          .collect();

      #[cfg(not(target_arch = "wasm32"))]
      let embeddings: Result<Vec<Vec<f32>>, String> = token_chunks
          .iter()
          .enumerate()
          .map(|(index, ids)| {
              info!("🚀 Processing chunk {} (desktop - already async)", index);
              model.embed_tokens(ids.clone())
          })
          .collect();

      let embeddings = embeddings?;

      let results = embeddings
          .into_iter()
          .enumerate()
          .map(|(index, embedding)| ChunkEmbeddingResult {
              chunk_index: index,
              token_count: token_chunks[index].len(),
              embedding,
          })
          .collect();

      Ok(results)
  }
  ```

- [ ] **3.2 Remove worker module** (no longer needed)
  - Delete `src/worker/` directory
  - Remove from `main.rs` module list
  - Clean up components.rs to not reference worker types

- [ ] **3.3 Update components.rs**
  - Remove `embed_text_chunks_worker` function
  - File processing can call `embed_text_chunks` directly
  - Rayon handles parallelism automatically

### Phase 4: Testing & Validation ✅

- [ ] **4.1 Build verification**
  ```bash
  dx build --platform web
  # Check for: No errors about atomics, SharedArrayBuffer
  # Binary size may increase slightly due to threading support
  ```

- [ ] **4.2 Runtime testing**
  ```bash
  dx serve
  # Open browser DevTools → Console
  # Look for: "🧵 Initializing Rayon thread pool with N threads"
  # Look for: "✓ Rayon thread pool ready"
  ```

- [ ] **4.3 Upload test file**
  - Upload foo.txt (or similar)
  - **Expected:** Console shows chunks being processed on different threads
  - **Expected:** UI remains responsive (can scroll, click during processing)
  - **Expected:** Processing time similar or faster than sequential

- [ ] **4.4 Benchmark parallel vs sequential**
  - Test with large file (many chunks)
  - Compare processing time
  - Expected: 2-4x speedup on quad-core CPU

- [ ] **4.5 Cross-platform verification**
  ```bash
  dx serve --platform desktop
  # Desktop should work unchanged (already uses tokio::spawn_blocking)
  ```

### Phase 5: Known Issues & Troubleshooting 🐛

**Issue 1: Build fails with "feature `build-std` is unstable"**
- **Cause:** Using stable Rust instead of nightly
- **Fix:** Ensure `rust-toolchain.toml` exists and specifies nightly

**Issue 2: WASM module fails to load with "SharedArrayBuffer not available"**
- **Cause:** COOP/COEP headers not set properly
- **Fix:** Verify COI service worker is loaded (check Network tab)
- **Verify:** `crossOriginIsolated` should be `true` in browser console

**Issue 3: "Cannot start thread pool twice" error**
- **Cause:** `initThreadPool()` called multiple times
- **Fix:** Call only once during app initialization
- **Pattern:** Use `once_cell::sync::OnceCell` or static flag

**Issue 4: Performance worse than sequential**
- **Cause:** Too few chunks (overhead dominates)
- **Fix:** Only use parallel for N > 4-8 chunks
- **Pattern:** ```rust
  if token_chunks.len() > 8 {
      // Use parallel
  } else {
      // Use sequential
  }
  ```

**Issue 5: Memory usage high**
- **Cause:** Each thread needs stack space
- **Expected:** +10-20MB per thread
- **Monitor:** Browser Task Manager → Memory column

### Phase 6: Cleanup & Documentation 📝

- [ ] **6.1 Remove old worker code**
  - Delete `public/jinabert-embedding-worker.js`
  - Delete `src/worker/` directory
  - Remove worker-related types from components.rs

- [ ] **6.2 Update CLAUDE.md**
  - Remove Web Worker architecture section
  - Add wasm-bindgen-rayon section
  - Document nightly Rust requirement
  - Update module organization

- [ ] **6.3 Update this roadmap**
  - Mark Milestone 1.5 as complete
  - Document final performance numbers
  - Add lessons learned

- [ ] **6.4 Add inline code comments**
  ```rust
  // Parallel processing with Rayon
  // Each chunk is embedded concurrently on the thread pool
  // Requires: wasm-bindgen-rayon, nightly Rust, COOP/COEP headers
  ```

**Architecture Diagram (After Implementation):**
```
┌─────────────────────────────────────────────────────────┐
│                    Main Thread (UI)                     │
│  ┌──────────────┐        ┌──────────────┐              │
│  │  UI (Dioxus) │◄──────►│ embed_text() │              │
│  └──────────────┘        └───────┬──────┘              │
└───────────────────────────────────┼──────────────────────┘
                                    │ .par_iter()
                                    ▼
┌─────────────────────────────────────────────────────────┐
│              Rayon Thread Pool (WASM Workers)           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ Thread 1│  │ Thread 2│  │ Thread 3│  │ Thread 4│   │
│  │ Chunk 1 │  │ Chunk 2 │  │ Chunk 3 │  │ Chunk 4 │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
│                                                         │
│  Shared Read-Only:                                      │
│  ├─ JinaBERT Model (262MB)                             │
│  └─ Tokenizer                                           │
└─────────────────────────────────────────────────────────┘
```

**Key Differences from Web Worker Approach:**
- ✅ No separate worker binary needed
- ✅ No postMessage serialization overhead
- ✅ Shared memory for model weights (via SharedArrayBuffer)
- ✅ Automatic work distribution by Rayon
- ✅ Simpler code: just `.par_iter()` instead of manual worker management

**Performance Expectations:**
- **UI Responsiveness:** Main thread yields during computation
- **Throughput:** 2-4x speedup on quad-core (linear scaling with cores)
- **Memory:** +10-20MB per thread for stack space
- **Latency:** No message passing overhead (shared memory)
- **Baseline:** Sequential ~2s for 37 chunks → Parallel ~0.5-1s (4 cores)

**Success Criteria:**
- ✅ Build succeeds with nightly + atomics + build-std
- ✅ Rayon thread pool initializes successfully in browser
- ✅ UI remains responsive during embedding (can scroll, click)
- ✅ Browser console shows parallel chunk processing
- ✅ Performance improves 2-4x vs sequential (benchmark documented)
- ✅ Works on web platform (desktop unchanged)
- ✅ CLAUDE.md updated with architecture changes
- ✅ Old worker code removed and cleaned up
- ✅ Passes all items in **Milestone Completion Checklist** (see top of document)

**References:**
- [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon) - Official repo
- [Candle PR #3063](https://github.com/huggingface/candle/pull/3063) - 3x speedup example
- [WASM Threads](https://web.dev/webassembly-threads/) - Technical overview
- [SharedArrayBuffer](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer) - Browser API docs

**Estimated Effort:** 4-8 hours
- Setup & config: 1-2 hours
- Code changes: 1-2 hours
- Testing & debugging: 2-3 hours
- Documentation: 1 hour

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
