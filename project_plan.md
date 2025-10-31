# File Processing & Vector Store - Project Plan

## Overview

Build a Dioxus web application (with future desktop support) that:
1. Allows users to select multiple files
2. Streams file content efficiently (no full file load in memory)
3. Chunks text into embedding-friendly segments
4. Computes embeddings (parallelized)
5. Stores vectors in a local vector database for retrieval

## Architecture Decision

**Selected: Worker Pool Architecture (Option B)**

**Rationale:**
- Works on any static host (no COOP/COEP headers required)
- Portable across browsers
- Easier debugging than WASM threads
- Clean separation: UI thread → coordinator worker → embedding workers
- Desktop migration path is straightforward

**Data Flow:**
```
Main Thread (Dioxus UI)
    ↓ file selection
Streaming Reader (wasm_streams)
    ↓ byte chunks via Blob.stream()
Chunker (Rust/WASM, main or dedicated worker)
    ↓ text chunks (transfer list, zero-copy)
Worker Pool (N embedding workers)
    ↓ embeddings (Float32Array)
Coordinator/Writer
    ↓ batch writes
Vector Store (SQLite on OPFS)
```

## Tech Stack

### Rust Dependencies
```toml
dioxus = { version = "0.6", features = ["web"] }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
wasm-streams = "0.4"
web-sys = { version = "0.3", features = ["File", "FileList", "Blob", "ReadableStream", "HtmlInputElement"] }
js-sys = "0.3"
futures = "0.3"
encoding_rs = "0.8"
flume = "0.11"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"

# For chunking (Phase 2+)
unicode-segmentation = "1.11"  # sentence boundaries
# Consider: text-splitter or tiktoken-rs for token-aware chunking

# For desktop later
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
rusqlite = { version = "0.31", features = ["bundled"] }
rayon = "1.8"
```

### JavaScript Dependencies
```json
{
  "dependencies": {
    "@xenova/transformers": "^2.17.0",
    "wa-sqlite": "^0.9.9"
  }
}
```

### Embedding Model
- **MVP:** Xenova/all-MiniLM-L6-v2 (384 dims, ~23MB, fast)
- **Alternative:** Use remote API (OpenAI, Cohere) for testing without local model download

### Vector Store
- **Phase 1 (MVP):** IndexedDB via `idb` crate (simpler to set up)
- **Phase 3:** wa-sqlite with OPFS VFS (better batch write performance)

## Implementation Phases

### Phase 1: Prove the Pipeline (MVP)
**Goal:** End-to-end flow with minimal complexity

**Learning Approach:** Incremental milestones where each step produces a working app

#### Milestone 1: Hello Dioxus + File Selection
**Goal:** Get comfortable with Dioxus basics and file input
**Status:** In Progress

**Tasks:**
- [x] Scaffold Dioxus web app with `dx new --template web`
- [ ] Create basic UI with file input component
- [ ] Handle file selection events and display file metadata (name, size, type)
- [ ] Learn: Dioxus components, `rsx!` macro, signals for reactive state

**Success Criteria:**
- App runs with `dx serve`
- Can select multiple files
- See file names and sizes displayed in a list

**New Concepts:**
- Dioxus component structure
- Signal-based reactivity
- Event handlers (onclick, oninput)
- web-sys integration basics

---

#### Milestone 2: Stream One File
**Goal:** Understand streaming architecture and wasm_streams
**Status:** Not started

**Tasks:**
- [ ] Add `wasm_streams` and `web-sys` dependencies
- [ ] Implement streaming reader for single file using `Blob.stream()`
- [ ] Display byte count as data streams (e.g., "Read 1024 bytes...")
- [ ] Add basic error handling for stream failures
- [ ] Learn: Rust async/await in WASM, streaming APIs, futures

**Success Criteria:**
- Select a file and see real-time byte count updates
- Can pause/resume reading
- Handle large files (>10MB) without blocking UI

**New Concepts:**
- `wasm_streams::ReadableStream`
- Async tasks in Dioxus (spawn, use_future)
- Stream error handling

---

#### Milestone 3: Decode to Text
**Goal:** Convert byte streams to text incrementally
**Status:** Not started

**Tasks:**
- [ ] Add `encoding_rs` for UTF-8 decoding
- [ ] Implement incremental decoder that handles split characters
- [ ] Display decoded text in chunks (scrollable view)
- [ ] Handle encoding errors gracefully (show warnings, skip invalid bytes)
- [ ] Learn: UTF-8 decoding challenges, stateful decoders

**Success Criteria:**
- Select a .txt file and see its content appear incrementally
- No garbled text at chunk boundaries
- Binary files show warning/error message

**New Concepts:**
- Stateful UTF-8 decoding
- Handling multi-byte character boundaries
- Text display in Dioxus (scrollable, virtualized lists)

---

#### Milestone 4: Chunk the Text
**Goal:** Split text into embedding-sized chunks
**Status:** Not started

**Tasks:**
- [ ] Implement character-based chunking (~4KB target, ~400 char overlap)
- [ ] Display chunks in numbered list with metadata (size, overlap)
- [ ] Add controls to adjust chunk size/overlap dynamically
- [ ] Learn: Chunking strategies, why overlap matters for retrieval

**Success Criteria:**
- See file split into ~10-20 chunks (for typical 50KB file)
- Chunks are roughly equal size
- Can see overlap regions highlighted

**New Concepts:**
- Sliding window chunking
- Chunk metadata (start_byte, end_byte, ord)
- Dynamic UI controls in Dioxus

---

#### Milestone 5: Mock Embeddings + IndexedDB Storage
**Goal:** Complete the pipeline with fake embeddings
**Status:** Not started

**Tasks:**
- [ ] Generate random Float32Array (384 dims) per chunk
- [ ] Add IndexedDB wrapper (use `rexie` crate - safer than raw JS)
- [ ] Implement schema: docs, chunks, vectors tables
- [ ] Store chunks and vectors with batch writes
- [ ] Display storage confirmation and stats
- [ ] Learn: IndexedDB transactions, binary data storage

**Success Criteria:**
- Process a file end-to-end: select → stream → chunk → embed → store
- See "Stored 15 chunks with 15 vectors" message
- Can inspect IndexedDB in browser DevTools

**New Concepts:**
- IndexedDB transactions and stores
- Storing binary data (Float32Arrays)
- Batch operations for performance
- `rexie` crate API

---

#### Milestone 6: Multi-File + Progress UI
**Goal:** Polish the MVP with proper UX
**Status:** Not started

**Tasks:**
- [ ] Support processing multiple files sequentially
- [ ] Add progress bars (per-file and overall)
- [ ] Show statistics: files processed, total chunks, total vectors, time elapsed
- [ ] Add error recovery (skip failed files, continue processing)
- [ ] Add "Clear Database" button for testing
- [ ] Learn: Complex UI state management, progress tracking

**Success Criteria:**
- Process 10 files in sequence with live progress
- Smooth UI (no blocking on main thread)
- Professional-looking interface

**New Concepts:**
- Complex state management patterns
- Progress tracking across async tasks
- Error boundaries and recovery
- Database management operations

---

**Phase 1 Complete When:**
- ✅ Can process multiple text files end-to-end
- ✅ Vectors stored in IndexedDB (mock embeddings)
- ✅ Clean, responsive UI with progress feedback
- ✅ Ready to swap mock embeddings for real ones (Phase 2)

**Unknowns/Investigation:**
- [ ] Best chunk size for streaming: emit per paragraph? Per N chars? Per N bytes?
- [ ] `wasm_streams` error handling patterns (EOF, invalid UTF-8, etc.)
- [ ] IndexedDB write throughput: can we sustain 1000+ writes/sec with batching?
- [ ] `rexie` vs other IndexedDB wrappers - which has best DX?

---

### Phase 2: Parallelism + Real Embeddings
**Goal:** Add worker pool and compute real embeddings

**Tasks:**
1. **Coordinator worker setup**
   - Create `/js/coordinator.js` (or Rust worker via `gloo-worker`)
   - Receives text chunks from main thread
   - Distributes to worker pool via round-robin
   - **Key:** Use transfer lists `postMessage(buf, [buf])` for zero-copy

2. **Embedding worker pool**
   - Create `/js/embed_worker.js` (N instances)
   - Load Transformers.js pipeline:
     ```js
     import { pipeline, env } from '@xenova/transformers';
     env.backends.onnx.wasm.proxy = false; // worker context
     const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
     ```
   - On message: embed text → return `Float32Array` via transfer list
   - **Pool size:** `min(4, navigator.hardwareConcurrency || 4)`

3. **Rust ↔ JS worker bindings**
   - `wasm_bindgen` extern for `dispatchToWorkers(text, meta)`
   - Listen for worker results via callbacks
   - Handle backpressure: pause chunking if worker queue is full

4. **Backpressure control**
   - Use `flume::bounded` channel (capacity = num_workers * 2)
   - Block stream reading when channel is full
   - Prevents memory buildup

**Unknowns/Investigation:**
- [ ] Transformers.js cold start time (first model load)
- [ ] Optimal worker pool size (need to benchmark: 2 vs 4 vs 8)
- [ ] Memory usage with 4 simultaneous ONNX inference sessions
- [ ] Should coordinator be Rust (gloo-worker) or plain JS? (Tradeoff: type safety vs simplicity)

---

### Phase 3: Production Storage
**Goal:** Migrate to SQLite for better durability and performance

**Tasks:**
1. **wa-sqlite integration**
   - Add wa-sqlite npm package
   - Create `/js/db.js` wrapper with OPFS VFS:
     ```js
     import SQLiteESMFactory from 'wa-sqlite/dist/wa-sqlite.mjs';
     import { OPFSCoopSyncVFS } from 'wa-sqlite/src/examples/OPFSCoopSyncVFS.js';
     ```
   - Schema (same as Phase 1 but in SQL):
     ```sql
     CREATE TABLE docs (id TEXT PRIMARY KEY, name TEXT, size INTEGER, mime TEXT);
     CREATE TABLE chunks (id TEXT PRIMARY KEY, doc_id TEXT, ord INTEGER, text TEXT, start_byte INTEGER, end_byte INTEGER);
     CREATE TABLE vectors (chunk_id TEXT PRIMARY KEY, dim INTEGER, vec BLOB);
     CREATE INDEX idx_chunks_doc ON chunks(doc_id, ord);
     ```

2. **Batch writer**
   - Accumulate 100 vectors in-memory
   - Single transaction for batch INSERT
   - Target: <10ms per batch

3. **Brute-force vector search**
   - Load all vectors (or top N docs by recency)
   - Compute cosine similarity in WASM with SIMD
   - Maintain min-heap for top-k results
   - Return `{ chunk_id, score, text, doc_name }`

**Unknowns/Investigation:**
- [ ] wa-sqlite OPFS VFS setup complexity (docs are sparse)
- [ ] Write throughput: batches/sec achievable in OPFS?
- [ ] Max practical vector count for brute-force search (<10K? <100K?)
- [ ] Do we need HNSW index? If so, which library? (usearch-wasm? custom?)

---

### Phase 4: Optimize Chunking
**Goal:** Improve chunk quality for embeddings

**Tasks:**
1. **Sentence-aware splitting**
   - Use `unicode-segmentation` for sentence boundaries
   - Respect semantic units (don't split mid-sentence)

2. **Token-based chunking**
   - Integrate `tiktoken-rs` (if using OpenAI models) or similar
   - Target: 512 tokens per chunk, 50-100 token overlap
   - Alternative: use byte-pair encoding approximation

3. **Metadata extraction**
   - Extract document title, headers (if markdown/HTML)
   - Store in `docs` table for better retrieval context

**Unknowns/Investigation:**
- [ ] Does Xenova/all-MiniLM-L6-v2 have a specific tokenizer? Or is char-based OK?
- [ ] Optimal chunk size for this model (test 256/512/1024 tokens)
- [ ] Should we preserve document structure (headers, lists) as separate chunks?

---

### Phase 5: Desktop Migration
**Goal:** Port to Dioxus Desktop with native I/O and threads

**Tasks:**
1. **Conditional compilation**
   - `#[cfg(target_arch = "wasm32")]` for web paths
   - `#[cfg(not(target_arch = "wasm32"))]` for desktop paths

2. **Native file I/O**
   - Replace `web_sys::File` with `std::fs::File`
   - Use `std::io::BufReader` for streaming

3. **Native threads**
   - Replace worker pool with Rayon:
     ```rust
     chunks.par_iter().map(|chunk| embed(chunk)).collect()
     ```

4. **Native SQLite**
   - Replace wa-sqlite with `rusqlite`
   - Same schema, much faster writes

5. **Shared code abstraction**
   - Trait for `FileReader`: `async fn read_stream() -> impl Stream<Item = Vec<u8>>`
   - Trait for `VectorStore`: `async fn insert_batch(&self, vecs: Vec<Vector>)`
   - Implement for web and desktop

**Unknowns/Investigation:**
- [ ] Does Dioxus 0.6 Desktop use Tauri under the hood? Or webview only?
- [ ] Can we reuse Transformers.js on desktop, or do we need a Rust embedding lib?
- [ ] What's the deployment story? Single binary with bundled SQLite?

---

## Open Questions & Decisions Needed

### High Priority
1. **Embedding strategy:**
   - Local (Transformers.js) vs Remote API (OpenAI/Cohere)?
   - Decision: Start local for offline capability; add remote as option later

2. **Chunk size:**
   - Need to test: what gives best retrieval quality with all-MiniLM-L6-v2?
   - Decision: Start with 512 tokens, benchmark against 256 and 1024

3. **File types:**
   - Text only? PDF? DOCX? Images (OCR)?
   - Decision: Phase 1 = plain text (.txt, .md, .json); PDF in Phase 2+

### Medium Priority
4. **Error handling:**
   - What if a file is binary? Encoding errors? Partial reads?
   - Need robust fallback/skip logic

5. **Storage limits:**
   - OPFS quota on web (usually ~1GB, varies by browser)
   - How to handle quota exceeded? Warn user? LRU eviction?

6. **Search UX:**
   - Real-time search as user types? Or explicit "search" button?
   - Display top-k results: just text snippets, or full document preview?

### Low Priority
7. **Advanced indexing:**
   - At what vector count does brute-force become too slow? (<10K? <100K?)
   - If needed, evaluate: usearch-wasm, hnswlib-wasm, or custom Rust HNSW

8. **Model switching:**
   - Should users be able to swap embedding models?
   - Re-embed all documents on model change?

---

## Performance Targets

### Phase 1 (MVP)
- 10 files (~1MB each) → chunked + stored in <10 seconds
- Smooth UI (no blocking on main thread)

### Phase 2 (Parallelism)
- 50 files (~1MB each) → embedded + stored in <30 seconds
- Worker pool utilization >80%

### Phase 3 (Production)
- 1000+ chunks stored with <1s write latency
- Search across 1000 chunks in <100ms

### Desktop
- 100 files (~10MB each) → embedded + stored in <60 seconds
- Search across 10K chunks in <50ms

---

## File Structure (Proposed)

```
workspace/
├── Cargo.toml
├── src/
│   ├── main.rs              # Dioxus app entry
│   ├── file_reader.rs       # Streaming reader (wasm_streams)
│   ├── chunker.rs           # Text chunking logic
│   ├── storage.rs           # VectorStore trait + impls
│   ├── worker_bridge.rs     # JS worker bindings
│   └── ui/
│       ├── mod.rs
│       ├── file_select.rs   # File input component
│       ├── progress.rs      # Progress display
│       └── search.rs        # Search UI
├── js/
│   ├── coordinator.js       # Worker pool coordinator
│   ├── embed_worker.js      # Embedding worker
│   └── db.js                # wa-sqlite wrapper
├── package.json
└── README.md
```

---

## Next Steps

1. **Scaffold Dioxus app:** `dx new --template web`
2. **Implement Phase 1, Task 1:** File selection UI
3. **Implement Phase 1, Task 2:** Streaming reader with `wasm_streams`
4. **Test:** Single file (1MB .txt) → stream → decode → print chunks

Then iterate through Phase 1 tasks sequentially.

---

## References

- Dioxus docs: https://dioxuslabs.com/
- wasm_streams: https://docs.rs/wasm-streams/
- Transformers.js: https://huggingface.co/docs/transformers.js/
- wa-sqlite: https://github.com/rhashimoto/wa-sqlite
