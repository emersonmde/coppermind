# Coppermind Roadmap

This document outlines implemented features, planned development, and potential future explorations for Coppermind.

---

## Project Status

Coppermind is a local-first cross-platform hybrid search engine built with Rust and WebAssembly.

**Current Platform Support:**
- ✅ Web (WASM) - Fully functional
- ✅ Desktop (macOS/Linux/Windows) - Fully functional with hardware acceleration
- 🚧 Mobile (iOS/iPadOS) - In development

**Core Architecture:**
- **Full Rust ML Stack**: Dioxus (UI) + Candle (ML inference) + tokenizers-rs (tokenization)
- **Cross-Platform**: Shared codebase with platform-specific optimizations via `cfg` attributes
- **Privacy-First**: Pure local inference, no cloud API calls, works offline
- **Type Safety**: Compile-time guarantees catch bugs before runtime

### Completed Features

#### ✅ Hybrid Search System
Successfully implemented hybrid search combining:
- **Vector Search**: HNSW semantic similarity via `hnsw` crate (512D embeddings, incremental indexing)
- **Keyword Search**: BM25 full-text search for exact keyword matching
- **RRF Fusion**: Reciprocal Rank Fusion merging both rankings without score normalization

**Result**: Catches both semantic matches (paraphrases, synonyms) and exact keyword matches with all-Rust implementation.

#### ✅ Platform-Specific Hardware Acceleration
- **macOS Desktop**: Metal GPU + Accelerate BLAS/LAPACK
- **iOS/iPadOS**: Accelerate CPU optimized inference (Metal compatibility issues in simulator)
- **Linux/Windows x86**: Intel MKL CPU optimizations
- **Web (WASM)**: CPU only (WebGPU backend in development by Candle team)

#### ✅ Intelligent Text Chunking
Automatic strategy selection based on file type:
- **Markdown**: Structure-aware chunking respecting document hierarchy (`pulldown-cmark`)
- **Code**: Syntax-aware chunking with tree-sitter (native platforms only)
- **Text**: Sentence-based chunking using ICU4X segmentation
- **Platform Behavior**: Web uses Markdown + Text; Desktop/mobile adds Code chunking

#### 🚧 Cross-Platform Storage
Storage backend infrastructure implemented via `StorageBackend` trait:
- **Web**: OPFS (Origin Private File System) implementation ready
- **Desktop/Mobile**: Native filesystem access via tokio::fs
- **Current Status**: Temporarily disabled (using `InMemoryStorage`) - see Phase 2 roadmap

#### ✅ Web Worker Architecture
Non-blocking ML inference on web platform:
- Module worker loads WASM and 65MB JinaBERT model in separate thread
- Prevents UI freezing during embedding computation
- Desktop uses `tokio::spawn_blocking` with native threading

---

## Roadmap

### Phase 1: Web Crawler (Desktop-First)

**Status:** Planned
**ADR:** [002-web-crawler-desktop-first.md](adrs/002-web-crawler-desktop-first.md)

**Description:**
Add web crawler feature to fetch and index web pages, initially for desktop platform only due to CORS restrictions on web.

**Goals:**
1. Paste URL (e.g., `https://example.com/docs`)
2. Fetch HTML pages and extract visible text
3. Find all links and follow same-origin links recursively
4. Feed extracted text to existing embedding pipeline
5. Store crawled pages alongside uploaded files

**Implementation Strategy:**
- Use cross-platform Rust crates (`reqwest` + `scraper`) for easy future web enablement
- Desktop-only initially (no CORS restrictions)
- Conditionally show UI with `#[cfg(not(target_arch = "wasm32"))]`
- Module structure ready for web platform when needed (CORS proxy option)

**Success Criteria:**
- Desktop: Crawler UI visible and functional (`dx serve --platform desktop`)
- Web: Crawler UI hidden, no errors (`dx serve`)
- Crawled pages indexed and searchable via hybrid search

**Future Enhancements:**
- Recursive crawling with depth limit
- Progress UI (pages crawled, queued)
- Robots.txt support (respect crawler directives)
- Rate limiting (politeness delays)
- Sitemap.xml parsing
- Optional browser extension for web platform support (different security context bypasses CORS)

---

### Phase 2: Persistence Layer

**Status:** Temporarily Disabled (In-Memory Only)
**ADR:** TBD

**Description:**
Persistence is currently disabled across all platforms using `InMemoryStorage` to allow testing of the crawler and other features in bundled apps (DMG, iOS). The storage backend infrastructure exists but is commented out due to path issues:
- **DMG builds**: Read-only filesystem errors (`./coppermind-storage` path invalid)
- **iOS**: Sandbox requirements for writable directories
- **Desktop (`dx serve --desktop`)**: Works but path not suitable for bundled apps
- **Web**: OPFS implementation exists but untested

**Re-enabling Persistence:**
To restore persistence, edit `src/components/mod.rs` line ~310:
- Uncomment platform-specific storage code
- Update `PlatformStorage` type alias to use `OpfsStorage` (web) or `NativeStorage` (desktop)
- Configure proper platform-specific paths (see below)

**Goals:**
1. Debug and fix current persistence issues across all platforms
2. Save indexed documents and embeddings to storage
3. Restore search index on application startup
4. Incremental updates (add/remove documents without full rebuild)
5. Platform-specific optimizations (OPFS for web, native filesystem for desktop/mobile)
6. Version compatibility and migration strategy

**Implementation Strategy:**
- Fix platform-specific storage paths (writable locations on DMG, iOS sandboxing)
- Leverage existing `StorageBackend` trait (OPFS on web, native fs on desktop/mobile)
- Serialize search index state (HNSW graph, BM25 statistics, document metadata)
- Lazy loading for large indices (load on demand, not all at startup)
- Clear user controls (clear storage, export/import)

**Success Criteria:**
- Index persists across app restarts on all platforms (web, desktop, iOS)
- DMG builds use correct writable paths (~/Library/Application Support)
- iOS respects sandbox requirements
- OPFS working correctly on web
- Incremental document addition without full rebuild
- Clear error handling for storage quota/permission issues
- Fast startup time even with large indices (lazy loading)

**Technical Considerations:**
- **Platform-specific paths**:
  - macOS: `~/Library/Application Support/com.coppermind.app/`
  - iOS: App sandbox documents directory
  - Linux: `~/.local/share/coppermind/`
  - Windows: `%APPDATA%/Coppermind/`
  - Web: OPFS (no filesystem paths)
- Storage format versioning (handle breaking changes gracefully)
- OPFS quota management on web (request persistent storage permission)
- Corruption detection and recovery (checksums, fallback to rebuild)
- Background persistence (don't block UI during saves)

---

## Backlog

Future explorations and potential projects, organized by priority and feasibility.

### 🚀 High Impact & Feasible

#### Candle WebGPU Backend for Browser ML
**Status:** In progress by Candle team

**Description:**
- Candle team actively working on WebGPU backend
- Monitor progress and test prerelease versions when available
- Potential 5-20x speedup for ML inference in browser

**Expected Outcome:**
- Embedding inference: 50-200ms → 5-20ms
- Enables real-time semantic search
- Larger models become feasible (BERT-base, MPNet)

**Next Steps:**
1. Track Candle GitHub for WebGPU backend progress
2. Test prerelease versions when available
3. Provide feedback based on browser ML use case
4. Update Coppermind to use WebGPU backend when stable

---

### 🧪 Experimental & High Impact

#### Quantized Models (F16/INT8) for WASM
**Status:** Candle has quantization, not well-tested for WASM

**Description:**
- Current: F32 weights
- F16: 2x smaller, minimal quality loss
- INT8: 4x smaller, some quality loss

**Benefits:**
- Larger models become feasible
- Faster downloads
- Could fit 2-3 models simultaneously (multi-lingual, domain-specific)

**Challenges:**
- INT8 operations slow on CPU (no SIMD in WASM yet)
- Need to benchmark quality vs size tradeoff

---

#### Turso/libSQL for Native Vector Search in Browser
**Status:** Needs investigation

**Description:**
- [Turso](https://turso.tech/) is SQLite fork (libSQL) with native vector similarity search
- Built-in `vec_distance_*` functions for cosine, L2, etc.
- Designed for edge computing, could work in browsers via WASM

**Example:**
```sql
-- Native vector search in SQL
SELECT chunk_id, text, vec_distance_cosine(embedding, :query_vector) AS similarity
FROM embeddings
ORDER BY similarity ASC
LIMIT 10;
```

**Benefits:**
- Eliminates separate vector index (HNSW, FAISS)
- Native SQL + vector search enables powerful queries
- Single database for everything (documents, chunks, embeddings, metadata)

**Implementation Strategy:**
1. Start with in-memory libSQL (no persistence initially)
2. Index small test dataset and verify search works
3. Benchmark vs brute-force cosine similarity
4. Add OPFS persistence once search is working

**Challenges:**
- libSQL WASM compilation status unknown
- Vector search performance in WASM vs native
- Need to verify compatibility

**Expected Outcome:**
- Search 10K vectors in <10ms (vs 50-100ms brute force)
- Simpler architecture than separate storage + vector index
- Single database solution

**References:**
- [Turso](https://turso.tech/)
- [libSQL GitHub](https://github.com/tursodatabase/libsql)
- [libSQL Vector Search Docs](https://turso.tech/blog/libsql-vector-search)

---

#### Desktop GPU Acceleration Benchmarks
**Description:**
- Leverage Candle's CUDA/Metal backends for desktop builds
- Benchmark vs browser WASM performance
- Demonstrate cross-platform power across all platforms

---

#### Multi-Model Support
**Description:**
- Load different embedding models for different use cases
- Examples: code embeddings, multilingual, domain-specific
- Model switching in browser

---

#### Export to Standard Formats
**Description:**
- Export embeddings to Parquet
- Compatible with Python ecosystem (FAISS, Pinecone, Weaviate)
- Enables hybrid workflows (browser → cloud)

---

### 💡 Speculative & Far Future

#### Federated Learning in Browser (Multi-User)
**Status:** Very experimental, privacy-tech research area

**Description:**
- Multiple users process documents locally
- Share only embeddings (not raw text) to build collective knowledge base
- Privacy-preserving semantic search across organizations

**How:**
- User A embeds their docs → shares embeddings (vectors only)
- User B does same
- Both can search across combined embedding space
- Neither sees other's raw documents

**Challenges:**
- Trust model (how to verify embeddings are safe?)
- De-duplication
- Malicious user detection
- Data leakage risks

**Note:** Out of scope for near/medium term, worth revisiting in far future.

---

#### ~~WASM Threading with Rayon~~
**Status:** ❌ Attempted and abandoned

**Description:**
Investigated enabling Rayon parallelism in WASM via `wasm-bindgen-rayon` to achieve 3x speedup for Candle inference (proven in Candle PR #3063).

**Result:**
After extensive investigation documented in [ADR 003](adrs/003-wasm-threading-workaround.md), determined that Rust WebAssembly atomics are fundamentally incomplete and unsuitable for production use:

- **Issue:** Rust tracking issue [#77839](https://github.com/rust-lang/rust/issues/77839) confirms WebAssembly atomics are "broken" with "significant workarounds required"
- **Problems:** TLS destructors ignored, `std::thread` incompatible, no dedicated WASM target, manual setup required
- **Toolchain conflicts:** Latest nightly supporting `__wasm_init_tls` can't compile modern dependencies
- **Memory cloning:** `WebAssembly.Memory` objects cannot be cloned via `postMessage`

**Conclusion:**
WebAssembly threading remains experimental and production-unviable. Revisit when Rust WebAssembly atomics mature and stabilize.

**References:**
- [ADR 003: WASM Threading Investigation](adrs/003-wasm-threading-workaround.md)
- [Rust Issue #77839](https://github.com/rust-lang/rust/issues/77839)

---

### 🎯 Infrastructure & Community

#### Candle WASM Benchmarking Suite
**Description:**
- Comprehensive benchmarks for ML ops in WASM
- Compare CPU vs WebGPU vs SIMD
- Identify bottlenecks and track progress over time

**Deliverable:**
- Public dashboard showing performance metrics
- Helps identify optimization opportunities for Candle ecosystem

---

#### Dioxus + Candle Example Library
**Description:**
- Coppermind as reference implementation
- Additional examples:
  - Image classification (ConvNext)
  - Audio transcription (Whisper)
  - Text generation (small LLaMA)
- All using Dioxus + Candle + WASM

**Benefits:**
- Makes Rust browser ML more accessible
- Provides working examples for community

---

## References

- [Candle GitHub](https://github.com/huggingface/candle) - Rust ML framework
- [Dioxus Docs](https://dioxuslabs.com/) - Rust UI framework
- [WebGPU Spec](https://www.w3.org/TR/webgpu/) - GPU acceleration in browsers
- [OPFS Spec](https://fs.spec.whatwg.org/) - Origin Private File System
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin (2018)
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) - Cormack et al. (2009)

---

**Note:** This is a living document. Completed items move to "Project Status" section.
