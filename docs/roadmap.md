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

#### ✅ Cross-Platform Storage & Persistence
**ADR:** [007-document-storage-reupload-handling.md](adrs/007-document-storage-reupload-handling.md)

Persistent document storage via `DocumentStore` trait with platform-specific implementations:
- **Web**: IndexedDB for O(1) key lookups (browser-native, zero bundle cost)
- **Desktop/Mobile**: redb (pure Rust B-tree database) for O(log n) lookups with ACID transactions
- **In-Memory**: `InMemoryDocumentStore` for testing and development

**Features implemented:**
- Source tracking with SHA-256 content hashes for re-upload detection
- Intelligent update handling: SKIP (unchanged), UPDATE (modified), ADD (new)
- Tombstone-based HNSW deletion (soft-delete without index rebuild)
- Automatic compaction when tombstone ratio exceeds 30%
- Incomplete source recovery on startup (crash-safe)

#### ✅ Web Worker Architecture
Non-blocking ML inference on web platform:
- Module worker loads WASM and 65MB JinaBERT model in separate thread
- Prevents UI freezing during embedding computation
- Desktop uses `tokio::spawn_blocking` with native threading

#### ✅ Web Crawler (Desktop-First)
**ADR:** [002-web-crawler-desktop-first.md](adrs/002-web-crawler-desktop-first.md)

Desktop-only web crawler for fetching and indexing web pages:
- Paste URL and crawl same-origin pages recursively
- Configurable depth, page limits, and parallel requests
- BFS traversal with cycle detection and politeness delays
- HTML text extraction with `scraper` crate
- Crawled pages indexed via hybrid search pipeline

**Implementation:**
- `crates/coppermind/src/crawler/` - Engine, fetcher, parser modules
- `crates/coppermind/src/components/web_crawler.rs` - UI component
- Hidden on web platform via `#[cfg(not(target_arch = "wasm32"))]`

#### ✅ GPU Scheduler
**ADR:** [006-gpu-scheduler.md](adrs/006-gpu-scheduler.md)

Thread-safe GPU access for Metal backend (works around Candle threading bug):
- `SerialScheduler` with dedicated worker thread owning GPU device
- Priority queue: search queries (P0) before background work (P2)
- Multi-model support via model registry
- Batch processing for efficient background embedding

**Implementation:**
- `crates/coppermind/src/gpu/` - Scheduler, types, error handling
- Desktop only (WASM uses direct CPU execution)

---

## Roadmap

### Phase 1: Persistence Layer

**Status:** ✅ Complete
**ADR:** [007-document-storage-reupload-handling.md](adrs/007-document-storage-reupload-handling.md)

**Implemented:**
- Platform-specific persistent storage via `DocumentStore` trait
- **Web**: IndexedDB (browser-native key-value store)
- **Desktop/Mobile**: redb (pure Rust embedded database)
- Source tracking with content hash for re-upload detection (SKIP/UPDATE/ADD)
- Tombstone-based HNSW deletion with automatic compaction
- Index restoration on application startup
- Crash-safe recovery for incomplete indexing operations

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


## References

- [Candle GitHub](https://github.com/huggingface/candle) - Rust ML framework
- [Dioxus Docs](https://dioxuslabs.com/) - Rust UI framework
- [WebGPU Spec](https://www.w3.org/TR/webgpu/) - GPU acceleration in browsers
- [OPFS Spec](https://fs.spec.whatwg.org/) - Origin Private File System
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin (2018)
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) - Cormack et al. (2009)

---

**Note:** This is a living document. Completed items move to "Project Status" section.
