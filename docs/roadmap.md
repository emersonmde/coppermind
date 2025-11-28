# Coppermind Roadmap

This document tracks implemented features, pending work, and future exploration ideas.

---

## Project Status

Coppermind is a local-first cross-platform hybrid search engine built with Rust and WebAssembly.

**Platform Support:**
- Web (WASM) - Fully functional
- Desktop (macOS/Linux/Windows) - Fully functional with hardware acceleration
- Mobile (iOS/iPadOS) - Compiles and runs, file upload UI pending

**Core Architecture:**
- **Full Rust ML Stack**: Dioxus (UI) + Candle (ML inference) + tokenizers-rs (tokenization)
- **Cross-Platform**: Shared codebase with platform-specific optimizations via `cfg` attributes
- **Privacy-First**: Pure local inference, no cloud API calls, works offline

---

## Implemented Features

### Hybrid Search
- **Vector Search**: HNSW semantic similarity via `hnsw` crate (512D JinaBERT embeddings, incremental indexing)
- **Keyword Search**: BM25 full-text search for exact keyword matching
- **RRF Fusion**: Reciprocal Rank Fusion merging rankings without score normalization
- **Result Aggregation**: Chunk-level results grouped by source file with best-match highlighting

### Hardware Acceleration
| Platform | Backend |
|----------|---------|
| macOS Desktop | Metal GPU + Accelerate BLAS |
| iOS/iPadOS | Accelerate CPU (Metal has simulator issues) |
| Linux/Windows x86 | Intel MKL |
| Web (WASM) | CPU only |

### Text Chunking
Automatic strategy selection based on file type:
- **Markdown**: Structure-aware via `pulldown-cmark`
- **Code**: Syntax-aware via tree-sitter (native only, falls back to text on WASM)
- **Text**: Sentence-based via ICU4X segmentation

Supported code languages: Rust, Python, JavaScript, TypeScript, Java, C, C++, Go

### Storage & Persistence
**ADR:** [007-document-storage-reupload-handling.md](adrs/007-document-storage-reupload-handling.md)

Platform-specific `DocumentStore` implementations:
- **Web**: IndexedDB (O(1) lookups, browser-native)
- **Desktop/Mobile**: redb (O(log n) B-tree, ACID transactions)
- **Testing**: InMemoryDocumentStore

Features:
- Source tracking with SHA-256 content hashes
- Update detection: SKIP (unchanged), UPDATE (modified), ADD (new)
- Tombstone-based HNSW deletion (soft-delete without rebuild)
- Automatic compaction at 30% tombstone ratio
- Crash-safe recovery for incomplete indexing

### Web Worker Architecture
Non-blocking ML inference on web:
- Module worker loads WASM + 65MB JinaBERT model in separate thread
- Desktop uses `tokio::spawn_blocking` with native threading

### Web Crawler (Desktop Only)
**ADR:** [002-web-crawler-desktop-first.md](adrs/002-web-crawler-desktop-first.md)

- BFS traversal with configurable depth and page limits
- Same-origin restriction with cycle detection
- Politeness delays between requests
- HTML text extraction via `scraper` crate
- Hidden on web platform (CORS blocks browser crawling)

### GPU Scheduler (Desktop Only)
**ADR:** [006-gpu-scheduler.md](adrs/006-gpu-scheduler.md)

Thread-safe Metal access (works around Candle threading limitations):
- `SerialScheduler` with dedicated worker thread
- Priority queue: search (P0) > batch indexing (P2)
- Multi-model support via model registry

---

## Pending Work

### UI Placeholders

| Feature | Current State | Location |
|---------|---------------|----------|
| Mobile file upload | "Coming soon" placeholder | `components/index/upload_card.rs:198-224` |
| Token count display | Approximation (`doc_count * 400`) | `components/search/search_card.rs:17` |
| Pagination | Fixed top-20 results, "Load more" is non-functional | `components/search/search_view.rs:246` |
| Index selector | Disabled (single index only) | `components/search/search_card.rs:35` |

### Configuration

Hardcoded constants documented in [config-options.md](config-options.md) for future preferences UI:
- Crawler settings (batch size, depth, delays)
- Chunking parameters (target size, overlap)
- HNSW parameters (M, M0, ef_construction)
- Compaction threshold

### Testing

- No integration tests for end-to-end search pipeline
- No UI component tests (Dioxus components)
- No performance benchmarks

### Platform

- Android: Untested (Dioxus mobile support still maturing)
- iOS file picker: Needs native implementation

---

## Backlog

Future explorations organized by impact and feasibility.

### High Impact

#### WebGPU Backend
**Status:** In progress by Candle team

- Potential 5-20x speedup for browser ML inference
- Enables larger models and real-time search
- Track progress at [Candle GitHub](https://github.com/huggingface/candle)

#### Quantized Models (F16/INT8)
**Status:** Candle supports quantization, untested for WASM

- F16: 2x smaller, minimal quality loss
- INT8: 4x smaller, some quality loss (slow on CPU without SIMD)
- Enables multi-model support and faster downloads

### Experimental

#### Turso/libSQL Vector Search
Native vector similarity in SQL, potentially simpler than separate HNSW index.
See [libSQL docs](https://turso.tech/blog/libsql-vector-search).

#### Multi-Model Support
Load different embedding models for different use cases (code, multilingual, domain-specific).

#### Export to Standard Formats
Export embeddings to Parquet for Python ecosystem compatibility (FAISS, Pinecone, Weaviate).

---

## References

- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [Dioxus](https://dioxuslabs.com/) - Rust UI framework
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin (2018)
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) - Cormack et al. (2009)
