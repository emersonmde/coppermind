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

### Evaluation Framework

Implemented in `crates/coppermind-eval/`:
- **IR Metrics**: NDCG, MAP, MRR, Precision@k, Recall@k, F1@k
- **Statistical Rigor**: Bootstrap confidence intervals, paired t-tests, Cohen's d effect size
- **Benchmarks**: Indexing throughput, search latency, hybrid rebuild time (`crates/coppermind-core/benches/`)

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

## Future Exploration

Longer-term ideas that would significantly expand Coppermind's capabilities.

### Retrieval Enhancement

#### Speculative Retrieval with Re-ranking
Two-stage retrieval: fast HNSW recall of top-k candidates, then cross-encoder re-ranking on-device with a small model. Improves precision without sacrificing latency for the initial recall phase.

#### Learned Sparse Representations (SPLADE/ColBERT)
Add learned sparse vectors alongside dense embeddings. SPLADE produces interpretable term-level weights with neural semantics, enabling hybrid retrieval that outperforms pure dense approaches.

#### Query Expansion via Pseudo-Relevance Feedback
Automatically expand queries using terms from top-k initial results (Rocchio algorithm) or LLM-generated synonyms. Classic IR technique that improves recall for ambiguous queries.

#### Hypothetical Document Embeddings (HyDE)
Generate a hypothetical answer to the query using an LLM, then embed that synthetic document to find similar real documents. Counter-intuitive technique that bridges the query-document vocabulary gap.

#### Query Understanding / Intent Classification
Route queries to different retrieval strategies based on detected intent (factual lookup vs. exploratory vs. navigational). Enables strategy-specific optimization.

### On-Device Learning

#### Contrastive Learning from Click Data
Fine-tune embedding projections using implicit feedback from which results users click. On-device personalization that improves relevance without sending data anywhere.

#### Active Learning for Relevance Feedback
Let users mark results as relevant/irrelevant, use this signal to fine-tune embedding projections or train a lightweight re-ranker. Closes the loop between retrieval and user signal.

### Index Types

#### Graph Indexes (Knowledge Graph Construction)
Automatically extract entities and relationships from crawled content to build a local knowledge graph alongside vector embeddings. Enables multi-hop reasoning queries and relationship-aware search.

#### PageRank / Link Analysis
Implement authority scoring for crawled web pages based on link structure, use as a signal in RRF fusion alongside BM25 and vector similarity.

#### Incremental Index Updates with LSM-Tree Approach
Log-structured merge for the HNSW index - new vectors go to a small "hot" index, periodically merged into the main index. Better handles frequent updates without full rebuilds.

### LLM Integration

#### MCP (Model Context Protocol) Server
Expose Coppermind's search capabilities as an MCP server, allowing LLM agents to query the local semantic index. Enables integration with agentic workflows and AI assistants.

#### Retrieval-Augmented Generation (RAG) Pipeline
Add a local LLM (via Candle or llama.cpp bindings) that synthesizes answers from retrieved chunks. Full question-answering system running entirely on-device.

#### LLM-Assisted Chunking/Categorization
Use a small local LLM to intelligently segment documents at semantic boundaries and auto-generate metadata tags. Replaces heuristic chunking with learned preprocessing.

### Scalability

#### Embedding Quantization (Binary/Product Quantization)
Compress 512D float vectors to binary or PQ codes, trading accuracy for ~32x memory reduction. Essential for scaling to large collections on memory-constrained devices.

#### WebGPU Acceleration
GPU-accelerated inference using WebGPU when available, with automatic fallback to CPU. Enables real ML workloads in-browser as WebGPU support matures.

### Privacy & Distribution

#### Federated Search Across Instances
P2P protocol allowing multiple Coppermind instances to query each other's indexes without centralizing data. Privacy-preserving distributed search.

#### Differential Privacy for Index Statistics
Add DP noise to term frequencies and other index statistics before exposing them via APIs. Provides formal privacy guarantees for shared indexes.

### Multimodal

#### Multimodal Embeddings (Images + Text)
Add CLIP-style embeddings for images found in crawled pages. Enables cross-modal queries like "find pages with diagrams similar to this screenshot."

---

## References

- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [Dioxus](https://dioxuslabs.com/) - Rust UI framework
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin (2018)
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) - Cormack et al. (2009)
