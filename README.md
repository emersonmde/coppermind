# Coppermind

[![CI](https://github.com/emersonmde/coppermind/actions/workflows/ci.yml/badge.svg)](https://github.com/emersonmde/coppermind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![Made with Rust](https://img.shields.io/badge/Made%20with-Rust-1f425f.svg)](https://www.rust-lang.org/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-654FF0?logo=webassembly&logoColor=white)](https://webassembly.org/)

A cross-platform hybrid search engine written in Rust with local embedding, indexing, and retrieval.

**Try it here:** https://errorsignal.dev/coppermind/

## Features

- **Rust-First**: UI (Dioxus), ML inference (Candle), search algorithms, and storage, all written in Rust and compiled to WASM
- **Hybrid Search**: Combines semantic similarity (vector search) with keyword matching (BM25) using Reciprocal Rank Fusion
- **Browser ML**: Runs [JinaBERT](https://huggingface.co/jinaai/jina-embeddings-v2-small-en) 512-dimensional embeddings client-side with [Candle](https://github.com/huggingface/candle)
- **Cross-Platform**: Single Rust codebase compiles to web (WASM) and native desktop apps
- **Fully Local**: All processing happens on your device-no cloud APIs, works offline

## Getting Started

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Dioxus CLI
cargo install dioxus-cli --locked
```

### Clone & Setup

```bash
# Clone repository
git clone https://github.com/emersonmde/coppermind.git
cd coppermind

# Download ML models (~262MB model + 695KB tokenizer)
./download-models.sh
```

### Running Locally

```bash
# Web platform (development server)
dx serve
# Opens http://localhost:8080

# Desktop platform (native app)
dx serve --platform desktop
# Note: Desktop build is functional but not yet optimized

# Production build (web)
dx bundle --release
```

## Tech Stack

### Core Framework
- **[Dioxus 0.7](https://dioxuslabs.com/)** - Reactive UI framework (React-like component model, cross-platform rendering)
- **[Candle 0.8](https://github.com/huggingface/candle)** - ML inference framework (Hugging Face Rust ML library)

### Machine Learning
- **[JinaBERT-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)** - Embedding model (512-dimensional, ALiBi positional embeddings)
- **[tokenizers-rs 0.20](https://github.com/huggingface/tokenizers)** - Tokenization (Hugging Face Transformers tokenizer in Rust)

### Search Infrastructure
- **[instant-distance 0.6](https://github.com/instant-labs/instant-distance)** - Vector search (HNSW approximate nearest neighbor, rayon parallel indexing)
- **[bm25 2.3](https://github.com/Michael-JB/bm25)** - Keyword search (Okapi BM25 ranking with TF-IDF)
- **Reciprocal Rank Fusion** - Result fusion (rank-based merging of vector and keyword results)

### Storage & Serialization
- **OPFS** - Web storage (Origin Private File System for binary data on web)
- **tokio::fs** - Desktop storage (Async filesystem operations)
- **[bincode 1.3](https://github.com/bincode-org/bincode)** - Binary serialization (Compact index persistence)

### Browser Integration
- **Web Workers** - Background processing (ML inference off main thread)
- **Service Worker** - Cross-origin isolation (COEP/COIP headers for SharedArrayBuffer)

## How It Works

### Hybrid Search Architecture

Coppermind combines two complementary search approaches to deliver better results than either method alone. **Vector search** uses [instant-distance](https://github.com/instant-labs/instant-distance)'s HNSW (Hierarchical Navigable Small World) implementation to find documents semantically similar to your query-catching paraphrases, synonyms, and conceptual matches that keyword search would miss. **Keyword search** uses the [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm (the same ranking function used by Elasticsearch and Lucene) to find exact keyword matches, ensuring precise terms aren't buried by semantic noise. These two result sets are merged using [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (RRF), a rank-based fusion algorithm that operates purely on document positions rather than raw scores. This means you can combine vector similarity scores (0-1 range) with BM25 scores (unbounded) without normalizing their different scales-RRF simply ranks documents by their positions in each list, giving you the best of both semantic understanding and exact matching.

### Browser ML with Candle

Traditional browser-based ML uses JavaScript frameworks like TensorFlow.js or ONNX Runtime Web, but Coppermind takes a different approach: [Candle](https://github.com/huggingface/candle), a minimalist Rust ML framework from Hugging Face. The JinaBERT embedding model (safetensors format) is loaded directly in the browser, tokenized with [tokenizers-rs](https://github.com/huggingface/tokenizers) (the same Rust tokenizer library used by Hugging Face Transformers), and executed in WebAssembly. This delivers near-native performance without sending data to external APIs, keeping your documents private and enabling offline operation once the model is cached. The model outputs 512-dimensional embeddings with a maximum sequence length of 2048 tokens (configurable up to 8192), striking a balance between context window and memory usage in the browser's WASM heap.

### Cross-Origin Isolation & Web Workers

Running ML in the browser requires careful architectural decisions to prevent UI freezing and enable parallelization. Coppermind uses a [Cross-Origin Isolation (COI) Service Worker](https://web.dev/coop-coep/) to inject `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers, which enable `SharedArrayBuffer` support-a requirement for multi-threaded WASM execution. This benefits the HNSW vector search implementation in [instant-distance](https://github.com/instant-labs/instant-distance), which uses [rayon](https://github.com/rayon-rs/rayon) for parallel graph construction and search operations. The embedding computation runs in a dedicated [Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API), offloading CPU-intensive JinaBERT inference to a background thread via message passing with the WASM binary. This architecture keeps the main thread responsive for UI interactions, and the COI setup enables multi-threaded WASM execution across CPU cores for search operations.

### Why Dioxus

[Dioxus](https://dioxuslabs.com/) is a React-like UI framework for Rust that provides a familiar component model with hooks, props, and reactive state management-but compiles to native code instead of running in a JavaScript runtime. Unlike JavaScript frameworks that require bundlers, transpilers, and runtime overhead, Dioxus components are statically typed at compile time and generate optimized WASM or native binaries with zero-cost abstractions. The framework handles platform-specific rendering transparently: DOM manipulation on web, native windows on desktop, and mobile views on iOS/Android. This means you write your UI logic once with full Rust type safety, and Dioxus handles the cross-platform rendering details.

### Cross-Platform Compilation

Platform-specific features-like storage (OPFS on web vs. tokio::fs on desktop), asset loading (HTTP fetch vs. filesystem), and threading (Web Workers vs. native threads)-are handled with conditional compilation (`#[cfg(target_arch = "wasm32")]`). The Rust compiler generates optimized code for each target: compact WASM binaries for browsers, and native executables for desktop that can leverage full GPU acceleration via Candle's CUDA/Metal backends.

### Storage & Persistence

Documents and search indexes are persisted locally using platform-appropriate storage backends: [OPFS (Origin Private File System)](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) on web and native filesystem access on desktop. OPFS is a modern browser API designed for high-performance binary data storage, offering 3-4x faster read/write operations than IndexedDB and supported by all modern browsers (Chrome, Firefox, Safari, Edge) in standard browsing mode. The vector index (HNSW graph), keyword index (BM25 term frequencies), and document metadata are serialized with [bincode](https://github.com/bincode-org/bincode) for efficient binary storage, enabling the search engine to persist across sessions without rebuilding indexes on every page load.

## License

MIT License - see [LICENSE](LICENSE) for details.
