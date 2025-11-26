# Coppermind

<div align="center">

[![CI](https://github.com/emersonmde/coppermind/actions/workflows/ci.yml/badge.svg)](https://github.com/emersonmde/coppermind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![WASM](https://img.shields.io/badge/WASM-Ready-654FF0?logo=webassembly&logoColor=white)](https://webassembly.org/)

**Client-Side Hybrid Search Engine.**  
*Built with Rust. Runs everywhere.*

[Live Demo](https://errorsignal.dev/coppermind/)

</div>

---

**Coppermind** is a client-side hybrid search engine that runs entirely on your device. It combines the semantic understanding of **Vector Search** with the precision of **Keyword Search** (BM25), fused using **Reciprocal Rank Fusion**.

Unlike traditional search engines that rely on heavy server-side infrastructure, Coppermind compiles to a single, high-performance binary that runs in your browser (WASM), on your desktop, or on your phone. It brings transformer-based embeddings (JinaBERT) to the edge, ensuring your data never leaves your machine.

## Features

- **Rust-First**: UI (Dioxus), ML inference (Candle), search algorithms, and storage - all written in Rust and compiled to WASM or native code
- **Hybrid Search**: Combines semantic similarity (vector search) with keyword matching (BM25) using Reciprocal Rank Fusion
- **Browser ML**: Runs [JinaBERT](https://huggingface.co/jinaai/jina-embeddings-v2-small-en) embeddings client-side with [Candle](https://github.com/huggingface/candle)
- **Cross-Platform**: Single Rust codebase targets web (WASM), desktop (macOS/Linux/Windows), and iOS
- **Platform-Specific Features**:
  - **Web**: Background embedding via Web Workers, OPFS storage
  - **Desktop**: Web crawler with parallel requests, native GPU acceleration (Metal/CUDA)
- **Fully Local**: All processing happens on your device - no cloud APIs, works offline

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
dx serve -p coppermind
# Opens http://localhost:8080

# Desktop platform (native app with web crawler)
dx serve -p coppermind --platform desktop

# iOS platform (experimental)
dx serve -p coppermind --platform ios

# Production build (web)
dx bundle -p coppermind --release

# Production build (desktop)
dx bundle -p coppermind --release --platform desktop
```

### Running macOS Desktop App


**Quarantine:**

1. Open the DMG and drag **Coppermind** to your Applications folder
2. Open Terminal and run:
   ```bash
   sudo xattr -rd com.apple.quarantine /Applications/Coppermind.app
   ```
3. Launch Coppermind from Applications

## Platform Comparison

| Feature | Web | Desktop                    | iOS                        |
|---------|-----|----------------------------|----------------------------|
| **Hybrid Search** (Vector + BM25) | ✅ | ✅                          | ✅                          |
| **Local Embedding** (JinaBERT) | ✅ | ✅                          | ✅                          |
| **Web Worker** (background ML) | ✅ | N/A                        | N/A                        |
| **Web Crawler** | ❌ (CORS) | ✅                          | ❌                          |
| **GPU Acceleration** | ❌ | ✅ (Metal/CUDA)             | ⚠️ (CPU via Accelerate)    |
| **Storage** | OPFS | Native FS                  | Native FS                  |
| **Text Chunking** | Markdown + Sentence | Markdown + Code + Sentence | Markdown + Code + Sentence |

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

### Web Crawler (Desktop Only)
- **[scraper 0.22](https://github.com/causal-agent/scraper)** - HTML parsing (CSS selector-based extraction)
- **[reqwest 0.12](https://github.com/seanmonstar/reqwest)** - HTTP client (async fetching with TLS)
- **BFS traversal** - Crawl strategy (breadth-first with cycle detection, same-origin filtering)

### Text Processing
- **[text-splitter 0.18](https://github.com/benbrandt/text-splitter)** - Smart chunking (ICU4X sentence segmentation)
- **[pulldown-cmark 0.12](https://github.com/pulldown-cmark/pulldown-cmark)** - Markdown parsing (structure-aware chunking)
- **[tree-sitter](https://tree-sitter.github.io/)** - Code parsing (syntax-aware chunking, desktop/iOS only)

### Storage & Serialization
- **OPFS** - Web storage (Origin Private File System for binary data)
- **tokio::fs** - Desktop/iOS storage (async filesystem operations)
- **[bincode 1.3](https://github.com/bincode-org/bincode)** - Binary serialization (compact index persistence)

## How It Works

### Hybrid Search Architecture

Coppermind combines two complementary search approaches. **Vector search** uses [instant-distance](https://github.com/instant-labs/instant-distance)'s HNSW (Hierarchical Navigable Small World) implementation to find documents semantically similar to your query - catching paraphrases, synonyms, and conceptual matches that keyword search would miss. **Keyword search** uses the [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm (the same ranking function used by Elasticsearch and Lucene) to find exact keyword matches, ensuring precise terms aren't buried by semantic noise. These two result sets are merged using [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (RRF), a rank-based fusion algorithm that operates purely on document positions. This means you can combine vector similarity scores (0-1 range) with BM25 scores (unbounded) without normalizing their scales - RRF simply ranks documents by their positions in each list.

### Browser ML with Candle

Traditional browser-based ML uses JavaScript frameworks like TensorFlow.js or ONNX Runtime Web, but Coppermind takes a different approach: [Candle](https://github.com/huggingface/candle), a minimalist Rust ML framework from Hugging Face. The JinaBERT embedding model (safetensors format) is loaded directly in the browser, tokenized with [tokenizers-rs](https://github.com/huggingface/tokenizers) (the same Rust tokenizer library used by Hugging Face Transformers), and executed in WebAssembly. This delivers near-native performance without sending data to external APIs, keeping documents private and enabling offline operation. The model outputs 512-dimensional embeddings with a maximum sequence length of 2048 tokens (configurable up to 8192), balancing context window and memory usage.

### Web Crawler (Desktop Only)

The desktop version includes a built-in web crawler for indexing documentation sites and web content. CORS (Cross-Origin Resource Sharing) restrictions prevent web browsers from fetching arbitrary URLs, so this feature is only available in native desktop builds. The crawler implements BFS (breadth-first search) traversal with cycle detection (deduplicates URLs by normalizing trailing slashes), same-origin filtering (restricts crawling to the starting domain and path), and configurable parallel requests (1-16 concurrent fetches). HTML parsing uses [scraper](https://github.com/causal-agent/scraper) (built on html5ever) with CSS selectors to extract visible text while filtering scripts and styles. Crawled pages are automatically chunked and indexed for semantic search.

### Text Chunking Strategies

Documents are split into semantically meaningful chunks before embedding to improve search relevance and reduce token counts. The chunking strategy is selected automatically based on file type:

- **Markdown** ([pulldown-cmark](https://github.com/pulldown-cmark/pulldown-cmark)): Structure-aware splitting that preserves headings, lists, and code blocks
- **Code** ([tree-sitter](https://tree-sitter.github.io/), desktop/iOS only): Syntax-aware splitting that respects function boundaries, classes, and module structure (supports Rust, Python, JavaScript, Java, C/C++, Go)
- **Text** ([text-splitter](https://github.com/benbrandt/text-splitter)): ICU4X sentence segmentation for natural language

On web (WASM), code files fall back to text chunking since tree-sitter requires native code compilation. Desktop and iOS get full syntax-aware chunking.

### Why Dioxus

[Dioxus](https://dioxuslabs.com/) is a React-like UI framework for Rust that provides a familiar component model with hooks, props, and reactive state management - but compiles to native code instead of running in a JavaScript runtime. Unlike JavaScript frameworks that require bundlers, transpilers, and runtime overhead, Dioxus components are statically typed at compile time and generate optimized WASM or native binaries with zero-cost abstractions. The framework handles platform-specific rendering transparently: DOM manipulation on web, native windows on desktop, and mobile views on iOS/Android. This means you write UI logic once with full Rust type safety, and Dioxus handles the cross-platform rendering.

### Cross-Platform Compilation

Platform-specific features are handled with conditional compilation (`#[cfg(target_arch = "wasm32")]`, `#[cfg(feature = "desktop")]`). The Rust compiler generates optimized code for each target:

- **Web (WASM)**: Compact binaries, CPU-only inference, Web Worker for background embedding
- **Desktop**: Native executables, full GPU acceleration (Metal on macOS, CUDA on Linux/Windows), web crawler support
- **iOS**: Native app, CPU-only inference (Accelerate framework), no web crawler (could be added but requires handling app sandboxing)

Storage, asset loading, and threading are also platform-specific: OPFS/HTTP fetch/Web Workers on web vs. native filesystem/direct file access/native threads on desktop/iOS.

### Storage & Persistence (WIP)

Documents and search indexes are persisted locally using platform-appropriate backends: [OPFS (Origin Private File System)](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) on web, native filesystem on desktop/iOS. OPFS is a modern browser API designed for high-performance binary data storage, offering 3-4x faster read/write than IndexedDB and supported by all modern browsers (Chrome, Firefox, Safari, Edge) in standard browsing mode. The vector index (HNSW graph), keyword index (BM25 term frequencies), and document metadata are serialized with [bincode](https://github.com/bincode-org/bincode) for efficient binary storage, enabling the search engine to persist across sessions without rebuilding indexes.

## License

MIT License - see [LICENSE](LICENSE) for details.
