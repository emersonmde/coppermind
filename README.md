# Coppermind

Browser-based semantic search using Rust, WASM, and local ML inference.

[![CI Status](https://github.com/emersonmde/coppermind/workflows/Dioxus%20CI/badge.svg)](https://github.com/emersonmde/coppermind/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

[**Live Demo**](https://emersonmde.github.io/coppermind) | [**Documentation**](docs/README.md)

---

## Overview

Browser-based semantic search built with Rust and WASM:
- ML inference using [Candle](https://github.com/huggingface/candle)
- UI built with [Dioxus](https://dioxuslabs.com/)
- Cross-platform support (web, desktop)
- All processing happens locally

### Technical Approach

Uses Rust for all components except the COOP/COEP service worker:
- **UI:** Dioxus
- **ML:** Candle + tokenizers-rs
- **Search:** instant-distance (HNSW) + BM25 + RRF fusion
- **Platform:** WASM (web), native (desktop)

---

## Features

**Embedding & ML:**
- 🤖 JinaBERT v2 embedding model (512-dim, supports 8192 tokens)
- 📄 File upload and text chunking
- 🔢 Embedding generation with configurable sequence lengths

**Hybrid Search:**
- 🔍 Vector search using HNSW (instant-distance) for semantic similarity
- 🔎 Keyword search using BM25 for exact term matching
- 🔀 Reciprocal Rank Fusion (RRF) to merge rankings
- 💾 Cross-platform storage (OPFS for web, filesystem for desktop)

**Platform:**
- 🖥️ Works on web and desktop
- ⚡ WebGPU compute shader support
- 👷 Web Workers for parallel processing
- 🔒 Cross-Origin Isolation via Service Worker (enables SharedArrayBuffer)

---

## Quick Start

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Dioxus CLI
cargo install dioxus-cli --locked

# Install cargo-audit (for security checks)
cargo install cargo-audit --locked
```

### Download ML Model

```bash
# Downloads JinaBERT model (~262MB) to assets/models/
./download-models.sh
```

### Run Development Server

```bash
# Web (default)
dx serve

# Desktop
dx serve --platform desktop
```

Open your browser to the URL shown (usually `http://localhost:8080`).

### Build for Production

```bash
# Web
dx bundle --release

# Desktop app
dx bundle --release --platform desktop
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Dioxus UI (Rust)                                          │
│  ├─ File Upload                                            │
│  ├─ Embedding Controls                                     │
│  └─ Results Display                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Candle ML (Rust → WASM)                                   │
│  ├─ JinaBERT Model Loading                                 │
│  ├─ Tokenization (tokenizers-rs)                           │
│  ├─ Embedding Inference (CPU or WebGPU future)             │
│  └─ Mean Pooling + L2 Normalization                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Platform Layer (Rust)                                     │
│  ├─ Web: IndexedDB (future), Web Workers, WebGPU          │
│  └─ Desktop: SQLite (future), Rayon, Native GPU (future)   │
└─────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `src/main.rs` - App entry point, COI setup
- `src/embedding.rs` - JinaBERT inference with Candle
- `src/components.rs` - Dioxus UI components
- `src/cpu.rs` - Web Worker demos
- `src/wgpu.rs` - WebGPU compute shader demos

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for detailed implementation plan.

**Completed:**
1. ✅ JinaBERT inference in WASM
2. ✅ Hybrid search system (vector + keyword + RRF fusion)
3. ✅ Cross-platform storage backend

**Next:**
1. Integrate embeddings with search index
2. Search UI for document queries
3. Multi-file batch processing
4. Index persistence across sessions

---

## Technical Details

**Rust Stack:**
- Dioxus for UI, Candle for ML inference
- Compiles to WASM for web, native for desktop
- Service worker (JavaScript) for COOP/COEP headers

**Cross-Platform:**
```rust
// Platform-specific implementations
#[cfg(target_arch = "wasm32")]
use OpfsStorage;  // Browser

#[cfg(not(target_arch = "wasm32"))]
use NativeStorage;  // Desktop
```

**Local Processing:**
- All computation happens on device
- No cloud API dependencies
- Works offline once loaded

---

## Documentation

See [docs/](docs/) for comprehensive technical documentation:

- **[Roadmap](docs/roadmap.md)** - Development plan and milestones
- **[Model Optimization](docs/model-optimization.md)** - WASM memory & sequence length tuning
- **[Browser ML Architecture](docs/browser-ml-architecture.md)** - COOP/COEP, WebGPU, Workers
- **[Ecosystem & Limitations](docs/ecosystem-and-limitations.md)** - Comparisons, resources, constraints

---

## Development

### Running Tests

```bash
cargo test --verbose
```

### Quality Checks

```bash
# Run all quality checks (what CI runs)
./.githooks/pre-commit

# Or individually:
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --verbose
cargo audit
cargo doc --no-deps
```

### Enable Pre-commit Hook

```bash
git config core.hooksPath .githooks
```

This runs all quality checks before each commit.

---

## Performance

**Current (M1 MacBook, Web):**
- Cold start: 3-7s (model download + init)
- Warm start: Instant (model cached)
- Embedding (512 tokens): ~50-200ms
- Embedding (2048 tokens): ~200-500ms (projected)

**Desktop (Projected):**
- Cold start: 1-2s
- Embedding (512 tokens): ~20-50ms
- 2-4x faster than web with native threading

---

## Future Directions

See [docs/roadmap.md](docs/roadmap.md) and [docs/innovation-ideas.md](docs/innovation-ideas.md) for planned work and experimental ideas.

---

## Tech Stack

- **UI:** [Dioxus 0.6](https://dioxuslabs.com/) - React-like Rust framework
- **ML:** [Candle 0.8](https://github.com/huggingface/candle) - Rust ML framework
- **Model:** [JinaBERT v2 Small](https://huggingface.co/jinaai/jina-embeddings-v2-small-en) - 512-dim embeddings, 8192 token support
- **Tokenizer:** [tokenizers-rs](https://github.com/huggingface/tokenizers) - HuggingFace tokenizers in Rust
- **Platform:** WASM (web), Native (desktop), Planned (mobile)

---

## License

Apache 2.0 - See [LICENSE](LICENSE) file.

### Third-Party Attributions

- **COI Service Worker:** [gzuidhof/coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) (MIT License)
- **JinaBERT Model:** [jinaai/jina-embeddings-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en) (Apache 2.0)

---

## Citation

If you use this work in research or build upon it:

```bibtex
@software{coppermind2025,
  author = {Emerson, Matthew},
  title = {Coppermind: Browser-based Semantic Search with Rust + WASM},
  year = {2025},
  url = {https://github.com/emersonmde/coppermind}
}
```

---

## Acknowledgments

- [Dioxus](https://dioxuslabs.com/) team for excellent Rust UI framework
- [Candle](https://github.com/huggingface/candle) team at HuggingFace for Rust ML framework
- [Jina AI](https://jina.ai/) for JinaBERT embedding models
- Browser vendors for pushing WASM, WebGPU, and Web APIs forward

---

**Built with 🦀 Rust**
