# Coppermind

Browser-based semantic search using Rust, WASM, and local ML inference.

[![CI Status](https://github.com/emersonmde/coppermind/workflows/Dioxus%20CI/badge.svg)](https://github.com/emersonmde/coppermind/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Live Demo](http://errorsignal.dev/coppermind)** | **[Documentation](docs/)**

---

## Overview

Local-first semantic search engine that runs entirely in your browser. Built with Rust and compiled to WebAssembly for web and native desktop platforms.

- ML inference using Candle (Rust ML framework)
- JinaBERT embeddings for semantic search
- Hybrid search: vector similarity + keyword (BM25) + fusion (RRF)
- All processing happens locally, no cloud dependencies

---

## Quick Start

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Dioxus CLI
cargo install dioxus-cli --locked
```

### Run

```bash
# Download ML model (~262MB)
./download-models.sh

# Start development server
dx serve
```

Open browser to `http://localhost:8080`

### Build

```bash
# Web
dx bundle --release

# Desktop
dx bundle --release --platform desktop
```

---

## Documentation

See [docs/](docs/) for technical details, architecture documentation, and development guides.

---

## License

MIT - See [LICENSE](LICENSE) file.
