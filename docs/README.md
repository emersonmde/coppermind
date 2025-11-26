# Coppermind Documentation

Technical documentation for Coppermind, a local-first hybrid search engine built with Rust and WebAssembly.

## Documentation Index

### [Design Document](architecture-design.md)
**Comprehensive technical design document covering the entire system.**

**Topics:**
- Architecture overview and module structure
- Hybrid search system (HNSW vector search, BM25 keyword search, RRF fusion)
- Browser ML with Candle (JinaBERT embeddings, model loading, inference)
- Web Worker architecture for non-blocking ML inference
- Cross-platform compilation (web vs desktop)
- Storage & persistence (IndexedDB for web, redb for desktop)

**Read this:** To understand how the system works and implementation details.

---

### [Ecosystem & Resources](ecosystem-and-limitations.md)
**External resources, community implementations, and ecosystem integration.**

**Topics:**
- Candle + WASM integration patterns
- Dioxus patterns for async and asset handling
- Known limitations and workarounds
- Community implementations and examples
- Useful blog posts and tutorials

**Read this:** When evaluating alternatives, learning from community patterns, or troubleshooting browser/WASM limitations.

---

### [Roadmap](roadmap.md)
**Project status, completed features, and future directions.**

**Topics:**
- Completed features (hybrid search, crawler, GPU scheduler, persistence)
- Backlog (WebGPU, quantization, multi-model support)
- Experimental ideas and future directions

**Read this:** To understand project status and future directions.

---

## Quick Links

### Official Documentation
- [Dioxus](https://dioxuslabs.com/learn/0.6/) - UI framework docs
- [Candle](https://github.com/huggingface/candle) - ML framework repository
- [hnsw](https://github.com/rust-cv/hnsw) - HNSW vector search (incremental indexing)
- [tokenizers-rs](https://github.com/huggingface/tokenizers) - Tokenization library

### Academic Papers
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin (2018)
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) - Cormack et al. (2009)

### Web Standards
- [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) - Browser storage for web platform
- [Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
