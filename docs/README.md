# Coppermind Documentation

Technical documentation for browser-based ML inference with Rust, WASM, and WebGPU.

## Documentation Index

### 🗺️ [Roadmap & Implementation Plan](roadmap.md)
**Focus:** Development milestones and project strategy

Cross-platform implementation plan with detailed milestones. Each milestone is self-contained, passes all quality checks, and represents a fully working iteration. Includes platform strategy (Web → Desktop → Mobile), data structure definitions, and completion checklists.

**Read this when:**
- Planning next features to implement
- Understanding project goals and direction
- Checking milestone completion criteria
- Reviewing platform-specific implementation patterns
- Understanding data models (Document, Chunk, Embedding, etc.)

**Key Sections:**
- Platform Strategy (Web/Desktop/Mobile)
- UI Architecture (POC tests → Semantic search)
- Data Structures & Storage Schema
- Detailed Milestones with Completion Checklists
- Architecture Evolution
- Performance Targets
- Development Workflow

---

### 💡 [Experimental Ideas](innovation-ideas.md)
**Focus:** Experimental approaches and directions

Documents project characteristics (full Rust ML stack in browsers) and explores experimental ideas. Includes high-priority feasible ideas and speculative high-risk/high-reward concepts.

**Read this when:**
- Looking for experimental features to implement
- Understanding why Rust-first approach matters
- Exploring WebGPU, WASM threading, or other advanced features
- Planning future work

**Key Sections:**
- Project Characteristics
- Ideas to Explore (Candle WebGPU, WASM threading, etc.)
- Speculative Ideas
- Metrics for Success

---

### 📊 [Model Optimization Guide](model-optimization.md)
**Focus:** Performance tuning and memory optimization

Learn how to optimize WASM memory configuration and JinaBERT sequence length settings. Includes detailed ALiBi memory calculations, tradeoff analysis, and recommended configurations for different use cases.

**Read this when:**
- You need to increase sequence length beyond 1024 tokens
- Memory errors occur during inference
- You want to understand WASM memory limits
- Planning to support longer documents

**Key Sections:**
- WASM Memory Configuration (512MB → 4GB)
- JinaBERT Sequence Length (1024 → 8192 tokens)
- ALiBi Memory Calculations
- Recommended Configuration Presets
- Implementation Checklist

---

### 🏗️ [Browser ML Architecture](browser-ml-architecture.md)
**Focus:** System design and implementation patterns

Deep dive into the architecture for running ML models in browsers, including cross-origin isolation, Web Workers, WebGPU compute shaders, and model loading patterns.

**Read this when:**
- Implementing new ML features
- Debugging WASM or WebGPU issues
- Understanding how COOP/COEP works
- Planning parallelization strategies
- Evaluating GPU vs CPU inference

**Key Sections:**
- Cross-Origin Isolation (COOP/COEP)
- Web Workers for Parallel Processing
- WebGPU Compute Shaders
- ML Inference Architecture
- Model Loading Patterns
- Performance Characteristics

---

### 🌐 [Ecosystem & Limitations](ecosystem-and-limitations.md)
**Focus:** Community resources, alternatives, and constraints

Comprehensive coverage of the browser ML ecosystem, known limitations, alternative approaches, and learning resources. Includes comparisons with Transformers.js, ONNX Runtime, and other tools.

**Read this when:**
- Evaluating different browser ML frameworks
- Hitting WASM or browser limitations
- Looking for example implementations
- Learning from community patterns
- Planning future architecture improvements

**Key Sections:**
- Technology Stack Integration
- Known Limitations (WASM, Candle, Dioxus, WebGPU)
- Community Implementations
- Blog Posts & Resources
- Performance Comparisons
- Future Directions
- Debugging Tips

---

## Quick Reference

### Critical Constraints

**Current (Suboptimal):**
- WASM Memory: 512MB max
- Sequence Length: 1024 tokens
- Inference: CPU only

**Recommended:**
- WASM Memory: 4GB max (full wasm32 support)
- Sequence Length: 2048-4096 tokens (balanced)
- Inference: CPU (WebGPU backend planned)

### Key Files Referenced

- `.cargo/config.toml` - WASM memory limits
- `src/embedding.rs` - JinaBERT configuration
- `public/coi-serviceworker.min.js` - Cross-origin isolation
- `src/main.rs` - Service Worker loading
- `clippy.toml` - Dioxus signal safety rules

### External Resources

- [Candle Examples](https://github.com/huggingface/candle/tree/main/candle-wasm-examples)
- [JinaBERT Model Card](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)
- [V8: 4GB WASM Memory](https://v8.dev/blog/4gb-wasm-memory)
- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [Dioxus Docs](https://dioxuslabs.com/)

---

## Document Updates

These documents should be updated when:
- [ ] WASM memory configuration changes
- [ ] JinaBERT model or config changes
- [ ] New browser ML patterns discovered
- [ ] Major dependency updates (Candle, Dioxus)
- [ ] WebGPU backend implementation
- [ ] Community identifies new limitations or workarounds

---

**Last Updated:** 2025-01-11
